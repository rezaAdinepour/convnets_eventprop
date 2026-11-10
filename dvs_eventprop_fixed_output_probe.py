"""TASK 10: fixed-spike, frozen-weight output-gradient isolation.

Synthetic input only. No DVS arrays, checkpoints or training updates.
"""

import argparse
from pathlib import Path

import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import Callback
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.neurons import SpikeInput, LeakyIntegrate
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import PreprocessedSpikes

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_output_gradient_probe as previous
from dvs_data import file_hash, require
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_probe import emit, pull


class Capture(Callback):
    def __init__(self, output):
        self.output = output
        # Mutable container survives CallbackList's shallow copy.
        self.results = []

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_end(self, cn, batch, metric_state):
        if batch == 0:
            return
        require(batch == 1, "Unexpected backward batch")
        pop = cn.neuron_populations[self.output.population()]
        np.testing.assert_array_equal(
            pull(pop.vars["YTrueBack"]).reshape(-1), [0]
        )
        group = cn.connection_populations[self.output.connection()]
        gradient = pull(group.vars["weightGradient"]).astype(np.float64)
        require(np.isfinite(gradient).all(), "Nonfinite gradient")
        self.results.append(gradient)
        emit("FIXED_CAPTURE", {
            "backward_batch": batch,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "optimizer_updates": 0,
        })
        # No GradientLearn custom update is invoked.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt-ms", type=float, choices=(1.0, 0.5, 0.25), default=1.0
    )
    args = parser.parse_args()
    steps = int(128.0 / args.dt_ms)

    payload = PreprocessedSpikes(
        np.array([3, 5], dtype=np.uint32),
        np.array([8, 32, 96, 16, 64], dtype=np.float32),
    )
    empty = PreprocessedSpikes(
        np.zeros(2, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    )
    expected_spikes = [(0, 8.0), (0, 32.0), (0, 96.0),
                       (1, 16.0), (1, 64.0)]
    label = np.array([0], dtype=np.int64)
    weight = np.array([
        [.2, -.1, .3, -.2, .1, 0., -.3, .15, -.15, .25, -.25],
        [-.1, .3, -.2, .2, 0., -.15, .1, -.25, .25, -.05, .05],
    ], dtype=np.float32)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "helper_source_sha256": file_hash(Path(previous.__file__)),
    })
    emit("FIXED_CONFIG", {
        "architecture": "SpikeInput(2)->Dense(11)->LI",
        "dt_ms": args.dt_ms,
        "simulation_timesteps": steps,
        "physical_horizon_ms": 128.0,
        "tau_mem_ms": 20.0,
        "tau_syn_ms": 5.0,
        "readout": "avg_var_exp_weight",
        "loss": "sparse_categorical_crossentropy",
        "tape_capacity": 2 * steps + 1,
        "strict_buffer_checking": True,
        "scheduled_spikes": expected_spikes,
        "initial_weights": weight.tolist(),
        "optimizer_updates": 0,
    })

    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=5), shape=(2,),
            name="fixed_input", record_spikes=True,
        )
        out = Layer(
            Dense(weight.copy(), delay=0),
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=11, synapse=Exponential(tau=5.0),
            name="fixed_output",
        )

    cn = EventPropCompiler(
        example_timesteps=steps,
        losses="sparse_categorical_crossentropy",
        dt=args.dt_ms,
        batch_size=1,
        rng_seed=42,
        max_spikes=2 * steps + 1,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        grad_limit=100.0,
        softmax_temperature=1.0,
    ).compile(
        network, name="FixedOutputGradient",
        optimisers={out: {"weight": Adam(alpha=0.001)}},
        regularisers={},
    )

    matches = [
        i for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    capture = Capture(out)
    cn.base_train_callbacks[matches[0]] = capture

    with cn:
        initial = checkpoint.training_state(cn, [out])
        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks, compiled_network=cn,
            num_batches=2, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        warmup = None
        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
            cn._train_batch(
                batch, 0, {inp: [sample]}, {out: label},
                {out: metric}, state, callbacks,
            )
            if batch == 0:
                warmup = np.array(cn.get_readout(out), copy=True)[0]
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)

        require(state[out].total == 1, "Drain entered metrics")
        require(len(capture.results) == 1, "Wrong capture count")
        for old, new in zip(initial, checkpoint.training_state(cn, [out])):
            np.testing.assert_array_equal(old, new)

        # Reuse the checked forward helper; record inp in its recorder slot.
        baseline, spikes = previous.forward(
            cn, inp, inp, out, payload, label
        )
        require(spikes == expected_spikes, "Fixed input delivery mismatch")
        np.testing.assert_allclose(
            baseline, warmup, rtol=1e-5, atol=1e-6
        )

        gradient = capture.results[0]
        norm = float(np.linalg.norm(gradient))
        require(norm > 1e-8, "Gradient too small")
        direction = gradient / norm
        var = cn.connection_populations[out.connection()].vars["weight"]
        original = pull(var)
        require(original.shape == gradient.shape, "Storage shape mismatch")
        np.testing.assert_array_equal(
            original.reshape(2, 11), weight
        )

        probability = np.exp(baseline - baseline.max())
        probability /= probability.sum()
        probability[0] -= 1.0

        # Independent continuous-time solution:
        # instantaneous current jumps at the scheduled spike times,
        # zero initial states, no transmission delay, full 128 ms readout.
        horizon = 128.0
        tau_mem = 20.0
        tau_syn = 5.0

        sensitivity = np.zeros(2, dtype=np.float64)
        rate_mem = 1.0 / horizon + 1.0 / tau_mem
        rate_syn = 1.0 / horizon + 1.0 / tau_syn

        for neuron, time_ms in expected_spikes:
            remaining = horizon - time_ms
            integral_mem = -np.expm1(-rate_mem * remaining) / rate_mem
            integral_syn = -np.expm1(-rate_syn * remaining) / rate_syn
            sensitivity[neuron] += (
                tau_syn / (tau_mem - tau_syn)
                * np.exp(-time_ms / horizon)
                / horizon
                * (integral_mem - integral_syn)
            )

        continuous_scores = sensitivity @ weight.astype(np.float64)
        continuous_probability = np.exp(
            continuous_scores - continuous_scores.max()
        )
        continuous_probability /= continuous_probability.sum()
        continuous_probability[0] -= 1.0

        # True gradient of the continuous-time model's own CE.
        continuous_gradient = (
            sensitivity[:, None] * continuous_probability[None, :]
        )

        # Diagnostic only: exact continuous response sensitivity with the
        # compiler's grid-forward softmax drive. This separates forward
        # score error from backward response/integration error.
        grid_drive_reference = (
            sensitivity[:, None] * probability[None, :]
        )

        actual_gradient = gradient.reshape(2, 11)
        unit_direction = direction.reshape(2, 11)

        def compare_reference(reference):
            reference_norm = float(np.linalg.norm(reference))
            actual_norm = float(np.linalg.norm(actual_gradient))
            return {
                "reference_l2": reference_norm,
                "relative_l2_error": float(
                    np.linalg.norm(actual_gradient - reference)
                    / max(reference_norm, 1e-12)
                ),
                "max_absolute_error": float(
                    np.max(np.abs(actual_gradient - reference))
                ),
                "cosine_similarity": float(
                    np.sum(actual_gradient * reference)
                    / max(actual_norm * reference_norm, 1e-12)
                ),
            }

        analytic_direction = float(
            np.sum(continuous_gradient * unit_direction)
        )
        score_direction = sensitivity @ unit_direction
        analytic_epsilon = 0.001
        analytic_fd = (
            previous.ce(
                continuous_scores + analytic_epsilon * score_direction, 0
            )
            - previous.ce(
                continuous_scores - analytic_epsilon * score_direction, 0
            )
        ) / (2.0 * analytic_epsilon)

        np.testing.assert_allclose(
            analytic_fd, analytic_direction, rtol=1e-7, atol=1e-9,
            err_msg="Analytical reference CE derivative inconsistency",
        )

        emit("CONTINUOUS_REFERENCE", {
            "dt_ms": args.dt_ms,
            "sensitivity_per_input": sensitivity.tolist(),
            "continuous_scores": continuous_scores.tolist(),
            "compiled_scores": baseline.tolist(),
            "score_max_absolute_error": float(
                np.max(np.abs(baseline - continuous_scores))
            ),
            "captured_gradient": actual_gradient.tolist(),
            "continuous_ce_gradient": continuous_gradient.tolist(),
            "continuous_ce_comparison": compare_reference(
                continuous_gradient
            ),
            "grid_softmax_drive_comparison": compare_reference(
                grid_drive_reference
            ),
            "continuous_directional_derivative": analytic_direction,
            "continuous_ce_finite_difference": analytic_fd,
            "reference_derivative_self_check": "pass",
            "scope": "continuous-time linear output model only",
        })

        def set_weight(value):
            var.view[...] = np.asarray(value, dtype=np.float32)
            var.push_to_device()

        results = []
        try:
            for epsilon in (0.01, 0.003, 0.001):
                scores = []
                for sign in (1.0, -1.0):
                    set_weight(original.astype(np.float64)
                               + sign * epsilon * direction)
                    value, spikes = previous.forward(
                        cn, inp, inp, out, payload, label
                    )
                    require(spikes == expected_spikes,
                            "Fixed spike schedule changed")
                    scores.append(value)

                plus, minus = scores
                np.testing.assert_allclose(
                    (plus + minus) / 2, baseline,
                    rtol=1e-5, atol=1e-6,
                )
                fd = (
                    previous.ce(plus, 0) - previous.ce(minus, 0)
                ) / (2 * epsilon)
                chain = float(
                    probability @ ((plus - minus) / (2 * epsilon))
                )
                scale = max(abs(norm), abs(fd), 1e-12)
                result = {
                    "epsilon": epsilon,
                    "eventprop_derivative": norm,
                    "forward_ce_finite_difference": fd,
                    "forward_score_chain_rule": chain,
                    "relative_error": abs(norm - fd) / scale,
                    "within_1pct": bool(
                        abs(norm - fd) <= 0.01 * scale + 1e-6
                    ),
                }
                results.append(result)
                emit("FIXED_DIRECTION", result)
        finally:
            set_weight(original)

        for old, new in zip(initial, checkpoint.training_state(cn, [out])):
            np.testing.assert_array_equal(old, new)

        derivatives = [
            r["forward_ce_finite_difference"] for r in results
        ]
        plateau = (
            max(derivatives) - min(derivatives)
            <= 0.005 * max(abs(v) for v in derivatives) + 1e-6
        )
        emit("FIXED_DIAGNOSTIC", {
            "dt_ms": args.dt_ms,
            "structural_checks": "pass",
            "fixed_input_spikes": 5,
            "optimizer_updates": 0,
            "weights_and_moments_unchanged": True,
            "finite_difference_plateau": bool(plateau),
            "all_within_1pct": all(r["within_1pct"] for r in results),
            "scope": "synthetic output path only",
        })


if __name__ == "__main__":
    main()