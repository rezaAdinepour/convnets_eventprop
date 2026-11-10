"""TASK 10: single-spike hidden LIF EventProp gradient diagnostic.

Synthetic input only. One forward example and empty backward drain.
Frozen weights, monitored limiter, no checkpoints or dataset access.
"""

import argparse
from pathlib import Path

import numpy as np

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.connectivity import Dense
from ml_genn.neurons import SpikeInput, LeakyIntegrateFire, LeakyIntegrate
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import PreprocessedSpikes

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_lif_reference as reference
import dvs_eventprop_limiter_probe as audit
import dvs_eventprop_output_gradient_probe as output_probe
from dvs_data import file_hash, require
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_probe import emit, pull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dt-ms", type=float,
        choices=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.0078125),
        default=1.0,
    )
    args = parser.parse_args()
    dt = args.dt_ms
    steps = int(reference.HORIZON_MS / dt)
    require(steps * dt == reference.HORIZON_MS, "Horizon mismatch")

    exact = reference.reference(reference.INPUT_WEIGHT)
    input_weight = np.array([[reference.INPUT_WEIGHT]], np.float32)
    output_weight = reference.OUTPUT_WEIGHTS.reshape(1, 11).copy()
    payload = PreprocessedSpikes(
        np.array([1], dtype=np.uint32),
        np.array([reference.INPUT_TIME_MS], dtype=np.float32),
    )
    empty = PreprocessedSpikes(
        np.zeros(1, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    )
    label = np.array([0], dtype=np.int64)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "reference_source_sha256": file_hash(Path(reference.__file__)),
        "limiter_source_sha256": file_hash(Path(audit.__file__)),
        "spike_helper_source_sha256": file_hash(
            Path(output_probe.__file__)
        ),
    })
    emit("LIF_CONFIG", {
        "architecture": "SpikeInput(1)->Dense->LIF(1)->Dense->LI(11)",
        "dt_ms": dt,
        "simulation_timesteps": steps,
        "physical_horizon_ms": reference.HORIZON_MS,
        "input_time_ms": reference.INPUT_TIME_MS,
        "input_weight": input_weight.tolist(),
        "output_weight": output_weight.tolist(),
        "tau_mem_ms": reference.TAU_MEM_MS,
        "tau_syn_ms": reference.TAU_SYN_MS,
        "threshold": 1.0,
        "absolute_reset": 0.0,
        "delay": 0,
        "readout": "avg_var_exp_weight",
        "loss": "sparse_categorical_crossentropy",
        "tape_capacity": 2 * steps + 1,
        "strict_buffer_checking": True,
        "grad_limit": 100.0,
        "optimizer_updates": 0,
    })

    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=1), shape=(1,),
            name="single_input", record_spikes=True,
        )
        hid = Layer(
            Dense(input_weight.copy(), delay=0),
            LeakyIntegrateFire(
                v_thresh=1.0, v_reset=0.0, v=0.0,
                tau_mem=20.0, relative_reset=False,
            ),
            shape=(1,), synapse=Exponential(tau=5.0),
            name="single_hidden", record_spikes=True,
        )
        out = Layer(
            Dense(output_weight.copy(), delay=0),
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=(11,), synapse=Exponential(tau=5.0),
            name="single_output",
        )

    compiler = audit.AuditCompiler(
        example_timesteps=steps,
        losses="sparse_categorical_crossentropy",
        dt=dt,
        batch_size=1,
        rng_seed=42,
        max_spikes=2 * steps + 1,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network, name="SingleLIFGradient",
        optimisers={
            hid: {"weight": Adam(alpha=0.001)},
            out: {"weight": Adam(alpha=0.001)},
        },
        regularisers={},
    )
    require(len(compiler.instrumentation) == 1,
            "Expected one instrumented hidden population")
    emit("INSTRUMENTATION", compiler.instrumentation)

    matches = [
        i for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    layers = [hid, out]
    capture = audit.CaptureBoth(layers, out, 0)
    cn.base_train_callbacks[matches[0]] = capture

    with cn:
        for layer, expected in zip(
            layers, (input_weight, output_weight)
        ):
            var = cn.connection_populations[layer.connection()].vars["weight"]
            np.testing.assert_array_equal(
                pull(var).reshape(expected.shape), expected
            )

        initial_state = checkpoint.training_state(cn, layers)
        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks + [
                SpikeRecorder(inp, key="input"),
                SpikeRecorder(hid, key="hidden"),
            ],
            compiled_network=cn, num_batches=2, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        scores = None
        replay_bound = None
        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
            if batch == 1:
                replay_bound = pull(
                    cn.neuron_populations[hid.population()]
                    .vars["LambdavLimit"]
                )
            cn._train_batch(
                batch, 0, {inp: [sample]}, {out: label},
                {out: metric}, state, callbacks,
            )
            input_pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(
                pull(input_pop.vars["StartSpike"]),
                pull(input_pop.vars["EndSpike"]),
            )
            if batch == 0:
                scores = np.array(cn.get_readout(out), copy=True)[0]

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1, "Drain entered real metrics")
        require(len(capture.results) == 1, "Wrong capture count")
        require(scores.shape == (11,) and np.isfinite(scores).all(),
                "Invalid scores")

        data = callbacks.get_data()
        require(
            output_probe.spike_pairs(data["input"], 0)
            == [(0, reference.INPUT_TIME_MS)],
            "Input delivery mismatch",
        )
        require(output_probe.spike_pairs(data["input"], 1) == [],
                "Input spikes in drain")
        hidden_spikes = output_probe.spike_pairs(data["hidden"], 0)
        require(len(hidden_spikes) == 1,
                f"Expected one hidden spike; observed {hidden_spikes}")
        require(output_probe.spike_pairs(data["hidden"], 1) == [],
                "Hidden spikes in drain")

        pop = cn.neuron_populations[hid.population()]
        measurements = {
            name: pull(pop.vars[name]).reshape(-1)[0].item()
            for name in audit.COUNTERS + audit.SCALARS
        }
        emit("LIF_LIMITER", {
            **measurements,
            "bound_before_replay": replay_bound.tolist(),
        })
        require(measurements["AuditJumps"] == 1,
                "Expected one replayed hidden jump")
        require(measurements["AuditClips"] == 0,
                "Clipping prevents unconstrained reference comparison")
        require(measurements["AuditNonfinite"] == 0,
                "Nonfinite hidden jump")
        require(measurements["AuditBadLimit"] == 0,
                "Invalid limiter bound")

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        hidden_gradient = float(capture.results[0][0].reshape(-1)[0])
        output_gradient = capture.results[0][1].reshape(11).astype(np.float64)

    expected_hidden = exact["hidden_weight_gradient"]
    expected_output = exact["output_weight_gradient"]
    hidden_scale = max(
        abs(hidden_gradient), abs(expected_hidden), 1e-12
    )
    output_error = float(
        np.linalg.norm(output_gradient - expected_output)
        / np.linalg.norm(expected_output)
    )
    hidden_within = (
        abs(hidden_gradient - expected_hidden)
        <= 0.01 * hidden_scale + 1e-6
    )
    output_within = bool(np.all(
        np.abs(output_gradient - expected_output)
        <= 0.01 * np.abs(expected_output) + 1e-6
    ))

    emit("LIF_GRADIENT_COMPARISON", {
        "dt_ms": dt,
        "continuous_hidden_spike_time_ms": exact["hidden_spike_time_ms"],
        "genn_hidden_spike_time_ms": hidden_spikes[0][1],
        "continuous_crossing_slope_per_ms":
            exact["crossing_slope_per_ms"],
        "continuous_hidden_weight_gradient": expected_hidden,
        "genn_hidden_weight_gradient": hidden_gradient,
        "hidden_gradient_same_sign": bool(
            np.sign(hidden_gradient) == np.sign(expected_hidden)
        ),
        "hidden_gradient_relative_error":
            abs(hidden_gradient - expected_hidden) / hidden_scale,
        "hidden_gradient_within_1pct_plus_1e6_absolute":
            bool(hidden_within),
        "continuous_output_weight_gradient": expected_output.tolist(),
        "genn_output_weight_gradient": output_gradient.tolist(),
        "output_gradient_relative_l2_error": output_error,
        "output_gradient_within_1pct_plus_1e6_absolute": output_within,
        "continuous_scores": exact["scores"].tolist(),
        "genn_scores": scores.tolist(),
        "score_max_absolute_difference": float(
            np.max(np.abs(scores - exact["scores"]))
        ),
    })
    emit("LIF_DIAGNOSTIC", {
        "structural_checks": "pass",
        "hidden_spikes": 1,
        "replayed_hidden_jumps": 1,
        "clipped_jumps": 0,
        "optimizer_updates": 0,
        "weights_and_moments_unchanged": True,
        "numerical_criteria_passed": bool(hidden_within and output_within),
        "scope": "one synthetic hidden-LIF example",
    })


if __name__ == "__main__":
    main()