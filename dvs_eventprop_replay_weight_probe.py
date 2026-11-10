"""Observe weight-state dependence of single-LIF backward replay.

Synthetic input only; no optimizer updates or dataset access.
The scale-2 condition deliberately changes output weights between
forward simulation and the empty backward drain.
"""

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


def run(scale):
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

    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=1), shape=(1,),
            name="input", record_spikes=True,
        )
        hid = Layer(
            Dense(input_weight.copy(), delay=0),
            LeakyIntegrateFire(
                v_thresh=1.0, v_reset=0.0, v=0.0,
                tau_mem=20.0, relative_reset=False,
            ),
            shape=(1,), synapse=Exponential(tau=5.0),
            name="hidden", record_spikes=True,
        )
        out = Layer(
            Dense(output_weight.copy(), delay=0),
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=(11,), synapse=Exponential(tau=5.0),
            name="output",
        )

    compiler = audit.AuditCompiler(
        example_timesteps=128,
        losses="sparse_categorical_crossentropy",
        dt=1.0, batch_size=1, rng_seed=42,
        max_spikes=257, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network, name=f"ReplayWeightScale{scale}",
        optimisers={
            hid: {"weight": Adam(alpha=0.001)},
            out: {"weight": Adam(alpha=0.001)},
        },
        regularisers={},
    )
    require(len(compiler.instrumentation) == 1,
            "Expected one instrumented hidden population")

    matches = [
        i for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    layers = [hid, out]
    capture = audit.CaptureBoth(layers, out, 0)
    cn.base_train_callbacks[matches[0]] = capture

    with cn:
        initial_state = checkpoint.training_state(cn, layers)
        for actual, expected in zip(
            initial_state[:2], (input_weight, output_weight)
        ):
            np.testing.assert_array_equal(
                actual.reshape(expected.shape), expected
            )

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

        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
            if batch == 1:
                bound = pull(
                    cn.neuron_populations[hid.population()]
                    .vars["LambdavLimit"]
                )
                if scale != 1:
                    var = cn.connection_populations[
                        out.connection()
                    ].vars["weight"]
                    var.view[...] = (
                        output_weight * np.float32(scale)
                    ).reshape(var.view.shape)
                    var.push_to_device()
                np.testing.assert_array_equal(
                    pull(cn.connection_populations[
                        out.connection()
                    ].vars["weight"]).reshape(1, 11),
                    output_weight * np.float32(scale),
                )

            cn._train_batch(
                batch, 0, {inp: [sample]}, {out: label},
                {out: metric}, state, callbacks,
            )
            if batch == 0:
                scores = np.array(cn.get_readout(out), copy=True)[0]
            else:
                np.testing.assert_allclose(
                    cn.get_readout(out), 0, rtol=0, atol=1e-7
                )

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1, "Drain entered real metrics")
        require(len(capture.results) == 1, "Wrong gradient capture count")

        data = callbacks.get_data()
        require(
            output_probe.spike_pairs(data["input"], 0)
            == [(0, reference.INPUT_TIME_MS)],
            "Input delivery mismatch",
        )
        require(output_probe.spike_pairs(data["input"], 1) == [],
                "Input spikes in drain")
        hidden = output_probe.spike_pairs(data["hidden"], 0)
        require(len(hidden) == 1, "Expected one hidden forward spike")
        require(output_probe.spike_pairs(data["hidden"], 1) == [],
                "Hidden spikes in drain")

        pop = cn.neuron_populations[hid.population()]
        statistics = {
            name: pull(pop.vars[name]).reshape(-1)[0].item()
            for name in audit.COUNTERS + audit.SCALARS
        }
        require(statistics["AuditJumps"] == 1, "Wrong replay jump count")
        for name in ("AuditClips", "AuditNonfinite", "AuditBadLimit"):
            require(statistics[name] == 0, f"Invalid replay: {name}")

        # training_state layout: two weights, then optimizer M/V arrays.
        expected_state = [value.copy() for value in initial_state]
        expected_state[1] *= np.float32(scale)
        final_state = checkpoint.training_state(cn, layers)
        require(len(final_state) == len(expected_state), "State layout changed")
        for expected, actual in zip(expected_state, final_state):
            np.testing.assert_array_equal(expected, actual)

        gradients = [
            value.astype(np.float64).reshape(-1)
            for value in capture.results[0]
        ]

    emit("REPLAY_WEIGHT_RUN", {
        "output_weight_scale_after_forward": scale,
        "forward_scores": scores.tolist(),
        "hidden_forward_spikes": hidden,
        "replay_limit": bound.tolist(),
        "hidden_weight_gradient": gradients[0].tolist(),
        "output_weight_gradient": gradients[1].tolist(),
        "limiter_statistics": statistics,
        "optimizer_updates": 0,
        "manual_weight_interventions": int(scale != 1),
        "state_changes_match_only_declared_intervention": True,
    })
    return scores, hidden, bound, gradients


def main():
    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_sha256": file_hash(Path(__file__).resolve()),
        "limiter_helper_sha256": file_hash(Path(audit.__file__).resolve()),
        "reference_helper_sha256": file_hash(Path(reference.__file__).resolve()),
        "spike_helper_sha256": file_hash(Path(output_probe.__file__).resolve()),
        "scope": "synthetic dt1ms replay-weight intervention; no learning",
    })
    control = run(1)
    changed = run(2)

    np.testing.assert_allclose(
        changed[0], control[0], rtol=1e-5, atol=1e-6
    )
    require(changed[1] == control[1], "Forward hidden spike changed")
    np.testing.assert_array_equal(changed[2], control[2])
    require(abs(control[3][0][0]) > 1e-8, "Degenerate control gradient")
    np.testing.assert_allclose(
        changed[3][0], 2 * control[3][0], rtol=2e-5, atol=1e-9
    )
    np.testing.assert_allclose(
        changed[3][1], control[3][1], rtol=2e-5, atol=2e-7
    )

    emit("PASS_REPLAY_WEIGHT_DEPENDENCE", {
        "hidden_gradient_ratio":
            float(changed[3][0][0] / control[3][0][0]),
        "output_gradient_max_absolute_difference":
            float(np.max(np.abs(changed[3][1] - control[3][1]))),
        "optimizer_updates": 0,
        "scope": "one synthetic trajectory; output weights doubled for replay",
        "ordinary_adam_staleness_magnitude_measured": False,
        "eventprop_gradient_correctness_certified": False,
    })


if __name__ == "__main__":
    main()