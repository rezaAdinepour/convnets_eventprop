"""Tiny shared-convolution LIF gradient comparison.

Synthetic inputs; frozen weights; one forward presentation plus empty drain.
Two sequential models: shared Conv2D and explicitly untied Dense.
"""

from pathlib import Path

import argparse

import numpy as np
from pygenn import SynapseMatrixType

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.connectivity import Conv2D, Dense
from ml_genn.neurons import SpikeInput, LeakyIntegrateFire, LeakyIntegrate
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import PreprocessedSpikes

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_limiter_probe as audit
import dvs_eventprop_output_gradient_probe as output_probe
import dvs_eventprop_shared_lif_reference as reference
from dvs_data import file_hash, require
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_probe import emit, pull


DT = 0.0078125
STEPS = 16384
CAPACITY = 2 * STEPS + 1


def run(kind, kernel, dense):
    input_weights = (
        kernel.reshape(reference.KERNEL_SHAPE).copy()
        if kind == "conv" else dense.copy()
    )
    output_weights = reference.OUTPUT_WEIGHTS.astype(np.float32)

    counts = np.zeros(45, dtype=np.uint32)
    counts[[0, 4]] = 1
    payload = PreprocessedSpikes(
        np.cumsum(counts, dtype=np.uint32),
        np.array([16.0, 16.0], dtype=np.float32),
    )
    empty = PreprocessedSpikes(
        np.zeros(45, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    )
    label = np.array([0], dtype=np.int64)

    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=2),
            shape=(5, 9, 1) if kind == "conv" else (45,),
            name=f"{kind}_input", record_spikes=True,
        )
        connectivity = (
            Conv2D(
                input_weights, filters=1, conv_size=5,
                flatten=True, conv_strides=4,
                conv_padding="valid", delay=0,
            )
            if kind == "conv" else Dense(input_weights, delay=0)
        )
        hid = Layer(
            connectivity,
            LeakyIntegrateFire(
                v_thresh=1.0, v_reset=0.0, v=0.0,
                tau_mem=20.0, relative_reset=False,
            ),
            shape=(2,), synapse=Exponential(tau=5.0),
            name=f"{kind}_hidden", record_spikes=True,
        )
        out = Layer(
            Dense(output_weights.copy(), delay=0),
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=(11,), synapse=Exponential(tau=5.0),
            name=f"{kind}_output",
        )

    require(hid.population().shape == (2,), "Incorrect hidden shape")
    compiler = audit.AuditCompiler(
        example_timesteps=STEPS,
        losses="sparse_categorical_crossentropy",
        dt=DT, batch_size=1, rng_seed=42,
        max_spikes=CAPACITY, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network, name=f"SharedLIF{kind.title()}",
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
        first_group = cn.connection_populations[hid.connection()]
        expected_matrix = (
            SynapseMatrixType.PROCEDURAL_KERNELG
            if kind == "conv" else SynapseMatrixType.DENSE
        )
        require(first_group.matrix_type == expected_matrix,
                "Unexpected first-projection matrix type")
        for layer, expected in zip(
            layers, (input_weights, output_weights)
        ):
            actual = pull(cn.connection_populations[
                layer.connection()
            ].vars["weight"])
            np.testing.assert_array_equal(
                actual.reshape(expected.shape), expected
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
            value = np.array(cn.get_readout(out), copy=True)
            if batch == 0:
                scores = value.reshape(11)
            else:
                np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1, "Drain entered real metrics")
        require(len(capture.results) == 1, "Wrong gradient capture count")

        data = callbacks.get_data()
        require(
            output_probe.spike_pairs(data["input"], 0)
            == [(0, 16.0), (4, 16.0)],
            "Input delivery mismatch",
        )
        require(output_probe.spike_pairs(data["input"], 1) == [],
                "Input spikes in drain")
        hidden = output_probe.spike_pairs(data["hidden"], 0)
        require([pair[0] for pair in hidden] == [0, 1],
                "Expected exactly one spike from each hidden neuron")
        require(output_probe.spike_pairs(data["hidden"], 1) == [],
                "Hidden spikes in drain")

        pop = cn.neuron_populations[hid.population()]
        measurements = {
            name: pull(pop.vars[name]).reshape(-1)
            for name in audit.COUNTERS + audit.SCALARS
        }
        require(all(v.shape == (2,) for v in measurements.values()),
                "Unexpected audit shape")
        require(all(np.isfinite(v).all() for v in measurements.values()),
                "Nonfinite audit telemetry")
        np.testing.assert_array_equal(
            measurements["AuditJumps"], [1, 1]
        )
        for name in ("AuditClips", "AuditNonfinite", "AuditBadLimit"):
            require(bool((measurements[name] == 0).all()),
                    f"Invalid unconstrained comparison: {name}")
        require(np.isfinite(replay_bound).all()
                and (replay_bound > 0).all(), "Invalid replay limit")

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        first_gradient = capture.results[0][0].reshape(
            input_weights.shape
        ).astype(np.float64)
        output_gradient = capture.results[0][1].reshape(
            2, 11
        ).astype(np.float64)
        require(
            np.isfinite(scores).all()
            and np.isfinite(first_gradient).all()
            and np.isfinite(output_gradient).all(),
            "Nonfinite model result",
        )

        emit("SHARED_LIF_MODEL", {
            "kind": kind,
            "matrix_type": str(first_group.matrix_type),
            "scores": scores.tolist(),
            "hidden_spikes": hidden,
            "first_gradient": first_gradient.tolist(),
            "output_gradient": output_gradient.tolist(),
            "replay_limit": replay_bound.tolist(),
            "audit_per_neuron": {
                name: value.tolist()
                for name, value in measurements.items()
            },
            "optimizer_updates": 0,
            "weights_and_moments_unchanged": True,
        })
        return scores, hidden, first_gradient, output_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlap-weight", type=float, choices=(0.0, 0.1), default=0.0
    )
    args = parser.parse_args()

    require(STEPS * DT == 128.0, "Physical horizon mismatch")
    kernel = np.zeros(25, dtype=np.float32)
    kernel[0] = 7.0
    kernel[4] = args.overlap_weight
    dense, mapping = reference.explicit_dense(kernel)
    dense = dense.astype(np.float32)
    mapping = np.asarray(mapping, dtype=np.int64)
    exact = reference.reference(kernel)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_sha256": file_hash(Path(__file__).resolve()),
        "reference_sha256": file_hash(Path(reference.__file__).resolve()),
        "single_lif_reference_sha256":
            file_hash(Path(reference.single.__file__).resolve()),
        "limiter_helper_sha256": file_hash(Path(audit.__file__).resolve()),
        "spike_helper_sha256": file_hash(Path(output_probe.__file__).resolve()),
        "dt_ms": DT,
        "simulation_timesteps": STEPS,
        "physical_horizon_ms": 128.0,
        "tape_capacity": CAPACITY,
        "scope": "tiny synthetic shared-LIF comparison",
        "overlap_kernel_weight": float(kernel[4]),
        "effective_input_weights": exact["effective_input_weights"].tolist(),
    })

    if args.overlap_weight != 0.0:
        require(
            exact["hidden_spike_times_ms"][0]
            < exact["hidden_spike_times_ms"][1],
            "Expected earlier continuous spike from hidden neuron 0",
        )
    
        directions = {}
        for slot in (0, 4):
            direction = np.zeros(25, dtype=np.float64)
            direction[slot] = 1.0
            directions[f"kernel_{slot}"] = direction
        directions["coupled_0_4"] = (
            directions["kernel_0"] + directions["kernel_4"]
        ) / np.sqrt(2.0)

        base_kernel = kernel.astype(np.float64)
        for name, direction in directions.items():
            analytic = float(np.dot(exact["kernel_gradient"], direction))
            for epsilon in (0.01, 0.003, 0.001):
                plus = reference.reference(base_kernel + epsilon * direction)
                minus = reference.reference(base_kernel - epsilon * direction)
                require(
                    plus["hidden_spikes"] == minus["hidden_spikes"] == [1, 1],
                    "Perturbation changed the single-spike regime",
                )
                finite_difference = (
                    plus["ce"] - minus["ce"]
                ) / (2.0 * epsilon)
                emit("ASYMMETRIC_REFERENCE_FD", {
                    "direction": name,
                    "epsilon": epsilon,
                    "analytic": analytic,
                    "finite_difference": finite_difference,
                    "relative_error": (
                        abs(finite_difference - analytic)
                        / max(abs(analytic), 1e-12)
                    ),
                })
                np.testing.assert_allclose(
                    finite_difference, analytic, rtol=2e-4, atol=1e-9
                )

    conv = run("conv", kernel, dense)
    untied = run("dense", kernel, dense)
    require(conv[1] == untied[1], "Shared/untied hidden spike times differ")
    if args.overlap_weight != 0.0:
        require(
            conv[1][0][1] < conv[1][1][1],
            "Expected distinct spikes, with hidden neuron 0 earlier",
        )
    np.testing.assert_allclose(
        conv[0], untied[0], rtol=2e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        conv[3], untied[3], rtol=2e-5, atol=1e-9
    )

    summed = np.zeros(25, dtype=np.float64)
    np.add.at(
        summed, mapping[:, 0],
        untied[2][mapping[:, 1], mapping[:, 2]],
    )
    shared = conv[2].reshape(25)
    expected = exact["kernel_gradient"]
    kernel_error = float(
        np.linalg.norm(shared - expected) / np.linalg.norm(expected)
    )
    output_error = float(
        np.linalg.norm(conv[3] - exact["output_weight_gradient"])
        / np.linalg.norm(exact["output_weight_gradient"])
    )
    emit("SHARED_LIF_COMPARISON", {
        "shared_vs_summed_max_absolute_difference":
            float(np.max(np.abs(shared - summed))),
        "shared_kernel_gradient": shared.tolist(),
        "summed_untied_gradient": summed.tolist(),
        "continuous_kernel_gradient": expected.tolist(),
        "continuous_kernel_relative_l2_error": kernel_error,
        "continuous_output_relative_l2_error": output_error,
        "continuous_hidden_spike_times_ms": exact["hidden_spike_times_ms"],
        "genn_hidden_spikes": conv[1],
    })

    np.testing.assert_allclose(
        shared, summed, rtol=2e-5, atol=1e-9,
        err_msg="Shared gradient differs from summed untied contributions",
    )
    np.testing.assert_allclose(
        shared, expected, rtol=0.01, atol=1e-9,
        err_msg="Kernel gradient misses the continuous-reference criterion",
    )
    np.testing.assert_allclose(
        conv[3], exact["output_weight_gradient"], rtol=0.01, atol=1e-9,
        err_msg="Output gradient misses the continuous-reference criterion",
    )
    require(kernel_error < 0.01 and output_error < 0.01,
            "Relative L2 error exceeds 1 percent")

    emit("PASS_TINY_SHARED_LIF_GRADIENT", {
        "shared_parameters_compared": 25,
        "mapped_untied_edges": 50,
        "dt_ms": DT,
        "optimizer_updates": 0,
        "clipped_jumps": 0,
        "scope": "one synthetic operating point at fine timestep",
        "original_dt1ms_gradient_gate_passed": False,
        "full_dvs_gradient_correctness_certified": False,
        "overlap_kernel_weight": float(kernel[4]),
    })


if __name__ == "__main__":
    main()