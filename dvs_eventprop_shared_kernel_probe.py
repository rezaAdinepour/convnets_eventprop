"""TASK 10: procedural shared-kernel gradients versus untied contributions.

Synthetic input only. Two sequential compiled contexts, no optimizer updates.
No datasets, checkpoints or caches are read/written.
"""

from pathlib import Path

import numpy as np
from pygenn import SynapseMatrixType

from ml_genn import InputLayer, Layer, SequentialNetwork
from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Conv2D, Dense
from ml_genn.neurons import SpikeInput, LeakyIntegrate
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.utils.callback_list import CallbackList
from ml_genn.utils.data import PreprocessedSpikes

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_fixed_output_probe as fixed
import dvs_eventprop_output_gradient_probe as output_probe
from dvs_data import file_hash, object_hash, require
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_probe import emit, pull


KERNEL_SHAPE = (5, 5, 2, 2)
INPUT_NEURONS = 5 * 9 * 2
OUTPUT_NEURONS = 4


def make_reference(kernel):
    """Explicit HWC map, independent of Conv2D's connectivity snippet."""
    dense = np.zeros((INPUT_NEURONS, OUTPUT_NEURONS), np.float32)
    mapping = []
    flat_kernel = kernel.reshape(-1)

    for output_x in range(2):
        for ky in range(5):
            for kx in range(5):
                for polarity in range(2):
                    input_x = output_x * 4 + kx
                    pre = ((ky * 9) + input_x) * 2 + polarity
                    for filt in range(2):
                        post = output_x * 2 + filt
                        slot = (
                            ((ky * 5 + kx) * 2 + polarity) * 2 + filt
                        )
                        dense[pre, post] = flat_kernel[slot]
                        mapping.append((slot, pre, post))

    mapping = np.asarray(mapping, dtype=np.int64)
    require(mapping.shape == (200, 3), "Wrong expanded edge count")
    require(
        len(set(map(tuple, mapping[:, 1:]))) == 200,
        "Duplicate expanded connection",
    )
    np.testing.assert_array_equal(
        np.bincount(mapping[:, 0], minlength=100),
        np.full(100, 2),
    )
    return dense, mapping


def make_input():
    # Every input neuron is active. Timing varies with its HWC ID.
    # Some neurons also emit a second, later spike.
    times_by_neuron = [
        [float(8 + (7 * i + 3 * (i // 2)) % 90)]
        + ([112.0] if i % 7 == 0 else [])
        for i in range(INPUT_NEURONS)
    ]
    payload = PreprocessedSpikes(
        np.cumsum(
            [len(times) for times in times_by_neuron],
            dtype=np.uint32,
        ),
        np.asarray(
            [t for times in times_by_neuron for t in times],
            dtype=np.float32,
        ),
    )
    expected = sorted(
        (i, t)
        for i, times in enumerate(times_by_neuron)
        for t in times
    )
    return payload, expected


def run(kind, kernel, dense, payload, expected):
    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=len(payload.spike_times)),
            shape=(5, 9, 2) if kind == "conv" else (INPUT_NEURONS,),
            name=f"{kind}_input",
            record_spikes=True,
        )
        connectivity = (
            Conv2D(
                kernel.copy(),
                filters=2,
                conv_size=5,
                flatten=True,
                conv_strides=4,
                conv_padding="valid",
                delay=0,
            )
            if kind == "conv" else
            Dense(dense.copy(), delay=0)
        )
        out = Layer(
            connectivity,
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=(OUTPUT_NEURONS,),
            synapse=Exponential(tau=5.0),
            name=f"{kind}_output",
        )

    require(out.population().shape == (4,), "Wrong flattened output shape")
    cn = EventPropCompiler(
        example_timesteps=128,
        losses="sparse_categorical_crossentropy",
        dt=1.0,
        batch_size=1,
        rng_seed=42,
        max_spikes=257,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        grad_limit=100.0,
        softmax_temperature=1.0,
    ).compile(
        network,
        name=f"SharedKernel{kind.title()}",
        optimisers={out: {"weight": Adam(alpha=0.001)}},
        regularisers={},
    )

    matches = [
        i for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    # Existing, verified callback captures label-0 gradients without learning.
    capture = fixed.Capture(out)
    cn.base_train_callbacks[matches[0]] = capture

    empty = PreprocessedSpikes(
        np.zeros(INPUT_NEURONS, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    )
    label = np.array([0], dtype=np.int64)

    with cn:
        group = cn.connection_populations[out.connection()]
        expected_matrix = (
            SynapseMatrixType.PROCEDURAL_KERNELG
            if kind == "conv" else SynapseMatrixType.DENSE
        )
        require(group.matrix_type == expected_matrix,
                f"Unexpected matrix type: {group.matrix_type}")

        expected_weight = kernel if kind == "conv" else dense
        np.testing.assert_array_equal(
            pull(group.vars["weight"]).reshape(expected_weight.shape),
            expected_weight,
        )
        initial_state = checkpoint.training_state(cn, [out])

        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks
            + [SpikeRecorder(inp, key="input")],
            compiled_network=cn,
            num_batches=2,
            num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        scores = None
        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
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
                scores = np.array(cn.get_readout(out), copy=True)
                require(scores.shape == (1, 4), "Wrong score axes")
                require(np.isfinite(scores).all(), "Nonfinite scores")

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)

        require(state[out].total == 1, "Drain contaminated metrics")
        require(len(capture.results) == 1, "Wrong gradient capture count")
        recorded = callbacks.get_data()["input"]
        require(output_probe.spike_pairs(recorded, 0) == expected,
                f"{kind}: input delivery mismatch")
        require(output_probe.spike_pairs(recorded, 1) == [],
                f"{kind}: input spikes in drain")

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, [out])
        ):
            np.testing.assert_array_equal(old, new)

        gradient = capture.results[0].reshape(expected_weight.shape)
        require(np.isfinite(gradient).all(), "Nonfinite gradient")
        require(np.linalg.norm(gradient) > 1e-8, "Gradient too small")

        emit("MODEL_RESULT", {
            "kind": kind,
            "matrix_type": str(group.matrix_type),
            "weight_shape": list(expected_weight.shape),
            "scores": scores.tolist(),
            "gradient_l2": float(np.linalg.norm(gradient)),
            "input_delivery": "pass",
            "optimizer_updates": 0,
            "weights_and_moments_unchanged": True,
        })
        return scores.copy(), gradient.copy()


def main():
    kernel = (
        ((np.arange(100, dtype=np.int32) * 7) % 23 - 11)
        .astype(np.float32) / np.float32(40.0)
    ).reshape(KERNEL_SHAPE)
    dense, mapping = make_reference(kernel)
    payload, expected = make_input()

    directory = Path(__file__).resolve().parent
    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "helper_source_sha256": {
            name: file_hash(directory / name)
            for name in (
                "dvs_eventprop_fixed_output_probe.py",
                "dvs_eventprop_output_gradient_probe.py",
            )
        },
    })
    emit("SHARED_CONFIG", {
        "input_hwc": [5, 9, 2],
        "output_hwc": [1, 2, 2],
        "output_flat": 4,
        "kernel_hwio": list(KERNEL_SHAPE),
        "kernel_weights": kernel.tolist(),
        "stride": [4, 4],
        "padding": "valid",
        "mapped_connections": 200,
        "uses_per_kernel_parameter": 2,
        "mapping_sha256": object_hash(mapping.tolist()),
        "scheduled_spikes": expected,
        "dt_ms": 1.0,
        "physical_horizon_ms": 128.0,
        "tau_mem_ms": 20.0,
        "tau_syn_ms": 5.0,
        "tape_capacity": 257,
        "strict_buffer_checking": True,
        "label": 0,
        "optimizer_updates": 0,
        "scope": "shared accumulation, nonspiking outputs",
    })

    conv_scores, conv_gradient = run(
        "conv", kernel, dense, payload, expected
    )
    dense_scores, dense_gradient = run(
        "dense", kernel, dense, payload, expected
    )

    summed = np.zeros(100, dtype=np.float64)
    np.add.at(
        summed,
        mapping[:, 0],
        dense_gradient[mapping[:, 1], mapping[:, 2]],
    )
    shared = conv_gradient.reshape(-1)
    reference_norm = float(np.linalg.norm(summed))
    require(reference_norm > 1e-8, "Summed reference too small")

    emit("SHARED_COMPARISON", {
        "score_max_absolute_difference": float(
            np.max(np.abs(conv_scores - dense_scores))
        ),
        "gradient_max_absolute_difference": float(
            np.max(np.abs(shared - summed))
        ),
        "gradient_relative_l2_error": float(
            np.linalg.norm(shared - summed) / reference_norm
        ),
        "gradient_cosine_similarity": float(
            np.dot(shared, summed)
            / (np.linalg.norm(shared) * reference_norm)
        ),
        "shared_gradient": shared.tolist(),
        "summed_untied_gradient": summed.tolist(),
    })

    np.testing.assert_allclose(
        conv_scores, dense_scores, rtol=2e-5, atol=2e-7,
        err_msg="Shared and expanded forward scores differ",
    )
    np.testing.assert_array_equal(
        conv_scores.argmax(axis=1), dense_scores.argmax(axis=1)
    )
    np.testing.assert_allclose(
        shared, summed, rtol=2e-5, atol=2e-7,
        err_msg="Shared gradient differs from summed untied contributions",
    )
    emit("PASS_SHARED_KERNEL", {
        "kernel_parameters_compared": 100,
        "mapped_connections": 200,
        "contributions_per_parameter": 2,
        "optimizer_updates": 0,
        "scope": "procedural stride-4 sharing with LI outputs",
        "hidden_lif_gradient_certified": False,
        "original_numerical_gradient_gate_passed": False,
    })


if __name__ == "__main__":
    main()