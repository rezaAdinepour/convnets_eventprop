"""TASK 10: frozen DVS EventProp tape-capacity parity, M257 versus M514.

Three fixed training examples plus an empty backward drain.
No optimizer updates, checkpoints, caches or test-event access.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from ml_genn.callbacks import Callback, SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.optimisers import Adam
from ml_genn.utils.callback_list import CallbackList

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_output_gradient_probe as output_probe
from dvs_data import (
    Config as DataConfig,
    PreprocessedSpikes,
    file_hash,
    inventory,
    load_sample,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_model import build_model
from dvs_eventprop_probe import emit, expected_pairs, pull


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class CaptureAndClear(Callback):
    def __init__(self, layers, hidden, output, labels, capacity):
        self.layers = layers
        self.hidden = hidden
        self.output = output
        self.labels = labels
        self.capacity = capacity
        self.records = []

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_end(self, cn, batch, metric_state):
        if batch == 0:
            return

        index = batch - 1
        require(index < len(self.labels), "Unexpected backward example")
        pop = cn.neuron_populations[self.output.population()]
        np.testing.assert_array_equal(
            pull(pop.vars["YTrueBack"]).reshape(-1),
            [int(self.labels[index])],
        )
        gradients = [
            pull(cn.connection_populations[layer.connection()]
                 .vars["weightGradient"])
            for layer in self.layers
        ]
        require(all(np.isfinite(g).all() for g in gradients),
                "Nonfinite gradient")
        limit = pull(
            cn.neuron_populations[self.hidden.population()]
            .vars["LambdavLimit"]
        )
        require(np.isfinite(limit).all() and (limit > 0).all(),
                "Invalid replay limiter bound")

        self.records.append({
            "index": index,
            "gradients": gradients,
            "limit": limit,
        })
        emit("GRADIENT_CAPTURE", {
            "capacity": self.capacity,
            "backward_example_index": index,
            "backward_label": int(self.labels[index]),
            "gradient_l2_hidden_output": [
                float(np.linalg.norm(g.astype(np.float64)))
                for g in gradients
            ],
            "gradient_sha256_hidden_output": [
                array_hash(g) for g in gradients
            ],
            "replay_limit": limit.tolist(),
            "optimizer_updates": 0,
        })

        # Source-checked compiler reset group; do not invoke GradientLearn.
        cn.custom_update("ZeroGradient")
        for layer in self.layers:
            gradient = pull(
                cn.connection_populations[layer.connection()]
                .vars["weightGradient"]
            )
            require(bool((gradient == 0).all()),
                    "Gradient reset failed")


def run(
    config, capacity, inputs, labels, initial_weights=None,
    *, compiler_class=EventPropCompiler, capture_class=CaptureAndClear,
):
    network, inp, hid, out = build_model(config, 1)
    layers = [hid, out]
    shapes = [(5, 5, 2, 8), (392, 11)]
    if initial_weights is not None:
        for layer, values, shape in zip(layers, initial_weights, shapes):
            layer.connection().connectivity.weight = (
                values.reshape(shape).copy()
            )

    cn = compiler_class(
        example_timesteps=128,
        losses="sparse_categorical_crossentropy",
        dt=1.0,
        batch_size=1,
        rng_seed=config.seed,
        max_spikes=capacity,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        grad_limit=100.0,
        softmax_temperature=1.0,
    ).compile(
        network,
        name=f"TapeCapacity{capacity}",
        optimisers={
            hid: {"weight": Adam(alpha=config.learning_rate)},
            out: {"weight": Adam(alpha=config.learning_rate)},
        },
        regularisers={},
    )

    learning = [
        i for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    zeroing = [
        cb for cb in cn.base_train_callbacks
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "ZeroGradient"
    ]
    require(len(learning) == len(zeroing) == 1,
            "Unexpected learning/reset callback structure")
    capture = capture_class(layers, hid, out, labels, capacity)
    cn.base_train_callbacks[learning[0]] = capture

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    presentations = inputs + [empty]
    targets = np.r_[labels, np.int64(0)]

    with cn:
        start_weights = [
            pull(cn.connection_populations[layer.connection()]
                 .vars["weight"])
            for layer in layers
        ]
        if initial_weights is not None:
            for old, new in zip(initial_weights, start_weights):
                np.testing.assert_array_equal(old, new)

        initial_state = checkpoint.training_state(cn, layers)
        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks + [
                SpikeRecorder(inp, key="input"),
                SpikeRecorder(hid, key="hidden"),
            ],
            compiled_network=cn, num_batches=4, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []

        for batch, payload in enumerate(presentations):
            metric.real_presentation = batch < 3
            cn._train_batch(
                batch, 0, {inp: [payload]},
                {out: targets[batch:batch + 1]},
                {out: metric}, state, callbacks,
            )
            pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(
                pull(pop.vars["StartSpike"]),
                pull(pop.vars["EndSpike"]),
            )
            value = np.array(cn.get_readout(out), copy=True)
            require(value.shape == (1, 11) and np.isfinite(value).all(),
                    "Invalid scores")
            scores.append(value[0])

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 3, "Drain contaminated metrics")
        require(
            [r["index"] for r in capture.records] == [0, 1, 2],
            "Missing or duplicated backward contribution",
        )
        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        recorded = callbacks.get_data()
        require(len(recorded["input"][0]) == 4,
                "Wrong input recording count")
        require(len(recorded["hidden"][0]) == 4,
                "Wrong hidden recording count")
        hidden = []
        for index, payload in enumerate(presentations):
            require(
                output_probe.spike_pairs(recorded["input"], index)
                == expected_pairs(payload),
                f"Input delivery mismatch at presentation {index}",
            )
            hidden.append(
                output_probe.spike_pairs(recorded["hidden"], index)
            )
        require(hidden[-1] == [], "Hidden spikes in empty drain")
        np.testing.assert_allclose(scores[-1], 0.0, rtol=0, atol=1e-7)

        emit("CAPACITY_RUN", {
            "capacity": capacity,
            "real_examples": 3,
            "backward_contributions": len(capture.records),
            "scores": np.asarray(scores).tolist(),
            "hidden_spikes_per_presentation": [len(s) for s in hidden],
            "initial_weight_sha256": [array_hash(w) for w in start_weights],
            "input_delivery": "pass",
            "weights_and_moments_unchanged": True,
            "optimizer_updates": 0,
        })
        return {
            "initial_weights": start_weights,
            "scores": np.asarray(scores),
            "hidden": hidden,
            "records": capture.records,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    root = args.root.resolve(strict=True)
    rows = pilot(inventory(root)["train"], 2)[:3]
    loaded = [
        load_sample(root, row, DataConfig(128)) for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    inputs = [to_ml_genn(payload) for payload in payloads]
    labels = np.asarray(labels, dtype=np.int64)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "spike_helper_source_sha256": file_hash(
            Path(output_probe.__file__)
        ),
    })
    plan = config.record()
    plan.pop("compiler_max_spikes")
    emit("CAPACITY_PLAN", {
        **plan,
        "compiler_capacities": [257, 514],
        "training_order": [row.stable_id for row in rows],
        "real_examples": 3,
        "empty_drains": 1,
        "weights_frozen": True,
        "per_example_gradients_cleared_after_capture": True,
        "optimizer_updates": 0,
    })
    for item in metadata:
        emit("SAMPLE_AUDIT", item)

    first = run(config, 257, inputs, labels)
    second = run(
        config, 514, inputs, labels,
        initial_weights=first["initial_weights"],
    )

    require(first["hidden"] == second["hidden"],
            "Hidden spike IDs/times depend on tape capacity")
    np.testing.assert_allclose(
        first["scores"], second["scores"], rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        first["scores"][:3].argmax(axis=1),
        second["scores"][:3].argmax(axis=1),
    )

    comparisons = []
    for old, new in zip(first["records"], second["records"]):
        require(old["index"] == new["index"], "Backward ownership mismatch")
        np.testing.assert_allclose(
            old["limit"], new["limit"], rtol=2e-5, atol=2e-7,
        )
        differences = []
        for a, b in zip(old["gradients"], new["gradients"]):
            differences.append(float(np.max(np.abs(a - b))))
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)
        comparisons.append({
            "backward_example_index": old["index"],
            "gradient_max_absolute_difference_hidden_output": differences,
        })

    emit("PASS_CAPACITY_PARITY", {
        "capacities": [257, 514],
        "real_examples": 3,
        "matched_backward_contributions": 3,
        "hidden_spike_times": "exact match",
        "score_max_absolute_difference": float(
            np.max(np.abs(first["scores"] - second["scores"]))
        ),
        "per_example_comparisons": comparisons,
        "optimizer_updates": 0,
        "scope": "frozen three-example DVS diagnostic at 128ms, dt1ms, B1",
        "memory_headroom_certified": False,
        "numerical_gradient_correctness_certified": False,
    })


if __name__ == "__main__":
    main()