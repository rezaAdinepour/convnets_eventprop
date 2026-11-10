"""TASK 09 Stage A: input delivery and stock-loop ownership diagnostics.

Run only in a new experiment directory: GeNN generates build files there.
This program does not save checkpoints or report scientific accuracy.
"""

import argparse
import importlib.metadata
import json
import logging
from pathlib import Path
import sys

import numpy as np
import ml_genn
import pygenn

from ml_genn.callbacks import Callback, SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd

from dvs_data import (
    Config as DataConfig,
    PreprocessedSpikes,
    file_hash,
    inventory,
    load_sample,
    pack_occupancy,
    pilot,
    require,
    to_ml_genn,
    validate_payload,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_model import compile_model


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def pull(var):
    var.pull_from_device()
    return np.array(var.view, copy=True)


def expected_pairs(payload):
    ids = np.repeat(
        np.arange(2048, dtype=np.int64),
        np.diff(np.r_[0, payload.end_spikes.astype(np.int64)]),
    )
    return sorted(zip(ids.tolist(), payload.spike_times.tolist()))


class Observe(Callback):
    def __init__(self, input_layer, output, labels):
        self.input_layer = input_layer
        self.output = output
        self.labels = labels

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_begin(self, cn, batch):
        # User callbacks follow compiler callbacks: Reset has already rotated
        # the previous forward label into YTrueBack. Current y is pushed later.
        pop = cn.neuron_populations[self.output.population()]
        back = pull(pop.vars["YTrueBack"]).reshape(-1).astype(int).tolist()
        emit("BATCH_BEGIN", {"batch": batch, "YTrueBack": back})

        if cn.genn_model.batch_size == 1 and batch > 0:
            require(
                back == [int(self.labels[batch - 1])],
                "Backward label does not match previous forward sample",
            )

    def on_batch_end(self, cn, batch, metric_state):
        pop = cn.neuron_populations[self.input_layer.population()]
        start = pull(pop.vars["StartSpike"])
        end = pull(pop.vars["EndSpike"])
        require(np.array_equal(start, end), "Unread scheduled input spikes")

        scores = np.array(cn.get_readout(self.output), copy=True)
        require(scores.shape == (cn.genn_model.batch_size, 11),
                "Unexpected score axes")
        require(scores.dtype == np.float32, "Expected FP32 scores")
        require(bool(np.isfinite(scores).all()), "Nonfinite scores")
        emit("BATCH_END", {"batch": batch, "scores": scores.tolist()})


class LearnTrace(CustomUpdateOnBatchEnd):
    """Observe the real existing GradientLearn callback, without extra calls."""

    def __init__(self, original, layers):
        super().__init__(original.name, original.filter_fn)
        self.layers = layers
        self.calls = []

    def on_batch_end(self, cn, batch, metric_state):
        enabled = self.filter_fn is None or self.filter_fn(batch)
        if not enabled:
            super().on_batch_end(cn, batch, metric_state)
            return

        before = []
        gradient_norms = []
        for layer in self.layers:
            group = cn.connection_populations[layer.connection()]
            before.append(pull(group.vars["weight"]))
            gradient = pull(group.vars["weightGradient"])
            require(bool(np.isfinite(gradient).all()), "Nonfinite gradient")
            gradient_norms.append(
                float(np.linalg.norm(gradient.astype(np.float64)))
            )

        # Exactly the compiler's existing learning call.
        super().on_batch_end(cn, batch, metric_state)

        changes = []
        for layer, previous in zip(self.layers, before):
            group = cn.connection_populations[layer.connection()]
            current = pull(group.vars["weight"])
            require(bool(np.isfinite(current).all()), "Nonfinite weights")
            changes.append(
                float(np.linalg.norm(
                    current.astype(np.float64) - previous.astype(np.float64)
                ))
            )

        self.calls.append(batch)
        emit("GRADIENT_LEARN", {
            "presentation_batch": batch,
            "backward_sample_index": batch - 1,
            "stock_zero_based_adam_step": batch,
            "gradient_l2_hidden_output": gradient_norms,
            "weight_change_l2_hidden_output": changes,
        })


def sentinel(frames):
    # Base32 HWC IDs: x=1,y=2,p=1 -> 131; p=0 -> 130.
    # Includes duplicate bin occupancy and excluded right boundary.
    payload, audit = pack_occupancy(
        np.array([0, 999, 1000, 127999, 128000], dtype=np.int64),
        np.ones(5, dtype=np.int64),
        np.full(5, 2, dtype=np.int64),
        np.array([1, 1, 0, 1, 1], dtype=np.int64),
        width=32,
        height=32,
        frames=frames,
    )
    require(
        expected_pairs(payload) == [(130, 1.0), (131, 0.0), (131, 127.0)],
        "Unexpected CPU sentinel",
    )
    emit("SENTINEL_AUDIT", audit)
    return payload


def delivery(config, batch_size):
    signal = sentinel(config.frames)
    empty = PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    )

    # Repeated signal after different predecessors; B2 also swaps lane.
    samples = (
        [signal, empty, signal, signal]
        if batch_size == 1 else
        [signal, empty, empty, signal, signal, signal]
    )
    labels = np.zeros(len(samples), dtype=np.int64)

    for sample in samples:
        validate_payload(sample, neurons=2048, frames=config.frames)

    _, cn, inp, hid, out = compile_model(
        config, batch_size, learn=False,
        name=f"DVSDeliveryB{batch_size}",
    )

    with cn:
        _, data = cn.train(
            {inp: [to_ml_genn(p) for p in samples]},
            {out: labels},
            num_epochs=1,
            shuffle=False,
            callbacks=[
                SpikeRecorder(inp, key="input_spikes"),
                Observe(inp, out, labels),
            ],
        )

    times, ids = data["input_spikes"]
    require(len(times) == len(samples), "Wrong recorded sample count")
    for index, (sample, actual_t, actual_id) in enumerate(
        zip(samples, times, ids)
    ):
        actual = sorted(zip(
            np.asarray(actual_id).astype(np.int64).tolist(),
            np.asarray(actual_t).astype(np.float64).tolist(),
        ))
        expected = expected_pairs(sample)
        require(actual == expected,
                f"Delivery mismatch sample {index}: {actual} != {expected}")

    emit("PASS_DELIVERY", {
        "batch_size": batch_size,
        "samples": len(samples),
        "accepted_times_ms": [0, 1, 127],
        "right_boundary_ms": 128,
        "optimizer_updates": 0,
    })


def ownership(config, root):
    # Three fixed members from the prescribed balanced training pilot.
    # This is deliberately a stock-loop diagnostic, not a balanced experiment.
    partitions = inventory(root)
    rows = pilot(partitions["train"], 2)[:3]
    loaded = [load_sample(root, row, DataConfig(config.frames)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    labels = np.asarray(labels, dtype=np.int64)

    for item in metadata:
        emit("SAMPLE_AUDIT", item)

    _, cn, inp, hid, out = compile_model(
        config, 1, learn=True, name="DVSStockOwnershipB1",
    )

    matches = [
        (i, callback)
        for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected compiler callback structure")
    index, original = matches[0]
    tracer = LearnTrace(original, [hid, out])
    # Changes this compiled object's callback list only; no library patch.
    cn.base_train_callbacks[index] = tracer

    emit("OWNERSHIP_ORDER", [row.stable_id for row in rows])

    with cn:
        cn.train(
            {inp: [to_ml_genn(p) for p in payloads]},
            {out: labels},
            num_epochs=1,
            shuffle=False,
            callbacks=[Observe(inp, out, labels)],
        )

    require(tracer.calls == [1, 2], "Stock update schedule differs from source")
    emit("PASS_STOCK_DIAGNOSIS", {
        "real_presentations": 3,
        "learning_calls": 2,
        "updated_sample_indices": [0, 1],
        "missing_final_sample_index": 2,
        "stock_adam_steps": [1, 2],
        "complete_training_coverage": False,
        "next_action": "STOP; inspect logs before implementing drain/clock repair",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("delivery", "ownership"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    logging.basicConfig(level=logging.INFO)

    emit("CONFIG", config.record())
    emit("ENVIRONMENT", {
        "python": sys.version,
        "executable": sys.executable,
        "numpy_version": np.__version__,
        "numpy_path": np.__file__,
        "ml_genn_version": importlib.metadata.version("ml_genn"),
        "ml_genn_path": ml_genn.__file__,
        "pygenn_version": importlib.metadata.version("pygenn"),
        "pygenn_path": pygenn.__file__,
        "working_directory": str(Path.cwd()),
        "numpy_seed": config.seed,
        "genn_seed": config.seed,
        "shuffle": False,
    })
    for name in (
        "dvs_data.py",
        "dvs_eventprop_config.py",
        "dvs_eventprop_model.py",
        "dvs_eventprop_probe.py",
    ):
        path = Path(__file__).resolve().parent / name
        emit("SOURCE", {"path": str(path), "sha256": file_hash(path)})

    if args.mode == "delivery":
        delivery(config, args.batch_size)
    else:
        require(args.batch_size == 1, "Ownership diagnostic is B1 only")
        require(args.root is not None, "--root is required")
        ownership(config, args.root.resolve(strict=True))


if __name__ == "__main__":
    main()
