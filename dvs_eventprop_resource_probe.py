"""TASK 10: phase-labelled frozen DVS resource diagnostic.

Three training examples plus empty drain, zero optimizer updates.
Run under Nsight Systems with CUDA/NVTX tracing.
"""

import argparse
from contextlib import contextmanager, ExitStack
import csv
import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.optimisers import Adam
from ml_genn.utils.callback_list import CallbackList

import dvs_eventprop_capacity_probe as capacity
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


NVTX_LIBRARY = Path(
    "/usr/local/cuda-13.5/targets/x86_64-linux/lib/"
    "libnvtx3interop.so.1.1.0"
)


class ObserveResources:
    def __init__(self):
        self.nvtx = ctypes.CDLL(str(NVTX_LIBRARY))
        self.nvtx.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self.nvtx.nvtxRangePushA.restype = ctypes.c_int
        self.nvtx.nvtxRangePop.argtypes = []
        self.nvtx.nvtxRangePop.restype = ctypes.c_int
        self.snapshots = []
        self.phases = []

    @contextmanager
    def phase(self, name):
        self.nvtx.nvtxRangePushA(f"task10/{name}".encode())
        start = time.perf_counter_ns()
        emit("PHASE_BEGIN", {"name": name})
        try:
            yield
        finally:
            elapsed = time.perf_counter_ns() - start
            self.nvtx.nvtxRangePop()
            entry = {"name": name, "profiled_wall_ms": elapsed / 1e6}
            self.phases.append(entry)
            emit("PHASE_END", entry)

    def snapshot(self, name):
        start = time.time_ns()
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                "--format=csv",
            ],
            capture_output=True, text=True, check=True, timeout=20,
        )
        rows = list(csv.DictReader(io.StringIO(gpu.stdout)))
        require(len(rows) == 1, "This probe requires one visible physical GPU")
        row = {key.strip(): value.strip() for key, value in rows[0].items()}

        def mib(key):
            return int(row[key].split()[0])

        processes = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            capture_output=True, text=True, check=True, timeout=20,
        )
        entry = {
            "name": name,
            "query_begin_wall_ns": start,
            "query_end_wall_ns": time.time_ns(),
            "gpu_index": int(row["index"]),
            "gpu_uuid": row["uuid"],
            "total_MiB": mib("memory.total [MiB]"),
            "used_MiB": mib("memory.used [MiB]"),
            "free_MiB": mib("memory.free [MiB]"),
            "compute_processes_csv": processes.stdout,
        }
        entry["free_fraction"] = entry["free_MiB"] / entry["total_MiB"]
        if self.snapshots:
            require(entry["gpu_uuid"] == self.snapshots[0]["gpu_uuid"],
                    "GPU identity changed")
        self.snapshots.append(entry)
        emit("GPU_MEMORY_SNAPSHOT", entry)


def normalize(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--recording", choices=("on", "off"), default="on")
    parser.add_argument("--initial-from", type=Path)
    args = parser.parse_args()

    require(os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "0"),
            "CUDA device mapping needs review")
    observer = ObserveResources()
    observer.snapshot("baseline")

    config = ProbeConfig()
    np.random.seed(config.seed)
    recording = args.recording == "on"
    identity = {
        **checkpoint.identity(),
        "resource_probe_sha256": file_hash(Path(__file__)),
        "capacity_helper_sha256": file_hash(Path(capacity.__file__)),
        "spike_helper_sha256": file_hash(Path(output_probe.__file__)),
        "nvtx_library": str(NVTX_LIBRARY),
        "nvtx_library_sha256": file_hash(NVTX_LIBRARY),
    }

    with observer.phase("data_preparation"):
        root = args.root.resolve(strict=True)
        rows = pilot(inventory(root)["train"], 2)[:3]
        loaded = [
            load_sample(root, row, DataConfig(128)) for row in rows
        ]
        payloads, labels, metadata = zip(*loaded)
        inputs = [to_ml_genn(p) for p in payloads]
        labels = np.asarray(labels, dtype=np.int64)
        metadata = normalize(list(metadata))

    condition = {
        **config.record(),
        "batch_size": 1,
        "real_examples": 3,
        "empty_drains": 1,
        "training_order": [row.stable_id for row in rows],
    }
    previous = None
    if args.initial_from is not None:
        with args.initial_from.resolve(strict=True).open() as stream:
            previous = json.load(stream)
        require(previous["identity"] == identity,
                "Initial-weight artifact source/environment identity changed")
        require(previous["condition"] == condition, "Condition changed")
        require(previous["sample_metadata"] == metadata,
                "Input/provenance changed")

    emit("RESOURCE_PLAN", {
        "identity": identity,
        "condition": condition,
        "recording": args.recording,
        "initial_from": str(args.initial_from) if args.initial_from else None,
        "pid": os.getpid(),
        "optimizer_updates": 0,
        "scope": "profiled frozen diagnostic with gradient telemetry",
    })

    with observer.phase("model_definition"):
        network, inp, hid, out = build_model(config, 1)
        layers = [hid, out]
        shapes = [(5, 5, 2, 8), (392, 11)]
        for layer in (inp, hid):
            layer.population().record_spikes = recording

        if previous is not None:
            for layer, values, shape in zip(
                layers, previous["initial_weights"], shapes
            ):
                layer.connection().connectivity.weight = (
                    np.asarray(values, dtype=np.float32).reshape(shape).copy()
                )

        cn = EventPropCompiler(
            example_timesteps=128,
            losses="sparse_categorical_crossentropy",
            dt=1.0, batch_size=1, rng_seed=config.seed,
            max_spikes=257, strict_buffer_checking=True,
            per_timestep_loss=False, grad_limit=100.0,
            softmax_temperature=1.0,
        ).compile(
            network,
            name=f"ResourceRecording{args.recording.title()}",
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
    capture = capacity.CaptureAndClear(layers, hid, out, labels, 257)
    cn.base_train_callbacks[learning[0]] = capture

    with observer.phase("build"):
        cn.genn_model.build()
    observer.snapshot("after_build")

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    presentations = inputs + [empty]
    targets = np.r_[labels, np.int64(0)]
    stack = ExitStack()

    try:
        with observer.phase("load_and_initial_sync"):
            stack.enter_context(cn)
            cn.get_readout(out)
        observer.snapshot("after_load")

        initial_weights = [
            pull(cn.connection_populations[layer.connection()].vars["weight"])
            for layer in layers
        ]
        if previous is not None:
            for actual, expected in zip(
                initial_weights, previous["initial_weights"]
            ):
                np.testing.assert_array_equal(
                    actual.reshape(-1),
                    np.asarray(expected, np.float32).reshape(-1),
                )
        initial_state = checkpoint.training_state(cn, layers)

        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        extras = (
            [
                SpikeRecorder(inp, key="input"),
                SpikeRecorder(hid, key="hidden"),
            ] if recording else []
        )
        callbacks = CallbackList(
            cn.base_train_callbacks + extras,
            compiled_network=cn, num_batches=4, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []

        for batch, payload in enumerate(presentations):
            metric.real_presentation = batch < 3
            with observer.phase(f"presentation_{batch}"):
                cn._train_batch(
                    batch, 0, {inp: [payload]},
                    {out: targets[batch:batch + 1]},
                    {out: metric}, state, callbacks,
                )
                value = np.array(cn.get_readout(out), copy=True)
                require(value.shape == (1, 11) and np.isfinite(value).all(),
                        "Invalid scores")
                scores.append(value[0])
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(
                    pull(pop.vars["StartSpike"]),
                    pull(pop.vars["EndSpike"]),
                )
            observer.snapshot(f"after_presentation_{batch}")

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 3, "Drain contaminated metrics")
        require([r["index"] for r in capture.records] == [0, 1, 2],
                "Missing/duplicate backward contributions")
        np.testing.assert_allclose(scores[-1], 0, rtol=0, atol=1e-7)

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        hidden_counts = None
        recorded_array_bytes = 0
        if recording:
            data = callbacks.get_data()
            hidden_counts = []
            for index, payload in enumerate(presentations):
                require(
                    output_probe.spike_pairs(data["input"], index)
                    == expected_pairs(payload),
                    "Recorded input mismatch",
                )
                hidden_counts.append(len(
                    output_probe.spike_pairs(data["hidden"], index)
                ))
            require(hidden_counts[-1] == 0, "Hidden spikes in drain")
            recorded_array_bytes = sum(
                np.asarray(array).nbytes
                for key in ("input", "hidden")
                for component in data[key]
                for array in component
            )
    finally:
        with observer.phase("unload"):
            stack.close()

    observer.snapshot("after_unload")
    result = {
        "identity": identity,
        "condition": condition,
        "recording": args.recording,
        "sample_metadata": metadata,
        "initial_weights": [w.tolist() for w in initial_weights],
        "initial_weight_sha256": [capacity.array_hash(w) for w in initial_weights],
        "scores": np.asarray(scores).tolist(),
        "per_example": [
            {
                "index": r["index"],
                "limit": r["limit"].tolist(),
                "gradients": [g.tolist() for g in r["gradients"]],
            }
            for r in capture.records
        ],
        "hidden_spikes_per_presentation": hidden_counts,
        "host_recorded_array_bytes_excluding_python_overhead":
            recorded_array_bytes,
        "phases": observer.phases,
        "memory_snapshots": observer.snapshots,
        "optimizer_updates": 0,
    }
    destination = Path.cwd() / "resource_result.json"
    with destination.open("x") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)

    emit("PASS_RESOURCE_CAPTURE", {
        "recording": args.recording,
        "result": str(destination),
        "initial_weight_sha256": result["initial_weight_sha256"],
        "hidden_spikes_per_presentation": hidden_counts,
        "min_snapshot_free_fraction": min(
            s["free_fraction"] for s in observer.snapshots
        ),
        "optimizer_updates": 0,
        "weights_and_moments_unchanged": True,
        "continuous_device_headroom_certified": False,
        "profiled_times_are_uninstrumented_benchmarks": False,
    })


if __name__ == "__main__":
    main()