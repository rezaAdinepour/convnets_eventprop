"""TASK 09: tiny training, validation, and checkpoint-integrity probe.

Train and reload must be separate Python processes to preserve stable names.
Uses only development samples. No official test event arrays are loaded.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import InferenceCompiler
from ml_genn.metrics.sparse_categorical_accuracy import (
    SparseCategoricalAccuracy,
)
from ml_genn.serialisers import Numpy
from ml_genn.utils.callback_list import CallbackList

from dvs_data import (
    Config as DataConfig,
    PreprocessedSpikes,
    audit_development_content,
    file_hash,
    inventory,
    load_sample,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_model import build_model, compile_model
from dvs_eventprop_probe import Observe, emit, pull
from dvs_eventprop_drain_probe import DrainTrace, RealOnlyAccuracy, weights


LOGICAL_SHAPES = {
    "hidden": (5, 5, 2, 8),
    "output": (392, 11),
}


class ExclusiveNumpy(Numpy):
    """Preserve Numpy key semantics while refusing existing output files."""

    def serialise(self, keys, data):
        filename = Path(self.path) / (
            "-".join(str(key) for key in keys) + ".npy"
        )
        with filename.open("xb") as stream:
            np.save(stream, data, allow_pickle=False)


class ScoreMetric(SparseCategoricalAccuracy):
    def __init__(self):
        self.scores = []

    def update(self, state, y_true, y_pred, communicator):
        require(y_pred.shape == (1, 11), "Unexpected validation score shape")
        require(y_pred.dtype == np.float32, "Expected FP32 scores")
        require(bool(np.isfinite(y_pred).all()), "Nonfinite validation scores")
        self.scores.append(y_pred.copy())
        super().update(state, y_true, y_pred, communicator)


def identity():
    directory = Path(__file__).resolve().parent
    source = {}
    for name in (
        "dvs_data.py",
        "dvs_eventprop_config.py",
        "dvs_eventprop_model.py",
        "dvs_eventprop_probe.py",
        "dvs_eventprop_drain_probe.py",
        "dvs_eventprop_checkpoint_probe.py",
    ):
        source[name] = file_hash(directory / name)

    def git_head(path):
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()

    return {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "ml_genn": importlib.metadata.version("ml_genn"),
        "pygenn": importlib.metadata.version("pygenn"),
        "convnets_commit": git_head(directory),
        "ml_genn_commit": git_head(directory.parent / "ml_genn"),
        "source_sha256": source,
    }


def load_rows(root, rows, config):
    loaded = [
        load_sample(root, row, DataConfig(config.frames))
        for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    return (
        [to_ml_genn(p) for p in payloads],
        np.asarray(labels, dtype=np.int64),
        list(metadata),
    )


def summary(scores, labels):
    s = scores.astype(np.float64)
    maximum = s.max(axis=1)
    ce = (
        maximum
        + np.log(np.exp(s - maximum[:, None]).sum(axis=1))
        - s[np.arange(len(labels)), labels]
    )
    return {
        "correct": int(np.sum(scores.argmax(axis=1) == labels)),
        "total": len(labels),
        "mean_ce": float(ce.mean()),
        "top_score_ties": int(np.sum(
            (scores == scores.max(axis=1, keepdims=True)).sum(axis=1) > 1
        )),
    }


def training_state(cn, layers):
    state = weights(cn, layers)
    for _, custom_update in cn.optimisers:
        state.extend([
            pull(custom_update.vars["M"]),
            pull(custom_update.vars["V"]),
        ])
    return state


def validate_pass(cn, inp, out, inputs, labels):
    metric = ScoreMetric()
    state = {out: metric.create_state()}
    callbacks = CallbackList(
        cn.base_validate_callbacks,
        compiled_network=cn,
        num_batches=len(inputs),
        num_epochs=1,
    )
    callbacks.on_test_begin()
    callbacks.on_epoch_begin(0)
    for batch, payload in enumerate(inputs):
        cn._validate_batch(
            batch,
            {inp: [payload]},
            {out: labels[batch:batch + 1]},
            {out: metric},
            state,
            callbacks,
        )
    callbacks.on_epoch_end(0, state)
    callbacks.on_test_end(state)
    require(state[out].total == len(inputs), "Wrong validation denominator")
    return np.concatenate(metric.scores, axis=0)


def save_checkpoint(cn, layers, serialiser, keys):
    expected_variables = {
        (layer.connection(), "weight") for layer in layers
    }
    require(
        set(cn.checkpoint_connection_vars) == expected_variables,
        "Unexpected checkpoint connection variables",
    )
    require(not cn.checkpoint_population_vars,
            "Unexpected checkpoint population variables")

    cn.save(keys, serialiser)
    files = {}

    for label, layer in zip(("hidden", "output"), layers):
        require(str(layer.connection()) == label, "Unexpected connection name")
        path = Path(serialiser.path) / (
            "-".join(str(key) for key in keys)
            + f"-{label}-weight.npy"
        )
        array = np.load(path, allow_pickle=False)
        logical = LOGICAL_SHAPES[label]
        require(array.dtype == np.float32, "Checkpoint must contain FP32")
        require(array.size == int(np.prod(logical)),
                "Wrong checkpoint weight count")
        require(bool(np.isfinite(array).all()), "Nonfinite saved weights")

        var = cn.connection_populations[layer.connection()].vars["weight"]
        var.pull_from_device()
        np.testing.assert_array_equal(array, var.values)

        files[label] = {
            "filename": path.name,
            "sha256": file_hash(path),
            "storage_shape": list(array.shape),
            "logical_shape": list(logical),
            "dtype": str(array.dtype),
        }

    return {"keys": list(keys), "files": files}


def train(root, directory, config):
    # The directory must be absent: never reuse a checkpoint destination.
    directory.mkdir(parents=False, exist_ok=False)
    serialiser = ExclusiveNumpy(str(directory))

    partitions = inventory(root)
    train_rows = pilot(partitions["train"], 2)[:3]
    val_rows = pilot(partitions["validation"], 1)

    # This is a content-duplication audit of the selected 3/11 samples only.
    content_audit = audit_development_content(
        root, {"train": train_rows, "validation": val_rows}
    )
    x, y, train_metadata = load_rows(root, train_rows, config)
    vx, vy, val_metadata = load_rows(root, val_rows, config)

    provenance = identity()
    emit("IDENTITY", provenance)
    emit("CONFIG", config.record())
    for item in train_metadata + val_metadata:
        emit("SAMPLE_AUDIT", item)

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    inputs = x + [empty]
    targets = np.r_[y, np.int64(0)]

    _, cn, inp, hid, out = compile_model(
        config, 1, learn=True, name="DVSCheckpointTraining",
    )
    layers = [hid, out]
    context = {"epoch": 0, "step": 0, "phase": "warmup"}
    matches = [
        (i, callback)
        for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    index, original = matches[0]
    trace = DrainTrace(original, layers, context, config.learning_rate)
    cn.base_train_callbacks[index] = trace

    metric = RealOnlyAccuracy()
    updates = 0
    epochs = []

    with cn:
        callbacks = CallbackList(
            cn.base_train_callbacks + [Observe(inp, out, targets)],
            compiled_network=cn,
            num_batches=4,
            num_epochs=2,
        )
        callbacks.on_train_begin()

        for epoch in range(2):
            context["epoch"] = epoch
            state = {out: metric.create_state()}
            callbacks.on_epoch_begin(epoch)

            for batch in range(4):
                context["step"] = updates
                context["phase"] = (
                    "drain" if batch == 3 else
                    "warmup" if batch == 0 else "real_forward"
                )
                metric.real_presentation = batch != 3
                before = weights(cn, layers) if batch == 0 else None

                cn._train_batch(
                    batch, updates,
                    {inp: [inputs[batch]]},
                    {out: targets[batch:batch + 1]},
                    {out: metric}, state, callbacks,
                )
                if batch == 0:
                    for old, new in zip(before, weights(cn, layers)):
                        np.testing.assert_array_equal(old, new)
                else:
                    updates += 1

            callbacks.on_epoch_end(epoch, state)
            require(state[out].total == 3, "Drain entered training metrics")

            before_validation = training_state(cn, layers)
            call_count = len(trace.calls)

            scores = validate_pass(cn, inp, out, vx, vy)
            changed_labels = validate_pass(
                cn, inp, out, vx, (vy + 1) % 11
            )
            reversed_scores = validate_pass(
                cn, inp, out, list(reversed(vx)), vy[::-1]
            )[::-1]

            for candidate in (changed_labels, reversed_scores):
                np.testing.assert_allclose(
                    candidate, scores, rtol=1e-5, atol=1e-6,
                    err_msg="Validation label/order score mismatch",
                )
                np.testing.assert_array_equal(
                    candidate.argmax(axis=1), scores.argmax(axis=1)
                )

            for old, new in zip(
                before_validation, training_state(cn, layers)
            ):
                np.testing.assert_array_equal(
                    old, new,
                    err_msg="Validation changed weights or Adam moments",
                )
            require(len(trace.calls) == call_count,
                    "Validation triggered learning")

            entry = save_checkpoint(
                cn, layers, serialiser, ("epoch", epoch)
            )
            entry.update({
                "epoch": epoch,
                "validation_scores": scores.tolist(),
                "validation": summary(scores, vy),
            })
            epochs.append(entry)
            emit("VALIDATION", {
                "epoch": epoch,
                **entry["validation"],
                "label_order_checks": "pass",
                "weights_and_adam_unchanged": True,
                "cumulative_updates": updates,
            })
            emit("CHECKPOINT", entry["files"])

        callbacks.on_train_end(state)
        final = save_checkpoint(cn, layers, serialiser, ("final",))

    require(
        trace.calls == [
            (0, 0, 0), (0, 1, 1), (0, 2, 2),
            (1, 0, 3), (1, 1, 4), (1, 2, 5),
        ],
        "Unexpected update ownership",
    )

    # Earliest epoch wins ties in validation correct count.
    best = max(range(2), key=lambda e: (
        epochs[e]["validation"]["correct"], -e
    ))
    final.update({
        "epoch": 1,
        "validation_scores": epochs[1]["validation_scores"],
        "validation": epochs[1]["validation"],
    })

    manifest = {
        "config": config.record(),
        "identity": provenance,
        "train_ids": [row.stable_id for row in train_rows],
        "validation_ids": [row.stable_id for row in val_rows],
        "validation_labels": vy.tolist(),
        "train_metadata": train_metadata,
        "validation_metadata": val_metadata,
        "selected_content_audit": content_audit,
        "epochs": epochs,
        "best_epoch": best,
        "selection_rule": "highest validation correct; earliest epoch on ties",
        "final": final,
        "updates": updates,
        "weight_only": True,
    }
    with (directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)

    emit("PASS_SAVE_VALIDATION", {
        "updates": updates,
        "validation_samples": len(vy),
        "best_epoch": best,
        "final_epoch": 1,
        "manifest": str(directory / "manifest.json"),
        "reload_verified": False,
    })


def reload_checkpoint(root, directory, selection, config):
    with (directory / "manifest.json").open() as stream:
        manifest = json.load(stream)

    require(manifest["config"] == config.record(), "Configuration changed")
    current_identity = identity()
    require(current_identity == manifest["identity"],
            "Environment/source identity changed")

    entries = manifest["epochs"] + [manifest["final"]]
    expected_files = set()
    for entry in entries:
        for label, info in entry["files"].items():
            path = directory / info["filename"]
            require(path.name == info["filename"], "Unsafe checkpoint filename")
            require(file_hash(path) == info["sha256"], "Checkpoint hash changed")
            array = np.load(path, allow_pickle=False)
            require(list(array.shape) == info["storage_shape"],
                    "Checkpoint storage shape changed")
            require(array.dtype == np.float32, "Checkpoint dtype changed")
            require(array.size == int(np.prod(LOGICAL_SHAPES[label])),
                    "Checkpoint weight count changed")
            require(bool(np.isfinite(array).all()), "Nonfinite saved weights")
            expected_files.add(path.name)

    require(
        {p.name for p in directory.glob("*.npy")} == expected_files,
        "Missing or unexpected checkpoint arrays",
    )

    # Explicitly verify final and last-epoch arrays match.
    for label in LOGICAL_SHAPES:
        a = np.load(
            directory / manifest["epochs"][-1]["files"][label]["filename"],
            allow_pickle=False,
        )
        b = np.load(
            directory / manifest["final"]["files"][label]["filename"],
            allow_pickle=False,
        )
        np.testing.assert_array_equal(a, b)

    entry = (
        manifest["epochs"][manifest["best_epoch"]]
        if selection == "best" else manifest["final"]
    )
    partitions = inventory(root)
    rows = pilot(partitions["validation"], 1)
    require(
        [row.stable_id for row in rows] == manifest["validation_ids"],
        "Validation membership changed",
    )
    vx, vy, metadata = load_rows(root, rows, config)
    # Compare the same JSON representation stored in the manifest.
    normalized_metadata = json.loads(
        json.dumps(metadata, allow_nan=False)
    )
    require(normalized_metadata == manifest["validation_metadata"],
            "Validation input/provenance changed")
    require(vy.tolist() == manifest["validation_labels"], "Labels changed")

    # First model constructed in this process: names must match saved names.
    network, inp, hid, out = build_model(config, 1)
    serialiser = Numpy(str(directory))
    network.load(tuple(entry["keys"]), serialiser)

    for label, layer in zip(("hidden", "output"), (hid, out)):
        require(str(layer.connection()) == label, "Network names changed")
        loaded = np.asarray(layer.connection().connectivity.weight)
        saved = np.load(
            directory / entry["files"][label]["filename"],
            allow_pickle=False,
        )
        np.testing.assert_array_equal(loaded, saved)
        # Restore explicit logical axes from GeNN's saved storage layout.
        layer.connection().connectivity.weight = loaded.reshape(
            LOGICAL_SHAPES[label]
        ).copy()

    compiler = InferenceCompiler(
        evaluate_timesteps=config.frames,
        dt=config.dt_ms,
        batch_size=1,
        rng_seed=config.seed,
        reset_time_between_batches=True,
        reset_vars_between_batches=True,
        reset_in_syn_between_batches=True,
    )
    cn = compiler.compile(network, name=f"DVSReload{selection}")

    with cn:
        predictions, _ = cn.predict(
            {inp: vx}, [out], callbacks=[]
        )
        scores = predictions[out]

        reversed_predictions, _ = cn.predict(
            {inp: list(reversed(vx))}, [out], callbacks=[]
        )
        np.testing.assert_allclose(
            reversed_predictions[out][::-1], scores,
            rtol=1e-5, atol=1e-6,
            err_msg="Inference predecessor/order dependence",
        )

    expected = np.asarray(entry["validation_scores"], dtype=np.float32)
    require(scores.dtype == np.float32 and scores.shape == (11, 11),
            "Unexpected inference scores")
    np.testing.assert_allclose(
        scores, expected, rtol=1e-5, atol=1e-6,
        err_msg="Fresh inference differs from saved validation scores",
    )
    np.testing.assert_array_equal(
        scores.argmax(axis=1), expected.argmax(axis=1)
    )
    emit("PASS_RELOAD", {
        "selection": selection,
        "epoch": entry["epoch"],
        "max_absolute_score_difference": float(
            np.max(np.abs(scores.astype(np.float64) - expected))
        ),
        **summary(scores, vy),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "reload"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--selection", choices=("best", "final"), default="best")
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    root = args.root.resolve(strict=True)
    directory = args.checkpoints.resolve()

    if args.mode == "train":
        train(root, directory, config)
    else:
        reload_checkpoint(root, directory, args.selection, config)


if __name__ == "__main__":
    main()
