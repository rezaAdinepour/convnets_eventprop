"""TASK 09: one-epoch EventProp smoke on the fixed 22/11 DVS pilot.

Train and reload are separate processes. Official test events stay held out.
This is an engineering diagnostic, not a generalization experiment.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.utils.callback_list import CallbackList

import dvs_eventprop_checkpoint_probe as checkpoint
from dvs_data import (
    PreprocessedSpikes,
    audit_development_content,
    file_hash,
    inventory,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_model import compile_model
from dvs_eventprop_probe import Observe, emit, expected_pairs, pull
from dvs_eventprop_drain_probe import DrainTrace, RealOnlyAccuracy, weights


class GradientTrace(DrainTrace):
    def __init__(self, original, layers, context, alpha):
        super().__init__(original, layers, context, alpha)
        self.gradient_norms = []

    def on_batch_end(self, cn, batch, metric_state):
        if batch > 0:
            norms = []
            for layer in self.layers:
                group = cn.connection_populations[layer.connection()]
                gradient = pull(group.vars["weightGradient"])
                norms.append(float(np.linalg.norm(
                    gradient.astype(np.float64)
                )))
            self.gradient_norms.append(norms)
        super().on_batch_end(cn, batch, metric_state)


def train(root, directory, config):
    directory.mkdir(parents=False, exist_ok=False)
    serialiser = checkpoint.ExclusiveNumpy(str(directory))

    partitions = inventory(root)
    selected_train = pilot(partitions["train"], 2)
    val_rows = pilot(partitions["validation"], 1)
    require(len(selected_train) == 22 and len(val_rows) == 11,
            "Unexpected pilot sizes")

    # Explicit, recorded permutation of the fixed pilot; no sample selection.
    rng = np.random.default_rng(config.seed)
    order = rng.permutation(22)
    train_rows = tuple(selected_train[int(i)] for i in order)

    content_audit = audit_development_content(
        root, {"train": train_rows, "validation": val_rows}
    )
    x, y, train_metadata = checkpoint.load_rows(root, train_rows, config)
    vx, vy, val_metadata = checkpoint.load_rows(root, val_rows, config)

    provenance = checkpoint.identity()
    runner_hash = file_hash(Path(__file__))
    emit("IDENTITY", {
        **provenance,
        "smoke_runner_sha256": runner_hash,
    })
    emit("SMOKE_PLAN", {
        **config.record(),
        "training_samples": 22,
        "validation_samples": 11,
        "epochs": 1,
        "batch_size": 1,
        "shuffle_generator": type(rng.bit_generator).__name__,
        "shuffle_seed": config.seed,
        "permutation_of_sorted_pilot": order.tolist(),
        "training_order": [row.stable_id for row in train_rows],
        "validation_order": [row.stable_id for row in val_rows],
    })
    for item in train_metadata + val_metadata:
        emit("SAMPLE_AUDIT", item)

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    inputs = x + [empty]
    targets = np.r_[y, np.int64(0)]

    _, cn, inp, hid, out = compile_model(
        config, 1, learn=True, name="DVSSmoke22",
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
    trace = GradientTrace(original, layers, context, config.learning_rate)
    cn.base_train_callbacks[index] = trace

    metric = RealOnlyAccuracy()
    state = {out: metric.create_state()}
    updates = 0

    with cn:
        initial_weights = weights(cn, layers)
        emit("CONNECTIVITY", {
            label: str(cn.connection_populations[layer.connection()].matrix_type)
            for label, layer in zip(("hidden", "output"), layers)
        })

        callbacks = CallbackList(
            cn.base_train_callbacks + [
                SpikeRecorder(inp, key="input_spikes"),
                SpikeRecorder(hid, key="hidden_counts", record_counts=True),
                Observe(inp, out, targets),
            ],
            compiled_network=cn,
            num_batches=23,
            num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        for batch in range(23):
            context["step"] = updates
            context["phase"] = (
                "drain" if batch == 22 else
                "warmup" if batch == 0 else "real_forward"
            )
            metric.real_presentation = batch != 22

            cn._train_batch(
                batch, updates,
                {inp: [inputs[batch]]},
                {out: targets[batch:batch + 1]},
                {out: metric}, state, callbacks,
            )

            if batch == 0:
                for old, new in zip(initial_weights, weights(cn, layers)):
                    np.testing.assert_array_equal(old, new)
            else:
                updates += 1

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)

        require(state[out].total == 22, "Drain contaminated metrics")
        require(trace.calls == [(0, i, i) for i in range(22)],
                "Missing or duplicate real-sample update")

        recorded = callbacks.get_data()
        times, ids = recorded["input_spikes"]
        require(len(times) == len(ids) == 23, "Wrong input recording count")
        for index, payload in enumerate(inputs):
            actual = sorted(zip(
                np.asarray(ids[index]).astype(np.int64).tolist(),
                np.asarray(times[index]).astype(np.float64).tolist(),
            ))
            require(actual == expected_pairs(payload),
                    f"Real-input delivery mismatch at presentation {index}")

        hidden_counts = np.asarray(recorded["hidden_counts"])
        require(hidden_counts.shape == (23, 392),
                "Unexpected hidden count axes")
        require(bool((hidden_counts >= 0).all()), "Negative spike count")
        require(bool((hidden_counts <= config.frames).all()),
                "Hidden activity exceeds one spike/neuron/step")
        require(bool((hidden_counts[-1] == 0).all()),
                "Hidden spikes during empty drain")

        real_counts = hidden_counts[:22]
        activity = {
            "count_axes": ["sample", "flattened_y_x_filter"],
            "shape": [22, 392],
            "spikes_per_sample": real_counts.sum(axis=1).astype(int).tolist(),
            "spikes_per_filter": real_counts.reshape(
                22, 7, 7, 8
            ).sum(axis=(0, 1, 2)).astype(int).tolist(),
            "max_spikes_per_neuron_trial": int(real_counts.max()),
            "silent_hidden_trials": int(np.sum(
                real_counts.sum(axis=1) == 0
            )),
        }
        emit("HIDDEN_ACTIVITY", activity)

        gradient_sums = np.asarray(trace.gradient_norms).sum(axis=0)
        require(bool((gradient_sums > 0).all()),
                "A projection received no aggregate task gradient")
        final_weights = weights(cn, layers)
        weight_changes = [
            float(np.linalg.norm(
                new.astype(np.float64) - old.astype(np.float64)
            ))
            for old, new in zip(initial_weights, final_weights)
        ]
        require(all(change > 0 for change in weight_changes),
                "A learned projection did not change")

        # All following scores use the fixed final weights.
        before_evaluation = checkpoint.training_state(cn, layers)
        calls_before = len(trace.calls)

        train_scores = checkpoint.validate_pass(cn, inp, out, x, y)
        val_scores = checkpoint.validate_pass(cn, inp, out, vx, vy)
        changed_labels = checkpoint.validate_pass(
            cn, inp, out, vx, (vy + 1) % 11
        )
        reversed_scores = checkpoint.validate_pass(
            cn, inp, out, list(reversed(vx)), vy[::-1]
        )[::-1]

        for candidate in (changed_labels, reversed_scores):
            np.testing.assert_allclose(
                candidate, val_scores, rtol=1e-5, atol=1e-6
            )
            np.testing.assert_array_equal(
                candidate.argmax(axis=1), val_scores.argmax(axis=1)
            )

        for old, new in zip(
            before_evaluation, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)
        require(len(trace.calls) == calls_before,
                "Evaluation triggered learning")

        entry = checkpoint.save_checkpoint(
            cn, layers, serialiser, ("epoch", 0)
        )
        entry.update({
            "epoch": 0,
            "validation_scores": val_scores.tolist(),
            "validation": checkpoint.summary(val_scores, vy),
        })
        final = checkpoint.save_checkpoint(
            cn, layers, serialiser, ("final",)
        )
        final.update({
            "epoch": 0,
            "validation_scores": val_scores.tolist(),
            "validation": entry["validation"],
        })

    train_result = checkpoint.summary(train_scores, y)
    manifest = {
        "config": config.record(),
        "identity": provenance,
        "smoke_runner_sha256": runner_hash,
        "train_ids": [row.stable_id for row in train_rows],
        "validation_ids": [row.stable_id for row in val_rows],
        "validation_labels": vy.tolist(),
        "train_metadata": train_metadata,
        "validation_metadata": val_metadata,
        "selected_content_audit": content_audit,
        "permutation_of_sorted_pilot": order.tolist(),
        "epochs": [entry],
        "best_epoch": 0,
        "selection_rule": "highest validation correct; earliest epoch on ties",
        "final": final,
        "updates": updates,
        "weight_only": True,
        "fresh_training": train_result,
        "hidden_activity": activity,
        "gradient_l2_sum_hidden_output": gradient_sums.tolist(),
        "weight_change_l2_hidden_output": weight_changes,
    }
    with (directory / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)

    emit("PASS_SMOKE_SAVE", {
        "updates": updates,
        "real_online_metric_examples": int(state[out].total),
        "fresh_training": train_result,
        "validation": entry["validation"],
        "gradient_l2_sum_hidden_output": gradient_sums.tolist(),
        "weight_change_l2_hidden_output": weight_changes,
        "input_delivery": "pass",
        "validation_label_order_checks": "pass",
        "evaluation_weights_and_moments_unchanged": True,
        "manifest": str(directory / "manifest.json"),
        "reload_verified": False,
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
        with (directory / "manifest.json").open() as stream:
            manifest = json.load(stream)
        require(
            file_hash(Path(__file__)) == manifest["smoke_runner_sha256"],
            "Smoke runner changed",
        )
        checkpoint.reload_checkpoint(
            root, directory, args.selection, config
        )


if __name__ == "__main__":
    main()
