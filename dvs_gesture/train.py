"""Train the DVS Gesture EventProp baseline or validate a selected checkpoint."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from ml_genn.compilers import InferenceCompiler
from ml_genn.metrics import SparseCategoricalAccuracy
from ml_genn.serialisers import Numpy

from data import TRIAL_STEPS, encode_parent, load_development_split
from model import build_model, compile_model


class CheckedAccuracy(SparseCategoricalAccuracy):
    def create_state(self):
        state = super().create_state()
        state.scores = []
        return state

    def update(self, state, y_true, y_pred, communicator):
        # argmax can return a class even when the scores contain NaNs.
        if not np.isfinite(y_pred).all():
            raise FloatingPointError("Nonfinite model readout")
        state.scores.append(y_pred.copy())
        super().update(state, y_true, y_pred, communicator)


def encode_subset(dataset, indices, split):
    inputs, labels = [], []
    for index in indices:
        events, label = dataset[index]
        spikes, _ = encode_parent(events)
        print(f"{split}: {dataset.data[index]}, label={label}, spikes={len(spikes.spike_times)}")
        inputs.append(spikes)
        labels.append(label)
    return inputs, np.asarray(labels, dtype=np.int64)


def balanced_indices(dataset, indices, per_class, rng):
    selected = []
    for label in range(len(dataset.classes)):
        candidates = [i for i in indices if dataset.targets[i] == label]
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def pilot_indices(dataset, train_indices, val_indices):
    rng = np.random.default_rng(0)
    return (balanced_indices(dataset, train_indices, 2, rng),
            balanced_indices(dataset, val_indices, 1, rng))


def check_weights(checkpoints, network, previous, current):
    changed = 0
    for connection in network.connections:
        before = checkpoints.deserialise((previous, connection, "weight"))
        after = checkpoints.deserialise((current, connection, "weight"))
        if before.shape != after.shape or not before.size:
            raise ValueError(f"Unexpected checkpoint shape in {connection}")
        if not np.isfinite(before).all() or not np.isfinite(after).all():
            raise FloatingPointError(f"Nonfinite weights in {connection}")
        count = np.count_nonzero(after != before)
        changed += count
        print(f"{connection}: {count}/{after.size} weights changed")
    if changed == 0:
        raise RuntimeError(f"No stored weight changed between {previous} and {current}")


def run(data_root, run_dir, pilot, full_epochs=None):
    dataset, train_indices, val_indices = load_development_split(data_root.resolve())
    run_dir.mkdir()
    os.chdir(run_dir)
    if pilot:
        train_indices, val_indices = pilot_indices(dataset, train_indices, val_indices)
    elif full_epochs is not None:
        # Mix subjects and labels while keeping the first full epoch reproducible.
        np.random.default_rng(0).shuffle(train_indices)
    else:
        train_indices, val_indices = train_indices[:2], val_indices[:1]
    train_x, train_y = encode_subset(dataset, train_indices, "train")
    val_x, val_y = encode_subset(dataset, val_indices, "validation")

    network = build_model()
    input_pop, output_pop = network.populations[0], network.populations[-1]
    compiled = compile_model(network)
    checkpoints = Numpy("checkpoints")
    history = []
    previous = "initial"
    num_epochs = full_epochs if full_epochs is not None else (2 if pilot else 1)
    with compiled:
        compiled.save((previous,), checkpoints)
        for epoch in range(1, num_epochs + 1):
            # Move the preceding epoch's last parent to the front for backward coverage.
            order = np.roll(np.arange(len(train_y)), epoch - 1)
            train_metrics, val_metrics, _, _ = compiled.train_validate(
                {input_pop: [train_x[i] for i in order]}, {output_pop: train_y[order]},
                num_epochs=1, start_epoch=epoch - 1, shuffle=False,
                validation_x={input_pop: val_x}, validation_y={output_pop: val_y},
                metrics=CheckedAccuracy(), callbacks=[], validation_callbacks=[])
            train = train_metrics[output_pop]
            val = val_metrics[output_pop]
            if train.total != len(train_y) or val.total != len(val_y):
                raise RuntimeError("Metric totals do not match the supplied parents")
            current = f"epoch{epoch}"
            compiled.save((current,), checkpoints)
            print(f"Epoch {epoch} | Online train: {train.correct}/{train.total} | "
                  f"Fixed-weight validation: {val.correct}/{val.total}")
            check_weights(checkpoints, network, previous, current)
            if full_epochs is not None:
                scores = np.concatenate(val.scores)
                predicted = scores.argmax(axis=1)
                if scores.shape != (len(val_y), len(dataset.classes)) or np.sum(predicted == val_y) != val.correct:
                    raise ValueError("Saved validation scores disagree with the epoch metric")
                np.savez(f"{current}-validation.npz", paths=[dataset.data[i] for i in val_indices],
                         labels=val_y, scores=scores, predictions=predicted, checkpoint=current)
            previous = current

            history.append(dict(epoch=epoch, checkpoint=current,
                                train_correct=int(train.correct), train_total=int(train.total),
                                validation_correct=int(val.correct), validation_total=int(val.total),
                                final_training_parent=dataset.data[train_indices[order[-1]]]))
            # max keeps the first epoch on a tie. Publish only checked checkpoints.
            best = max(history, key=lambda row: row["validation_correct"])
            selection = dict(planned_epochs=num_epochs, epochs=history,
                             best=best["checkpoint"], latest=current)
            pending = Path("selection.json.tmp")
            pending.write_text(json.dumps(selection, indent=2) + "\n")
            pending.replace("selection.json")
            print(f"Best: {best['checkpoint']} | Latest: {current}")
    if full_epochs == 1:
        print("EventProp full development epoch: OK")
    elif full_epochs is not None:
        print(f"EventProp full development training: {num_epochs} epochs completed")
    else:
        print("EventProp class pilot: OK" if pilot else "EventProp training smoke: OK")


def validate_selected(data_root, run_dir, source_run, *, pilot):
    source_run = source_run.resolve(strict=True)
    saved = json.loads((source_run / "selection.json").read_text())
    best = min(saved["epochs"], key=lambda row: (-row["validation_correct"], row["epoch"]))
    key = best["checkpoint"]
    if key != saved["best"]:
        raise ValueError("Selected checkpoint disagrees with the epoch selection rule")
    dataset, train_indices, val_indices = load_development_split(data_root.resolve())
    if pilot:
        _, val_indices = pilot_indices(dataset, train_indices, val_indices)
    if len(val_indices) != best["validation_total"]:
        raise ValueError("Validation split size disagrees with the selected epoch")
    paths = [dataset.data[i] for i in val_indices]
    if not pilot:
        with np.load(source_run / f"{key}-validation.npz", allow_pickle=False) as ref:
            if ref["checkpoint"].item() != key or not np.array_equal(ref["paths"], paths):
                raise ValueError("Reference checkpoint or ordered validation paths differ")
            reference_labels = ref["labels"]
            reference_scores = ref["scores"]
            reference_predictions = ref["predictions"]
        shape = (len(paths), len(dataset.classes))
        if reference_scores.shape != shape or not np.isfinite(reference_scores).all():
            raise ValueError("Invalid reference scores")
        if not np.array_equal(reference_scores.argmax(axis=1), reference_predictions):
            raise ValueError("Reference predictions differ from the saved score argmax")
        if not np.array_equal(reference_labels, [dataset.targets[i] for i in val_indices]):
            raise ValueError("Reference labels differ from the ordered Tonic labels")
        if np.count_nonzero(reference_predictions == reference_labels) != best["validation_correct"]:
            raise ValueError("Reference correct count disagrees with the selected epoch")
    val_x, val_y = encode_subset(dataset, val_indices, "validation")
    source = source_run / "checkpoints"
    if not source.is_dir():
        raise FileNotFoundError(source)
    weights = Numpy(str(source))
    network = build_model()
    network.load((key,), weights)

    run_dir.mkdir()
    os.chdir(run_dir)
    compiler = InferenceCompiler(evaluate_timesteps=TRIAL_STEPS, backend="cuda",
                                 reset_in_syn_between_batches=True)
    compiled = compiler.compile(network, name="DVSGesture")
    output = network.populations[-1]
    with compiled:
        for connection in network.connections:
            expected = weights.deserialise((key, connection, "weight"))
            device_weight = compiled.connection_populations[connection].vars["g"]
            device_weight.pull_from_device()
            if not np.isfinite(expected).all() or not np.array_equal(device_weight.values, expected):
                raise ValueError(f"Reloaded weights differ for {connection}")
        predictions, _ = compiled.predict({network.populations[0]: val_x}, output, callbacks=[])
    scores = predictions[output]
    if scores.shape != (len(val_y), len(dataset.classes)) or not np.isfinite(scores).all():
        raise ValueError("Unexpected validation scores")
    predicted = scores.argmax(axis=1)
    correct = int(np.count_nonzero(predicted == val_y))
    np.savez("validation.npz", paths=paths,
             labels=val_y, scores=scores, predictions=predicted, checkpoint=key)
    print(f"Reloaded {key}: {len(network.connections)} weight arrays matched")
    original = "Pilot" if pilot else "Training-run validation"
    print(f"Frozen validation: {correct}/{len(val_y)} | {original}: {best['validation_correct']}/{best['validation_total']}")
    if not pilot:
        matches = predicted == reference_predictions
        difference = np.abs(scores.astype(np.float64) - reference_scores)
        print(f"Prediction matches: {np.count_nonzero(matches)}/{len(matches)}")
        print(f"Score matches (exact): {np.count_nonzero(difference == 0)}/{difference.size}")
        print(f"Absolute score difference: max={difference.max():.9g}, mean={difference.mean():.9g}")
        for i in np.flatnonzero(~matches):
            print(f"Mismatch: {paths[i]}, label={val_y[i]}, "
                  f"training-run prediction={reference_predictions[i]}, reload prediction={predicted[i]}")
        if not matches.all():
            raise RuntimeError("Per-parent validation predictions differ; see validation.npz")
    if correct != best["validation_correct"]:
        raise RuntimeError("Frozen inference and the selected epoch differ in validation correct count")
    print("Selected pilot checkpoint: OK" if pilot else "Full validation predictions: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Parent of the DVSGesture directory")
    parser.add_argument("--run-dir", type=Path, required=True, help="New output directory")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true", help="Use all eleven classes for two epochs")
    mode.add_argument("--full-epoch", dest="full_epochs", action="store_const", const=1,
                      help="Run one fresh epoch on the full training split")
    mode.add_argument("--full-epochs", type=int, metavar="N",
                      help="Run N epochs on the full training split with validation after each epoch")
    mode.add_argument("--validate-pilot", type=Path, metavar="PILOT_RUN",
                      help="Reload the best pilot checkpoint in a fresh inference model")
    mode.add_argument("--validate-full", type=Path, metavar="TRAINING_RUN",
                      help="Reload the selected full checkpoint and compare each validation prediction")
    args = parser.parse_args()
    if args.full_epochs is not None and args.full_epochs < 1:
        parser.error("--full-epochs must be at least 1")
    if args.validate_pilot is not None or args.validate_full is not None:
        validate_selected(args.data_root, args.run_dir, args.validate_pilot or args.validate_full,
                          pilot=args.validate_pilot is not None)
    else:
        run(args.data_root, args.run_dir, args.pilot, args.full_epochs)
