"""Train the DVS Gesture EventProp baseline and evaluate its selected checkpoint."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ml_genn.metrics import SparseCategoricalAccuracy
from ml_genn.serialisers import Numpy

from data import (EVDOWNSAMPLING, HISTORICAL_VALIDATION_SUBJECTS, OCCUPANCY, PREPROCESSORS,
                  TRIAL_STEPS, VALIDATION_SUBJECTS, encode_trial, load_development_split, load_partition)
from evdownsampling import PARAMETERS
from model import MODEL_NAME, build_model, compile_model, inference_compiler, tape_slots

SHUFFLE_SEED = 0
PILOT_SEED = 0
PILOT_TRAIN_PER_CLASS = 2
PILOT_VALIDATION_PER_CLASS = 1
PILOT_EPOCHS = 2
SMOKE_TRAIN_PARENTS = 2
SMOKE_VALIDATION_PARENTS = 1
SMOKE_EPOCHS = 1


def mean_cross_entropy(scores, labels):
    """Mean sparse categorical cross-entropy from per-parent readout logits."""
    logits = np.asarray(scores, dtype=np.float64)
    if logits.ndim != 2 or labels.shape != (len(logits),) or len(labels) == 0:
        raise ValueError("Expected one class label per score vector")
    if np.any((labels < 0) | (labels >= logits.shape[1])) or not np.isfinite(logits).all():
        raise ValueError("Invalid labels or nonfinite class scores")
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    return float(nll.mean())


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


def encode_subset(dataset, indices, split, *, whole=False, preprocessing=OCCUPANCY):
    inputs, labels, lengths = [], [], []
    for index in indices:
        events, label = dataset[index]
        spikes, _, steps = encode_trial(events, whole=whole, preprocessing=preprocessing)
        print(f"{split}: {dataset.data[index]}, label={label}, frames={steps}, spikes={len(spikes.spike_times)}")
        inputs.append(spikes)
        labels.append(label)
        lengths.append(steps)
    return inputs, np.asarray(labels, dtype=np.int64), np.asarray(lengths)


def balanced_indices(dataset, indices, per_class, rng):
    selected = []
    for label in range(len(dataset.classes)):
        candidates = [i for i in indices if dataset.targets[i] == label]
        selected.extend(rng.choice(candidates, per_class, replace=False).tolist())
    rng.shuffle(selected)
    return selected


def pilot_indices(dataset, train_indices, val_indices):
    rng = np.random.default_rng(PILOT_SEED)
    return (balanced_indices(dataset, train_indices, PILOT_TRAIN_PER_CLASS, rng),
            balanced_indices(dataset, val_indices, PILOT_VALIDATION_PER_CLASS, rng))


def check_weights(checkpoints, network, previous, current, *, require_change=True):
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
    if require_change and changed == 0:
        raise RuntimeError(f"No stored weight changed between {previous} and {current}")


def run(data_root, run_dir, pilot, full_epochs=None, *, whole=False,
        preprocessing=OCCUPANCY, initial_from=None):
    dataset, train_indices, val_indices = load_development_split(data_root.resolve())
    train_subjects = sorted(set(dataset.users[i] for i in train_indices))
    val_subjects = sorted(VALIDATION_SUBJECTS)
    print(f"Training subjects {train_subjects}: {len(train_indices)} parents")
    print(f"Validation subjects {val_subjects}: {len(val_indices)} parents")
    if initial_from is not None:
        initial_from = initial_from.resolve(strict=True)
        initial_weights = Numpy(str((initial_from / "checkpoints").resolve(strict=True)))
    run_dir.mkdir()
    os.chdir(run_dir)
    if pilot:
        train_indices, val_indices = pilot_indices(dataset, train_indices, val_indices)
    elif full_epochs is not None:
        # Mix subjects and labels while keeping the first full epoch reproducible.
        np.random.default_rng(SHUFFLE_SEED).shuffle(train_indices)
    elif not whole:
        train_indices = train_indices[:SMOKE_TRAIN_PARENTS]
        val_indices = val_indices[:SMOKE_VALIDATION_PARENTS]
    train_x, train_y, train_lengths = encode_subset(
        dataset, train_indices, "train", whole=whole, preprocessing=preprocessing)
    val_x, val_y, val_lengths = encode_subset(
        dataset, val_indices, "validation", whole=whole, preprocessing=preprocessing)
    steps = int(max(train_lengths.max(), val_lengths.max()))
    capacity = max(1, max(len(x.spike_times) for x in train_x + val_x)) if whole else None
    protocol = dict(whole_gesture=whole, trial_steps=steps, input_capacity=capacity,
                    preprocessing=preprocessing, validation_subjects=val_subjects,
                    train_subjects=train_subjects, shuffle_seed=SHUFFLE_SEED)
    if preprocessing == EVDOWNSAMPLING:
        protocol["evdownsampling"] = dict(PARAMETERS)
    if initial_from is not None:
        protocol["initial_weights_from"] = str(initial_from)
    if whole:
        print(f"Development horizon: {steps} steps | Input capacity: {capacity} | Ring slots: {tape_slots(steps)}")
    if whole and full_epochs is None:
        # Exercise the production allocation and delayed update with two long parents.
        longest = np.argsort(train_lengths, kind="stable")[-SMOKE_TRAIN_PARENTS:][::-1]
        train_indices = [train_indices[i] for i in longest]
        train_x, train_y = [train_x[i] for i in longest], train_y[longest]
        v = int(val_lengths.argmax())
        val_indices, val_x, val_y = [val_indices[v]], [val_x[v]], val_y[v:v + 1]

    network = build_model(capacity)
    if initial_from is not None:
        network.load(("initial",), initial_weights)
    input_pop, output_pop = network.populations[0], network.populations[-1]
    compiled = compile_model(network, steps, whole=whole)
    checkpoints = Numpy("checkpoints")
    history = []
    previous = "initial"
    num_epochs = full_epochs if full_epochs is not None else (PILOT_EPOCHS if pilot else SMOKE_EPOCHS)
    rng = np.random.default_rng(SHUFFLE_SEED)
    order = np.arange(len(train_y))
    with compiled:
        compiled.save((previous,), checkpoints)
        if initial_from is not None:
            for conn in network.connections:
                actual = checkpoints.deserialise(("initial", conn, "weight"))
                expected = initial_weights.deserialise(("initial", conn, "weight"))
                if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                    raise ValueError(f"Initial weights differ in {conn}")
            print(f"Matched initial weights in {len(network.connections)} connections")
        for epoch in range(1, num_epochs + 1):
            if epoch > 1:
                if whole:
                    # Give the previous final parent a backward pass, shuffle the rest.
                    order = np.concatenate((order[-1:], rng.permutation(order[:-1])))
                else:
                    order = np.roll(order, 1)
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
            train_ce = mean_cross_entropy(np.concatenate(train.scores), train_y[order])
            scores = np.concatenate(val.scores)
            val_ce = mean_cross_entropy(scores, val_y)
            print(f"Online train CE: {train_ce:.6f} | Fixed-weight validation CE: {val_ce:.6f}")
            check_weights(checkpoints, network, previous, current,
                          require_change=(full_epochs is None or epoch == 1))
            if full_epochs is not None:
                predicted = scores.argmax(axis=1)
                if scores.shape != (len(val_y), len(dataset.classes)) or np.sum(predicted == val_y) != val.correct:
                    raise ValueError("Saved validation scores disagree with the epoch metric")
                np.savez(f"{current}-validation.npz", paths=[dataset.data[i] for i in val_indices],
                         labels=val_y, scores=scores, predictions=predicted, checkpoint=current)
            previous = current

            history.append(dict(epoch=epoch, checkpoint=current,
                                train_correct=int(train.correct), train_total=int(train.total),
                                validation_correct=int(val.correct), validation_total=int(val.total),
                                train_cross_entropy=train_ce, validation_cross_entropy=val_ce,
                                final_training_parent=dataset.data[train_indices[order[-1]]]))
            # max keeps the first epoch on a tie. Publish only checked checkpoints.
            best = max(history, key=lambda row: row["validation_correct"])
            lowest_loss = min(history, key=lambda row: row["validation_cross_entropy"])
            selection = dict(planned_epochs=num_epochs, epochs=history, protocol=protocol,
                             best=best["checkpoint"], latest=current,
                             lowest_validation_loss=lowest_loss["checkpoint"])
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


def assert_loaded_weights(compiled, network, weights, key):
    for connection in network.connections:
        expected = weights.deserialise((key, connection, "weight"))
        device_weight = compiled.connection_populations[connection].vars["g"]
        device_weight.pull_from_device()
        if not np.isfinite(expected).all() or not np.array_equal(device_weight.values, expected):
            raise ValueError(f"Reloaded weights differ for {connection}")


def validate_selected(data_root, run_dir, source_run, *, pilot, final_test=False):
    data_root = data_root.resolve(strict=True)
    source_run = source_run.resolve(strict=True)
    saved = json.loads((source_run / "selection.json").read_text())
    if final_test:
        completed = [row["epoch"] for row in saved["epochs"]]
        if completed != list(range(1, saved["planned_epochs"] + 1)):
            raise ValueError("Complete all planned epochs before final testing")
    best = min(saved["epochs"], key=lambda row: (-row["validation_correct"], row["epoch"]))
    key = best["checkpoint"]
    if key != saved["best"]:
        raise ValueError("Selected checkpoint disagrees with the epoch selection rule")
    protocol = saved.get("protocol", dict(whole_gesture=False, trial_steps=TRIAL_STEPS, input_capacity=None))
    val_subjects = protocol.get("validation_subjects", HISTORICAL_VALIDATION_SUBJECTS)
    dataset, train_indices, val_indices = load_development_split(data_root, val_subjects)
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
    whole, steps = protocol["whole_gesture"], protocol["trial_steps"]
    preprocessing = protocol.get("preprocessing", OCCUPANCY)
    if preprocessing == EVDOWNSAMPLING and protocol.get("evdownsampling") != PARAMETERS:
        raise ValueError("Saved EvDownsampling parameters differ from this encoder")
    val_x, val_y, val_lengths = encode_subset(
        dataset, val_indices, "validation", whole=whole, preprocessing=preprocessing)
    if val_lengths.max() > steps:
        raise ValueError("Validation duration exceeds the recorded training horizon")
    source = source_run / "checkpoints"
    if not source.is_dir():
        raise FileNotFoundError(source)
    weights = Numpy(str(source))
    network = build_model(protocol["input_capacity"])
    network.load((key,), weights)

    run_dir.mkdir()
    os.chdir(run_dir)
    compiler = inference_compiler(steps)
    compiled = compiler.compile(network, name=MODEL_NAME)
    with compiled:
        assert_loaded_weights(compiled, network, weights, key)
        print(f"Reloaded {key}: {len(network.connections)} weight arrays matched")
        scores, predicted, metrics = evaluate_split(compiled, network, val_x, val_y, paths,
                                                   key, "validation")
        original = "Pilot" if pilot else "Training-run validation"
        print(f"Frozen validation: {metrics['correct']}/{metrics['total']} | "
              f"{original}: {best['validation_correct']}/{best['validation_total']}")
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
        if metrics["correct"] != best["validation_correct"]:
            raise RuntimeError("Frozen inference and the selected epoch differ in validation correct count")
        print("Selected pilot checkpoint: OK" if pilot else "Full validation predictions: OK")

    if final_test:
        test_set, indices = load_partition(data_root, train=False)
        test_x, test_y, lengths = encode_subset(
            test_set, indices, "test", whole=whole, preprocessing=preprocessing)
        # For longer unseen recordings, extend only that parent's inference trial.
        horizons = np.maximum(steps, lengths)
        test_scores = np.empty((len(indices), len(test_set.classes)), dtype=np.float32)
        for horizon in np.unique(horizons):
            group = np.flatnonzero(horizons == horizon)
            inputs = [test_x[i] for i in group]
            network.populations[0].neuron.max_spikes = max(1, max(len(x.spike_times) for x in inputs))
            compiler = inference_compiler(int(horizon))
            test_model = compiler.compile(network, name=f"DVSGestureTest{horizon}")
            with test_model:
                assert_loaded_weights(test_model, network, weights, key)
                predictions, _ = test_model.predict({network.populations[0]: inputs},
                                                    network.populations[-1], callbacks=[])
                scores = predictions[network.populations[-1]]
                if scores.shape != (len(group), len(test_set.classes)):
                    raise ValueError(f"Unexpected test score shape: {scores.shape}")
                test_scores[group] = scores
        _, test = save_predictions(test_scores, test_y, [test_set.data[i] for i in indices],
                                   key, "test", frames=lengths, trial_steps=horizons)
        result = dict(training_run=str(source_run), checkpoint=key, protocol=protocol,
                      epoch=best["epoch"], validation=metrics, test=test)
        pending = Path("test_result.json.tmp")
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace("test_result.json")
        print(f"Official test: {test['correct']}/{test['total']} "
              f"({test['accuracy']:.2%}) | CE: {test['cross_entropy']:.6f}")


def evaluate_split(compiled, network, inputs, labels, paths, key, name):
    output = network.populations[-1]
    predictions, _ = compiled.predict({network.populations[0]: inputs}, output, callbacks=[])
    scores = predictions[output]
    if scores.shape != (len(labels), output.shape[0]):
        raise ValueError(f"Unexpected {name} score shape: {scores.shape}")
    predicted, metrics = save_predictions(scores, labels, paths, key, name)
    return scores, predicted, metrics


def save_predictions(scores, labels, paths, key, name, **timing):
    loss = mean_cross_entropy(scores, labels)
    predicted = scores.argmax(axis=1)
    correct = int(np.count_nonzero(predicted == labels))
    np.savez(f"{name}.npz", paths=paths, labels=labels, scores=scores,
             predictions=predicted, checkpoint=key, **timing)
    return predicted, dict(correct=correct, total=len(labels),
                           accuracy=correct / len(labels), cross_entropy=loss)


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
    mode.add_argument("--test-full", type=Path, metavar="TRAINING_RUN",
                      help="Verify the selected checkpoint on validation, then evaluate the official test set")
    parser.add_argument("--whole-gesture", action="store_true",
                        help="Train on all complete frames per recording; alone, check two training parents")
    parser.add_argument("--preprocessing", choices=PREPROCESSORS,
                        help="Fresh training encoder (default: occupancy); reload reads the saved protocol")
    parser.add_argument("--initial-from", type=Path, metavar="TRAINING_RUN",
                        help="Use the five pre-training weight arrays from this run with fresh Adam")
    args = parser.parse_args()
    if args.full_epochs is not None and args.full_epochs < 1:
        parser.error("--full-epochs must be at least 1")
    source_run = args.validate_pilot or args.validate_full or args.test_full
    if source_run and (args.preprocessing is not None or args.initial_from is not None):
        parser.error("--preprocessing and --initial-from apply only to fresh training")
    if args.whole_gesture and (source_run or args.pilot):
        parser.error("--whole-gesture applies to fresh full training or a two-parent training check")
    if source_run is not None:
        validate_selected(args.data_root, args.run_dir, source_run,
                          pilot=args.validate_pilot is not None, final_test=args.test_full is not None)
    else:
        run(args.data_root, args.run_dir, args.pilot, args.full_epochs, whole=args.whole_gesture,
            preprocessing=args.preprocessing or OCCUPANCY, initial_from=args.initial_from)
