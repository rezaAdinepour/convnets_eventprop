"""Train the DVS Gesture EventProp baseline and evaluate its selected checkpoint."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ml_genn.metrics import SparseCategoricalAccuracy
from ml_genn.serialisers import Numpy
from ml_genn.utils.data import PreprocessedSpikes

from data import (EVDOWNSAMPLING, HISTORICAL_VALIDATION_SUBJECTS, OCCUPANCY, PREPROCESSORS,
                  TRIAL_STEPS, VALIDATION_SUBJECTS, encode_trial, load_development_split, load_partition)
from evdownsampling import PARAMETERS
from model import (BATCH_SIZE, BATCH_TAPE_SLOTS, GRAD_LIMIT, MODEL_NAME, SINGLE_EXAMPLE, STRICT_BUFFER_CHECKING,
                   build_model, compile_model, inference_compiler, tape_slots)
from training_log import epoch_callbacks, log_spikes, spike_callbacks

SHUFFLE_SEED = 0
PILOT_SEED = 0
PILOT_TRAIN_PER_CLASS = 2
PILOT_VALIDATION_PER_CLASS = 1
PILOT_EPOCHS = 2
SMOKE_TAIL_PARENTS = 6
SMOKE_EPOCHS = 2
INITIAL_HIDDEN_WEIGHT_SCALE = 1.0


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
        state.loss_sum = 0.0
        return state

    def update(self, state, y_true, y_pred, communicator):
        # argmax can return a class even when the scores contain NaNs.
        if not np.isfinite(y_pred).all():
            raise FloatingPointError("Nonfinite model readout")
        valid = ~np.ma.getmaskarray(y_true)
        if np.any(y_pred[~valid] != 0):
            raise RuntimeError("Silent padding produced nonzero logits in the bias-free model")
        y_true, y_pred = np.asarray(y_true)[valid], y_pred[valid]
        if len(y_true) == 0:
            return
        state.loss_sum += len(y_true) * mean_cross_entropy(y_pred, y_true)
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


def input_capacity(inputs, batch_size):
    sizes = sorted((len(x.spike_times) for x in inputs), reverse=True)
    return max(1, sum(sizes[:batch_size]))


def training_epoch(inputs, labels, order, batch_size):
    """Flush the stock one-batch-delayed backward pass before validation."""
    ordered = [inputs[i] for i in order]
    silent = PreprocessedSpikes(np.zeros_like(ordered[0].end_spikes), ordered[0].spike_times[:0])
    # Fill the partial real batch, then start one final silent forward batch.
    extra = (-len(order)) % batch_size + 1
    targets = np.pad(labels[order], (0, extra))
    # The compiler receives valid zero labels for silent lanes. Only real rows enter metrics.
    targets = np.ma.array(targets, mask=np.arange(len(targets)) >= len(order))
    return ordered + [silent] * extra, targets


def run(data_root, run_dir, pilot, full_epochs=None, *, whole=False,
        preprocessing=OCCUPANCY, initial_from=None, batch_size=BATCH_SIZE, ring_slots=None,
        activity_check=False, hidden_weight_scale=INITIAL_HIDDEN_WEIGHT_SCALE, grad_limit=GRAD_LIMIT):
    if not np.isfinite(grad_limit) or grad_limit <= 0:
        raise ValueError("grad_limit must be finite and positive")
    if not np.isfinite(hidden_weight_scale) or hidden_weight_scale <= 0:
        raise ValueError("Hidden weight scale must be finite and positive")
    if hidden_weight_scale != INITIAL_HIDDEN_WEIGHT_SCALE and initial_from is None:
        raise ValueError("Scaling requires --initial-from for a controlled initial-weight comparison")
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
    elif full_epochs is None and not whole:
        train_indices = train_indices[:batch_size + SMOKE_TAIL_PARENTS]
        val_indices = val_indices[:batch_size + 1]
    train_x, train_y, train_lengths = encode_subset(
        dataset, train_indices, "train", whole=whole, preprocessing=preprocessing)
    val_x, val_y, val_lengths = encode_subset(
        dataset, val_indices, "validation", whole=whole, preprocessing=preprocessing)
    steps = int(max(train_lengths.max(), val_lengths.max()))
    capacity = max(input_capacity(train_x, batch_size), input_capacity(val_x, batch_size))
    peak_input = max(int(np.diff(x.end_spikes.astype(np.int64), prepend=0).max())
                     for x in train_x + val_x)
    input_ring_bound = tape_slots(peak_input)
    if ring_slots is None:
        bound = tape_slots(steps)
        ring_slots = bound if batch_size == SINGLE_EXAMPLE else max(input_ring_bound, min(bound, BATCH_TAPE_SLOTS))
    print(f"Peak spikes at one input neuron in one gesture: {peak_input}")
    if not activity_check and ring_slots < input_ring_bound:
        raise ValueError(f"The two-gesture input bound is {input_ring_bound} ring slots; configured {ring_slots}.")
    protocol = dict(whole_gesture=whole, trial_steps=steps, input_capacity=capacity,
                    grad_limit=grad_limit,
                    initial_hidden_weight_scale=hidden_weight_scale,
                    batch_size=batch_size, ring_slots=ring_slots, strict_buffer_checking=STRICT_BUFFER_CHECKING,
                    peak_input_spikes_per_neuron=peak_input,
                    input_ring_bound=input_ring_bound,
                    training_order="seeded permutation each epoch; silent final forward batch",
                    partial_batch_gradient_denominator=batch_size,
                    preprocessing=preprocessing, validation_subjects=val_subjects,
                    train_subjects=train_subjects, shuffle_seed=SHUFFLE_SEED)
    if preprocessing == EVDOWNSAMPLING:
        protocol["evdownsampling"] = dict(PARAMETERS)
    if initial_from is not None:
        protocol["initial_weights_from"] = str(initial_from)
    print(f"Development horizon: {steps} steps | Input capacity: {capacity}")
    if not activity_check:
        print(f"Ring slots: {ring_slots}")
    if whole and full_epochs is None:
        # Exercise the production allocation with long parents and a partial batch.
        longest = np.argsort(train_lengths, kind="stable")[-(batch_size + SMOKE_TAIL_PARENTS):][::-1]
        train_indices = [train_indices[i] for i in longest]
        train_x, train_y = [train_x[i] for i in longest], train_y[longest]
        longest_val = np.argsort(val_lengths, kind="stable")[-(batch_size + 1):][::-1]
        val_indices = [val_indices[i] for i in longest_val]
        val_x, val_y = [val_x[i] for i in longest_val], val_y[longest_val]

    network = build_model(capacity)
    for pop in network.populations[1:-1]:
        pop.record_spikes = True
    if initial_from is not None:
        network.load(("initial",), initial_weights)
        for connection in network.connections[:-1]:
            connection.connectivity.weight = connection.connectivity.weight * hidden_weight_scale
        print(f"Initial hidden weight scale: {hidden_weight_scale:g}")
    input_pop, output_pop = network.populations[0], network.populations[-1]
    if activity_check:
        compiled = inference_compiler(steps, batch_size).compile(network, name=f"{MODEL_NAME}Activity")
        summary = dict(initial_weights_from=str(initial_from), checkpoint="initial",
                       initial_hidden_weight_scale=hidden_weight_scale,
                       mode="frozen_initial_activity", trial_steps=steps, batch_size=batch_size,
                       preprocessing=preprocessing, selection="longest gestures in each development split",
                       input_capacity=capacity, peak_input_spikes_per_neuron=peak_input, splits={})
        if preprocessing == EVDOWNSAMPLING:
            summary["evdownsampling"] = dict(PARAMETERS)
        with compiled:
            assert_loaded_weights(compiled, network, initial_weights, "initial",
                                  hidden_weight_scale=hidden_weight_scale)
            print(f"Activity backend={compiled.genn_model.backend_name}, batch_size={batch_size}")
            for split, inputs, labels, indices in (
                    ("train", train_x, train_y, train_indices),
                    ("validation", val_x, val_y, val_indices)):
                predictions, recordings = compiled.predict(
                    {input_pop: inputs}, output_pop, callbacks=spike_callbacks(network, len(inputs)))
                scores = predictions[output_pop]
                if not np.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite activity-check readout")
                log_spikes(0, split, recordings, network, len(inputs))
                counts = {name: np.asarray(values) for name, values in recordings.items()}
                np.savez(f"{split}-spike-counts.npz", paths=[dataset.data[i] for i in indices],
                         labels=labels, scores=scores, **counts)
                summary["splits"][split] = {
                    name: dict(max_spikes=int(values.max()), mean_spikes=float(values.mean()),
                               silent_neuron_fraction=float(np.mean(~values.any(axis=0))))
                    for name, values in counts.items()}
                for name, values in summary["splits"][split].items():
                    print(f"{split} {name}: max={values['max_spikes']} spikes/neuron/gesture, "
                          f"mean={values['mean_spikes']:.3f}, "
                          f"silent={values['silent_neuron_fraction']:.2%}")
        Path("activity.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("Frozen initial activity check: OK")
        return
    compiled = compile_model(network, steps, whole=whole, batch_size=batch_size,
                             ring_slots=ring_slots, grad_limit=grad_limit)
    checkpoints = Numpy("checkpoints")
    history = []
    previous = "initial"
    num_epochs = full_epochs if full_epochs is not None else (PILOT_EPOCHS if pilot else SMOKE_EPOCHS)
    rng = np.random.default_rng(SHUFFLE_SEED)
    with compiled:
        print(f"Training backend={compiled.genn_model.backend_name}, batch_size={compiled.genn_model.batch_size}, "
              f"strict_buffer_checking={STRICT_BUFFER_CHECKING}, grad_limit={grad_limit:g}")
        compiled.save((previous,), checkpoints)
        if initial_from is not None:
            for conn in network.connections:
                actual = checkpoints.deserialise(("initial", conn, "weight"))
                expected = initial_weights.deserialise(("initial", conn, "weight"))
                if conn in network.connections[:-1]:
                    expected = expected * hidden_weight_scale
                if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                    raise ValueError(f"Initial weights differ in {conn}")
            print(f"Matched initial weights in {len(network.connections)} connections")
        for epoch in range(1, num_epochs + 1):
            order = rng.permutation(len(train_y))
            epoch_x, epoch_y = training_epoch(train_x, train_y, order, batch_size)
            # Shuffle real parents above so the backward flush always stays last.
            train_metrics, val_metrics, train_spikes, val_spikes = compiled.train_validate(
                {input_pop: epoch_x}, {output_pop: epoch_y},
                num_epochs=1, start_epoch=epoch - 1, shuffle=False,
                validation_x={input_pop: val_x}, validation_y={output_pop: val_y},
                metrics=CheckedAccuracy(), callbacks=epoch_callbacks(network, "train", len(train_y)),
                validation_callbacks=epoch_callbacks(network, "validation", len(val_y)))
            train = train_metrics[output_pop]
            val = val_metrics[output_pop]
            if train.total != len(train_y) or val.total != len(val_y):
                raise RuntimeError("Metric totals do not match the supplied parents")
            log_spikes(epoch, "train", train_spikes, network, len(train_y))
            log_spikes(epoch, "validation", val_spikes, network, len(val_y))
            current = f"epoch{epoch}"
            compiled.save((current,), checkpoints)
            print(f"Epoch {epoch} | Online train: {train.correct}/{train.total} | "
                  f"Fixed-weight validation: {val.correct}/{val.total}")
            train_ce = train.loss_sum / train.total
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
                                final_training_parent=dataset.data[train_indices[order[-1]]],
                                scheduled_updates=(len(order) + batch_size - 1) // batch_size))
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


def assert_loaded_weights(compiled, network, weights, key, *,
                          hidden_weight_scale=INITIAL_HIDDEN_WEIGHT_SCALE):
    for connection in network.connections:
        expected = weights.deserialise((key, connection, "weight"))
        if connection in network.connections[:-1]:
            expected = expected * hidden_weight_scale
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
    batch_size = protocol.get("batch_size", SINGLE_EXAMPLE)
    compiler = inference_compiler(steps, batch_size)
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
            network.populations[0].neuron.max_spikes = input_capacity(inputs, batch_size)
            compiler = inference_compiler(int(horizon), batch_size)
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
    mode.add_argument("--activity-check", action="store_true",
                      help="Measure initial forward spikes on long gestures using inference")
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
                        help="Use all complete frames; alone, run a short check with long recordings")
    parser.add_argument("--batch-size", type=int, help=f"Training batch size (default: {BATCH_SIZE})")
    parser.add_argument("--ring-slots", type=int, help="EventProp tape slots per neuron and batch lane")
    parser.add_argument("--grad-limit", type=float,
                        help=f"Multiplier for the adaptive adjoint bound (default: {GRAD_LIMIT:g})")
    parser.add_argument("--preprocessing", choices=PREPROCESSORS,
                        help="Fresh training encoder (default: occupancy); reload reads the saved protocol")
    parser.add_argument("--initial-from", type=Path, metavar="TRAINING_RUN",
                        help="Use the five pre-training weight arrays from this run with fresh Adam")
    parser.add_argument("--hidden-weight-scale", type=float,
                        help="Multiply the four hidden-layer initial weight arrays from --initial-from")
    args = parser.parse_args()
    if args.full_epochs is not None and args.full_epochs < 1:
        parser.error("--full-epochs must be at least 1")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.ring_slots is not None and args.ring_slots < 2:
        parser.error("--ring-slots must be at least 2")
    if args.grad_limit is not None and (not np.isfinite(args.grad_limit) or args.grad_limit <= 0):
        parser.error("--grad-limit must be finite and positive")
    if args.hidden_weight_scale is not None:
        if not np.isfinite(args.hidden_weight_scale) or args.hidden_weight_scale <= 0:
            parser.error("--hidden-weight-scale must be finite and positive")
        if args.initial_from is None:
            parser.error("--hidden-weight-scale requires --initial-from")
    if args.activity_check and (not args.whole_gesture or args.initial_from is None):
        parser.error("--activity-check requires --whole-gesture and --initial-from")
    if args.activity_check and args.ring_slots is not None:
        parser.error("--activity-check uses inference without an EventProp ring")
    source_run = args.validate_pilot or args.validate_full or args.test_full
    if args.grad_limit is not None and (source_run or args.activity_check):
        parser.error("--grad-limit applies only to fresh training")
    if source_run and (args.preprocessing is not None or args.initial_from is not None):
        parser.error("--preprocessing and --initial-from apply only to fresh training")
    if source_run and (args.batch_size is not None or args.ring_slots is not None):
        parser.error("Reload uses the saved training protocol")
    if args.whole_gesture and (source_run or args.pilot):
        parser.error("--whole-gesture applies to fresh full training or a training check")
    if source_run is not None:
        validate_selected(args.data_root, args.run_dir, source_run,
                          pilot=args.validate_pilot is not None, final_test=args.test_full is not None)
    else:
        run(args.data_root, args.run_dir, args.pilot, args.full_epochs, whole=args.whole_gesture,
            preprocessing=args.preprocessing or OCCUPANCY, initial_from=args.initial_from,
            batch_size=args.batch_size or BATCH_SIZE, ring_slots=args.ring_slots,
            activity_check=args.activity_check,
            grad_limit=GRAD_LIMIT if args.grad_limit is None else args.grad_limit,
            hidden_weight_scale=(INITIAL_HIDDEN_WEIGHT_SCALE if args.hidden_weight_scale is None
                                 else args.hidden_weight_scale))
