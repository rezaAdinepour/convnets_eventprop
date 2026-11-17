import argparse
import gzip
import json
import os
from pathlib import Path

import numpy as np
from ml_genn.callbacks import Callback
from ml_genn.compilers import InferenceCompiler
from ml_genn.serialisers import Numpy

from data import load_development_split
from model import build_model, compile_model
from train import (CheckedAccuracy, assert_loaded_weights, balanced_indices, encode_subset,
                   evaluate_split, save_predictions)


def pull(group, name):
    var = group.vars[name]
    var.pull_from_device()
    return var.values.reshape(-1).copy()


def peak(values):
    return float(np.max(np.abs(values)))


class BatchTrace(Callback):
    def __init__(self, network, steps, paths, stream, epoch=1):
        self.network = network
        self.steps = steps
        self.paths = paths
        self.stream = stream
        self.epoch = epoch

    def create_state(self, compiled_network, **kwargs):
        return {"net": compiled_network}

    def on_timestep_end(self, state, timestep):
        if timestep != self.steps - 1:
            return
        net = state["net"]
        state["layers"] = {}
        state["weights"] = {}
        for conn in self.network.connections:
            group = net.connection_populations[conn]
            g = pull(group, "weightGradient")
            state["weights"][conn] = pull(group, "weight")
            state["layers"][conn.name] = dict(gradient_nonzero=int(np.count_nonzero(g)),
                                              gradient_absmax=peak(g))
        for pop in self.network.populations[1:-1]:
            group = net.neuron_populations[pop]
            # Reset stores the index just before the first spike of this trial.
            counts = ((pull(group, "RingWriteOffset") - pull(group, "RingWriteStartOffset") - 1)
                      % (2 * self.steps + 2))
            state["layers"][pop.name].update(forward_spikes=int(counts.sum()),
                                             limit_before=float(pull(group, "LambdavLimit").item()),
                                             adjoint_abs_sum=float(pull(group, "LambdavAbsSum").sum(dtype=np.float64)))

    def on_batch_end(self, state, batch, metric_state):
        net = state["net"]
        for conn, (optimiser, adam) in zip(self.network.connections, net.optimisers, strict=True):
            before = state["weights"][conn]
            after = pull(net.connection_populations[conn], "weight")
            m, v = pull(adam, "M"), pull(adam, "V")
            if before.shape != m.shape or m.shape != v.shape:
                raise ValueError(f"Adam and weight shape mismatch for {conn}")
            row = state["layers"][conn.name]
            row.update(changed=int(np.count_nonzero(after != before)),
                       update_absmax=peak(after.astype(np.float64) - before),
                       m_absmax=peak(m), v_absmax=peak(v),
                       m_nonfinite=int((~np.isfinite(m)).sum()),
                       v_nonfinite=int((~np.isfinite(v)).sum()))
            if batch and np.isfinite(m).all() and np.isfinite(v).all() and (v >= 0).all():
                step = (self.epoch - 1) * len(self.paths) + batch
                m_hat = m.astype(np.float64) / (1 - optimiser.beta1 ** (step + 1))
                v_hat = v.astype(np.float64) / (1 - optimiser.beta2 ** (step + 1))
                delta = -optimiser.alpha * m_hat / (np.sqrt(v_hat) + optimiser.epsilon)
                rounded = (before.astype(np.float64) + delta).astype(before.dtype)
                row["nominal_nonzero_rounded_away"] = int(
                    np.count_nonzero((delta != 0) & (rounded == before)))
                row["nominal_changed"] = int(np.count_nonzero(rounded != before))
        for pop in self.network.populations[1:-1]:
            state["layers"][pop.name]["limit_after"] = float(
                pull(net.neuron_populations[pop], "LambdavLimit").item())
        record = dict(epoch=self.epoch, batch=batch, forward=self.paths[batch],
                      backward=self.paths[batch - 1] if batch else None, layers=state["layers"])
        # Keep nonfinite measurements identifiable in valid JSON.
        for row in record["layers"].values():
            for name, value in row.items():
                if not np.isfinite(value):
                    row[name] = str(value)
        print(json.dumps(record, allow_nan=False), file=self.stream)


def verify_start(compiled, network, weights, key):
    for conn in network.connections:
        expected = weights.deserialise((key, conn, "weight")).ravel()
        actual = pull(compiled.connection_populations[conn], "weight")
        np.testing.assert_array_equal(actual, expected)


def continuous_replay(dataset, train, validation, history, protocol, weights, run_dir):
    """Replay the development schedule through the first historically frozen epoch."""
    np.random.default_rng(0).shuffle(train)
    x, y, lengths = encode_subset(dataset, train, "train", whole=True)
    vx, vy, vlengths = encode_subset(dataset, validation, "validation", whole=True)
    steps = protocol["trial_steps"]
    if max(lengths.max(), vlengths.max()) != steps:
        raise ValueError("Development horizon differs from the source run")
    run_dir.mkdir()
    os.chdir(run_dir)
    network = build_model(protocol["input_capacity"])
    network.load(("initial",), weights)
    compiled = compile_model(network, steps, whole=True)
    inp, out = network.populations[0], network.populations[-1]
    order = np.arange(len(y))
    rng = np.random.default_rng(0)
    with compiled, open("epochs.jsonl", "x") as summary:
        verify_start(compiled, network, weights, "initial")
        previous = [pull(compiled.connection_populations[c], "weight") for c in network.connections]
        for epoch in range(1, 36):
            original = history[epoch - 1]
            if epoch > 1:
                order = np.concatenate((order[-1:], rng.permutation(order[:-1])))
            paths = [dataset.data[train[i]] for i in order]
            if original["epoch"] != epoch or paths[-1] != original["final_training_parent"]:
                raise ValueError(f"Training order disagrees with source at epoch {epoch}")
            with gzip.open(f"epoch{epoch:02d}-trace.jsonl.gz", "xt", compresslevel=1) as stream:
                trace = BatchTrace(network, steps, paths, stream, epoch)
                tm, vm, _, _ = compiled.train_validate(
                    {inp: [x[i] for i in order]}, {out: y[order]}, num_epochs=1,
                    start_epoch=epoch - 1, shuffle=False, validation_x={inp: vx},
                    validation_y={out: vy}, metrics=CheckedAccuracy(),
                    callbacks=[trace], validation_callbacks=[])
            if tm[out].total != len(y) or vm[out].total != len(vy):
                raise ValueError("Development metric totals differ from the supplied data")
            changed, historical_diff = {}, {}
            for i, conn in enumerate(network.connections):
                current = pull(compiled.connection_populations[conn], "weight")
                expected = weights.deserialise((original["checkpoint"], conn, "weight")).ravel()
                if not np.isfinite(current).all():
                    raise FloatingPointError(f"Nonfinite weights in {conn} at epoch {epoch}")
                changed[conn.name] = int(np.count_nonzero(previous[i] != current))
                historical_diff[conn.name] = peak(current.astype(np.float64) - expected)
                previous[i] = current
            row = dict(epoch=epoch, changed=changed, historical_max_abs_difference=historical_diff,
                       train_correct=int(tm[out].correct), validation_correct=int(vm[out].correct))
            print(json.dumps(row), file=summary, flush=True)
            print(f"Epoch {epoch}: changed={changed}, validation={vm[out].correct}/{vm[out].total}",
                  flush=True)
            if all(changed[c] == 0 for c in ("conv1", "conv2", "conv3")) and changed["hidden"] and changed["output"]:
                print("Observed a full epoch with unchanged convolutional checkpoints.")
                return
        print("The historical freeze pattern was not observed through epoch 35 in this replay.")


def order_probe_indices(trace_path, dataset, train, saved, variant, *, full=False):
    # Read paths from the measured replay so even the final forward companion is known.
    with gzip.open(trace_path, "rt") as stream:
        paths = []
        for batch, line in enumerate(stream):
            record = json.loads(line)
            if (record["epoch"] != 34 or record["batch"] != batch
                    or record["backward"] != (paths[-1] if paths else None)):
                raise ValueError("Expected the intact, ordered epoch-34 trace")
            paths.append(record["forward"])
    lookup = {dataset.data[i]: i for i in train}
    if (len(paths) != len(train) or set(paths) != set(lookup)
            or paths[-1] != saved["epochs"][33]["final_training_parent"]):
        raise ValueError("Trace paths disagree with the original development split or epoch")
    # Batches 1..400 cover both the zero-change interval and its recovery.
    indices = [lookup[path] for path in (paths if full else paths[:401])]
    if variant == "reverse":
        # Keep the first and last learned parents and the final companion fixed.
        indices[1:-2] = indices[1:-2][::-1]
    return indices


def score_full_order(network, trained_weights, dataset, validation, protocol, train_steps):
    inputs, labels, lengths = encode_subset(dataset, validation, "validation", whole=True)
    eval_steps = protocol["trial_steps"]
    if lengths.max() > eval_steps:
        raise ValueError("A validation recording exceeds the common evaluation horizon")
    network.load(("after",), trained_weights)
    compiler = InferenceCompiler(evaluate_timesteps=eval_steps, backend="cuda",
                                 reset_in_syn_between_batches=True)
    frozen = compiler.compile(network, name="DVSGestureOrderValidation")
    with frozen:
        assert_loaded_weights(frozen, network, trained_weights, "after")
        _, _, metrics = evaluate_split(frozen, network, inputs, labels,
                                       [dataset.data[i] for i in validation], "after", "validation")
    result = dict(start_checkpoint="epoch33", train_steps=train_steps, validation_steps=eval_steps,
                  validation=metrics)
    Path("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Frozen validation: {metrics['correct']}/{metrics['total']} | CE: {metrics['cross_entropy']:.6f}")


def rescore_validation(data_root, source, run_dir):
    """Shorten validation padding where the entire gesture fits the training-set maximum."""
    protocol = json.loads((source / "protocol.json").read_text())
    original = json.loads((source / "result.json").read_text())
    short = protocol["longest_recording_steps"]  # All 934 training recordings in this probe.
    long = original["validation_steps"]
    if not 0 < short < long or long != protocol["development_steps"]:
        raise ValueError("Expected the full-order probe scored at the development horizon")
    if original["train_steps"] != protocol["probe_steps"] or original["train_steps"] not in (short, long):
        raise ValueError("Inconsistent source training horizon")
    dataset, _, indices = load_development_split(data_root.resolve())
    inputs, labels, lengths = encode_subset(dataset, indices, "validation", whole=True)
    paths = [dataset.data[i] for i in indices]
    with np.load(source / "validation.npz", allow_pickle=False) as ref:
        reference = ref["scores"]
        if (ref["checkpoint"].item() != "after" or not np.array_equal(paths, ref["paths"])
                or not np.array_equal(labels, ref["labels"]) or reference.shape != (len(labels), len(dataset.classes))
                or not np.isfinite(reference).all() or not np.array_equal(ref["predictions"], reference.argmax(axis=1))):
            raise ValueError("Validation reference does not match the ordered split")
    old_predictions = reference.argmax(axis=1)
    if int(np.sum(old_predictions == labels)) != original["validation"]["correct"]:
        raise ValueError("Archived predictions disagree with the reference correct count")
    mask = lengths <= short
    if not mask.any() or lengths.max() > long:
        raise ValueError("Validation durations do not fit the two prescribed horizons")
    weights = Numpy(str((source / "checkpoints").resolve(strict=True)))
    run_dir.mkdir()
    os.chdir(run_dir)
    network = build_model(protocol["input_capacity"])
    network.load(("after",), weights)
    compiler = InferenceCompiler(evaluate_timesteps=int(short), backend="cuda", reset_in_syn_between_batches=True)
    frozen = compiler.compile(network, name="DVSGestureShortValidation")
    scores = reference.copy()
    with frozen:
        assert_loaded_weights(frozen, network, weights, "after")
        out = network.populations[-1]
        prediction, _ = frozen.predict({network.populations[0]: [inputs[i] for i in np.flatnonzero(mask)]},
                                       out, callbacks=[])
        if prediction[out].shape != (int(mask.sum()), len(dataset.classes)):
            raise ValueError("Unexpected inference output shape")
        scores[mask] = prediction[out]
    # For longer parents both protocols use exactly the archived long-horizon prediction.
    predicted, metrics = save_predictions(scores, labels, paths, "after", "validation",
                                          frames=lengths, trial_steps=np.where(mask, short, long))
    fixed = int(np.sum((predicted == labels) & (old_predictions != labels)))
    lost = int(np.sum((predicted != labels) & (old_predictions == labels)))
    result = dict(source=str(source), train_steps=original["train_steps"],
                  short_steps=int(short), long_steps=int(long), rescored=int(mask.sum()),
                  original=original["validation"], validation=metrics, corrected=fixed, lost=lost)
    Path("result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Validation: {metrics['correct']}/{metrics['total']} | CE: {metrics['cross_entropy']:.6f} | "
          f"corrected={fixed}, lost={lost}, rescored={mask.sum()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("training_run", type=Path, help="Baseline run, or completed full-order probe for rescore")
    parser.add_argument("checkpoint", choices=["epoch33", "epoch34", "continuous", "rescore"])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epoch34-trace", type=Path, help="Complete compressed trace for the order probe")
    parser.add_argument("--order", choices=["recorded", "reverse"])
    parser.add_argument("--fit-horizon", action="store_true",
                        help="Use the longest complete gesture in this order probe as the common trial length")
    parser.add_argument("--validate-full-order", action="store_true",
                        help="Use all 934 training parents in recorded order and evaluate on full validation")
    args = parser.parse_args()
    if bool(args.epoch34_trace) != bool(args.order) or (args.order and args.checkpoint != "epoch33"):
        parser.error("Use --epoch34-trace and --order together with epoch33")
    if args.fit_horizon and not args.order:
        parser.error("--fit-horizon requires an order probe")
    if args.validate_full_order and args.order != "recorded":
        parser.error("--validate-full-order requires --order recorded")
    source = args.training_run.resolve(strict=True)
    if args.checkpoint == "rescore":
        rescore_validation(args.data_root, source, args.run_dir)
        return
    saved = json.loads((source / "selection.json").read_text())
    protocol = saved["protocol"]
    if not protocol["whole_gesture"]:
        raise ValueError("Expected the whole-gesture training run")
    steps = protocol["trial_steps"]
    dataset, train, validation = load_development_split(args.data_root.resolve())
    weights = Numpy(str((source / "checkpoints").resolve(strict=True)))
    if args.checkpoint == "continuous":
        continuous_replay(dataset, train, validation, saved["epochs"], protocol, weights, args.run_dir)
        return
    if args.order:
        indices = order_probe_indices(args.epoch34_trace, dataset, train, saved, args.order,
                                      full=args.validate_full_order)
    else:
        indices = balanced_indices(dataset, train, 1, np.random.default_rng(0))
        # Last forward supplies the backward pass for the eleventh selected parent.
        indices.append(indices[0])
    paths = [dataset.data[i] for i in indices]
    inputs, labels, lengths = encode_subset(dataset, indices, "train", whole=True)
    if lengths.max() > steps:
        raise ValueError("Gesture exceeds the saved trial length")
    if args.fit_horizon:
        steps = int(lengths.max())
    args.run_dir.mkdir()
    os.chdir(args.run_dir)
    if args.order:
        timing = dict(checkpoint=args.checkpoint, order=args.order,
                      development_steps=protocol["trial_steps"], probe_steps=steps,
                      longest_recording_steps=int(lengths.max()),
                      input_capacity=protocol["input_capacity"], ring_slots=2 * steps + 2)
        Path("protocol.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"Probe horizon: {steps} ms | Longest recording: {lengths.max()} ms")
    network = build_model(protocol["input_capacity"])
    network.load((args.checkpoint,), weights)
    compiled = compile_model(network, steps, whole=True)
    with compiled, open("trace.jsonl", "x") as stream:
        verify_start(compiled, network, weights, args.checkpoint)
        trace = BatchTrace(network, steps, paths, stream)
        compiled.train({network.populations[0]: inputs}, {network.populations[-1]: labels},
                       num_epochs=1, shuffle=False, metrics=CheckedAccuracy(), callbacks=[trace])
        if args.validate_full_order:
            trained_weights = Numpy("checkpoints")
            compiled.save(("after",), trained_weights)
    print(f"Finished {args.checkpoint}: {len(indices)} forward trials, {len(indices) - 1} scheduled updates")
    if args.validate_full_order:
        score_full_order(network, trained_weights, dataset, validation, protocol, steps)


if __name__ == "__main__":
    main()
