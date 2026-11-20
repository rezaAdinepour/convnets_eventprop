import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_genn.callbacks import Callback, Checkpoint, SpikeRecorder
from ml_genn.serialisers import Numpy
from ml_genn.utils.data import calc_max_spikes

from data import (ENCODINGS, encode_recordings, load_dataset, max_spikes_per_neuron,
                  split_train_validation)
from model import (BATCH_SIZE, GRAD_LIMIT, MAX_SPIKES, REG_STRENGTH, REG_TARGET, SEED,
                   build_network, compile_for_training)


class CSVLog:
    """Create a CSV file with a header, then append one row per call."""

    def __init__(self, filename, fieldnames):
        self.filename = filename
        self.fieldnames = fieldnames
        with open(filename, "w", newline="") as f:
            csv.DictWriter(f, fieldnames).writeheader()

    def write(self, **row):
        with open(self.filename, "a", newline="") as f:
            csv.DictWriter(f, self.fieldnames).writerow(row)


class AdjointLimitLog(Callback):

    def __init__(self, log, hidden_layers):
        self.log = log
        self.hidden_layers = hidden_layers

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_epoch_end(self, compiled_network, epoch, metrics):
        for layer in self.hidden_layers:
            limit = compiled_network.neuron_populations[layer].vars["LambdavLimit"]
            limit.pull_from_device()
            self.log.write(epoch=epoch, layer=layer.name, limit=float(np.max(limit.values)))


def spike_recorders(layers, split, num_examples):
    # example_filter excludes the unused batch lanes of the last, partial batch.
    return [SpikeRecorder(layer, key=(split, layer.name), record_counts=True,
                          example_filter=np.arange(num_examples))
            for layer in layers]


def log_spike_counts(log, epoch, split, recordings, layers):
    for layer in layers:
        counts = np.array(recordings[(split, layer.name)])  # (examples, neurons)
        log.write(epoch=epoch, split=split, layer=layer.name,
                  mean_per_neuron=counts.mean(), max_per_neuron=int(counts.max()),
                  silent_fraction=np.mean(counts.sum(axis=0) == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Directory that contains DVSGesture/")
    parser.add_argument("run_dir", type=Path, help="New directory for this run's outputs")
    parser.add_argument("--encoding", choices=ENCODINGS, default="occupancy")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad-limit", type=float, default=GRAD_LIMIT,
                        help="EventProp adjoint limit multiplier (default: mlGeNN's 100)")
    parser.add_argument("--reg-target", type=float, default=REG_TARGET,
                        help="SpikeCount target spikes per hidden neuron per gesture")
    parser.add_argument("--reg-strength", type=float, nargs=2, default=REG_STRENGTH,
                        metavar=("BELOW", "ABOVE"),
                        help="SpikeCount strength below and above the target")
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--trial-ms", type=int, default=None,
                        help="Compress every recording into this many ms (default: real time)")
    length.add_argument("--crop-ms", type=int, default=None,
                        help="Keep only the first this many ms of every recording, in real time")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Reuse encoded recordings stored in this directory")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for weight initialisation and shuffling (must not be 0)")
    parser.add_argument("--train-all", action="store_true",
                        help="Train on all official training subjects, without validation")
    args = parser.parse_args()
    args.reg_strength = tuple(args.reg_strength)
    if args.seed == 0:
        parser.error("--seed 0 makes GeNN pick a random seed")

    np.random.seed(args.seed)

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(dataset)
    cache_dir = args.cache_dir.resolve() if args.cache_dir else None
    train_x, train_y, train_len = encode_recordings(dataset, train_idx, args.encoding,
                                                    "Encoding training", args.trial_ms,
                                                    cache_dir, args.crop_ms)
    val_x, val_y, val_len = encode_recordings(dataset, val_idx, args.encoding,
                                              "Encoding validation", args.trial_ms,
                                              cache_dir, args.crop_ms)

    example_timesteps = int(max(train_len.max(), val_len.max()))
    if args.train_all:
        train_x = train_x + val_x
        train_y = np.concatenate((train_y, val_y))
        val_x, val_y = [], val_y[:0]
    input_buffer = BATCH_SIZE * calc_max_spikes(train_x + val_x)
    peak_input = max_spikes_per_neuron(train_x + val_x)
    print(f"Train: {len(train_x)}, validation: {len(val_x)}, encoding: {args.encoding}")
    print(f"Trial: {example_timesteps} ms, input buffer: {input_buffer} spikes, "
          f"busiest input neuron: {peak_input} spikes per gesture")

    if 2 * peak_input + 2 > MAX_SPIKES:
        raise ValueError(f"An input neuron fires {peak_input} times; "
                         f"MAX_SPIKES={MAX_SPIKES} must be at least {2 * peak_input + 2}")

    args.run_dir.mkdir(parents=True)
    os.chdir(args.run_dir)

    Path("run_config.json").write_text(json.dumps(dict(
        encoding=args.encoding, trial_ms=args.trial_ms, crop_ms=args.crop_ms,
        epochs=args.epochs,
        example_timesteps=example_timesteps,
        input_buffer=input_buffer, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        grad_limit=args.grad_limit, reg_target=args.reg_target,
        reg_strength=args.reg_strength, seed=args.seed, train_all=args.train_all,
        cache_dir=str(cache_dir) if cache_dir else None), indent=2))

    network = build_network(input_buffer)
    input_pop, output_pop = network.populations[0], network.populations[-1]
    hidden_layers = network.populations[1:-1]
    compiled = compile_for_training(network, example_timesteps, args.grad_limit,
                                    args.reg_strength, args.reg_target, args.seed)

    metrics_log = CSVLog("metrics.csv", ["epoch", "train_correct", "train_total", "train_loss",
                                         "val_correct", "val_total", "seconds"])
    spikes_log = CSVLog("spikes.csv", ["epoch", "split", "layer", "mean_per_neuron",
                                       "max_per_neuron", "silent_fraction"])
    limit_log = CSVLog("adjoint_limits.csv", ["epoch", "layer", "limit"])
    checkpoint = Checkpoint(Numpy("checkpoints"))

    with compiled:
        compiled.save(("initial",), Numpy("checkpoints"))
        for epoch in range(args.epochs):
            start = perf_counter()
            callbacks = ["batch_progress_bar", checkpoint,
                         AdjointLimitLog(limit_log, hidden_layers),
                         *spike_recorders(hidden_layers, "train", len(train_x))]
            if args.train_all:
                train_metrics, train_data = compiled.train(
                    {input_pop: train_x}, {output_pop: train_y},
                    num_epochs=1, start_epoch=epoch, shuffle=True, callbacks=callbacks)
                val_metrics, val_data = None, None
            else:
                train_metrics, val_metrics, train_data, val_data = compiled.train_validate(
                    {input_pop: train_x}, {output_pop: train_y},
                    validation_x={input_pop: val_x}, validation_y={output_pop: val_y},
                    num_epochs=1, start_epoch=epoch, shuffle=True, callbacks=callbacks,
                    validation_callbacks=spike_recorders(hidden_layers, "validation",
                                                         len(val_x)))

            train = train_metrics[output_pop]

            train_loss = float(np.mean(train_data["loss"]))
            if not np.isfinite(train_loss):
                raise FloatingPointError(f"Training loss is {train_loss} in epoch {epoch}")
            val_correct = val_total = ""
            if val_metrics is not None:
                val_correct = val_metrics[output_pop].correct
                val_total = val_metrics[output_pop].total
            metrics_log.write(epoch=epoch, train_correct=train.correct, train_total=train.total,
                              train_loss=train_loss, val_correct=val_correct,
                              val_total=val_total, seconds=round(perf_counter() - start, 1))
            log_spike_counts(spikes_log, epoch, "train", train_data, hidden_layers)
            if val_data is not None:
                log_spike_counts(spikes_log, epoch, "validation", val_data, hidden_layers)
            validation = f", validation {val_correct}/{val_total}" if val_total else ""
            print(f"Epoch {epoch}: train {train.correct}/{train.total}{validation}, "
                  f"train loss {train_loss:.4f}")


if __name__ == "__main__":
    main()
