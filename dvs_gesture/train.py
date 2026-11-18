"""Train the DVS Gesture ConvSNN with EventProp and log every epoch to CSV.

Each epoch:
  * trains on the 934 training recordings in a new random order (shuffle=True),
  * evaluates the 143 validation recordings with the updated weights,
  * saves a checkpoint (checkpoints/<epoch>-<layer>-weight.npy),
  * appends one row to metrics.csv and one row per layer to spikes.csv and
    adjoint_limits.csv.

Usage (from the dvs_gesture directory):
    python train.py DATA_ROOT RUN_DIR [--encoding evdownsampling] [--epochs 100]
"""

import argparse
import csv
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_genn.callbacks import Callback, Checkpoint, SpikeRecorder
from ml_genn.serialisers import Numpy

from data import (ENCODINGS, encode_recordings, load_dataset, max_spikes_per_batch,
                  max_spikes_per_neuron, split_train_validation)
from model import BATCH_SIZE, GRAD_LIMIT, MAX_SPIKES, build_network, compile_for_training

SEED = 0


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
    """Log the adaptive clamp mlGeNN applies to each hidden layer's adjoint jumps.

    EventProp clips the jump of lambda_V at every spike to +/-LambdavLimit,
    and recomputes the limit after each batch from the mean |lambda_V| of that
    batch. If the limit reaches zero the layer receives no gradient any more,
    so this is the first thing to look at if a layer's weights stop changing.
    """

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
    args = parser.parse_args()

    np.random.seed(SEED)  # train_validate(shuffle=True) uses NumPy's global RNG

    print("Encoding recordings...")
    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(dataset)
    train_x, train_y, train_len = encode_recordings(dataset, train_idx, args.encoding)
    val_x, val_y, val_len = encode_recordings(dataset, val_idx, args.encoding)

    # One trial length for all recordings: the longest one. Shorter recordings
    # are followed by silence.
    example_timesteps = int(max(train_len.max(), val_len.max()))
    input_buffer = max(max_spikes_per_batch(train_x, BATCH_SIZE),
                       max_spikes_per_batch(val_x, BATCH_SIZE))
    peak_input = max_spikes_per_neuron(train_x + val_x)
    print(f"Train: {len(train_x)}, validation: {len(val_x)}, encoding: {args.encoding}")
    print(f"Trial: {example_timesteps} ms, input buffer: {input_buffer} spikes, "
          f"busiest input neuron: {peak_input} spikes per gesture")

    # Input neurons share the per-neuron spike buffer with the hidden layers.
    # strict_buffer_checking only guards hidden layers, so check inputs here.
    if 2 * peak_input + 2 > MAX_SPIKES:
        raise ValueError(f"An input neuron fires {peak_input} times; "
                         f"MAX_SPIKES={MAX_SPIKES} must be at least {2 * peak_input + 2}")

    args.run_dir.mkdir(parents=True)
    os.chdir(args.run_dir)
    # evaluate.py reads this to rebuild the same network and trial length.
    Path("run_config.json").write_text(json.dumps(dict(
        encoding=args.encoding, epochs=args.epochs, example_timesteps=example_timesteps,
        input_buffer=input_buffer, batch_size=BATCH_SIZE, max_spikes=MAX_SPIKES,
        grad_limit=args.grad_limit, seed=SEED), indent=2))

    network = build_network(input_buffer)
    input_pop, output_pop = network.populations[0], network.populations[-1]
    hidden_layers = network.populations[1:-1]
    compiled = compile_for_training(network, example_timesteps, args.grad_limit)

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
            train_metrics, val_metrics, train_data, val_data = compiled.train_validate(
                {input_pop: train_x}, {output_pop: train_y},
                validation_x={input_pop: val_x}, validation_y={output_pop: val_y},
                num_epochs=1, start_epoch=epoch, shuffle=True,
                callbacks=[checkpoint, AdjointLimitLog(limit_log, hidden_layers),
                           *spike_recorders(hidden_layers, "train", len(train_x))],
                validation_callbacks=spike_recorders(hidden_layers, "validation", len(val_x)))

            train, val = train_metrics[output_pop], val_metrics[output_pop]
            # Loss of each batch is recorded during the next batch's backward
            # pass, so the last batch of the epoch is missing from this mean.
            train_loss = float(np.mean(train_data["loss"]))
            if not np.isfinite(train_loss):
                raise FloatingPointError(f"Training loss is {train_loss} in epoch {epoch}")
            metrics_log.write(epoch=epoch, train_correct=train.correct, train_total=train.total,
                              train_loss=train_loss, val_correct=val.correct, val_total=val.total,
                              seconds=round(perf_counter() - start, 1))
            log_spike_counts(spikes_log, epoch, "train", train_data, hidden_layers)
            log_spike_counts(spikes_log, epoch, "validation", val_data, hidden_layers)
            print(f"Epoch {epoch}: train {train.correct}/{train.total}, "
                  f"validation {val.correct}/{val.total}, train loss {train_loss:.4f}")


if __name__ == "__main__":
    main()
