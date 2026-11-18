"""Reload the best checkpoint of a training run and evaluate it.

The best epoch is the one with the most correct validation predictions
(earliest epoch on a tie). Without --test only the validation split is used,
which should reproduce the val_correct value logged during training.
--test evaluates the official test set (subjects 24-29); only use it once the
model has been chosen.

Usage (from the dvs_gesture directory):
    python evaluate.py DATA_ROOT RUN_DIR [--test]
"""

import argparse
import csv
import json
import os
from pathlib import Path

from ml_genn.serialisers import Numpy

from data import encode_recordings, load_dataset, max_spikes_per_batch, split_train_validation
from model import BATCH_SIZE, build_network, compile_for_inference


def best_epoch(run_dir):
    with open(run_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda row: (int(row["val_correct"]), -int(row["epoch"])))
    return int(best["epoch"]), int(best["val_correct"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Directory that contains DVSGesture/")
    parser.add_argument("run_dir", type=Path, help="Directory written by train.py")
    parser.add_argument("--test", action="store_true", help="Evaluate the official test set")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    epoch, logged_correct = best_epoch(run_dir)

    if args.test:
        dataset = load_dataset(args.data_root.resolve(), train=False)
        indices = list(range(len(dataset)))
    else:
        dataset = load_dataset(args.data_root.resolve(), train=True)
        _, indices = split_train_validation(dataset)
    x, y, lengths = encode_recordings(dataset, indices, config["encoding"])

    # Use the training trial length. Test recordings longer than that would
    # be cut short, so report it rather than silently changing the readout.
    timesteps = config["example_timesteps"]
    if lengths.max() > timesteps:
        print(f"Warning: {(lengths > timesteps).sum()} recordings are longer than the "
              f"{timesteps} ms training trial and will be truncated")

    network = build_network(max_spikes_per_batch(x, BATCH_SIZE))
    network.load((epoch,), Numpy(str(run_dir / "checkpoints")))
    os.chdir(run_dir)  # keep GeNN's generated code inside the run directory
    compiled = compile_for_inference(network, timesteps)
    with compiled:
        metrics, _ = compiled.evaluate({network.populations[0]: x},
                                       {network.populations[-1]: y})
    result = metrics[network.populations[-1]]

    split = "Test" if args.test else "Validation"
    print(f"Epoch {epoch}: {split} {result.correct}/{result.total} ({100 * result.result:.2f}%)")
    if not args.test:
        print(f"Logged during training: {logged_correct}/{result.total}")


if __name__ == "__main__":
    main()
