import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from ml_genn.serialisers import Numpy
from ml_genn.utils.data import calc_max_spikes
from tonic.datasets import DVSGesture

from data import (EVDOWNSAMPLING_PARAMS, NUM_CLASSES, encode_recordings, load_dataset,
                  split_train_validation)
from model import (BATCH_SIZE, MAX_DELAY_MS, TAU_MEM_MS, TAU_SYN_MS, build_network,
                   compile_for_inference, delay_connections)


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
    if config.get("train_all"):
        if not args.test:
            parser.error("This run trained on the validation subjects; use --test")
        epoch = config["epochs"] - 1
    else:
        epoch, logged_correct = best_epoch(run_dir)

    if args.test:
        dataset = load_dataset(args.data_root.resolve(), train=False)
        indices = list(range(len(dataset)))
    else:
        dataset = load_dataset(args.data_root.resolve(), train=True)
        _, indices = split_train_validation(dataset)
    x, y, lengths = encode_recordings(dataset, indices, config["encoding"],
                                      trial_ms=config.get("trial_ms"),
                                      cache_dir=config.get("cache_dir"),
                                      crop_ms=config.get("crop_ms"),
                                      evd_params=config.get("evd_params", EVDOWNSAMPLING_PARAMS))

    timesteps = config["example_timesteps"]
    if lengths.max() > timesteps:
        print(f"Warning: {(lengths > timesteps).sum()} recordings are longer than the "
              f"{timesteps} ms training trial and will be truncated")

    network = build_network(BATCH_SIZE * calc_max_spikes(x), config.get("delay_layers", []),
                            config.get("max_delay_ms", MAX_DELAY_MS),
                            downsample=config.get("downsample", "pool"),
                            hidden=config.get("hidden", True),
                            tau_mem_ms=config.get("tau_mem_ms", TAU_MEM_MS),
                            tau_syn_ms=config.get("tau_syn_ms", TAU_SYN_MS))
    network.load((epoch,), Numpy(str(run_dir / "checkpoints")))
    for connection in delay_connections(network):
        # Training rounds the learned delays the same way (EventPropCompiler)
        connection.connectivity.delay = np.clip(np.round(connection.connectivity.delay),
                                                0, connection.max_delay_steps)
    os.chdir(run_dir)
    compiled = compile_for_inference(network, timesteps)
    output_pop = network.populations[-1]
    with compiled:
        readout, _ = compiled.predict({network.populations[0]: x}, output_pop)
    predicted = np.argmax(readout[output_pop], axis=1)
    correct = predicted == y

    split = "Test" if args.test else "Validation"
    print(f"Epoch {epoch}: {split} {correct.sum()}/{len(y)} ({100 * correct.mean():.2f}%)")
    if not args.test:
        print(f"Logged during training: {logged_correct}/{len(y)}")

    subjects = np.array([dataset.users[i] for i in indices])
    print("\nPer subject:")
    for subject in np.unique(subjects):
        mask = subjects == subject
        print(f"  subject {subject:2d}: {correct[mask].sum():3d}/{mask.sum():3d} "
              f"({100 * correct[mask].mean():.1f}%)")

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(confusion, (y, predicted), 1)
    print("\nPer class:")
    for c, name in enumerate(DVSGesture.classes):
        print(f"  {c:2d} {name:<16} {confusion[c, c]:3d}/{confusion[c].sum():3d}")
    print("\nConfusion matrix (rows: true class, columns: predicted class):")
    print("     " + "".join(f"{c:4d}" for c in range(NUM_CLASSES)))
    for c in range(NUM_CLASSES):
        print(f"  {c:2d} " + "".join(f"{n:4d}" for n in confusion[c]))


if __name__ == "__main__":
    main()
