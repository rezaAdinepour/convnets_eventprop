"""Validation accuracy of single runs, their weight averages (SWA) and seed ensembles.

    python ensemble.py DATA_ROOT WORK_DIR RUN_DIR [RUN_DIR ...]
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from ml_genn.utils.data import calc_max_spikes
from tonic.datasets import DVSGesture

from data import (COUNT_LEVELS, EVDOWNSAMPLING_PARAMS, VALIDATION_SUBJECTS, encode_recordings,
                  input_channels, load_dataset, split_train_validation)
from model import (BATCH_SIZE, CHANNELS, MAX_DELAY_MS, TAU_MEM_MS, TAU_SYN_MS, build_network,
                   compile_for_inference)

# for older runs that did not save these
DEFAULTS = dict(channels=list(CHANNELS), readout="avg_var_exp_weight", synapse="exponential",
                subtract_reset=False, output_lr_factor=1.0, count_levels=COUNT_LEVELS,
                recurrent_weight_sd=0.0, val_subjects=list(VALIDATION_SUBJECTS))
PER_RUN = ("seed", "run", "config")


def load_run_config(run_dir):
    config = {**DEFAULTS, **json.loads((run_dir / "run_config.json").read_text())}
    if config.get("train_all"):
        raise SystemExit(f"{run_dir} trained on the validation subjects")
    if config.get("delay_layers"):
        raise SystemExit(f"{run_dir} learns delays, which this script does not load")
    return config


def logged_validation(run_dir, epoch):
    with open(run_dir / "metrics.csv") as f:
        return next(int(row["val_correct"]) for row in csv.DictReader(f)
                    if int(row["epoch"]) == epoch)


def mean_weights(run_dir, connections, epochs):
    return {c: np.mean([np.load(run_dir / "checkpoints" / f"{e}-{c}-weight.npy")
                        for e in epochs], axis=0) for c in connections}


def softmax(readout):
    e = np.exp(readout - readout.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--epoch", type=int, help="default: last epoch")
    parser.add_argument("--swa-epochs", type=int, default=20,
                        help="0: no SWA")
    args = parser.parse_args()

    run_dirs = [d.resolve() / "train" if (d / "train" / "run_config.json").exists()
                else d.resolve() for d in args.run_dirs]
    configs = [load_run_config(d) for d in run_dirs]
    shared = [{k: v for k, v in c.items() if k not in PER_RUN} for c in configs]
    for d, s in zip(run_dirs[1:], shared[1:]):
        different = sorted(k for k in set(s) | set(shared[0]) if s.get(k) != shared[0].get(k))
        if different:
            raise SystemExit(f"{d} differs from {run_dirs[0]} in {different}")
    config = configs[0]
    epoch = config["epochs"] - 1 if args.epoch is None else args.epoch
    swa_epochs = range(epoch - args.swa_epochs + 1, epoch + 1)

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if (work_dir / "readouts.npz").exists():
        raise SystemExit(f"{work_dir / 'readouts.npz'} exists; choose a new work_dir")

    dataset = load_dataset(args.data_root.resolve(), train=True)
    _, indices = split_train_validation(dataset, config["val_subjects"])
    x, y, _ = encode_recordings(dataset, indices, config["encoding"],
                                trial_ms=config.get("trial_ms"), cache_dir=config.get("cache_dir"),
                                crop_ms=config.get("crop_ms"),
                                evd_params=config.get("evd_params", EVDOWNSAMPLING_PARAMS),
                                count_levels=config.get("count_levels", COUNT_LEVELS))
    subjects = np.array([dataset.users[i] for i in indices])

    # build once, a second build renames the layers
    network = build_network(BATCH_SIZE * calc_max_spikes(x), (), MAX_DELAY_MS,
                            downsample=config.get("downsample", "pool"),
                            hidden=config.get("hidden", True),
                            tau_mem_ms=config.get("tau_mem_ms", TAU_MEM_MS),
                            tau_syn_ms=config.get("tau_syn_ms", TAU_SYN_MS),
                            channels=config["channels"], synapse=config["synapse"],
                            subtract_reset=config["subtract_reset"], readout=config["readout"],
                            input_channels=input_channels(
                                config["encoding"], config.get("count_levels", COUNT_LEVELS)),
                            recurrent_weight_sd=config["recurrent_weight_sd"])
    connections = [c.name for c in network.connections]
    input_pop, output_pop = network.populations[0], network.populations[-1]
    os.chdir(work_dir)

    def predict(weights):
        for c in network.connections:
            c.connectivity.weight = weights[c.name]
        compiled = compile_for_inference(network, config["example_timesteps"])
        with compiled:
            readout, _ = compiled.predict({input_pop: x}, output_pop)
        return readout[output_pop]

    readouts = {}
    for d, c in zip(run_dirs, configs):
        name = f"seed{c['seed']}"
        readouts[name] = predict(mean_weights(d, connections, [epoch]))
        correct = (readouts[name].argmax(axis=1) == y).sum()
        print(f"{d.parent.name}, epoch {epoch}: {correct}/{len(y)} "
              f"(logged during training: {logged_validation(d, epoch)})")
        if args.swa_epochs > 0:
            readouts[f"{name}_swa"] = predict(mean_weights(d, connections, swa_epochs))
    np.savez(work_dir / "readouts.npz", y=y, subjects=subjects, indices=np.array(indices),
             **readouts)

    singles = [n for n in readouts if not n.endswith("_swa")]
    probabilities = {n: softmax(r) for n, r in readouts.items()}
    if len(singles) > 1:
        probabilities["ensemble"] = np.mean([probabilities[n] for n in singles], axis=0)
        if args.swa_epochs > 0:
            probabilities["ensemble_swa"] = np.mean(
                [probabilities[f"{n}_swa"] for n in singles], axis=0)
    correct = {n: p.argmax(axis=1) == y for n, p in probabilities.items()}

    if args.swa_epochs > 0:
        print(f"\n_swa: weights averaged over epochs {swa_epochs.start}-{swa_epochs.stop - 1}")
    width = max(len(n) for n in correct) + 1
    print(f"\n{'':<23}" +"".join(f"{n:>{width}}" for n in correct))
    rows = [("All", np.ones_like(y, dtype=bool))]
    rows += [(name, y == i) for i, name in enumerate(DVSGesture.classes)]
    rows += [(f"Subject {s}", subjects == s) for s in np.unique(subjects)]
    for label, mask in rows:
        print(f"{label:<16}{mask.sum():>4} of" +
              "".join(f"{correct[n][mask].sum():>{width}}" for n in correct))


if __name__ == "__main__":
    main()
