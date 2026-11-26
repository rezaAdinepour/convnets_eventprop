"""Spike rate per layer at each saved epoch of a run (validation set by default).

    python activity_over_training.py DATA_ROOT RUN_DIR WORK_DIR [--split validation train]

Epoch -1 is the initial weights.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from ml_genn.callbacks import SpikeRecorder
from ml_genn.serialisers import Numpy
from ml_genn.utils.data import calc_max_spikes

from activity import FIELDS, activity_stats, describe_run, plot_activity, read_activity
from data import (COUNT_LEVELS, EVDOWNSAMPLING_PARAMS, VALIDATION_SUBJECTS, encode_recordings,
                  input_channels, load_dataset, split_train_validation)
from model import (BATCH_SIZE, CHANNELS, MAX_DELAY_MS, TAU_MEM_MS, TAU_SYN_MS, build_network,
                   compile_for_inference)
from train import CSVLog

EPOCHS = [-1, 0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 99]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--split", nargs="+", choices=("validation", "train"),
                        default=["validation"])
    parser.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text())
    if config.get("delay_layers"):
        parser.error("Runs that learn delays are not supported")
    if config.get("train_all") and "validation" in args.split:
        parser.error("This run trained on the validation subjects; use --split train")
    epochs = [e for e in args.epochs if e < config["epochs"]]
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True)

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(
        dataset, config.get("val_subjects", VALIDATION_SUBJECTS))
    data = {}
    for split in args.split:
        x, _, _ = encode_recordings(
            dataset, val_idx if split == "validation" else train_idx, config["encoding"],
            description=f"Encoding {split}", trial_ms=config.get("trial_ms"),
            cache_dir=config.get("cache_dir"), crop_ms=config.get("crop_ms"),
            evd_params=config.get("evd_params", EVDOWNSAMPLING_PARAMS),
            count_levels=config.get("count_levels", COUNT_LEVELS))
        data[split] = x

    network = build_network(BATCH_SIZE * max(calc_max_spikes(x) for x in data.values()), (),
                            MAX_DELAY_MS, downsample=config.get("downsample", "pool"),
                            hidden=config.get("hidden", True),
                            tau_mem_ms=config.get("tau_mem_ms", TAU_MEM_MS),
                            tau_syn_ms=config.get("tau_syn_ms", TAU_SYN_MS),
                            channels=config.get("channels", CHANNELS),
                            synapse=config.get("synapse", "exponential"),
                            subtract_reset=config.get("subtract_reset", False),
                            readout=config.get("readout", "avg_var_exp_weight"),
                            input_channels=input_channels(
                                config["encoding"], config.get("count_levels", COUNT_LEVELS)),
                            recurrent_weight_sd=config.get("recurrent_weight_sd", 0.0))
    input_pop, output_pop = network.populations[0], network.populations[-1]
    layers = network.populations[1:-1]
    os.chdir(work_dir)
    log = CSVLog("activity.csv", FIELDS)
    checkpoints = Numpy(str(run_dir / "checkpoints"))

    for epoch in epochs:
        network.load(("initial" if epoch < 0 else epoch,), checkpoints)
        compiled = compile_for_inference(network, config["example_timesteps"])
        with compiled:
            for split, x in data.items():
                recorders = [SpikeRecorder(layer, key=layer.name, record_counts=True,
                                           example_filter=np.arange(len(x)))
                             for layer in layers]
                _, recordings = compiled.predict({input_pop: x}, output_pop,
                                                 callbacks=["batch_progress_bar", *recorders])
                for layer in layers:
                    stats = activity_stats(recordings[layer.name])
                    log.write(epoch=epoch, split=split, layer=layer.name, **stats)
                    print(f"Epoch {epoch:3d} {split:<10} {layer.name:<7} "
                          f"mean {stats['mean_per_neuron']:7.1f} "
                          f"sd {stats['sd_across_neurons']:7.1f} "
                          f"p5-p95 {stats['p5_per_neuron']:6.1f}-{stats['p95_per_neuron']:7.1f} "
                          f"silent {stats['silent_fraction']:.3f}")

    plot_activity(read_activity("activity.csv"), work_dir / "activity.png",
                  config["example_timesteps"], describe_run(config, args.split),
                  config.get("reg_target"), args.split)
    (work_dir / "source.json").write_text(json.dumps(
        {"run_dir": str(run_dir), "epochs": epochs, "splits": args.split}, indent=2))


if __name__ == "__main__":
    main()
