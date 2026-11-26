"""Plot spike rate per layer for one or more runs (e.g. seeds) in one figure.

    python activity.py OUTPUT.png --run CSV RUN_DIR [--run CSV RUN_DIR ...]

CSV is a run's spikes.csv or activity.csv (activity_over_training.py).
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from run_labels import layer_title, load_config

FIELDS = ["epoch", "split", "layer", "mean_per_neuron", "max_per_neuron", "silent_fraction",
          "sd_across_neurons", "p5_per_neuron", "p50_per_neuron", "p95_per_neuron"]
SPLITS = {"validation": "validation set", "train": "training set"}
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def activity_stats(counts):
    counts = np.asarray(counts)  # (recordings, neurons)
    per_neuron = counts.mean(axis=0)
    p5, p50, p95 = np.percentile(per_neuron, (5, 50, 95))
    return dict(mean_per_neuron=per_neuron.mean(), max_per_neuron=int(counts.max()),
                silent_fraction=np.mean(counts.sum(axis=0) == 0),
                sd_across_neurons=per_neuron.std(), p5_per_neuron=p5, p50_per_neuron=p50,
                p95_per_neuron=p95)


def read_activity(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def model_kind(config):
    return "Final" if config.get("train_all") else "Val. split"


def plot_activity(runs, output, trial_ms=4000.0, title="", reg_target=None, split="validation"):
    # runs: {legend label: csv rows}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    first = next(iter(runs.values()))
    layers = list(dict.fromkeys(r["layer"] for r in first))
    to_hz = 1000.0 / trial_ms
    colors = dict(zip(runs, COLORS))
    has_sd = all(rows[0].get("sd_across_neurons") for rows in runs.values())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for label, rows in runs.items():
            sel = [r for r in rows if r["layer"] == layer and r["split"] == split]
            epoch = np.array([float(r["epoch"]) for r in sel])
            mean = np.array([float(r["mean_per_neuron"]) for r in sel]) * to_hz
            ax.plot(epoch, mean, color=colors[label])
            if has_sd:
                sd = np.array([float(r["sd_across_neurons"]) for r in sel]) * to_hz
                ax.fill_between(epoch, mean - sd, mean + sd, color=colors[label], alpha=0.12)
                for p in ("p5_per_neuron", "p95_per_neuron"):
                    ax.plot(epoch, [float(r[p]) * to_hz for r in sel], color=colors[label],
                            linestyle="--", linewidth=0.7)
        if reg_target is not None:
            ax.axhline(reg_target * to_hz, color="black", linestyle=":", linewidth=1)
        ax.set_title(layer_title(layer))
        ax.set_xlabel("Epoch")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Spikes/s per neuron")

    handles = ([Line2D([], [], color=colors[label], label=label) for label in runs]
               if len(runs) > 1 else [])
    base = "grey" if len(runs) > 1 else next(iter(colors.values()))
    handles.append(Line2D([], [], color=base, label="Mean"))
    if has_sd:
        handles += [Patch(color=base, alpha=0.3, label="± 1 sd"),
                    Line2D([], [], color=base, linestyle="--", linewidth=0.7,
                           label="5th–95th percentile")]
    if reg_target is not None:
        handles.append(Line2D([], [], color="black", linestyle=":", label="Regulariser target"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", nargs=2, action="append", required=True, type=Path,
                        metavar=("CSV", "RUN_DIR"))
    parser.add_argument("--split", default="validation", choices=list(SPLITS))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")

    configs = [load_config(run_dir) for _, run_dir in args.run]
    runs = {f"Seed {c['seed']}": read_activity(path) for (path, _), c in zip(args.run, configs)}
    if len(runs) != len(args.run):
        parser.error("Two runs have the same seed")
    kinds = sorted({model_kind(c) for c in configs})
    title = f"Spike rate per layer, {' / '.join(kinds)}, {SPLITS[args.split]}"
    if len(runs) == 1:
        title += f" ({next(iter(runs))})"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_activity(runs, args.output, configs[0]["example_timesteps"], title,
                  configs[0].get("reg_target"), args.split)


if __name__ == "__main__":
    main()
