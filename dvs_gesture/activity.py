"""Spike activity across neurons per layer and epoch, and its plot.

    python activity.py SPIKES_CSV OUTPUT.png [--trial-ms 4000]

SPIKES_CSV is a run's spikes.csv (train.py) or activity.csv (activity_over_training.py).
Each neuron's rate is its spike count averaged over the recordings; the plot shows the
mean across neurons +- one sd (band) and the 5th-95th percentile (dashed) per layer.
Runs older than this script have no sd and percentile columns, so only the mean is drawn.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

FIELDS = ["epoch", "split", "layer", "mean_per_neuron", "max_per_neuron", "silent_fraction",
          "sd_across_neurons", "p5_per_neuron", "p50_per_neuron", "p95_per_neuron"]


def activity_stats(counts):
    """Statistics of a (recordings, neurons) array of spike counts."""
    counts = np.asarray(counts)
    per_neuron = counts.mean(axis=0)
    p5, p50, p95 = np.percentile(per_neuron, (5, 50, 95))
    return dict(mean_per_neuron=per_neuron.mean(), max_per_neuron=int(counts.max()),
                silent_fraction=np.mean(counts.sum(axis=0) == 0),
                sd_across_neurons=per_neuron.std(), p5_per_neuron=p5, p50_per_neuron=p50,
                p95_per_neuron=p95)


def read_activity(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_activity(rows, output, trial_ms=4000.0, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(dict.fromkeys(r["layer"] for r in rows))
    splits = list(dict.fromkeys(r["split"] for r in rows))
    to_hz = 1000.0 / trial_ms
    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.4), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for split, color in zip(splits, ("tab:blue", "tab:orange", "tab:green")):
            sel = [r for r in rows if r["layer"] == layer and r["split"] == split]
            epoch = np.array([float(r["epoch"]) for r in sel])
            mean = np.array([float(r["mean_per_neuron"]) for r in sel]) * to_hz
            ax.plot(epoch, mean, color=color, label=split)
            if sel and sel[0].get("sd_across_neurons"):
                sd = np.array([float(r["sd_across_neurons"]) for r in sel]) * to_hz
                ax.fill_between(epoch, mean - sd, mean + sd, color=color, alpha=0.2)
                for p in ("p5_per_neuron", "p95_per_neuron"):
                    ax.plot(epoch, [float(r[p]) * to_hz for r in sel], color=color,
                            linestyle="--", linewidth=0.8)
        ax.set_title(layer)
        ax.set_xlabel("Epoch (-1: initial weights)" if min(
            float(r["epoch"]) for r in rows) < 0 else "Epoch")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Spikes per neuron per second")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle((f"{title}: " if title else "") +
                 "mean across neurons +- sd (band), 5th-95th percentile (dashed)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, help="New PNG file")
    parser.add_argument("--trial-ms", type=float, default=4000.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")
    plot_activity(read_activity(args.csv), args.output, args.trial_ms,
                  args.csv.resolve().parent.parent.name)


if __name__ == "__main__":
    main()
