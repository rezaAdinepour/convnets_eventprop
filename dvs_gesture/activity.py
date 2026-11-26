"""Spike activity across neurons per layer and epoch, and its plot.

    python activity.py SPIKES_CSV OUTPUT.png [--run-dir RUN_DIR/train]

SPIKES_CSV is a run's spikes.csv (train.py) or activity.csv (activity_over_training.py).
Each neuron's rate is its spike count averaged over the recordings; the plot shows the
mean across neurons +- one sd (band) and the 5th-95th percentile (dashed) per layer, and the
SpikeCount regulariser's target. --run-dir (default: the CSV's directory, if it holds
run_config.json) gives the run's description, trial length and target. Runs older than
this script have no sd and percentile columns, so only the mean is drawn.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from run_labels import layer_title, load_config, run_label

FIELDS = ["epoch", "split", "layer", "mean_per_neuron", "max_per_neuron", "silent_fraction",
          "sd_across_neurons", "p5_per_neuron", "p50_per_neuron", "p95_per_neuron"]
SPLITS = {"validation": "validation recordings", "train": "training recordings (augmented)"}


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


def plot_activity(rows, output, trial_ms=4000.0, title="", reg_target=None, splits=None):
    """``reg_target``: SpikeCount target in spikes per neuron per trial (drawn as a line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    layers = list(dict.fromkeys(r["layer"] for r in rows))
    splits = splits or list(dict.fromkeys(r["split"] for r in rows))
    to_hz = 1000.0 / trial_ms
    colors = dict(zip(splits, ("tab:blue", "tab:orange", "tab:green")))
    has_sd = bool(rows[0].get("sd_across_neurons"))
    fig, axes = plt.subplots(1, len(layers), figsize=(4.2 * len(layers), 4.0), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        for split in splits:
            sel = [r for r in rows if r["layer"] == layer and r["split"] == split]
            epoch = np.array([float(r["epoch"]) for r in sel])
            mean = np.array([float(r["mean_per_neuron"]) for r in sel]) * to_hz
            ax.plot(epoch, mean, color=colors[split])
            if has_sd:
                sd = np.array([float(r["sd_across_neurons"]) for r in sel]) * to_hz
                ax.fill_between(epoch, mean - sd, mean + sd, color=colors[split], alpha=0.2)
                for p in ("p5_per_neuron", "p95_per_neuron"):
                    ax.plot(epoch, [float(r[p]) * to_hz for r in sel], color=colors[split],
                            linestyle="--", linewidth=0.8)
        if reg_target is not None:
            ax.axhline(reg_target * to_hz, color="black", linestyle=":", linewidth=1)
        ax.set_title(layer_title(layer), fontsize=10)
        initial = min(float(r["epoch"]) for r in rows) < 0
        ax.set_xlabel("Training epoch" + (" (-1: initial random weights)" if initial else ""))
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Spikes per neuron per second")

    handles = [Line2D([], [], color=colors[s], label=f"Mean across neurons, {SPLITS.get(s, s)}")
               for s in splits]
    if has_sd:
        handles += [Patch(color="grey", alpha=0.3, label="Mean +- 1 sd across neurons"),
                    Line2D([], [], color="grey", linestyle="--", linewidth=0.8,
                           label="5th and 95th percentile across neurons")]
    if reg_target is not None:
        handles.append(Line2D([], [], color="black", linestyle=":",
                              label=f"Activity regulariser target "
                                    f"({reg_target * to_hz:.0f} spikes/s)"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=8,
               frameon=False)
    fig.suptitle("Spiking activity per layer during training: each neuron's spike rate averaged "
                 "over recordings, summarised across the neurons of the layer" +
                 (f"\n{title}" if title else ""), fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved {output}")


def describe_run(config, splits):
    shown = ", ".join(SPLITS.get(s, s) for s in splits)
    text = run_label(config)
    if "validation" in splits and not config.get("train_all"):
        text += "; validation recordings come from the held-out subjects"
    return f"{text}\nShown: {shown}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path, help="New PNG file")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="The run's train/ directory (for its description)")
    parser.add_argument("--split", nargs="+", default=None, help="Splits to draw (default: all)")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")
    rows = read_activity(args.csv)
    splits = args.split or list(dict.fromkeys(r["split"] for r in rows))
    run_dir = args.run_dir or args.csv.resolve().parent
    title, trial_ms, target = "", 4000.0, None
    if (run_dir / "run_config.json").exists():
        config = load_config(run_dir)
        title = describe_run(config, splits)
        trial_ms, target = config["example_timesteps"], config.get("reg_target")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_activity(rows, args.output, trial_ms, title, target, splits)


if __name__ == "__main__":
    main()
