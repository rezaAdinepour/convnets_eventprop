"""Plot spike rate per layer from spikes.csv or activity.csv.

    python activity.py CSV OUTPUT.png [--run-dir RUN_DIR/train]
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from run_labels import layer_title, load_config, run_label

FIELDS = ["epoch", "split", "layer", "mean_per_neuron", "max_per_neuron", "silent_fraction",
          "sd_across_neurons", "p5_per_neuron", "p50_per_neuron", "p95_per_neuron"]
SPLITS = {"validation": "validation set", "train": "training set"}


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


def plot_activity(rows, output, trial_ms=4000.0, title="", reg_target=None, splits=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    layers = list(dict.fromkeys(r["layer"] for r in rows))
    splits = splits or list(dict.fromkeys(r["split"] for r in rows))
    to_hz = 1000.0 / trial_ms
    colors = dict(zip(splits, ("tab:blue", "tab:orange")))
    has_sd = bool(rows[0].get("sd_across_neurons"))
    fig, axes = plt.subplots(1, len(layers), figsize=(3.6 * len(layers), 3.4), squeeze=False)
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
        ax.set_title(layer_title(layer))
        ax.set_xlabel("Epoch")
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Spikes/s per neuron")

    handles = [Line2D([], [], color=colors[s], label=f"Mean ({SPLITS.get(s, s)})"
                      if len(splits) > 1 else "Mean") for s in splits]
    if has_sd:
        handles += [Patch(color="grey", alpha=0.3, label="± 1 sd"),
                    Line2D([], [], color="grey", linestyle="--", linewidth=0.8,
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


def describe_run(config, splits):
    return f"Spike rate per layer, {run_label(config)}, {', '.join(SPLITS.get(s, s) for s in splits)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--split", nargs="+", default=None)
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
