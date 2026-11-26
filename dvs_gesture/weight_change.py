"""How far each layer's weights move from their initial values over training.

    python weight_change.py OUTPUT_DIR RUN_DIR [RUN_DIR ...]

Reads only checkpoints (no GPU). Per run, connection and epoch:
  relative_change  ||W_e - W_init|| / ||W_init||
  correlation      Pearson correlation of W_e with W_init (1: unchanged pattern)
  sd               sd of W_e
  relative_step    ||W_e - W_(e-1)|| / ||W_init||, the size of one epoch's update
Writes OUTPUT_DIR/weight_change.csv and weight_change.png.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = ["run", "connection", "epoch", "relative_change", "correlation", "sd", "relative_step"]
METRICS = FIELDS[3:]
STYLES = ("-", "--", ":", "-.")


def train_dir(path):
    path = path.resolve()
    return path / "train" if (path / "train" / "run_config.json").exists() else path


def connections(checkpoints):
    return sorted(p.name[len("initial-"):-len("-weight.npy")]
                  for p in checkpoints.glob("initial-*-weight.npy"))


def run_rows(run_dir):
    name = run_dir.parent.name
    epochs = json.loads((run_dir / "run_config.json").read_text())["epochs"]
    checkpoints = run_dir / "checkpoints"
    rows = []
    for c in connections(checkpoints):
        initial = np.load(checkpoints / f"initial-{c}-weight.npy").ravel()
        norm = np.linalg.norm(initial)
        previous = initial
        for e in range(epochs):
            path = checkpoints / f"{e}-{c}-weight.npy"
            if not path.exists():
                break
            w = np.load(path).ravel()
            rows.append(dict(run=name, connection=c, epoch=e,
                             relative_change=np.linalg.norm(w - initial) / norm,
                             correlation=np.corrcoef(w, initial)[0, 1], sd=w.std(),
                             relative_step=np.linalg.norm(w - previous) / norm))
            previous = w
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="New directory for the outputs")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Directories written by train.py")
    args = parser.parse_args()
    if len(args.run_dirs) > len(STYLES):
        parser.error(f"At most {len(STYLES)} runs")
    args.output_dir.mkdir(parents=True)

    rows = [r for d in args.run_dirs for r in run_rows(train_dir(d))]
    with open(args.output_dir / "weight_change.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    runs = list(dict.fromkeys(r["run"] for r in rows))
    names = list(dict.fromkeys(r["connection"] for r in rows))
    print(f"{'Last epoch':<52}" + "".join(f"{m:>17}" for m in METRICS))
    for run in runs:
        for c in names:
            last = [r for r in rows if r["run"] == run and r["connection"] == c][-1]
            print(f"{run[-40:]:<42}{c:<10}" + "".join(f"{last[m]:17.3f}" for m in METRICS))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.6))
    colors = dict(zip(names, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, metric in zip(axes, METRICS):
        for run, style in zip(runs, STYLES):
            for c in names:
                sel = [r for r in rows if r["run"] == run and r["connection"] == c]
                ax.plot([r["epoch"] for r in sel], [r[metric] for r in sel], style,
                        color=colors[c], label=c if run == runs[0] else None)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("Epoch")
        if metric == "relative_step":
            ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("Weights against their initial values; line style per run: " +
                 ", ".join(f"{s} {r}" for r, s in zip(runs, STYLES)), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "weight_change.png", dpi=150)
    print(f"Saved {args.output_dir / 'weight_change.png'}")


if __name__ == "__main__":
    main()
