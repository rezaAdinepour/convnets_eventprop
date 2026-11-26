"""Weight change from initialisation per layer and epoch, from the checkpoints.

    python weight_change.py OUTPUT_DIR RUN_DIR [RUN_DIR ...]
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from run_labels import layer_title, load_config

FIELDS = ["run", "connection", "epoch", "relative_change", "correlation", "sd", "relative_step"]
METRICS = FIELDS[3:]
STYLES = ("-", "--", ":", "-.")
PANEL_TITLES = {"relative_change": r"$\Vert W - W_0 \Vert \, / \, \Vert W_0 \Vert$",
                "correlation": r"corr$(W, W_0)$",
                "sd": r"sd$(W)$",
                "relative_step": r"$\Vert W_t - W_{t-1} \Vert \, / \, \Vert W_0 \Vert$"}


def train_dir(path):
    path = path.resolve()
    return path / "train" if (path / "train" / "run_config.json").exists() else path


def connections(checkpoints):
    return sorted(p.name[len("initial-"):-len("-weight.npy")]
                  for p in checkpoints.glob("initial-*-weight.npy"))


def run_rows(run_dir):
    name = run_dir.parent.name
    epochs = load_config(run_dir)["epochs"]
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
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if len(args.run_dirs) > len(STYLES):
        parser.error(f"At most {len(STYLES)} runs")
    out = args.output_dir.resolve()
    csv_path, png_path = out / "weight_change_from_init.csv", out / "weight_change_from_init.png"
    if csv_path.exists() or png_path.exists():
        parser.error(f"{csv_path} or {png_path} exists")
    out.mkdir(parents=True, exist_ok=True)

    run_dirs = [train_dir(d) for d in args.run_dirs]
    seeds = {d.parent.name: load_config(d)["seed"] for d in run_dirs}
    rows = [r for d in run_dirs for r in run_rows(d)]
    with open(csv_path, "w", newline="") as f:
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

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.8 * len(METRICS), 3.4))
    colors = dict(zip(names, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, metric in zip(axes, METRICS):
        for run, style in zip(runs, STYLES):
            for c in names:
                sel = [r for r in rows if r["run"] == run and r["connection"] == c]
                ax.plot([r["epoch"] for r in sel], [r[metric] for r in sel], style,
                        color=colors[c])
        ax.set_title(PANEL_TITLES[metric])
        ax.set_xlabel("Epoch")
        if metric == "relative_step":
            ax.set_yscale("log")
    handles = [Line2D([], [], color=colors[c], label=layer_title(c)) for c in names]
    handles += [Line2D([], [], color="black", linestyle=s, label=f"Seed {seeds[r]}")
                for r, s in zip(runs, STYLES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(png_path, dpi=200)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
