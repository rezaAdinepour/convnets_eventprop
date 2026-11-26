"""Events per bin of every training and validation recording, and the silent lead and tail.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tonic import transforms
from tonic.datasets import DVSGesture
from tqdm import tqdm

from data import DENOISE_US, NUM_CLASSES, load_dataset, split_train_validation


def active_range(counts, fraction):
    active = np.flatnonzero(counts >= fraction * counts.max())
    return active[0], active[-1] + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--bin-ms", type=int, default=500)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, val_idx = split_train_validation(dataset)
    args.output_dir.mkdir(parents=True)
    bin_us = args.bin_ms * 1000

    rows, histograms = [], {}
    for split, indices in (("train", train_idx), ("validation", val_idx)):
        for i in tqdm(indices, desc=split, unit="recording"):
            events, label = dataset[i]
            events = transforms.Denoise(filter_time=DENOISE_US)(events)
            t = events["t"] - events["t"][0]
            counts = np.bincount(t // bin_us)
            histograms[i] = counts
            row = dict(index=i, split=split, user=dataset.users[i], label=label,
                       duration_ms=round(int(t[-1]) / 1000), events=len(t))
            for f in args.fractions:
                start, stop = active_range(counts, f)
                row[f"lead_ms_{f}"] = start * args.bin_ms
                row[f"trail_ms_{f}"] = (len(counts) - stop) * args.bin_ms
                row[f"active_ms_{f}"] = (stop - start) * args.bin_ms
                row[f"kept_events_{f}"] = round(counts[start:stop].sum() / len(t), 4)
            rows.append(row)

    with open(args.output_dir / "activity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    def column(name, split=None):
        return np.array([r[name] for r in rows if split is None or r["split"] == split])

    percentiles = (0, 5, 25, 50, 75, 95, 100)
    lines = [f"{len(rows)} recordings, {args.bin_ms} ms bins",
             f"{'':<24}" + "".join(f"{f'p{p}':>9}" for p in percentiles)]
    names = ["duration_ms"] + [f"{k}_{f}" for f in args.fractions
                               for k in ("lead_ms", "trail_ms", "active_ms", "kept_events")]
    for name in names:
        values = column(name)
        fmt = "{:9.3f}" if name.startswith("kept") else "{:9.0f}"
        lines.append(f"{name:<24}" + "".join(fmt.format(v) for v in np.percentile(values, percentiles)))
    middle = args.fractions[len(args.fractions) // 2]
    lines.append(f"\nMedian per class (fraction {middle}): duration / lead / trail / active ms")
    labels = column("label")
    for c, name in enumerate(DVSGesture.classes):
        mask = labels == c
        medians = [np.median(column(n)[mask]) for n in
                   ("duration_ms", f"lead_ms_{middle}", f"trail_ms_{middle}", f"active_ms_{middle}")]
        lines.append(f"  {c:2d} {name:<16}" + "".join(f"{m:8.0f}" for m in medians))
    summary = "\n".join(lines)
    print(summary)
    (args.output_dir / "summary.txt").write_text(summary + "\n")

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax in axes.flat[NUM_CLASSES:]:
        ax.axis("off")
    for c, ax in enumerate(axes.flat[:NUM_CLASSES]):
        i = next(i for i in train_idx if dataset.targets[i] == c)
        counts = histograms[i]
        bin_s = args.bin_ms / 1000
        ax.bar(np.arange(len(counts)) * bin_s, counts, width=bin_s, align="edge")
        start, stop = active_range(counts, middle)
        ax.axvspan(start * bin_s, stop * bin_s, color="tab:orange", alpha=0.25)
        ax.set_title(f"{DVSGesture.classes[c]} (user {dataset.users[i]})", fontsize=9)
        ax.set_xlabel("s")
    fig.suptitle(f"Events per {args.bin_ms} ms; shaded: bins >= {middle} x peak")
    fig.tight_layout()
    fig.savefig(args.output_dir / "examples.png", dpi=120)


if __name__ == "__main__":
    main()
