"""Compare conv1 filters of our runs with Arfa et al.'s checkpoint, each other and chance.

    python conv1_similarity.py OUTPUT_DIR RUN_DIR [RUN_DIR ...] [--epoch E] [--draws 2000]

Two banks are compared by the mean cosine similarity of their filters after one-to-one
matching (Hungarian). Chance comes from random Gaussian banks.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from compare_conv1_filters import REFERENCE, cosine, load_state_dict, reshape_conv_weight
from run_labels import PAPER, initial_label, load_config, run_label

POLARITIES = ("OFF", "ON")


def matched(a, b):
    similarity = cosine(a, b)
    rows, cols = linear_sum_assignment(-similarity)
    return similarity[rows, cols], cols


def spatial(w):
    return w - w.mean(axis=(0, 1), keepdims=True)


def raw(w):
    return w


def train_dir(path):
    path = path.resolve()
    return path / "train" if (path / "train" / "run_config.json").exists() else path


def similarity_report(banks, seeds, ours, transform, draws):
    paper = list(banks)[0]
    shape = banks[paper].shape
    theirs = transform(banks[paper])
    rng = np.random.default_rng(0)
    chance_theirs = [matched(theirs, transform(rng.standard_normal(shape)))[0]
                     for _ in range(draws)]
    chance_random = [matched(transform(rng.standard_normal(shape)),
                             transform(rng.standard_normal(shape)))[0].mean()
                     for _ in range(draws)]
    chance_means = np.array([c.mean() for c in chance_theirs])
    chance_filter_95 = np.percentile(np.concatenate(chance_theirs), 95)

    names = list(banks)
    matrix = np.array([[matched(transform(banks[a]), transform(banks[b]))[0].mean()
                        for b in names] for a in names])
    width = max(len(n) for n in names) + 2
    lines = [f"{i}: {n}" for i, n in enumerate(names)]
    lines += ["", " " * 4 + "".join(f"{i:>8}" for i in range(len(names)))]
    lines += [f"{i:<4}" + "".join(f"{v:8.3f}" for v in row) for i, row in enumerate(matrix)]
    lines += ["", f"Chance, random bank vs {PAPER}: {chance_means.mean():.3f} +- "
                  f"{chance_means.std():.3f} (95th percentile {np.percentile(chance_means, 95):.3f}, "
                  f"{draws} draws)",
              f"Chance, random vs random bank: {np.mean(chance_random):.3f} +- "
              f"{np.std(chance_random):.3f} (95th percentile "
              f"{np.percentile(chance_random, 95):.3f})",
              f"Chance, single matched filter vs {PAPER}: 95th percentile "
              f"{chance_filter_95:.3f}", ""]
    for n in ours:
        values, _ = matched(theirs, transform(banks[n]))
        p = np.mean(chance_means >= values.mean())
        lines.append(f"{n:<{width}} vs {PAPER}: mean {values.mean():.3f} (p = {p:.3f}); "
                     f"filters above the single-filter 95th percentile: "
                     f"{np.sum(values > chance_filter_95)}/{len(values)}")
    pairs = [(a, b) for i, a in enumerate(ours) for b in ours[i + 1:]]
    for label, keep in (("different seeds", lambda a, b: seeds[a] != seeds[b]),
                        ("same seed", lambda a, b: seeds[a] == seeds[b])):
        values = [matrix[names.index(a), names.index(b)] for a, b in pairs if keep(a, b)]
        if values:
            lines.append(f"Ours vs ours, {label}: mean {np.mean(values):.3f} "
                         f"(min {np.min(values):.3f}, max {np.max(values):.3f}, "
                         f"{len(values)} pairs)")
    to_theirs = [matrix[0, names.index(n)] for n in ours]
    lines.append(f"Ours vs {PAPER}: mean {np.mean(to_theirs):.3f}")
    return lines, matrix, chance_means.mean()


def plot_matrix(matrix, names, chance, path, title):
    fig, ax = plt.subplots(figsize=(0.8 * len(names) + 2.5, 0.7 * len(names) + 1.5))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title(f"{title} (chance ≈ {chance:.2f})")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_filter_pair(theirs, ours, our_name, path):
    values, order = matched(theirs, ours)
    n = theirs.shape[-1]
    fig = plt.figure(figsize=(n * 0.8 + 1.2, 4.6))
    grid = fig.add_gridspec(5, n, height_ratios=[1, 1, 0.3, 1, 1], left=0.09, right=0.99,
                            top=0.88, bottom=0.02, hspace=0.3, wspace=0.1)
    for group, (w, filters, first_row) in enumerate(((theirs, range(n), 0), (ours, order, 3))):
        for col, f in enumerate(filters):
            scale = np.abs(w[..., f]).max()
            for p in range(2):
                ax = fig.add_subplot(grid[first_row + p, col])
                ax.imshow(w[:, :, p, f] / scale, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(POLARITIES[p], fontsize=8)
                if p == 0 and group == 1:
                    ax.set_title(f"{values[col]:.2f}", fontsize=7)
    fig.text(0.01, 0.68, PAPER, fontsize=10, va="center", rotation=90)
    fig.text(0.01, 0.22, f"Ours\n{our_name}", fontsize=10, va="center", ha="left",
             rotation=90, multialignment="center")
    fig.suptitle(f"Conv1 filters, matched one-to-one (mean cosine {values.mean():.2f})")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def file_name(label):
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    return "_".join(part for part in keep.split("_") if part)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--epoch", type=int, default=None, help="Default: each run's last epoch")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()

    theirs = reshape_conv_weight(load_state_dict(args.reference)["0.weight"])
    banks, seeds = {PAPER: theirs}, {PAPER: None}
    for d in map(train_dir, args.run_dirs):
        config = load_config(d)
        epoch = config["epochs"] - 1 if args.epoch is None else args.epoch
        name = run_label(config)
        if name in banks:
            parser.error(f"Two runs are both '{name}'")
        banks[name] = np.load(d / "checkpoints" / f"{epoch}-conv1-weight.npy").reshape(theirs.shape)
        seeds[name] = config["seed"]
        if len(banks) == 2:
            initial = np.load(d / "checkpoints" / "initial-conv1-weight.npy")
            banks[initial_label(config)] = initial.reshape(theirs.shape)
            seeds[initial_label(config)] = config["seed"]
    names = list(banks)
    ours = [n for n in names if n != PAPER and not n.startswith("Initial")]

    out = args.output_dir.resolve()
    pair_files = {n: out / f"conv1_filters_{file_name(n)}.png" for n in ours}
    files = [out / "conv1_similarity.txt", out / "conv1_similarity_raw.png",
             out / "conv1_similarity_mean_removed.png", *pair_files.values()]
    existing = [str(f) for f in files if f.exists()]
    if existing:
        parser.error(f"Would overwrite {', '.join(existing)}")
    out.mkdir(parents=True, exist_ok=True)

    width = max(len(n) for n in names) + 2
    lines = ["Mean weight per polarity / sd of the bank:"]
    lines += [f"  {n:<{width}}" + "  ".join(
        f"{p} {banks[n][:, :, i].mean() / banks[n].std():+.2f}" for i, p in enumerate(POLARITIES))
        for n in names]
    for label, transform, filename, title in (
            ("raw weights", raw, "conv1_similarity_raw.png", "Conv1 similarity"),
            ("mean per filter and polarity removed", spatial,
             "conv1_similarity_mean_removed.png", "Conv1 similarity, mean removed")):
        report, matrix, chance = similarity_report(banks, seeds, ours, transform, args.draws)
        lines += ["", f"=== {label}", ""] + report
        plot_matrix(matrix, names, chance, out / filename, title)
    text = "\n".join(lines)
    print(text)
    (out / "conv1_similarity.txt").write_text(text + "\n")

    for name, path in pair_files.items():
        plot_filter_pair(theirs, banks[name], name, path)
    print(f"Saved {len(files)} files in {out}")


if __name__ == "__main__":
    main()
