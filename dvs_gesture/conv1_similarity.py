"""conv1 filters of several of our runs against Arfa et al.'s checkpoint, each other and chance.

    python conv1_similarity.py OUTPUT_DIR RUN_DIR [RUN_DIR ...] [--epoch E] [--draws 2000]

Reads only weights (no GPU). Filter order is arbitrary, so two filter banks are compared by
matching their 16 filters one-to-one to maximise the summed cosine similarity (Hungarian
algorithm) and averaging the matched cosines. Chance is the same score for banks of
independent Gaussian filters (our initialisation): the matching alone makes it positive.
Everything is reported twice: on the raw weights, and on the spatial pattern only (each
filter's mean per polarity removed), since a shared sign per polarity alone makes filters
look alike. Seed vs seed excludes runs that share a seed, as they start from the same
weights. Writes into OUTPUT_DIR (which may exist; files are never overwritten):
conv1_similarity.txt, conv1_cosine_matrix_raw.png, conv1_cosine_matrix_spatial.png and
one conv1_filters_paper_vs_<run>.png per run.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from compare_conv1_filters import REFERENCE, cosine, load_state_dict, reshape_conv_weight
from run_labels import PAPER, PAPER_SHORT, initial_label, load_config, run_label

POLARITIES = ("OFF events", "ON events")


def matched(a, b):
    """Matched cosines (in a's filter order) and b's filter index matched to each of a's."""
    similarity = cosine(a, b)
    rows, cols = linear_sum_assignment(-similarity)
    return similarity[rows, cols], cols


def spatial(w):
    """Remove each filter's mean per polarity, leaving its spatial pattern."""
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
    lines += ["", f"Chance, random Gaussian bank vs {PAPER_SHORT}: {chance_means.mean():.3f} +- "
                  f"{chance_means.std():.3f} (95th percentile {np.percentile(chance_means, 95):.3f}, "
                  f"{draws} draws)",
              f"Chance, random vs random bank: {np.mean(chance_random):.3f} +- "
              f"{np.std(chance_random):.3f} (95th percentile "
              f"{np.percentile(chance_random, 95):.3f})",
              f"Chance, a single matched filter vs {PAPER_SHORT}: 95th percentile "
              f"{chance_filter_95:.3f}", ""]
    for n in ours:
        values, _ = matched(theirs, transform(banks[n]))
        p = np.mean(chance_means >= values.mean())
        lines.append(f"{n:<{width}} vs {PAPER_SHORT}: mean {values.mean():.3f} (fraction of "
                     f"chance draws >= this: {p:.3f}); filters above the single-filter 95th "
                     f"percentile: {np.sum(values > chance_filter_95)}/{len(values)}")
    pairs = [(a, b) for i, a in enumerate(ours) for b in ours[i + 1:]]
    for label, keep in (("different seeds", lambda a, b: seeds[a] != seeds[b]),
                        ("same seed, so same initial weights", lambda a, b: seeds[a] == seeds[b])):
        values = [matrix[names.index(a), names.index(b)] for a, b in pairs if keep(a, b)]
        if values:
            lines.append(f"Ours vs ours, {label}: mean {np.mean(values):.3f} "
                         f"(min {np.min(values):.3f}, max {np.max(values):.3f}, "
                         f"{len(values)} pairs)")
    to_theirs = [matrix[0, names.index(n)] for n in ours]
    lines.append(f"Ours vs {PAPER_SHORT}: mean {np.mean(to_theirs):.3f}")
    return lines, matrix, chance_means.mean(), np.mean(chance_random)


def plot_matrix(matrix, names, chance_paper, chance_random, path, what):
    fig, ax = plt.subplots(figsize=(0.95 * len(names) + 4, 0.8 * len(names) + 2.2))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title(f"Similarity of conv1 filter banks ({what})\n"
                 f"mean cosine similarity of the 16 filters after one-to-one matching; "
                 f"chance: {chance_paper:.2f} vs {PAPER_SHORT}, {chance_random:.2f} between "
                 f"two random banks", fontsize=9)
    fig.colorbar(image, ax=ax, label="Mean matched cosine similarity (1 = identical patterns)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_filter_pair(theirs, ours, our_name, path):
    """Paper filters (top two rows) and our filters matched to them (bottom two rows)."""
    values, order = matched(theirs, ours)
    n = theirs.shape[-1]
    fig = plt.figure(figsize=(n * 0.85 + 1.6, 5.6))
    grid = fig.add_gridspec(5, n, height_ratios=[1, 1, 0.35, 1, 1], left=0.12, right=0.99,
                            top=0.80, bottom=0.02, hspace=0.35, wspace=0.12)
    for group, (w, filters, first_row) in enumerate(((theirs, range(n), 0), (ours, order, 3))):
        for col, f in enumerate(filters):
            scale = np.abs(w[..., f]).max()
            for p in range(2):
                ax = fig.add_subplot(grid[first_row + p, col])
                ax.imshow(w[:, :, p, f] / scale, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(POLARITIES[p], fontsize=7)
                if p == 0:
                    title = (f"#{f}" if group == 0 else f"#{f}\ncos {values[col]:.2f}")
                    ax.set_title(title, fontsize=6.5)
    fig.text(0.005, 0.66, PAPER_SHORT, fontsize=10, weight="bold", va="center",
             rotation=90, color="tab:red")
    fig.text(0.005, 0.20, "Ours", fontsize=10, weight="bold", va="center",
             rotation=90, color="tab:blue")
    fig.text(0.5, 0.965, f"conv1 filters: {PAPER} (top) vs {our_name} (bottom)",
             ha="center", fontsize=9, weight="bold")
    fig.text(0.5, 0.905,
             "Each column is one filter (5x5 pixels for OFF and for ON events). Each of our filters "
             "is paired one-to-one with the paper filter above it, maximising the summed cosine "
             f"similarity. #: filter index; cos: cosine similarity of the pair "
             f"(mean {values.mean():.2f}).\nRed: positive weight, blue: negative; each filter is "
             "scaled to its own largest |weight|.", ha="center", va="top", fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def file_name(label):
    keep = "".join(c if c.isalnum() else "_" for c in label.lower())
    return "_".join(part for part in keep.split("_") if part)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Directory for the outputs")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Directories written by train.py")
    parser.add_argument("--epoch", type=int, default=None, help="Default: each run's last epoch")
    parser.add_argument("--draws", type=int, default=2000, help="Random banks for chance")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()

    theirs = reshape_conv_weight(load_state_dict(args.reference)["0.weight"])
    banks, seeds = {PAPER: theirs}, {PAPER: None}
    for d in map(train_dir, args.run_dirs):
        config = load_config(d)
        epoch = config["epochs"] - 1 if args.epoch is None else args.epoch
        name = run_label(config, short=True)
        if name in banks:
            parser.error(f"Two runs are both '{name}'")
        banks[name] = np.load(d / "checkpoints" / f"{epoch}-conv1-weight.npy").reshape(theirs.shape)
        seeds[name] = config["seed"]
        if len(banks) == 2:
            initial = np.load(d / "checkpoints" / "initial-conv1-weight.npy")
            banks[initial_label(config)] = initial.reshape(theirs.shape)
            seeds[initial_label(config)] = config["seed"]
    names = list(banks)
    ours = [n for n in names if n != PAPER and "initial" not in n]

    out = args.output_dir.resolve()
    outputs = {n: out / f"conv1_filters_paper_vs_{file_name(n)}.png" for n in ours}
    files = [out / "conv1_similarity.txt", out / "conv1_cosine_matrix_raw.png",
             out / "conv1_cosine_matrix_spatial.png", *outputs.values()]
    existing = [str(f) for f in files if f.exists()]
    if existing:
        parser.error(f"Would overwrite {', '.join(existing)}")
    out.mkdir(parents=True, exist_ok=True)

    width = max(len(n) for n in names) + 2
    lines = [f"conv1 filter banks: {theirs.shape[-1]} filters of {np.prod(theirs.shape[:3])} "
             "weights (5x5 pixels x OFF/ON)", "",
             "Mean weight per polarity in units of the bank's sd (a shared sign per polarity):"]
    lines += [f"  {n:<{width}}" + "  ".join(
        f"{p.split()[0]} {banks[n][:, :, i].mean() / banks[n].std():+.2f}"
        for i, p in enumerate(POLARITIES)) for n in names]
    for label, what, transform, filename in (
            ("raw weights", "raw weights", raw, "conv1_cosine_matrix_raw.png"),
            ("spatial pattern only (mean per filter and polarity removed)",
             "spatial pattern only: each filter's mean per polarity removed", spatial,
             "conv1_cosine_matrix_spatial.png")):
        report, matrix, chance_paper, chance_random = similarity_report(
            banks, seeds, ours, transform, args.draws)
        lines += ["", f"=== Mean matched cosine similarity, {label}", ""] + report
        plot_matrix(matrix, names, chance_paper, chance_random, out / filename, what)
    text = "\n".join(lines)
    print(text)
    (out / "conv1_similarity.txt").write_text(text + "\n")

    for name, path in outputs.items():
        plot_filter_pair(theirs, banks[name], name, path)
    print(f"Saved {len(files)} files in {out}")


if __name__ == "__main__":
    main()
