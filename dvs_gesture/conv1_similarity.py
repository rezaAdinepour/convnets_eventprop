"""conv1 filters of several of our runs against Arfa et al.'s checkpoint, each other and chance.

    python conv1_similarity.py OUTPUT_DIR RUN_DIR [RUN_DIR ...] [--epoch E] [--draws 2000]

Reads only weights (no GPU). Filter order is arbitrary, so two filter banks are compared by
matching their 16 filters one-to-one to maximise the summed cosine similarity (Hungarian
algorithm, as in compare_conv1_filters.py) and averaging the matched cosines. Chance is the
same score for banks of independent Gaussian filters (our initialisation): the matching
alone makes it positive. Writes similarity.txt, filters.png (every run matched to theirs,
matched cosine above each filter) and matrix.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from compare_conv1_filters import (POLARITIES, REFERENCE, cosine, load_state_dict,
                                   reshape_conv_weight)


def matched(a, b):
    """Matched cosines (in a's filter order) and b's filter index matched to each of a's."""
    similarity = cosine(a, b)
    rows, cols = linear_sum_assignment(-similarity)
    return similarity[rows, cols], cols


def train_dir(path):
    path = path.resolve()
    return path / "train" if (path / "train" / "run_config.json").exists() else path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="New directory for the outputs")
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Directories written by train.py")
    parser.add_argument("--epoch", type=int, default=None, help="Default: each run's last epoch")
    parser.add_argument("--draws", type=int, default=2000, help="Random banks for chance")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)

    theirs = reshape_conv_weight(load_state_dict(args.reference)["0.weight"])
    banks = {"theirs": theirs}
    for d in map(train_dir, args.run_dirs):
        config = json.loads((d / "run_config.json").read_text())
        epoch = config["epochs"] - 1 if args.epoch is None else args.epoch
        name = f"{config.get('run') or d.parent.name[-15:]} (s{config['seed']})"
        if name in banks:
            parser.error(f"Two runs are both named {name}")
        banks[name] = np.load(d / "checkpoints" / f"{epoch}-conv1-weight.npy").reshape(theirs.shape)
        if len(banks) == 2:
            initial = np.load(d / "checkpoints" / "initial-conv1-weight.npy")
            banks[f"initial (s{config['seed']})"] = initial.reshape(theirs.shape)
    ours = [n for n in banks if n not in ("theirs",) and not n.startswith("initial")]

    rng = np.random.default_rng(0)
    chance_theirs = [matched(theirs, rng.standard_normal(theirs.shape))[0]
                     for _ in range(args.draws)]
    chance_random = [matched(rng.standard_normal(theirs.shape),
                             rng.standard_normal(theirs.shape))[0].mean()
                     for _ in range(args.draws)]
    chance_means = np.array([c.mean() for c in chance_theirs])
    chance_filter_95 = np.percentile(np.concatenate(chance_theirs), 95)

    names = list(banks)
    matrix = np.array([[matched(banks[a], banks[b])[0].mean() for b in names] for a in names])
    lines = [f"Mean matched cosine similarity of conv1 banks ({theirs.shape[-1]} filters of "
             f"{np.prod(theirs.shape[:3])} weights)", ""]
    width = max(len(n) for n in names) + 2
    lines.append(" " * width + "".join(f"{n[:14]:>16}" for n in names))
    lines += [f"{a:<{width}}" + "".join(f"{v:16.3f}" for v in row)
              for a, row in zip(names, matrix)]
    lines += ["", f"Chance, random Gaussian bank vs theirs: {chance_means.mean():.3f} +- "
                  f"{chance_means.std():.3f} (95th percentile {np.percentile(chance_means, 95):.3f}, "
                  f"{args.draws} draws)",
              f"Chance, random vs random bank: {np.mean(chance_random):.3f} +- "
              f"{np.std(chance_random):.3f}",
              f"Chance, a single matched filter vs theirs: 95th percentile {chance_filter_95:.3f}",
              ""]
    for n in ours:
        values, _ = matched(theirs, banks[n])
        p = np.mean(chance_means >= values.mean())
        lines.append(f"{n:<{width}} vs theirs: mean {values.mean():.3f} (fraction of chance "
                     f"draws >= this: {p:.3f}); filters above the single-filter 95th "
                     f"percentile: {np.sum(values > chance_filter_95)}/{len(values)}")
    pairs = [(a, b) for i, a in enumerate(ours) for b in ours[i + 1:]]
    if pairs:
        between = [matrix[names.index(a), names.index(b)] for a, b in pairs]
        to_theirs = [matrix[0, names.index(n)] for n in ours]
        lines += ["", f"Ours vs ours (seed vs seed): mean {np.mean(between):.3f} over {len(pairs)} "
                      f"pairs; ours vs theirs: mean {np.mean(to_theirs):.3f}"]
    text = "\n".join(lines)
    print(text)
    (args.output_dir / "similarity.txt").write_text(text + "\n")

    n_filters = theirs.shape[-1]
    shown = ["theirs"] + ours
    fig, axes = plt.subplots(2 * len(shown), n_filters,
                             figsize=(n_filters, 1.15 * 2 * len(shown) + 0.6))
    for i, name in enumerate(shown):
        w = banks[name]
        values, order = matched(theirs, w) if name != "theirs" else (None, range(n_filters))
        for col, f in enumerate(order):
            scale = np.abs(w[..., f]).max()
            for row in range(2):
                ax = axes[2 * i + row, col]
                ax.imshow(w[:, :, row, f] / scale, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{name}\n{POLARITIES[row]}" if row == 0 else POLARITIES[row],
                                  fontsize=6)
                if row == 0:
                    label = f"{f}" if values is None else f"{f}: {values[col]:.2f}"
                    ax.set_title(label, fontsize=6)
    fig.suptitle("conv1 filters, each scaled to its own max |w| (red +, blue -). Our filters are "
                 "reordered to match theirs; titles: filter index: matched cosine.", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "filters.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 2, 1.0 * len(names) + 1))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if matrix[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title(f"Mean matched cosine (chance vs theirs {chance_means.mean():.2f})", fontsize=8)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(args.output_dir / "matrix.png", dpi=150)
    print(f"Saved {args.output_dir / 'filters.png'} and {args.output_dir / 'matrix.png'}")


if __name__ == "__main__":
    main()
