"""Compare Base line and EvDownsampling using their per-epoch training logs."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


RUN_LABELS = ("Base line", "EvDownsampling")
COLORS = ("tab:blue", "tab:orange")
DEFAULT_SPLIT = "validation"
EPOCH_LINE = re.compile(r"^Epoch (\d+) \| Online train: (\d+)/(\d+) \| Fixed-weight validation: (\d+)/(\d+)$")
LOSS_LINE = re.compile(r"^Online train CE: (\S+) \| Fixed-weight validation CE: (\S+)$")


def read_log(path, split):
    epochs, losses, accuracies = [], [], []
    pending = None
    for line in path.read_text().splitlines():
        counts = EPOCH_LINE.fullmatch(line.strip())
        if counts:
            if pending is not None:
                raise ValueError(f"{path}: missing loss after epoch {pending[0]}")
            pending = tuple(map(int, counts.groups()))
        loss = LOSS_LINE.fullmatch(line.strip())
        if loss and pending is not None:
            epoch, train_ok, train_total, val_ok, val_total = pending
            if epochs and epoch != epochs[-1] + 1:
                raise ValueError(f"{path}: unexpected epoch {epoch}")
            epochs.append(epoch)
            if split == "validation":
                accuracies.append(100 * val_ok / val_total)
                losses.append(float(loss[2]))
            else:
                accuracies.append(100 * train_ok / train_total)
                losses.append(float(loss[1]))
            pending = None
    if not epochs or pending is not None:
        raise ValueError(f"{path}: expected complete per-epoch accuracy and loss")
    return epochs, losses, accuracies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_log", type=Path)
    parser.add_argument("evdownsampling_log", type=Path)
    parser.add_argument("--split", choices=("train", "validation"), default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    runs = [read_log(path, args.split) for path in (args.baseline_log, args.evdownsampling_log)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_label = "Validation" if args.split == "validation" else "Online training"
    for metric, ylabel, column in (("loss", "Cross-entropy loss", 1), ("accuracy", "Accuracy (%)", 2)):
        plt.figure()
        for run, label, color in zip(runs, RUN_LABELS, COLORS):
            plt.plot(run[0], run[column], label=label, color=color)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{split_label} {metric}")
        plt.legend()
        plt.tight_layout()
        output = args.output_dir / f"{args.split}_{metric}.svg"
        plt.savefig(output)
        plt.close()
        print(output)
