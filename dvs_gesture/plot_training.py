import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(run_dir):
    with open(run_dir / "metrics.csv") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    val_accuracy = [100 * int(r["val_correct"]) / int(r["val_total"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    return epochs, val_accuracy, train_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    runs = {run.name: read_metrics(run) for run in args.run_dirs}
    for column, ylabel, filename in ((1, "Validation accuracy (%)", "validation_accuracy.svg"),
                                     (2, "Training loss", "training_loss.svg")):
        plt.figure()
        for name, values in runs.items():
            plt.plot(values[0], values[column], label=name)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output / filename)
        plt.close()
        print(args.output / filename)


if __name__ == "__main__":
    main()
