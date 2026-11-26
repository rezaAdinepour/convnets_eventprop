"""Plot our conv1 filters next to the Arfa et al. checkpoint (read without PyTorch).

    python compare_conv1_filters.py RUN_DIR/train OUTPUT.png
"""
import argparse
import collections
import json
import pickle
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

REFERENCE = (Path(__file__).resolve().parents[2]
             / "deep-snn-deployment-on-spinnaker2-single-chip-using-nir/PTQ/SpiNNaker2"
             / "PTQ_time_window-1ms-snntorch_dvsgesture_model.pth")
STORAGE_DTYPES = {"FloatStorage": np.dtype("<f4"), "DoubleStorage": np.dtype("<f8"),
                  "IntStorage": np.dtype("<i4"), "LongStorage": np.dtype("<i8")}
POLARITIES = ("OFF", "ON")


def _rebuild_tensor(storage, offset, size, stride, *args):
    strides = [s * storage.itemsize for s in stride]
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size, strides=strides).copy()


class _Unpickler(pickle.Unpickler):
    # only lets tensors through

    def __init__(self, file, archive, prefix):
        super().__init__(file)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in STORAGE_DTYPES:
            return STORAGE_DTYPES[name]
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed")

    def persistent_load(self, pid):
        _, dtype, key, _, numel = pid
        data = self.archive.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(data, dtype=dtype, count=numel)


def load_state_dict(path):
    with zipfile.ZipFile(path) as archive:
        pkl = next(n for n in archive.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[:-len("/data.pkl")]
        with archive.open(pkl) as f:
            return _Unpickler(f, archive, prefix).load()


def reshape_conv_weight(weight):
    # (out, in, kh, kw) -> (kh, kw, in, out)
    return np.moveaxis(weight, (0, 1, 2, 3), (3, 2, 0, 1))


def cosine(a, b):
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    a = a / np.linalg.norm(a, axis=0)
    b = b / np.linalg.norm(b, axis=0)
    return a.T @ b


def on_off_correlation(w):
    return np.array([np.corrcoef(w[:, :, 0, f].ravel(), w[:, :, 1, f].ravel())[0, 1]
                     for f in range(w.shape[-1])])


def describe(name, w):
    print(f"{name:<22} mean |w| {np.abs(w).mean():.3f}, sd {w.std():.3f}, "
          f"ON/OFF correlation per filter {on_off_correlation(w).mean():+.2f} (mean)")


def draw(axes, w, title, order):
    for col, f in enumerate(order):
        scale = np.abs(w[..., f]).max()
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(w[:, :, row, f] / scale, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{title}\n{POLARITIES[row]}" if row == 0 else POLARITIES[row],
                              fontsize=8)
            if row == 0:
                ax.set_title(f"{f}", fontsize=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epoch", type=int, default=None, help="Default: the last epoch")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists")

    epoch = args.epoch
    if epoch is None:
        epoch = json.loads((args.run_dir / "run_config.json").read_text())["epochs"] - 1
    checkpoints = args.run_dir / "checkpoints"
    ours = np.load(checkpoints / f"{epoch}-conv1-weight.npy").reshape(5, 5, 2, -1)
    initial = np.load(checkpoints / "initial-conv1-weight.npy").reshape(5, 5, 2, -1)
    state = load_state_dict(args.reference)
    theirs = reshape_conv_weight(state["0.weight"])
    if theirs.shape != ours.shape:
        raise ValueError(f"Shapes differ: theirs {theirs.shape}, ours {ours.shape}")
    print(f"Reference neuron parameters: beta {float(state['1.beta']):.3f}, "
          f"threshold {float(state['1.threshold']):.3f}")

    describe("theirs", theirs)
    describe(f"ours, epoch {epoch}", ours)
    describe("ours, initial", initial)

    rows = []
    for name, w in ((f"ours, epoch {epoch}", ours), ("ours, initial", initial)):
        similarity = cosine(theirs, w)
        their_idx, our_idx = linear_sum_assignment(-similarity)
        matched = similarity[their_idx, our_idx]
        rows.append((name, w, our_idx))
        print(f"{name:<22} matched cosine similarity to theirs: mean {matched.mean():.2f}, "
              f"min {matched.min():.2f}, max {matched.max():.2f}")

    fig, axes = plt.subplots(6, theirs.shape[-1], figsize=(16, 7.2))
    draw(axes[0:2], theirs, "theirs", range(theirs.shape[-1]))
    for i, (name, w, order) in enumerate(rows, start=1):
        draw(axes[2 * i:2 * i + 2], w, name, order)
    fig.suptitle("Conv1 filters")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
