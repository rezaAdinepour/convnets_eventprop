"""Load development parents and encode one 200 ms prefix per gesture."""

from pathlib import Path

import numpy as np
from ml_genn.utils.data import preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture


INPUT_SHAPE = (34, 34, 2)  # The 32 x 32 event field has a silent one-pixel border.
TRIAL_STEPS = 200
BIN_US = 1000


def load_development_split(data_root: Path):
    """Return the Tonic dataset and deterministic train/validation indices."""
    dataset_dir = data_root / "DVSGesture"
    train_dir = dataset_dir / "ibmGestureTrain"
    # Tonic can download when either the archive or extracted files are missing.
    if not (dataset_dir / DVSGesture.train_filename).is_file():
        raise FileNotFoundError(f"Missing local training archive in {dataset_dir}")
    if len(list(train_dir.glob("**/*.npy"))) != 1077:
        raise ValueError(f"Expected 1077 extracted training parents in {train_dir}")

    dataset = DVSGesture(save_to=str(data_root), train=True)
    paths = dataset.data
    if len(paths) != 1077 or len(set(paths)) != 1077:
        raise ValueError("Tonic must identify exactly 1077 distinct parent paths")
    if set(dataset.users) != set(range(1, 24)):
        raise ValueError("Expected official training subjects 1 through 23")

    # Tonic's filesystem traversal order is not part of the split definition.
    indices = sorted(range(len(dataset)), key=paths.__getitem__)
    train = [i for i in indices if dataset.users[i] <= 20]
    validation = [i for i in indices if dataset.users[i] >= 21]
    if (len(train), len(validation)) != (934, 143):
        raise ValueError(f"Unexpected split sizes: {len(train)}, {len(validation)}")
    return dataset, train, validation


def encode_parent(events):
    """Return binary input spikes in ms and the number of in-window source events."""
    times = events["t"]
    if np.any(times[1:] < times[:-1]):
        raise ValueError("Parent timestamps must be nondecreasing")
    if any(np.any((events[axis] < 0) | (events[axis] >= 128)) for axis in ("x", "y")):
        raise ValueError("Event outside the 128 x 128 sensor")

    if len(events):
        events = transforms.Denoise(filter_time=10000)(events)
    if len(events):
        times = events["t"]
        stop = np.searchsorted(times, int(times[0]) + TRIAL_STEPS * BIN_US)
        events = transforms.Downsample(spatial_factor=0.25)(events[:stop])
        events["x"] += 1
        events["y"] += 1
        events["t"] = ((events["t"] - events["t"][0]) // BIN_US) * BIN_US

    # One event per (x, y, polarity, bin). Keeping explicit bin starts also
    # preserves a spike in the last occupied millisecond.
    occupied = np.unique(events)
    spikes = preprocess_tonic_spikes(occupied, DVSGesture.ordering, INPUT_SHAPE)
    if not np.isfinite(spikes.spike_times).all():
        raise ValueError("Nonfinite input spike time")
    return spikes, len(events)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Parent of the DVSGesture directory")
    parser.add_argument("--example", action="store_true", help="Encode the first training parent")
    args = parser.parse_args()
    dataset, train, validation = load_development_split(args.data_root)
    print(f"Train: {len(train)} parents, subjects 1-20")
    print(f"Validation: {len(validation)} parents, subjects 21-23")
    if args.example:
        index = train[0]
        events, label = dataset[index]
        spikes, in_window = encode_parent(events)
        count = len(spikes.spike_times)
        if spikes.end_spikes.shape != (int(np.prod(INPUT_SHAPE)),) or spikes.end_spikes[-1] != count:
            raise ValueError("Unexpected SpikeInput layout")
        if count > in_window or np.any((spikes.spike_times < 0) | (spikes.spike_times >= TRIAL_STEPS)):
            raise ValueError("Input spikes exceed the event count or the trial bounds")
        print(f"Parent: {dataset.data[index]}, label: {label}")
        print(f"In-window events: {in_window}, occupied bins: {count}, input neurons: {spikes.end_spikes.size}")
