"""Load DVS Gesture partitions and encode a prefix or a complete gesture."""

from pathlib import Path

import numpy as np
from ml_genn.utils.data import preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture

from evdownsampling import BIN_US, NUM_POLARITIES, adaptive_spikes

VALIDATION_SUBJECTS = (21, 22, 23)
HISTORICAL_VALIDATION_SUBJECTS = (21, 22, 23)
DEVELOPMENT_SUBJECTS = range(1, 24)
TEST_SUBJECTS = range(24, 30)
DEVELOPMENT_RECORDINGS = 1077
TEST_RECORDINGS = 264
SENSOR_SIZE = 128
SPATIAL_REDUCTION = 4
FIELD_SIZE = SENSOR_SIZE // SPATIAL_REDUCTION
BORDER = 1
INPUT_SHAPE = (FIELD_SIZE + 2 * BORDER, FIELD_SIZE + 2 * BORDER, NUM_POLARITIES)
DENOISE_US = 10_000
TRIAL_STEPS = 200
NUM_CLASSES = 11
OCCUPANCY = "occupancy"
EVDOWNSAMPLING = "evdownsampling"
PREPROCESSORS = (OCCUPANCY, EVDOWNSAMPLING)


def load_partition(data_root: Path, *, train: bool):
    """Open an already extracted official partition in deterministic path order."""
    dataset_dir = data_root / "DVSGesture"
    if train:
        archive, folder, count = DVSGesture.train_filename, "ibmGestureTrain", DEVELOPMENT_RECORDINGS
        users = DEVELOPMENT_SUBJECTS
    else:
        archive, folder, count = DVSGesture.test_filename, "ibmGestureTest", TEST_RECORDINGS
        users = TEST_SUBJECTS
    # Tonic can download when either the archive or extracted files are missing.
    if not (dataset_dir / archive).is_file():
        raise FileNotFoundError(dataset_dir / archive)
    if len(list((dataset_dir / folder).glob("**/*.npy"))) != count:
        raise ValueError(f"Expected {count} extracted parents in {folder}")
    dataset = DVSGesture(save_to=str(data_root), train=train)
    paths = dataset.data
    if len(paths) != count or len(set(paths)) != count or set(dataset.users) != set(users):
        raise ValueError(f"Unexpected parent paths or subjects in {folder}")
    if set(dataset.targets) != set(range(len(dataset.classes))):
        raise ValueError(f"Unexpected class labels in {folder}")
    return dataset, sorted(range(count), key=paths.__getitem__)


def load_development_split(data_root: Path, validation_subjects=VALIDATION_SUBJECTS):
    """Return the Tonic dataset and deterministic train/validation indices."""
    held_out = set(validation_subjects)
    if not held_out or not held_out < set(DEVELOPMENT_SUBJECTS):
        raise ValueError("Validation subjects must form a nonempty proper subset of development subjects")
    dataset, indices = load_partition(data_root, train=True)
    train = [i for i in indices if dataset.users[i] not in held_out]
    validation = [i for i in indices if dataset.users[i] in held_out]
    return dataset, train, validation


def encode_trial(events, *, whole=False, preprocessing=OCCUPANCY):
    """Return binary spikes, retained event count and frame count before padding."""
    times = events["t"]
    if np.any(times[1:] < times[:-1]):
        raise ValueError("Parent timestamps must be nondecreasing")
    if any(np.any((events[axis] < 0) | (events[axis] >= SENSOR_SIZE)) for axis in ("x", "y")):
        raise ValueError("Event outside the sensor")

    if len(events):
        events = transforms.Denoise(filter_time=DENOISE_US)(events)
    steps = 1 if whole else TRIAL_STEPS
    if len(events):
        times = events["t"]
        if whole:
            # ToFrame(time_window=1000) uses complete half-open bins, at least one.
            steps = max(1, (int(times[-1]) - int(times[0])) // BIN_US)
        stop = np.searchsorted(times, int(times[0]) + steps * BIN_US)
        events = transforms.Downsample(spatial_factor=1 / SPATIAL_REDUCTION)(events[:stop])
        events["x"] += BORDER
        events["y"] += BORDER
        events["t"] = ((events["t"] - events["t"][0]) // BIN_US) * BIN_US

    # One event per (x, y, polarity, bin). Keeping explicit bin starts also
    # preserves a spike in the last occupied millisecond.
    if preprocessing == OCCUPANCY:
        occupied = np.unique(events)
        spikes = preprocess_tonic_spikes(occupied, DVSGesture.ordering, INPUT_SHAPE)
    elif preprocessing == EVDOWNSAMPLING:
        # Counts include repeated raw events before polarity cancellation.
        spikes = adaptive_spikes(events, steps, INPUT_SHAPE)
    else:
        raise ValueError(f"Unknown preprocessing: {preprocessing}")
    if not np.isfinite(spikes.spike_times).all():
        raise ValueError("Nonfinite input spike time")
    return spikes, len(events), steps


def encode_parent(events):
    spikes, count, _ = encode_trial(events)
    return spikes, count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path, help="Parent of the DVSGesture directory")
    parser.add_argument("--example", action="store_true", help="Encode the first training parent")
    args = parser.parse_args()
    dataset, train, validation = load_development_split(args.data_root)
    for name, indices in (("Train", train), ("Validation", validation)):
        subjects = sorted({dataset.users[i] for i in indices})
        print(f"{name}: {len(indices)} parents, subjects {subjects}")
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
