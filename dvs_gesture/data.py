from pathlib import Path

import numpy as np
from ml_genn.utils.data import preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture
from tqdm import tqdm

from evdownsampling import BIN_MS, evdownsampling_spikes

VALIDATION_SUBJECTS = (21, 22, 23)
NUM_CLASSES = 11

# 128x128 sensor -> 32x32 after 4x spatial downsampling, plus a one-pixel
# silent border on each side -> 34x34 input with ON/OFF channels.
SENSOR_SIZE = 128
SPATIAL_FACTOR = 4
BORDER = 1
INPUT_SHAPE = (SENSOR_SIZE // SPATIAL_FACTOR + 2 * BORDER,
               SENSOR_SIZE // SPATIAL_FACTOR + 2 * BORDER, 2)
DENOISE_US = 10_000

ENCODINGS = ("occupancy", "evdownsampling")


def load_dataset(data_root: Path, train: bool):
    """Open an already-extracted official partition, without downloading."""
    folder = data_root / "DVSGesture" / ("ibmGestureTrain" if train else "ibmGestureTest")
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found; this script never downloads data")
    return DVSGesture(save_to=str(data_root), train=train)


def split_train_validation(dataset):
    """Return training and validation indices, sorted by file path."""
    order = sorted(range(len(dataset)), key=lambda i: dataset.data[i])
    train = [i for i in order if dataset.users[i] not in VALIDATION_SUBJECTS]
    validation = [i for i in order if dataset.users[i] in VALIDATION_SUBJECTS]
    return train, validation


def encode_gesture(events, encoding, trial_ms=None):
    """Return (spikes, num_bins) for one complete recording.

    occupancy:       one spike per (pixel, polarity, 1 ms bin) that has at least
                     one event, i.e. repeated events in a bin are merged.
    evdownsampling:  adaptive-threshold encoder, see evdownsampling.py.

    trial_ms=None keeps real time. Otherwise the whole recording is linearly
    compressed (or stretched) into exactly trial_ms ms.
    """
    events = transforms.Denoise(filter_time=DENOISE_US)(events)
    if len(events) == 0:
        raise ValueError("Recording is empty after denoising")

    bin_us = BIN_MS * 1000
    t0 = int(events["t"][0])
    if trial_ms is None:
        num_bins = max(1, (int(events["t"][-1]) - t0) // bin_us)
        events = events[events["t"] < t0 + num_bins * bin_us]
    else:
        num_bins = int(trial_ms) // BIN_MS
    events = transforms.Downsample(spatial_factor=1 / SPATIAL_FACTOR)(events)
    events["x"] += BORDER
    events["y"] += BORDER
    if trial_ms is None:
        events["t"] -= t0
    else:
        duration_us = int(events["t"][-1]) - t0 + 1
        events["t"] = (events["t"] - t0) * (num_bins * bin_us) // duration_us
    events["t"] = (events["t"] // bin_us) * bin_us

    if encoding == "occupancy":
        spikes = preprocess_tonic_spikes(np.unique(events), DVSGesture.ordering, INPUT_SHAPE)
    elif encoding == "evdownsampling":
        spikes = evdownsampling_spikes(events, num_bins, INPUT_SHAPE)
    else:
        raise ValueError(f"Unknown encoding {encoding!r}")
    return spikes, num_bins


def encode_recordings(dataset, indices, encoding, description="Encoding", trial_ms=None):
    """Encode the given recordings; return spikes, labels and lengths in bins."""
    spikes, labels, lengths = [], [], []
    for i in tqdm(indices, desc=description, unit="recording"):
        events, label = dataset[i]
        s, num_bins = encode_gesture(events, encoding, trial_ms)
        spikes.append(s)
        labels.append(label)
        lengths.append(num_bins)
    return spikes, np.array(labels), np.array(lengths)


def max_spikes_per_neuron(spikes):
    """Largest number of spikes any single input neuron emits in one recording."""
    return max(int(np.diff(s.end_spikes, prepend=0).max()) for s in spikes)


def max_spikes_per_batch(spikes, batch_size):
    """Spikes in the busiest possible batch; sizes the SpikeInput buffer."""
    sizes = sorted((len(s.spike_times) for s in spikes), reverse=True)
    return sum(sizes[:batch_size])
