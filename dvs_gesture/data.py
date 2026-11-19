import hashlib
import inspect
import os
from pathlib import Path

import numpy as np
from ml_genn.utils.data import PreprocessedSpikes, preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture
from tqdm import tqdm

import evdownsampling
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


def crop_events(events, crop_ms):
    """Keep the first crop_ms ms of real time, starting at the first event."""
    return events[events["t"] < events["t"][0] + crop_ms * 1000]


def encode_recordings(dataset, indices, encoding, description="Encoding", trial_ms=None,
                      cache_dir=None, crop_ms=None):
    """Encode the given recordings; return spikes, labels and lengths in bins.

    crop_ms keeps only the first crop_ms ms of each recording in real time
    (instead of compressing it with trial_ms).

    With cache_dir, the result is stored there and reused by later calls with
    the same recordings, encoding, trial_ms, crop_ms and encoder code.
    """
    if trial_ms is not None and crop_ms is not None:
        raise ValueError("Use either trial_ms or crop_ms, not both")
    if cache_dir is not None:
        key = _cache_key(dataset, indices, encoding, trial_ms, crop_ms)
        length = f"crop{crop_ms}" if crop_ms is not None else (trial_ms or "full")
        cache_file = Path(cache_dir) / f"{encoding}_{length}_{key}.npz"
        if cache_file.exists():
            print(f"{description}: loaded from {cache_file}")
            return _load_cache(cache_file)

    spikes, labels, lengths = [], [], []
    for i in tqdm(indices, desc=description, unit="recording"):
        events, label = dataset[i]
        if crop_ms is not None:
            events = crop_events(events, crop_ms)
        s, num_bins = encode_gesture(events, encoding, trial_ms)
        spikes.append(s)
        labels.append(label)
        lengths.append(num_bins)
    labels, lengths = np.array(labels), np.array(lengths)

    if cache_dir is not None:
        _save_cache(cache_file, spikes, labels, lengths)
        print(f"{description}: saved to {cache_file}")
    return spikes, labels, lengths


def _cache_key(dataset, indices, encoding, trial_ms, crop_ms=None):
    """Hash of everything that determines the encoded spikes."""
    key = hashlib.sha256()
    key.update(repr((encoding, trial_ms, DENOISE_US, SPATIAL_FACTOR, BORDER, INPUT_SHAPE,
                     [str(dataset.data[i]) for i in indices])).encode())
    key.update(inspect.getsource(encode_gesture).encode())
    key.update(Path(evdownsampling.__file__).read_bytes())
    if crop_ms is not None:
        key.update(repr(crop_ms).encode())
        key.update(inspect.getsource(crop_events).encode())
    return key.hexdigest()[:16]


def _save_cache(cache_file, spikes, labels, lengths):
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_suffix(".tmp.npz")
    np.savez(temporary, labels=labels, lengths=lengths,
             end_spikes=np.stack([s.end_spikes for s in spikes]),
             spike_counts=np.array([len(s.spike_times) for s in spikes]),
             spike_times=np.concatenate([s.spike_times for s in spikes]))
    os.replace(temporary, cache_file)


def _load_cache(cache_file):
    with np.load(cache_file) as cache:
        starts = np.concatenate(([0], np.cumsum(cache["spike_counts"])))
        times = cache["spike_times"]
        spikes = [PreprocessedSpikes(end, times[start:stop])
                  for end, start, stop in zip(cache["end_spikes"], starts[:-1], starts[1:])]
        return spikes, cache["labels"], cache["lengths"]


def max_spikes_per_neuron(spikes):
    """Largest number of spikes any single input neuron emits in one recording."""
    return max(int(np.diff(s.end_spikes, prepend=0).max()) for s in spikes)


def max_spikes_per_batch(spikes, batch_size):
    """Spikes in the busiest possible batch; sizes the SpikeInput buffer."""
    sizes = sorted((len(s.spike_times) for s in spikes), reverse=True)
    return sum(sizes[:batch_size])
