import contextlib
import hashlib
import inspect
import os
import sys
from pathlib import Path

import numpy as np
from ml_genn.utils.data import PreprocessedSpikes, preprocess_spikes, preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture
from tqdm import tqdm

VALIDATION_SUBJECTS = (21, 22, 23)
NUM_CLASSES = 11
BIN_MS = 1

# 128x128 sensor -> 32x32 after 4x spatial downsampling, plus a one-pixel
# silent border on each side -> 34x34 input with ON/OFF channels.
SENSOR_SIZE = 128
SPATIAL_FACTOR = 4
BORDER = 1
INPUT_SHAPE = (SENSOR_SIZE // SPATIAL_FACTOR + 2 * BORDER,
               SENSOR_SIZE // SPATIAL_FACTOR + 2 * BORDER, 2)
DENOISE_US = 10_000

ENCODINGS = ("occupancy", "evdownsampling", "counts")
COUNT_LEVELS = 4


def input_channels(encoding, count_levels=COUNT_LEVELS):
    return 2 * count_levels if encoding == "counts" else 2

# EvDownsampling parameters from the authors' test_script_genn.py
EVDOWNSAMPLING_REPO = Path(os.environ.get(
    "EVDOWNSAMPLING_REPO", Path(__file__).resolve().parents[2] / "EvDownsampling"))
TAU_THETA_MS = 420.0
TAU_ACCUMULATOR_MS = 16.0
BETA = 1.2
EVDOWNSAMPLING_PARAMS = (TAU_THETA_MS, TAU_ACCUMULATOR_MS, BETA)


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


def create_downsampler(build_dir, evd_params=EVDOWNSAMPLING_PARAMS):
    """Build the authors' real-time GeNN EvDownsampling model in build_dir.

    evd_params is (tau_theta in ms, tau_accumulator in ms, beta).
    """
    if str(EVDOWNSAMPLING_REPO) not in sys.path:
        sys.path.append(str(EVDOWNSAMPLING_REPO))
    from event_downsampling_genn import EventDownsampleGeNN

    tau_theta, tau_accumulator, beta = evd_params
    build_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(build_dir):
        return EventDownsampleGeNN(sensor_size=(SENSOR_SIZE, SENSOR_SIZE, 2),
                                   target_size=(SENSOR_SIZE // SPATIAL_FACTOR,) * 2,
                                   dt=BIN_MS, tau_theta=tau_theta,
                                   tau_accumulator=tau_accumulator, beta=beta)


def evdownsampling_events(downsampler, events, num_bins):
    """Run full-resolution events (times in us from 0) through the GeNN downsampler."""
    for pop, names in ((downsampler.neurons, ("V", "Theta0", "Theta1", "LastSpikePol")),
                       (downsampler.current_source, ("magnitude",))):
        for name in names:
            pop.vars[name].view[:] = 0
            pop.vars[name].push_to_device()

    offset = downsampler.model.timestep
    events = events.copy()
    events["t"] = events["t"] // (BIN_MS * 1000) + offset
    edges = np.searchsorted(events["t"], offset + np.arange(num_bins + 1))
    output = np.concatenate([
        downsampler.process_events(events[edges[b]:edges[b + 1]], iteration=offset + b)
        for b in range(num_bins)])
    output["t"] -= offset
    return output


def encode_gesture(events, encoding, trial_ms=None, downsampler=None):
    """Return (spikes, num_bins) for one complete recording.

    occupancy:       one spike per (pixel, polarity, 1 ms bin) that has at least
                     one event (ml_genn's preprocess_tonic_spikes histogram).
    evdownsampling:  the authors' GeNN EvDownsampling (needs downsampler).

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
        events["t"] -= t0
    else:
        num_bins = int(trial_ms) // BIN_MS
        duration_us = int(events["t"][-1]) - t0 + 1
        events["t"] = (events["t"] - t0) * (num_bins * bin_us) // duration_us

    if encoding == "occupancy":
        events = transforms.Downsample(spatial_factor=1 / SPATIAL_FACTOR)(events)
        time_scale, histogram = 1.0 / 1000.0, dict(dt=BIN_MS, histogram_thresh=1)
    elif encoding == "evdownsampling":
        events = evdownsampling_events(downsampler, events, num_bins)
        time_scale, histogram = 1.0, {}
    else:
        raise ValueError(f"Unknown encoding {encoding!r}")
    events["x"] += BORDER
    events["y"] += BORDER
    spikes = preprocess_tonic_spikes(events, events.dtype.names, INPUT_SHAPE,
                                     time_scale=time_scale, **histogram)
    return spikes, num_bins


def encode_counts(events, levels):
    """Return (spikes, num_bins): thermometer-coded event counts in real time.

    Events are denoised and downsampled as for occupancy. Channel p * levels + k of a
    pixel spikes in every 1 ms bin with more than k events of polarity p, so channels
    k = 0 are the occupancy encoding.
    """
    events = transforms.Denoise(filter_time=DENOISE_US)(events)
    if len(events) == 0:
        raise ValueError("Recording is empty after denoising")
    bin_us = BIN_MS * 1000
    t0 = int(events["t"][0])
    num_bins = max(1, (int(events["t"][-1]) - t0) // bin_us)
    events = events[events["t"] < t0 + num_bins * bin_us]
    events = transforms.Downsample(spatial_factor=1 / SPATIAL_FACTOR)(events)

    width, height, _ = INPUT_SHAPE
    pixel = ((events["y"].astype(np.int64) + BORDER) * width
             + events["x"].astype(np.int64) + BORDER)
    cell = ((pixel * 2 + events["p"].astype(np.int64)) * num_bins
            + (events["t"].astype(np.int64) - t0) // bin_us)
    cells, counts = np.unique(cell, return_counts=True)
    channel, bins = np.divmod(cells, num_bins)
    pixel, polarity = np.divmod(channel, 2)
    level = np.minimum(counts, levels)
    first = np.cumsum(level) - level
    ids = (np.repeat(pixel * 2 * levels + polarity * levels, level)
           + np.arange(level.sum()) - np.repeat(first, level))
    times = np.repeat(bins * BIN_MS, level).astype(float)
    return preprocess_spikes(times, ids, width * height * 2 * levels), num_bins


def count_level_spikes(spikes, levels):
    """Input spikes per count level, summed over recordings, pixels and polarities."""
    total = sum(np.diff(s.end_spikes, prepend=0) for s in spikes)
    return total.reshape(-1, levels).sum(axis=0)


def crop_events(events, crop_ms):
    """Keep the first crop_ms ms of real time, starting at the first event."""
    return events[events["t"] < events["t"][0] + crop_ms * 1000]


def encode_recordings(dataset, indices, encoding, description="Encoding", trial_ms=None,
                      cache_dir=None, crop_ms=None, evd_params=EVDOWNSAMPLING_PARAMS,
                      count_levels=COUNT_LEVELS):
    """Encode the given recordings; return spikes, labels and lengths in bins.

    crop_ms keeps only the first crop_ms ms of each recording in real time
    (instead of compressing it with trial_ms). evd_params are the EvDownsampling
    parameters, see create_downsampler; count_levels the levels of encode_counts.

    With cache_dir, the result is stored there and reused by later calls with
    the same recordings, encoding, trial_ms, crop_ms, evd_params and encoder code.
    """
    if trial_ms is not None and crop_ms is not None:
        raise ValueError("Use either trial_ms or crop_ms, not both")
    if encoding == "counts" and trial_ms is not None:
        raise ValueError("The counts encoding keeps real time; use crop_ms, not trial_ms")
    evd_params = tuple(float(p) for p in evd_params)
    if cache_dir is not None:
        key = _cache_key(dataset, indices, encoding, trial_ms, crop_ms, evd_params,
                         count_levels)
        length = f"crop{crop_ms}" if crop_ms is not None else (trial_ms or "full")
        cache_file = Path(cache_dir) / f"{encoding}_{length}_{key}.npz"
        if cache_file.exists():
            print(f"{description}: loaded from {cache_file}")
            return _load_cache(cache_file)

    downsampler = None
    if encoding == "evdownsampling":
        downsampler = create_downsampler(Path(cache_dir or ".").resolve() / "evdownsampling_genn",
                                         evd_params)

    spikes, labels, lengths = [], [], []
    for i in tqdm(indices, desc=description, unit="recording"):
        events, label = dataset[i]
        if crop_ms is not None:
            events = crop_events(events, crop_ms)
        if encoding == "counts":
            s, num_bins = encode_counts(events, count_levels)
        else:
            s, num_bins = encode_gesture(events, encoding, trial_ms, downsampler)
        spikes.append(s)
        labels.append(label)
        lengths.append(num_bins)
    labels, lengths = np.array(labels), np.array(lengths)
    if downsampler is not None:
        downsampler.model.unload()

    if cache_dir is not None:
        _save_cache(cache_file, spikes, labels, lengths)
        print(f"{description}: saved to {cache_file}")
    return spikes, labels, lengths


def _cache_key(dataset, indices, encoding, trial_ms, crop_ms, evd_params, count_levels):
    """Hash of everything that determines the encoded spikes."""
    key = hashlib.sha256()
    key.update(repr((encoding, trial_ms, DENOISE_US, SPATIAL_FACTOR, BORDER, INPUT_SHAPE,
                     [str(dataset.data[i]) for i in indices])).encode())
    key.update(inspect.getsource(encode_gesture).encode())
    if encoding == "counts":
        key.update(repr(count_levels).encode())
        key.update(inspect.getsource(encode_counts).encode())
    if encoding == "evdownsampling":
        key.update(repr(evd_params).encode())
        key.update(inspect.getsource(evdownsampling_events).encode())
        key.update((EVDOWNSAMPLING_REPO / "event_downsampling_genn.py").read_bytes())
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


def augment_spikes(spikes, rng, shift_px=0.0, zoom=(1.0, 1.0), rotate_deg=0.0):
    """Randomly shift, zoom and rotate each recording on the input grid.

    One draw per recording: shift uniform in +-shift_px pixels per axis, zoom
    factor uniform in zoom, rotation uniform in +-rotate_deg degrees about the
    centre. Each input neuron copies the spike train of the nearest source pixel
    (same channel), so no neuron fires more often than before; the border stays
    silent. Pixels pulled from outside the sensor are silent.
    """
    width, height, _ = INPUT_SHAPE
    channels = len(spikes[0].end_spikes) // (width * height)
    inner = SENSOR_SIZE // SPATIAL_FACTOR
    centre = (inner - 1) / 2
    grid_y, grid_x = np.meshgrid(np.arange(inner), np.arange(inner), indexing="ij")
    dest_pixel = ((grid_y + BORDER) * width + grid_x + BORDER).ravel()
    u_base, v_base = (grid_x - centre).ravel(), (grid_y - centre).ravel()
    augmented = []
    for s in spikes:
        dx, dy = rng.uniform(-shift_px, shift_px, 2)
        factor = rng.uniform(*zoom)
        angle = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg))
        u, v = u_base - dx, v_base - dy
        src_x = np.rint((np.cos(angle) * u + np.sin(angle) * v) / factor + centre).astype(int)
        src_y = np.rint((-np.sin(angle) * u + np.cos(angle) * v) / factor + centre).astype(int)
        valid = (src_x >= 0) & (src_x < inner) & (src_y >= 0) & (src_y < inner)
        src_pixel = (src_y[valid] + BORDER) * width + src_x[valid] + BORDER

        source = np.full(width * height * channels, -1)
        for c in range(channels):
            source[dest_pixel[valid] * channels + c] = src_pixel * channels + c
        counts = np.diff(s.end_spikes, prepend=0)
        starts = s.end_spikes - counts
        has_source = source >= 0
        source = np.where(has_source, source, 0)
        new_counts = np.where(has_source, counts[source], 0)
        new_end = np.cumsum(new_counts)
        index = (np.arange(new_end[-1]) - np.repeat(new_end - new_counts, new_counts)
                 + np.repeat(starts[source], new_counts))
        augmented.append(PreprocessedSpikes(new_end, s.spike_times[index]))
    return augmented


def augmentation_growth(zoom, rotate_deg):
    """Bound on how many input pixels can copy one source pixel."""
    return 1 if zoom[1] <= 1.0 and rotate_deg == 0.0 else 4
