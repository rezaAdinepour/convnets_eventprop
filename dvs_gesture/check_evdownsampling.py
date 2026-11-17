import argparse
from pathlib import Path
import sys

import numpy as np
from tonic import transforms

from data import (BORDER, DENOISE_US, EVDOWNSAMPLING, FIELD_SIZE, INPUT_SHAPE,
                  SPATIAL_REDUCTION, encode_trial, load_development_split)
from evdownsampling import BETA, BIN_MS, NUM_POLARITIES, TAU_ACCUMULATOR_MS, TAU_THETA_MS

REFERENCE_FRAMES = 1024
REFERENCE_BIN_MS = 1


def neuron_ids(spikes):
    counts = np.diff(np.r_[0, spikes.end_spikes.astype(np.int64)])
    return np.repeat(np.arange(len(counts)), counts)


def check(data_root, candidate_dir):
    sys.path.insert(0, str(candidate_dir.resolve(strict=True)))
    from dvs_data import Config, pack_evdownsampling

    dataset, train, _ = load_development_split(data_root)
    events, label = dataset[train[0]]
    occupancy, n, frames = encode_trial(events, whole=True)
    adaptive, m, steps = encode_trial(events, whole=True, preprocessing=EVDOWNSAMPLING)
    np.testing.assert_equal((m, steps), (n, frames))
    if frames < REFERENCE_FRAMES or BIN_MS != REFERENCE_BIN_MS:
        raise ValueError("The reference check requires a full prefix at the candidate's 1 ms bin width")
    filtered = transforms.Denoise(filter_time=DENOISE_US)(events)
    expected, info = pack_evdownsampling(
        filtered["t"] - filtered["t"][0], filtered["x"] // SPATIAL_REDUCTION,
        filtered["y"] // SPATIAL_REDUCTION, filtered["p"], width=FIELD_SIZE, height=FIELD_SIZE,
        config=Config(frames=REFERENCE_FRAMES, tau_accumulator_ms=TAU_ACCUMULATOR_MS,
                      tau_theta_ms=TAU_THETA_MS, beta=BETA))

    y, x, p = np.unravel_index(neuron_ids(adaptive), INPUT_SHAPE)
    t = adaptive.spike_times
    if not (np.all((x >= BORDER) & (x < BORDER + FIELD_SIZE) & (y >= BORDER) & (y < BORDER + FIELD_SIZE))
            and np.all((t >= 0) & (t < frames) & (t == np.floor(t)))):
        raise ValueError("Unexpected border spike or timestamp")
    prefix = t < REFERENCE_FRAMES
    ids = ((y[prefix] - BORDER) * FIELD_SIZE + x[prefix] - BORDER) * NUM_POLARITIES + p[prefix]
    np.testing.assert_array_equal(ids, neuron_ids(expected))
    np.testing.assert_array_equal(t[prefix], expected.spike_times)
    np.testing.assert_equal(np.count_nonzero(occupancy.spike_times < REFERENCE_FRAMES),
                            info["raw_binary_occupancy_spikes"])
    empty, count, length = encode_trial(events[:0], whole=True, preprocessing=EVDOWNSAMPLING)
    np.testing.assert_array_equal(empty.end_spikes, np.zeros(np.prod(INPUT_SHAPE)))
    np.testing.assert_equal((len(empty.spike_times), count, length), (0, 0, 1))
    print(f"Train parent: {dataset.data[train[0]]}, label={label}, frames={frames}")
    print(f"Whole gesture: retained={n}, occupancy={len(occupancy.spike_times)}, "
          f"adaptive={len(t)}")
    print(f"{REFERENCE_FRAMES} ms: occupancy={info['raw_binary_occupancy_spikes']}, "
          f"adaptive={len(expected.spike_times)}, cancelled={info['cancelled_opposite_polarity_events']}")
    print("EvDownsampling candidate parity: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("candidate_dir", type=Path, help="Directory containing the user's dvs_data.py")
    args = parser.parse_args()
    check(args.data_root, args.candidate_dir)
