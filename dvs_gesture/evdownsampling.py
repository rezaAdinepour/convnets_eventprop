"""EvDownsampling input encoder (Ghosh, Nowotny and Knight, Eq. 1).

Each downsampled pixel has a leaky accumulator of signed event counts and one
adaptive threshold per polarity. When the accumulator crosses the threshold of
either polarity the pixel emits one spike on that polarity channel and the
accumulator is reset. Adapted from
EvDownsampling/New_Approach_for_EvProp_CSNN/dvs_data.py.
"""

import numpy as np
from ml_genn.utils.data import preprocess_spikes

BIN_MS = 1
TAU_ACCUMULATOR_MS = 16.0
TAU_THETA_MS = 420.0
BETA = 1.2


def evdownsampling_spikes(events, num_bins, shape):
    """Encode events whose times are already shifted to start at zero (in us).

    ``shape`` is (height, width, 2). Spike IDs use the same (y, x, p) layout
    as ``preprocess_tonic_spikes`` so both encoders feed the same input layer.
    """
    height, width, num_polarities = shape
    num_pixels = height * width
    if num_polarities != 2 or not np.isin(events["p"], (0, 1)).all():
        raise ValueError("EvDownsampling needs two polarities with values 0 and 1")

    bins = events["t"] // (BIN_MS * 1000)
    ids = (events["y"].astype(np.int64) * width + events["x"]) * 2 + events["p"]
    bin_edges = np.searchsorted(bins, np.arange(num_bins + 1))

    alpha = np.exp(-BIN_MS / TAU_ACCUMULATOR_MS)
    rho = np.exp(-BIN_MS / TAU_THETA_MS)
    accumulator = np.zeros(num_pixels)
    threshold = np.zeros((2, num_pixels))  # [OFF, ON]

    spike_ids, spike_times = [], []
    for b in range(num_bins):
        counts = np.bincount(ids[bin_edges[b]:bin_edges[b + 1]],
                             minlength=2 * num_pixels).reshape(num_pixels, 2)
        accumulator = (1.0 - alpha) * accumulator + alpha * (counts[:, 1] - counts[:, 0])
        on = accumulator > threshold[1]
        off = -accumulator > threshold[0]

        threshold *= rho
        threshold[0, off] += BETA * np.abs(accumulator[off]) * (1.0 - rho)
        threshold[1, on] += BETA * np.abs(accumulator[on]) * (1.0 - rho)

        fired = np.flatnonzero(on | off)
        if fired.size:
            spike_ids.append(2 * fired + on[fired])
            spike_times.append(np.full(fired.size, b * BIN_MS, dtype=np.float32))
            accumulator[fired] = 0.0

    if not spike_ids:
        return preprocess_spikes(np.empty(0, np.float32), np.empty(0, np.int64), 2 * num_pixels)
    return preprocess_spikes(np.concatenate(spike_times), np.concatenate(spike_ids), 2 * num_pixels)
