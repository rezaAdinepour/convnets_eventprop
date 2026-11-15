"""Adaptive spatial downsampling for millisecond-binned DVS Gesture events.

Adapted from New_Approach_for_EvProp_CSNN/dvs_data.py (pack_evdownsampling)
and Ghosh, Nowotny and Knight, EvDownsampling, Eq. (1).

MIT License
Copyright (c) 2024 anindyaghosh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from ml_genn.utils.data import preprocess_spikes


BIN_MS = 1
MICROSECONDS_PER_MS = 1000
BIN_US = BIN_MS * MICROSECONDS_PER_MS
TAU_ACCUMULATOR_MS = 16.0
TAU_THETA_MS = 420.0
BETA = 1.2
NUM_POLARITIES = 2
PARAMETERS = dict(bin_ms=BIN_MS, tau_accumulator_ms=TAU_ACCUMULATOR_MS,
                  tau_theta_ms=TAU_THETA_MS, beta=BETA)


def adaptive_spikes(events, frames, shape):
    """Encode one parent after shared denoising, spatial mapping and binning."""
    height, width, channels = shape
    pixels = height * width
    bins = events["t"] // BIN_US
    if channels != NUM_POLARITIES or np.any((bins < 0) | (bins >= frames)):
        raise ValueError("Expected two polarities and in-window millisecond bins")
    p = events["p"]
    if np.any((p != 0) & (p != 1)):
        raise ValueError("EvDownsampling requires binary polarity")
    ids = (events["y"].astype(np.int64) * width + events["x"]) * channels + p
    edges = np.searchsorted(bins, np.arange(frames + 1))
    v = np.zeros(pixels, dtype=np.float64)
    theta = np.zeros((channels, pixels), dtype=np.float64)
    a = np.exp(-PARAMETERS["bin_ms"] / PARAMETERS["tau_accumulator_ms"])
    rho = np.exp(-PARAMETERS["bin_ms"] / PARAMETERS["tau_theta_ms"])
    spike_ids, times = [], []
    for k, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
        counts = np.bincount(ids[start:stop], minlength=channels * pixels).reshape(pixels, channels)
        v *= 1.0 - a
        v += a * (counts[:, 1] - counts[:, 0])
        positive, negative = v > theta[1], -v > theta[0]
        theta *= rho
        for channel, mask in ((0, negative), (1, positive)):
            theta[channel, mask] += PARAMETERS["beta"] * np.abs(v[mask]) * (1.0 - rho)
        fired = np.flatnonzero(negative | positive)
        if fired.size:
            spike_ids.append(channels * fired + positive[fired])
            times.append(np.full(fired.size, k * BIN_MS, dtype=np.float32))
            v[fired] = 0.0

    ids_out = np.concatenate(spike_ids) if spike_ids else np.empty(0, dtype=np.int64)
    times_out = np.concatenate(times) if times else np.empty(0, dtype=np.float32)
    return preprocess_spikes(times_out, ids_out, pixels * channels)
