"""Compare our encoders with the library's and the EvDownsampling authors' code.

occupancy:      encode_gesture vs ml_genn's preprocess_tonic_spikes(dt=1, histogram_thresh=1)
evdownsampling: evdownsampling.py vs EventDownsampleGeNN (real-time GeNN version) and
                event_downsample (NumPy version, with float ms and with integer us times)

Uses the first training recording of every class. Writes nothing except GeNN's
generated code, which goes into out_dir.
"""
import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from ml_genn.utils.data import preprocess_tonic_spikes
from tonic import transforms
from tonic.datasets import DVSGesture

from data import (BORDER, DENOISE_US, INPUT_SHAPE, NUM_CLASSES, SENSOR_SIZE, SPATIAL_FACTOR,
                  encode_gesture, load_dataset, split_train_validation)
from evdownsampling import BETA, TAU_ACCUMULATOR_MS, TAU_THETA_MS

TARGET_SIZE = (SENSOR_SIZE // SPATIAL_FACTOR, SENSOR_SIZE // SPATIAL_FACTOR)
SENSOR = (SENSOR_SIZE, SENSOR_SIZE, 2)


def prepare(events):
    """Denoise, cut to whole 1 ms bins and shift to the first event, as encode_gesture does."""
    events = transforms.Denoise(filter_time=DENOISE_US)(events)
    t0 = int(events["t"][0])
    num_bins = max(1, (int(events["t"][-1]) - t0) // 1000)
    events = events[events["t"] < t0 + num_bins * 1000]
    events["t"] -= t0
    return events, num_bins


def to_spikes(events, time_scale):
    events = events.copy()
    events["x"] += BORDER
    events["y"] += BORDER
    return preprocess_tonic_spikes(events, events.dtype.names, INPUT_SHAPE, time_scale=time_scale)


def library_occupancy(events):
    events = transforms.Downsample(spatial_factor=1 / SPATIAL_FACTOR)(events)
    events["x"] += BORDER
    events["y"] += BORDER
    return preprocess_tonic_spikes(events, DVSGesture.ordering, INPUT_SHAPE,
                                   dt=1.0, histogram_thresh=1)


def genn_evdownsampling(downsampler, events, num_bins):
    for pop, names in ((downsampler.neurons, ("V", "Theta0", "Theta1", "LastSpikePol")),
                       (downsampler.current_source, ("magnitude",))):
        for name in names:
            pop.vars[name].view[:] = 0
            pop.vars[name].push_to_device()

    offset = downsampler.model.timestep
    events = events.copy()
    events["t"] = events["t"] // 1000 + offset
    edges = np.searchsorted(events["t"], offset + np.arange(num_bins + 1))
    output = []
    for b in range(num_bins):
        out = downsampler.process_events(events[edges[b]:edges[b + 1]], iteration=offset + b)
        if len(out):
            output.append(out)
    output = np.concatenate(output)
    output["t"] -= offset
    return to_spikes(output, 1.0)


def numpy_evdownsampling(event_downsample, events, integer_us):
    if integer_us:
        time_scale = 1.0 / 1000.0
    else:
        dtype = [(n, np.float64 if n == "t" else events.dtype[n]) for n in events.dtype.names]
        events = events.astype(dtype)
        events["t"] /= 1000.0
        time_scale = 1.0
    try:
        output = event_downsample(events.copy(), SENSOR, TARGET_SIZE, dt=1,
                                  tau_theta=TAU_THETA_MS, tau_accumulator=TAU_ACCUMULATOR_MS,
                                  beta=BETA)
    except ValueError as error:
        print(f"  NumPy event_downsample failed ({'us' if integer_us else 'ms'}): {error}")
        return None
    return to_spikes(output, time_scale)


def describe(reference, spikes):
    if spikes is None:
        return f"{'fail':>6} {'-':>5}"
    return f"{len(spikes.spike_times):6d} {overlap(reference, spikes):5.3f}"


def spike_pairs(spikes):
    ids = np.repeat(np.arange(len(spikes.end_spikes)), np.diff(spikes.end_spikes, prepend=0))
    return set(zip(ids.tolist(), np.round(spikes.spike_times, 3).tolist()))


def overlap(a, b):
    a, b = spike_pairs(a), spike_pairs(b)
    return len(a & b) / max(1, len(a | b))


def identical(a, b):
    return (np.array_equal(a.end_spikes, b.end_spikes)
            and np.array_equal(a.spike_times, b.spike_times))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_root", type=Path, help="Directory that contains DVSGesture/")
    parser.add_argument("evdownsampling_repo", type=Path, help="Clone of anindyaghosh/EvDownsampling")
    parser.add_argument("out_dir", type=Path, help="New directory for GeNN's generated code")
    args = parser.parse_args()

    sys.path.insert(0, str(args.evdownsampling_repo.resolve()))
    from event_based_downsampling import event_downsample
    from event_downsampling_genn import EventDownsampleGeNN

    dataset = load_dataset(args.data_root.resolve(), train=True)
    train_idx, _ = split_train_validation(dataset)
    first = {}
    for i in train_idx:
        first.setdefault(dataset.targets[i], i)
    indices = [first[c] for c in range(NUM_CLASSES)]

    args.out_dir.mkdir(parents=True)
    os.chdir(args.out_dir)
    downsampler = EventDownsampleGeNN(sensor_size=SENSOR, target_size=TARGET_SIZE, dt=1,
                                      tau_theta=TAU_THETA_MS, tau_accumulator=TAU_ACCUMULATOR_MS,
                                      beta=BETA, use_gpu=True)

    print("\nOccupancy: ours vs ml_genn preprocess_tonic_spikes(dt=1, histogram_thresh=1)")
    print("EvDownsampling spikes per recording, and overlap |A and B| / |A or B| with ours\n")
    print(f"{'idx':>5} {'class':>5} {'ms':>6} | {'occ':>7} {'same':>5} | {'ours':>6} "
          f"{'genn':>6} {'ovl':>5} {'np_ms':>6} {'ovl':>5} {'np_us':>6} {'ovl':>5} | "
          f"{'s_ours':>6} {'s_genn':>6} {'s_np':>6}")
    totals = np.zeros(2)
    seconds = np.zeros(2)
    numpy_failures = 0
    all_same = True
    for i in indices:
        raw, label = dataset[i]
        events, num_bins = prepare(raw)

        occupancy, _ = encode_gesture(raw, "occupancy")
        same = identical(occupancy, library_occupancy(events))
        all_same &= same

        start = perf_counter()
        ours, _ = encode_gesture(raw, "evdownsampling")
        t_ours = perf_counter() - start
        start = perf_counter()
        genn = genn_evdownsampling(downsampler, events, num_bins)
        t_genn = perf_counter() - start
        start = perf_counter()
        np_ms = numpy_evdownsampling(event_downsample, events, integer_us=False)
        t_np = perf_counter() - start
        np_us = numpy_evdownsampling(event_downsample, events, integer_us=True)
        numpy_failures += (np_ms is None) + (np_us is None)

        totals += (len(ours.spike_times), len(genn.spike_times))
        seconds += (t_ours, t_genn)
        print(f"{i:5d} {label:5d} {num_bins:6d} | {len(occupancy.spike_times):7d} {str(same):>5} | "
              f"{len(ours.spike_times):6d} {describe(ours, genn)} "
              f"{describe(ours, np_ms)} {describe(ours, np_us)} | "
              f"{t_ours:6.1f} {t_genn:6.1f} {t_np:6.1f}")

    print(f"\nOccupancy identical for all recordings: {all_same}")
    print(f"Total EvDownsampling spikes (ours, genn): {int(totals[0])}, {int(totals[1])}")
    print(f"Total seconds (ours, genn): {seconds[0]:.1f}, {seconds[1]:.1f}")
    print(f"NumPy event_downsample failures: {numpy_failures} of {2 * len(indices)} calls")


if __name__ == "__main__":
    main()
