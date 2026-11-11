"""CPU-only continuous LIF trajectory for one frozen DVS hidden neuron.

Reads development input and saved JSON; no GeNN imports or filesystem writes.
This is a forward reference, not an EventProp gradient certification.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import dvs_data as data
import dvs_eventprop_lif_reference as single


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def emit(name, value):
    print(name, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def advance(v, current, elapsed):
    """Exact flow of dv/dt=(-v+I)/20, dI/dt=-I/5."""
    em, es = np.exp(-elapsed / 20.0), np.exp(-elapsed / 5.0)
    return float(v * em + current / 3.0 * (em - es)), float(current * es)


def first_crossing(v, current, duration, iterations):
    # Voltage is A*exp(-t/20)+B*exp(-t/5). Its derivative has at
    # most one interior zero; partition there to bracket the first root.
    require(v < 1.0, "Interval starts at or above threshold")
    a, b = v + current / 3.0, -current / 3.0
    boundaries = [0.0, duration]
    if a != 0.0 and b != 0.0 and -4.0 * b / a > 0.0:
        extremum = float(np.log(-4.0 * b / a) / 0.15)
        if 0.0 < extremum < duration:
            boundaries.insert(1, extremum)
    for low, high in zip(boundaries[:-1], boundaries[1:]):
        if advance(v, current, high)[0] < 1.0:
            continue
        require(advance(v, current, low)[0] < 1.0,
                "Threshold crossing was not the first root")
        for _ in range(iterations):
            middle = (low + high) / 2.0
            if advance(v, current, middle)[0] < 1.0:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0
    return None


def trajectory(jumps, horizon, iterations=80):
    v, current, now = 0.0, 0.0, 0.0
    spikes = []
    max_residual = 0.0
    # Input impulses change current, not voltage. Between impulses the
    # solution is analytic; absolute voltage resets leave current intact.
    for event_time in sorted(set(jumps) | {float(horizon)}):
        require(now <= event_time <= horizon, "Invalid input event order")
        while now < event_time:
            root = first_crossing(v, current, event_time - now, iterations)
            if root is None:
                v, current = advance(v, current, event_time - now)
                now = event_time
                break
            require(root > 1e-12, "Root too close to previous reset")
            v, current = advance(v, current, root)
            now += root
            residual = abs(v - 1.0)
            max_residual = max(max_residual, residual)
            require(residual < 1e-10 and current > 1.0,
                    "Non-transverse or inaccurate threshold root")
            if now < horizon:
                spikes.append({
                    "time_ms": now, "current_at_crossing": current,
                    "crossing_slope_per_ms": (current - 1.0) / 20.0,
                })
            require(len(spikes) < 10000, "Unexpected spike count")
            v = 0.0
        current += jumps.get(event_time, 0.0)
        require(np.isfinite(v) and np.isfinite(current), "Nonfinite state")
    return spikes, max_residual


def self_check():
    # Existing analytical single-impulse reference supplies an independent
    # bracket and peak-based single-spike guarantee.
    expected = single.reference(7.0)
    observed, _ = trajectory({16.0: 7.0}, 128.0)
    require(len(observed) == 1, "Single-impulse spike count mismatch")
    np.testing.assert_allclose(
        observed[0]["time_ms"], expected["hidden_spike_time_ms"],
        rtol=0, atol=1e-10,
    )
    for v, current in ((0.2, 3.0), (-0.4, -2.0), (0.8, -1.0)):
        middle = advance(v, current, 0.4)
        np.testing.assert_allclose(
            advance(*middle, 0.7), advance(v, current, 1.1),
            rtol=1e-13, atol=1e-14,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neuron", type=int, default=298)
    parser.add_argument("artifacts", type=Path, nargs=2)
    args = parser.parse_args()
    require(0 <= args.neuron < 392, "Hidden neuron ID out of range")
    self_check()
    paths = [p.resolve(strict=True) for p in args.artifacts]
    results = [json.loads(p.read_text()) for p in paths]
    base = results[0]
    for result in results:
        condition = result["condition"]
        for key, expected in {
            "tau_mem_ms": 20.0, "tau_syn_ms": 5.0, "threshold": 1.0,
            "absolute_reset_voltage": 0.0, "input_hwp": [32, 32, 2],
            "hidden_hwf": [7, 7, 8], "kernel_hwio": [5, 5, 2, 8],
            "conv_stride": [4, 4], "conv_padding": "valid",
            "frames": 1024, "physical_horizon_ms": 1024.0,
            "scalar_precision": "double",
        }.items():
            require(condition[key] == expected, f"Unsupported condition: {key}")
        require(result["optimizer_updates"] == 0
                and result["weights_and_moments_unchanged"] is True,
                "Artifact was not frozen")
        require(result["sample_metadata"] == base["sample_metadata"]
                and result["initial_weights"] == base["initial_weights"],
                "Artifacts do not share input and initial weights")
        require(result["identity"]["source_sha256"]["dvs_data.py"]
                == data.file_hash(Path(data.__file__)), "Loader source changed")

    root = args.root.resolve(strict=True)
    rows = data.inventory(root)["train"]
    matches = [row for row in rows
               if row.stable_id == base["sample_metadata"]["stable_id"]]
    require(len(matches) == 1, "Training parent was not uniquely identified")
    payload, _, metadata = data.load_sample(root, matches[0], data.Config(1024))
    require(json.loads(json.dumps(metadata)) == base["sample_metadata"],
            "Reconstructed input metadata differ from saved artifact")
    kernel = np.asarray(base["initial_weights"][0], dtype=np.float64)
    require(kernel.shape == (400,) and np.isfinite(kernel).all(),
            "Invalid frozen kernel")
    require(hashlib.sha256(kernel.tobytes()).hexdigest()
            == base["initial_weight_sha256"][0], "Frozen kernel hash mismatch")
    kernel = kernel.reshape(5, 5, 2, 8)
    oy, ox, feature = np.unravel_index(args.neuron, (7, 7, 8))
    ends = payload.end_spikes.astype(np.int64)
    starts = np.r_[0, ends[:-1]]
    jumps = {}
    incoming_spikes = 0
    for ky in range(5):
        for kx in range(5):
            for polarity in range(2):
                pre = (((int(oy) * 4 + ky) * 32
                        + int(ox) * 4 + kx) * 2 + polarity)
                weight = float(kernel[ky, kx, polarity, feature])
                for time in payload.spike_times[starts[pre]:ends[pre]]:
                    time = float(time)
                    jumps[time] = jumps.get(time, 0.0) + weight
                    incoming_spikes += 1
    continuous, residual = trajectory(jumps, 1024.0)
    repeated, _ = trajectory(jumps, 1024.0, iterations=60)
    times = np.array([row["time_ms"] for row in continuous])
    np.testing.assert_allclose(
        times, [row["time_ms"] for row in repeated], rtol=0, atol=1e-9
    )
    emit("CONTINUOUS_NEURON_REFERENCE", {
        "neuron": args.neuron, "output_yxf": [int(oy), int(ox), int(feature)],
        "input_origin_yx": [int(oy) * 4, int(ox) * 4],
        "incoming_spikes": incoming_spikes, "hidden_spikes": len(continuous),
        "spike_times_ms": times.tolist(), "max_threshold_residual": residual,
        "crossings_680_to_705ms": [r for r in continuous
                                    if 680 <= r["time_ms"] <= 705],
        "minimum_crossing_slope_per_ms": (
            min(r["crossing_slope_per_ms"] for r in continuous)
            if continuous else None
        ),
        "input_transmission_delay_ms": 0.0,
    })
    for path, result in zip(paths, results):
        recorded = sorted(float(t) for n, t in result["hidden_spike_pairs"]
                          if n == args.neuron)
        equal_counts = len(recorded) == len(times)
        emit("CONTINUOUS_VS_GRID_TIMING", {
            "artifact": str(path), "dt_ms": result["condition"]["dt_ms"],
            "recorded_spikes": len(recorded), "reference_spikes": len(times),
            "per_neuron_count_matches": equal_counts,
            "max_ordinal_time_difference_ms": (
                float(np.max(np.abs(np.asarray(recorded) - times)))
                if equal_counts and len(times) else None
            ),
            "recorded_times_680_to_705ms": [t for t in recorded if 680 <= t <= 705],
        })
    emit("PASS_NEURON_REFERENCE_SELF_CHECKS", {
        "single_impulse_reference": "pass", "flow_composition": "pass",
        "root_iteration_repeatability": "pass",
        "artifact_sha256": [data.file_hash(p) for p in paths],
        "diagnostic_sha256": data.file_hash(Path(__file__)),
        "loader_sha256": data.file_hash(Path(data.__file__)),
        "single_reference_sha256": data.file_hash(Path(single.__file__)),
        "genn_simulation_performed": False,
        "eventprop_gradient_correctness_certified": False,
        "scope": "continuous forward trajectory of one frozen hidden neuron",
    })


if __name__ == "__main__":
    main()
