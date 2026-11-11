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


def readout_response(time, horizon):
    """Continuous weighted LI readout per unit outgoing weight, and d/dtime."""
    remaining = horizon - time
    rm, rs = 1.0 / horizon + 1.0 / 20.0, 1.0 / horizon + 1.0 / 5.0
    integral = -np.expm1(-rm * remaining) / rm + np.expm1(-rs * remaining) / rs
    derivative_remaining = np.exp(-rm * remaining) - np.exp(-rs * remaining)
    scale = np.exp(-time / horizon) / (3.0 * horizon)
    return (float(scale * integral),
            float(scale * (-integral / horizon - derivative_remaining)))


def sensitive_trajectory(jumps, jump_derivatives, horizon):
    """Fixed-time state sensitivities with exact threshold/reset correction."""
    v, current, now = 0.0, 0.0, 0.0
    dimension = len(next(iter(jump_derivatives.values())))
    zv, zi = np.zeros(dimension), np.zeros(dimension)
    times, derivatives = [], []
    for event_time in sorted(set(jumps) | {float(horizon)}):
        while now < event_time:
            root = first_crossing(v, current, event_time - now, 80)
            elapsed = event_time - now if root is None else root
            require(root is None or root > 1e-12, "Degenerate sensitivity root")
            em, es = np.exp(-elapsed / 20.0), np.exp(-elapsed / 5.0)
            zv = zv * em + zi / 3.0 * (em - es)
            zi = zi * es
            v, current = advance(v, current, elapsed)
            now = event_time if root is None else now + elapsed
            if root is not None:
                require(abs(v - 1.0) < 1e-10 and current > 1.0,
                        "Invalid sensitivity threshold crossing")
                # At threshold: zv + f_minus * dt_spike/dw = 0.
                dt_dw = -zv / ((current - 1.0) / 20.0)
                if now < horizon:
                    times.append(now)
                    derivatives.append(dt_dw.copy())
                # Absolute reset has zero voltage Jacobian. Returning to
                # fixed-time sensitivities gives zv_plus=-f_plus*dt_dw.
                # Current is continuous through reset, so zi is unchanged.
                zv = -(current / 20.0) * dt_dw
                v = 0.0
                require(len(times) < 10000, "Unexpected sensitivity spike count")
        current += jumps.get(event_time, 0.0)
        if event_time in jump_derivatives:
            zi += jump_derivatives[event_time]
        require(np.isfinite(zv).all() and np.isfinite(zi).all(),
                "Nonfinite state sensitivity")
    return np.asarray(times), np.asarray(derivatives).reshape(-1, dimension)


def check_derivatives(jumps, jump_derivatives, horizon, expected_times):
    # Validate the reset-aware derivative against the earlier independently
    # derived single-impulse spike-time and readout derivatives first.
    test_times, test_derivative = sensitive_trajectory(
        {16.0: 7.0}, {16.0: np.ones(1)}, 128.0
    )
    single_exact = single.reference(7.0)
    np.testing.assert_allclose(test_derivative[0, 0],
                               single_exact["spike_time_dweight"], rtol=1e-12)
    np.testing.assert_allclose(
        readout_response(test_times[0], 128.0),
        single.readout_response(test_times[0]), rtol=1e-12, atol=1e-15,
    )
    times, time_derivatives = sensitive_trajectory(jumps, jump_derivatives, horizon)
    np.testing.assert_allclose(times, expected_times, rtol=0, atol=1e-9)
    response = np.array([readout_response(t, horizon) for t in times])
    require(len(times) > 0, "Derivative check requires spikes")
    coefficient = float(response[:, 0].sum())
    coefficient_gradient = response[:, 1] @ time_derivatives
    emit("CONTINUOUS_READOUT_SENSITIVITY", {
        "coefficient_per_unit_outgoing_weight": coefficient,
        "local_kernel_axes": ["kernel_y", "kernel_x", "polarity"],
        "gradient_shape": [5, 5, 2],
        "coefficient_gradient": coefficient_gradient.tolist(),
        "objective": "this neuron's additive weighted-LI readout coefficient",
        "cross_entropy_gradient": False,
    })
    directions = {}
    for slot in (28, 29):
        direction = np.zeros(50)
        direction[slot] = 1.0
        directions[f"kernel_2_4_{slot - 28}"] = direction
    directions["coupled_polarities"] = (
        directions["kernel_2_4_0"] + directions["kernel_2_4_1"]
    ) / np.sqrt(2.0)
    all_local = []
    for name, direction in directions.items():
        analytic_times = time_derivatives @ direction
        analytic_coefficient = float(coefficient_gradient @ direction)
        passed = []
        for epsilon in (1e-5, 3e-6, 1e-6, 3e-7):
            perturbed = []
            for sign in (1.0, -1.0):
                perturbed_jumps = {
                    t: value + sign * epsilon * float(jump_derivatives[t] @ direction)
                    for t, value in jumps.items()
                }
                spikes, _ = trajectory(perturbed_jumps, horizon)
                perturbed.append(np.array([s["time_ms"] for s in spikes]))
            stable_counts = all(len(t) == len(times) for t in perturbed)
            fd_coefficient = (
                sum(readout_response(t, horizon)[0] for t in perturbed[0])
                - sum(readout_response(t, horizon)[0] for t in perturbed[1])
            ) / (2.0 * epsilon)
            time_error = None
            time_pass = False
            if stable_counts:
                fd_times = (perturbed[0] - perturbed[1]) / (2.0 * epsilon)
                time_error = float(np.max(np.abs(fd_times - analytic_times)))
                time_pass = bool(np.allclose(
                    fd_times, analytic_times, rtol=5e-4, atol=1e-6
                ))
            coefficient_pass = bool(np.isclose(
                fd_coefficient, analytic_coefficient, rtol=5e-4, atol=1e-9
            ))
            passed.append(stable_counts and time_pass and coefficient_pass)
            emit("NEURON_REFERENCE_FINITE_DIFFERENCE", {
                "direction": name, "epsilon": epsilon,
                "perturbed_spike_counts": [len(t) for t in perturbed],
                "coefficient_derivative_analytic": analytic_coefficient,
                "coefficient_derivative_fd": float(fd_coefficient),
                "maximum_spike_time_derivative_absolute_error": time_error,
                "all_spike_time_derivatives_within_tolerance": time_pass,
                "coefficient_derivative_within_tolerance": coefficient_pass,
                "pass": passed[-1],
            })
        all_local.append(all(passed[-2:]))
    require(all(all_local), "Derivative reference failed at the two smallest perturbations")
    emit("PASS_NEURON_CONTINUOUS_DERIVATIVES", {
        "directions_checked": len(directions), "perturbations_per_direction": 4,
        "two_smallest_perturbations_passed_each_direction": True,
        "single_impulse_derivative_check": "pass",
        "scope": "one neuron; spike-time and additive readout derivatives",
        "eventprop_gradient_correctness_certified": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--neuron", type=int, default=298)
    parser.add_argument("--check-derivatives", action="store_true")
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
    jump_derivatives = {}
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
                    if time not in jump_derivatives:
                        jump_derivatives[time] = np.zeros(50)
                    jump_derivatives[time][(ky * 5 + kx) * 2 + polarity] += 1.0
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
    if args.check_derivatives:
        check_derivatives(jumps, jump_derivatives, 1024.0, times)


if __name__ == "__main__":
    main()
