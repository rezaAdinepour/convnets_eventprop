"""TASK 10: continuous-time reference for a single-spike hidden LIF.

CPU mathematics only. No mlGeNN imports, datasets or filesystem writes.
"""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HORIZON_MS = 128.0
INPUT_TIME_MS = 16.0
TAU_MEM_MS = 20.0
TAU_SYN_MS = 5.0
INPUT_WEIGHT = 7.0

OUTPUT_WEIGHTS = np.array(
    [.2, -.1, .3, -.2, .1, 0., -.3, .15, -.15, .25, -.25],
    dtype=np.float32,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(tag, value):
    print(
        tag,
        json.dumps(
            value, sort_keys=True, allow_nan=False,
            default=lambda item: item.tolist(),
        ),
        flush=True,
    )


def impulse_response(elapsed):
    """Voltage caused by a unit current jump, with initial voltage zero."""
    return TAU_SYN_MS / (TAU_MEM_MS - TAU_SYN_MS) * (
        np.exp(-elapsed / TAU_MEM_MS)
        - np.exp(-elapsed / TAU_SYN_MS)
    )


def impulse_response_derivative(elapsed):
    return TAU_SYN_MS / (TAU_MEM_MS - TAU_SYN_MS) * (
        -np.exp(-elapsed / TAU_MEM_MS) / TAU_MEM_MS
        + np.exp(-elapsed / TAU_SYN_MS) / TAU_SYN_MS
    )


def readout_response(spike_time):
    """Continuous weighted LI score per unit output weight, and d/dtime."""
    remaining = HORIZON_MS - spike_time
    require(0.0 < spike_time < HORIZON_MS,
            "Spike outside reference horizon")
    rate_mem = 1.0 / HORIZON_MS + 1.0 / TAU_MEM_MS
    rate_syn = 1.0 / HORIZON_MS + 1.0 / TAU_SYN_MS
    integral = (
        -np.expm1(-rate_mem * remaining) / rate_mem
        + np.expm1(-rate_syn * remaining) / rate_syn
    )
    integral_dremaining = (
        np.exp(-rate_mem * remaining)
        - np.exp(-rate_syn * remaining)
    )
    scale = (
        TAU_SYN_MS / (TAU_MEM_MS - TAU_SYN_MS)
        * np.exp(-spike_time / HORIZON_MS) / HORIZON_MS
    )
    coefficient = scale * integral
    derivative = scale * (
        -integral / HORIZON_MS - integral_dremaining
    )
    return float(coefficient), float(derivative)


def reference(weight, output_weights=OUTPUT_WEIGHTS):
    """Exact single-crossing solution and continuous CE derivatives."""
    require(np.isfinite(weight) and weight > 0.0,
            "Invalid input weight")
    output_weights = np.asarray(output_weights, dtype=np.float64)
    require(output_weights.shape == (11,), "Expected eleven output weights")
    require(np.isfinite(output_weights).all(), "Nonfinite output weights")

    peak_time = np.log(TAU_MEM_MS / TAU_SYN_MS) / (
        1.0 / TAU_SYN_MS - 1.0 / TAU_MEM_MS
    )
    first_peak = weight * impulse_response(peak_time)
    require(first_peak > 1.0, "No transverse rising threshold crossing")

    # Unique rising crossing: voltage rises monotonically before peak_time.
    low, high = 0.0, peak_time
    for _ in range(80):
        middle = (low + high) / 2.0
        if weight * impulse_response(middle) < 1.0:
            low = middle
        else:
            high = middle
    elapsed = (low + high) / 2.0
    spike_time = INPUT_TIME_MS + elapsed
    residual = weight * impulse_response(elapsed) - 1.0
    require(abs(residual) < 1e-12, "Threshold root residual too large")

    slope = weight * impulse_response_derivative(elapsed)
    require(slope > 0.0, "Crossing is not transverse and rising")

    # Following absolute reset, the residual exponential current is the
    # only remaining drive. Its maximum possible voltage must stay below 1.
    residual_current = weight * np.exp(-elapsed / TAU_SYN_MS)
    second_peak_bound = residual_current * impulse_response(peak_time)
    require(second_peak_bound < 1.0,
            "Reference cannot guarantee exactly one hidden spike")

    # Differentiate weight*h(elapsed)=1 implicitly.
    spike_time_dweight = -impulse_response(elapsed) / slope
    coefficient, coefficient_dtime = readout_response(spike_time)
    scores = coefficient * output_weights
    maximum = scores.max()
    probabilities = np.exp(scores - maximum)
    probabilities /= probabilities.sum()
    loss = maximum + np.log(np.exp(scores - maximum).sum()) - scores[0]

    ce_score_gradient = probabilities.copy()
    ce_score_gradient[0] -= 1.0
    hidden_gradient = (
        np.dot(ce_score_gradient, output_weights)
        * coefficient_dtime * spike_time_dweight
    )

    return {
        "hidden_spikes": 1,
        "hidden_spike_time_ms": float(spike_time),
        "threshold_residual": float(residual),
        "crossing_slope_per_ms": float(slope),
        "first_unreset_peak_voltage": float(first_peak),
        "post_reset_peak_voltage_upper_bound": float(second_peak_bound),
        "spike_time_dweight": float(spike_time_dweight),
        "readout_coefficient": coefficient,
        "readout_coefficient_dtime": coefficient_dtime,
        "scores": scores,
        "ce": float(loss),
        "hidden_weight_gradient": float(hidden_gradient),
        "output_weight_gradient": coefficient * ce_score_gradient,
    }


def main():
    emit("REFERENCE_CONFIG", {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "architecture": "SpikeInput(1)->Dense->LIF(1)->Dense->LI(11)",
        "horizon_ms": HORIZON_MS,
        "input_time_ms": INPUT_TIME_MS,
        "input_weight": INPUT_WEIGHT,
        "output_weights": OUTPUT_WEIGHTS.tolist(),
        "tau_mem_ms": TAU_MEM_MS,
        "tau_syn_ms": TAU_SYN_MS,
        "threshold": 1.0,
        "absolute_reset": 0.0,
        "transmission_delay_ms": 0.0,
        "label": 0,
        "scope": "continuous-time analytical reference only",
    })
    base = reference(INPUT_WEIGHT)
    emit("CONTINUOUS_LIF_REFERENCE", base)

    for epsilon in (0.01, 0.003, 0.001):
        plus = reference(INPUT_WEIGHT + epsilon)
        minus = reference(INPUT_WEIGHT - epsilon)
        spike_fd = (
            plus["hidden_spike_time_ms"] - minus["hidden_spike_time_ms"]
        ) / (2.0 * epsilon)
        ce_fd = (plus["ce"] - minus["ce"]) / (2.0 * epsilon)

        time = base["hidden_spike_time_ms"]
        coefficient_fd = (
            readout_response(time + epsilon)[0]
            - readout_response(time - epsilon)[0]
        ) / (2.0 * epsilon)

        comparisons = (
            (spike_fd, base["spike_time_dweight"]),
            (coefficient_fd, base["readout_coefficient_dtime"]),
            (ce_fd, base["hidden_weight_gradient"]),
        )
        emit("REFERENCE_FINITE_DIFFERENCE", {
            "epsilon": epsilon,
            "spike_time_derivative_fd": spike_fd,
            "spike_time_derivative_analytic": base["spike_time_dweight"],
            "readout_time_derivative_fd": coefficient_fd,
            "readout_time_derivative_analytic":
                base["readout_coefficient_dtime"],
            "hidden_ce_gradient_fd": ce_fd,
            "hidden_ce_gradient_analytic": base["hidden_weight_gradient"],
            "perturbed_hidden_spike_counts": [
                minus["hidden_spikes"], plus["hidden_spikes"]
            ],
        })
        for observed, predicted in comparisons:
            np.testing.assert_allclose(
                observed, predicted, rtol=5e-4, atol=1e-10,
                err_msg="Continuous reference derivative self-check failed",
            )

    emit("PASS_CONTINUOUS_LIF_REFERENCE", {
        "hidden_spikes": 1,
        "perturbation_sizes_checked": 3,
        "spike_time_derivative": "pass",
        "readout_time_derivative": "pass",
        "hidden_ce_derivative": "pass",
        "genn_simulation_performed": False,
        "eventprop_hidden_gradient_certified": False,
    })


if __name__ == "__main__":
    main()