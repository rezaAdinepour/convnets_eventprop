"""Continuous reference for a tiny shared convolution -> LIF -> LI model.

CPU mathematics only. No GeNN imports, simulation or dataset access.
"""

import hashlib
from pathlib import Path

import numpy as np

import dvs_eventprop_lif_reference as single

emit = single.emit
require = single.require

KERNEL_SHAPE = (5, 5, 1, 1)
ACTIVE_INPUTS = (0, 4)
OUTPUT_WEIGHTS = np.tile(
    single.OUTPUT_WEIGHTS.astype(np.float64) / 2.0, (2, 1)
)


def explicit_dense(kernel):
    """Independent HWC map: input (5,9,1), stride 4, valid, two outputs."""
    dense = np.zeros((45, 2), dtype=np.float64)
    mapping = []
    for output_x in range(2):
        for ky in range(5):
            for kx in range(5):
                slot = ky * 5 + kx
                pre = ky * 9 + output_x * 4 + kx
                dense[pre, output_x] = kernel[slot]
                mapping.append((slot, pre, output_x))
    require(len(mapping) == 50, "Incorrect expanded edge count")
    require(len(set((pre, post) for _, pre, post in mapping)) == 50,
            "Duplicate expanded edge")
    np.testing.assert_array_equal(
        np.bincount([slot for slot, _, _ in mapping], minlength=25),
        np.full(25, 2),
    )
    return dense, mapping


def reference(kernel):
    kernel = np.asarray(kernel, dtype=np.float64).reshape(25)
    require(np.isfinite(kernel).all(), "Nonfinite kernel")

    # Input IDs 0 and 4 fire simultaneously at 16 ms.
    # The windows start at x=0 and x=4.
    effective_weights = np.array([
        kernel[0] + kernel[4],
        kernel[0],
    ])
    dense, mapping = explicit_dense(kernel)
    np.testing.assert_allclose(
        dense[list(ACTIVE_INPUTS)].sum(axis=0),
        effective_weights, rtol=0, atol=1e-14,
    )

    neurons = [
        single.reference(float(weight), OUTPUT_WEIGHTS[index])
        for index, weight in enumerate(effective_weights)
    ]
    scores = sum(
        neuron["readout_coefficient"] * OUTPUT_WEIGHTS[index]
        for index, neuron in enumerate(neurons)
    )
    maximum = float(scores.max())
    probabilities = np.exp(scores - maximum)
    probabilities /= probabilities.sum()
    score_gradient = probabilities.copy()
    score_gradient[0] -= 1.0
    ce = maximum + np.log(np.exp(scores - maximum).sum()) - scores[0]

    drive_gradients = np.array([
        np.dot(score_gradient, OUTPUT_WEIGHTS[index])
        * neuron["readout_coefficient_dtime"]
        * neuron["spike_time_dweight"]
        for index, neuron in enumerate(neurons)
    ])

    # Direct differentiation of [k0+k4, k0].
    kernel_gradient = np.zeros(25, dtype=np.float64)
    kernel_gradient[0] = drive_gradients.sum()
    kernel_gradient[4] = drive_gradients[0]

    # Independent chain-rule accumulation through the explicit edge map.
    summed_untied = np.zeros(25, dtype=np.float64)
    for slot, pre, post in mapping:
        if pre in ACTIVE_INPUTS:
            summed_untied[slot] += drive_gradients[post]
    np.testing.assert_allclose(
        kernel_gradient, summed_untied, rtol=1e-13, atol=1e-15
    )

    return {
        "ce": float(ce),
        "scores": scores,
        "effective_input_weights": effective_weights,
        "hidden_spikes": [n["hidden_spikes"] for n in neurons],
        "hidden_spike_times_ms": [n["hidden_spike_time_ms"] for n in neurons],
        "crossing_slopes_per_ms":
            [n["crossing_slope_per_ms"] for n in neurons],
        "post_reset_peak_bounds":
            [n["post_reset_peak_voltage_upper_bound"] for n in neurons],
        "kernel_gradient": kernel_gradient,
        "summed_untied_gradient": summed_untied,
        "output_weight_gradient": np.array([
            n["readout_coefficient"] * score_gradient for n in neurons
        ]),
    }


def main():
    kernel = np.zeros(25, dtype=np.float64)
    kernel[0] = single.INPUT_WEIGHT
    base = reference(kernel)
    original = single.reference(single.INPUT_WEIGHT)

    np.testing.assert_allclose(
        base["scores"], original["scores"], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        base["kernel_gradient"][[0, 4]],
        [
            original["hidden_weight_gradient"],
            original["hidden_weight_gradient"] / 2.0,
        ],
        rtol=1e-12, atol=1e-15,
    )
    require(base["hidden_spikes"] == [1, 1], "Expected one spike per neuron")

    emit("SHARED_LIF_REFERENCE_CONFIG", {
        "diagnostic_sha256":
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "single_lif_reference_sha256":
            hashlib.sha256(Path(single.__file__).read_bytes()).hexdigest(),
        "input_hwc": [5, 9, 1],
        "hidden_hwc": [1, 2, 1],
        "kernel_hwio": list(KERNEL_SHAPE),
        "kernel": kernel.reshape(KERNEL_SHAPE),
        "stride": [4, 4],
        "padding": "valid",
        "scheduled_spikes": [[0, 16.0], [4, 16.0]],
        "output_weights": OUTPUT_WEIGHTS,
        "horizon_ms": 128.0,
        "tau_mem_ms": 20.0,
        "tau_syn_ms": 5.0,
        "threshold": 1.0,
        "absolute_reset": 0.0,
        "readout": "avg_var_exp_weight",
        "label": 0,
        "scope": "continuous mathematics; no timestep discretization",
    })
    emit("CONTINUOUS_SHARED_LIF_REFERENCE", base)

    directions = {}
    for slot in (0, 4):
        direction = np.zeros(25, dtype=np.float64)
        direction[slot] = 1.0
        directions[f"kernel_{slot}"] = direction
    directions["coupled_0_4"] = (
        directions["kernel_0"] + directions["kernel_4"]
    ) / np.sqrt(2.0)

    for name, direction in directions.items():
        analytic = float(np.dot(base["kernel_gradient"], direction))
        for epsilon in (0.01, 0.003, 0.001):
            plus = reference(kernel + epsilon * direction)
            minus = reference(kernel - epsilon * direction)
            require(
                plus["hidden_spikes"] == minus["hidden_spikes"] == [1, 1],
                "Perturbation changed the reference spike-count regime",
            )
            difference = (plus["ce"] - minus["ce"]) / (2.0 * epsilon)
            emit("SHARED_LIF_REFERENCE_FD", {
                "direction": name,
                "epsilon": epsilon,
                "analytic": analytic,
                "finite_difference": difference,
                "relative_error":
                    abs(difference - analytic) / max(abs(analytic), 1e-12),
                "plus_hidden_spike_times_ms": plus["hidden_spike_times_ms"],
                "minus_hidden_spike_times_ms": minus["hidden_spike_times_ms"],
            })
            np.testing.assert_allclose(
                difference, analytic, rtol=2e-4, atol=1e-9
            )

    emit("PASS_CONTINUOUS_SHARED_LIF_REFERENCE", {
        "hidden_spikes": [1, 1],
        "shared_parameters": 25,
        "expanded_edges": 50,
        "directions_checked": 3,
        "perturbation_sizes_per_direction": 3,
        "shared_vs_summed_untied_chain_rule": "pass",
        "genn_simulation_performed": False,
        "eventprop_shared_lif_gradient_certified": False,
    })


if __name__ == "__main__":
    main()