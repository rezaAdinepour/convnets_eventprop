"""CPU-only audit of the fixed-output probe's saved measurements.

Specific to its two inputs, eleven LI outputs, interior grid-aligned spikes,
zero delays, tau_mem=20 ms, tau_syn=5 ms and weighted-voltage readout.
Reads logs/generated source; writes only stdout.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(scores):
    p = np.exp(scores - scores.max())
    return p / p.sum()


def one(records, tag):
    values = records.get(tag, [])
    require(len(values) == 1, f"Expected one {tag} record")
    return values[0]


def audit(path):
    records = {}
    for line in path.read_text().splitlines():
        tag, separator, text = line.partition(" ")
        if separator and text.startswith("{"):
            records.setdefault(tag, []).append(json.loads(text))

    config = one(records, "FIXED_CONFIG")
    reference = one(records, "CONTINUOUS_REFERENCE")
    capture = one(records, "FIXED_CAPTURE")
    identity = one(records, "IDENTITY")

    require(config["architecture"] == "SpikeInput(2)->Dense(11)->LI",
            "Unsupported architecture")
    require(config["tau_mem_ms"] == 20.0, "Unsupported membrane tau")
    require(config["tau_syn_ms"] == 5.0, "Unsupported synapse tau")
    require(config["readout"] == "avg_var_exp_weight",
            "Unsupported readout")
    require(config["optimizer_updates"] == 0, "Unexpected updates")

    dt = float(config["dt_ms"])
    horizon = float(config["physical_horizon_ms"])
    steps = int(config["simulation_timesteps"])
    require(steps * dt == horizon, "Horizon mismatch")

    weight = np.asarray(config["initial_weights"], dtype=np.float64)
    actual_scores = np.asarray(reference["compiled_scores"], np.float64)
    actual_gradient = np.asarray(reference["captured_gradient"], np.float64)
    require(weight.shape == actual_gradient.shape == (2, 11),
            "Unsupported weight axes")

    delivery = np.zeros((steps, 2), dtype=np.float64)
    replay = np.zeros_like(delivery)
    for neuron, time_ms in config["scheduled_spikes"]:
        k = int(round(time_ms / dt))
        require(k * dt == time_ms, "Off-grid input")
        require(1 <= k <= steps - 2,
                "This audit excludes endpoint replay cases")
        # Spike emitted in neuron update k; delivered in synapse update k+1.
        delivery[k + 1, neuron] += 1.0

        # backT = T-t-dt; event condition subtracts another dt.
        # Detection is at N-k-2; synapse processing is one step later.
        replay[steps - k - 1, neuron] += 1.0

    a_mem = np.exp(-dt / 20.0)
    a_syn = np.exp(-dt / 5.0)

    # Linear response coefficients for each of the two input sources.
    current = np.zeros(2)
    voltage = np.zeros(2)
    sensitivity = np.zeros(2)

    for n in range(steps):
        current += delivery[n]
        voltage = a_mem * voltage + (1.0 - a_mem) * current
        current *= a_syn
        sensitivity += (
            np.exp(-(n * dt) / horizon)
            * dt / horizon * voltage
        )

    host_scores = sensitivity @ weight
    host_probability = softmax(host_scores)
    host_probability[0] -= 1.0
    discrete_gradient = sensitivity[:, None] * host_probability[None, :]

    # Reconstruct the generated continuous-adjoint discretization.
    # Positive unit loss drive; CE class factors are applied afterwards.
    lambda_v = 0.0
    lambda_i = 0.0
    backward_sensitivity = np.zeros(2)

    for n in range(steps):
        # Synapse update reads adjoints before this step's neuron update.
        backward_sensitivity += replay[n] * lambda_i

        # Synaptic adjoint uses the previous voltage adjoint.
        lambda_i = (
            a_syn * lambda_i
            + (1.0 - a_syn) * (5.0 / 20.0) * lambda_v
        )
        drive = np.exp(-1.0 + n * dt / horizon) / horizon
        lambda_v = (
            a_mem * lambda_v
            + 20.0 * (1.0 - a_mem) * drive
        )

    generated_gradient = (
        backward_sensitivity[:, None] * host_probability[None, :]
    )
    direction = actual_gradient / np.linalg.norm(actual_gradient)
    discrete_direction = float(np.sum(discrete_gradient * direction))
    generated_direction = float(np.sum(generated_gradient * direction))
    observed_directions = records.get("FIXED_DIRECTION", [])
    require(len(observed_directions) == 3, "Expected three finite differences")

    generated_dir = path.parent / "FixedOutputGradientEventPropCompiler_CODE"
    source_hashes = {
        name: digest(generated_dir / name)
        for name in ("runner.cc", "neuronUpdate.cc", "synapseUpdate.cc")
    }

    emit("HOST_RECURRENCE", {
        "log": str(path),
        "log_sha256": digest(path),
        "audit_source_sha256": digest(Path(__file__)),
        "original_identity": identity,
        "generated_source_sha256": source_hashes,
        "dt_ms": dt,
        "forward_sensitivity": sensitivity.tolist(),
        "backward_sensitivity": backward_sensitivity.tolist(),
        "forward_score_max_absolute_error": float(
            np.max(np.abs(host_scores - actual_scores))
        ),
        "generated_gradient_max_absolute_error": float(
            np.max(np.abs(generated_gradient - actual_gradient))
        ),
        "exact_discrete_directional_derivative": discrete_direction,
        "generated_adjoint_directional_derivative": generated_direction,
        "captured_directional_derivative": float(
            np.linalg.norm(actual_gradient)
        ),
        "captured_vs_exact_discrete_relative_l2": float(
            np.linalg.norm(actual_gradient - discrete_gradient)
            / np.linalg.norm(discrete_gradient)
        ),
    })

    # Float64 host reconstruction versus float32 GPU measurements.
    np.testing.assert_allclose(
        host_scores, actual_scores, rtol=2e-5, atol=2e-7,
        err_msg="Host forward recurrence does not reproduce GPU scores",
    )
    np.testing.assert_allclose(
        generated_gradient, actual_gradient, rtol=2e-5, atol=2e-7,
        err_msg="Host adjoint/replay recurrence does not reproduce GPU gradient",
    )
    np.testing.assert_allclose(
        generated_direction, capture["gradient_l2"],
        rtol=2e-5, atol=2e-7,
    )
    for observed in observed_directions:
        np.testing.assert_allclose(
            discrete_direction,
            observed["forward_ce_finite_difference"],
            rtol=2e-4, atol=1e-6,
            err_msg="Exact discrete derivative differs from GPU finite difference",
        )

    emit("PASS_HOST_RECURRENCE", {
        "log": str(path),
        "dt_ms": dt,
        "forward_reproduced": True,
        "captured_gradient_reproduced": True,
        "discrete_derivative_matches_finite_differences": True,
        "original_1pct_gradient_gate_passed": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.logs:
        audit(path.resolve(strict=True))


if __name__ == "__main__":
    main()