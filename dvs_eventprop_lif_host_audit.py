"""CPU-only reconstruction of the single-hidden-LIF diagnostic.

Reads existing logs and generated source hashes. No GeNN imports or builds.
Specific to one input spike, one hidden spike, absolute reset and label 0.
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


def read_records(path):
    result = {}
    for line in path.read_text().splitlines():
        tag, separator, value = line.partition(" ")
        if separator and value.startswith("{"):
            result.setdefault(tag, []).append(json.loads(value))
    return result


def one(records, tag):
    values = records.get(tag, [])
    require(len(values) == 1, f"Expected one {tag}")
    return values[0]


def audit(path):
    records = read_records(path)
    config = one(records, "LIF_CONFIG")
    measured = one(records, "LIF_GRADIENT_COMPARISON")
    limiter = one(records, "LIF_LIMITER")

    require(config["absolute_reset"] == 0.0, "Unsupported reset")
    require(config["threshold"] == 1.0, "Unsupported threshold")
    require(config["tau_mem_ms"] == 20.0, "Unsupported membrane tau")
    require(config["tau_syn_ms"] == 5.0, "Unsupported synapse tau")
    require(config["delay"] == 0, "Unsupported delay")
    require(limiter["AuditJumps"] == 1, "Expected one replayed jump")
    require(limiter["AuditClips"] == 0, "Expected inactive limiter")

    dt = float(config["dt_ms"])
    horizon = float(config["physical_horizon_ms"])
    steps = int(config["simulation_timesteps"])
    require(steps * dt == horizon, "Horizon mismatch")
    input_step = int(round(config["input_time_ms"] / dt))
    require(input_step * dt == config["input_time_ms"],
            "Input is not grid-aligned")

    weight = float(config["input_weight"][0][0])
    output_weight = np.asarray(config["output_weight"], np.float64).reshape(11)
    a_mem = np.exp(-dt / 20.0)
    a_syn = np.exp(-dt / 5.0)

    # Forward: synapses consume spikes from the preceding neuron update.
    hidden_current = 0.0
    hidden_voltage = 0.0
    output_current = np.zeros(11)
    output_voltage = np.zeros(11)
    scores = np.zeros(11)
    previous_input_spike = False
    previous_hidden_spike = False
    hidden_events = []

    for n in range(steps):
        hidden_current += weight * previous_input_spike
        output_current += output_weight * previous_hidden_spike
        current_at_hidden_update = hidden_current

        hidden_voltage = (
            a_mem * hidden_voltage
            + (1.0 - a_mem) * hidden_current
        )
        output_voltage = (
            a_mem * output_voltage
            + (1.0 - a_mem) * output_current
        )
        hidden_current *= a_syn
        output_current *= a_syn
        scores += (
            np.exp(-(n * dt) / horizon) * dt / horizon * output_voltage
        )

        hidden_spike = hidden_voltage >= 1.0
        if hidden_spike:
            hidden_events.append((n, current_at_hidden_update))
            hidden_voltage = 0.0

        previous_input_spike = n == input_step
        previous_hidden_spike = hidden_spike

    require(len(hidden_events) == 1,
            f"Host produced {len(hidden_events)} hidden spikes")
    hidden_step, saved_current = hidden_events[0]
    require(1 <= input_step <= steps - 2, "Endpoint input unsupported")
    require(1 <= hidden_step <= steps - 2, "Endpoint hidden event unsupported")

    probability = np.exp(scores - scores.max())
    probability /= probability.sum()
    probability[0] -= 1.0

    # Backward event detection happens one step before synapse processing.
    input_service = steps - input_step - 1
    hidden_service = steps - hidden_step - 1

    output_lambda_v = np.zeros(11)
    output_lambda_i = np.zeros(11)
    hidden_lambda_v = 0.0
    hidden_lambda_i = 0.0
    output_gradient = np.zeros(11)
    hidden_gradient = 0.0
    raw_jump = None
    bound = float(limiter["bound_before_replay"][0])

    for n in range(steps):
        reverse_input = 0.0

        # Synapse update reads adjoints before this step's neuron updates.
        if n == input_service:
            hidden_gradient -= hidden_lambda_i

        if n == hidden_service:
            output_gradient -= output_lambda_i
            reverse_input = float(np.dot(
                output_weight,
                -output_lambda_i / 5.0 + output_lambda_v / 20.0,
            ))

        # Both synaptic adjoints use preceding voltage-adjoint values.
        output_lambda_i = (
            a_syn * output_lambda_i
            + (1.0 - a_syn) * (5.0 / 20.0) * output_lambda_v
        )
        hidden_lambda_i = (
            a_syn * hidden_lambda_i
            + (1.0 - a_syn) * (5.0 / 20.0) * hidden_lambda_v
        )

        drive = (
            -probability * np.exp(-1.0 + n * dt / horizon) / horizon
        )
        output_lambda_v = (
            a_mem * output_lambda_v
            + 20.0 * (1.0 - a_mem) * drive
        )
        hidden_lambda_v *= a_mem

        if n == hidden_service:
            raw_jump = (
                hidden_lambda_v * saved_current + 20.0 * reverse_input
            ) / (saved_current - 1.0)
            hidden_lambda_v = float(np.clip(raw_jump, -bound, bound))

    require(raw_jump is not None, "No host hidden jump")
    require(abs(raw_jump) <= bound, "Host unexpectedly activated limiter")
    actual_scores = np.asarray(measured["genn_scores"], np.float64)
    actual_output = np.asarray(
        measured["genn_output_weight_gradient"], np.float64
    )
    actual_hidden = float(measured["genn_hidden_weight_gradient"])

    continuous_denominator = (
        20.0 * measured["continuous_crossing_slope_per_ms"]
    )
    # Analysis only: change the one jump's denominator while retaining all
    # other generated states/timings. This is not a proposed implementation.
    denominator_counterfactual = (
        hidden_gradient * (saved_current - 1.0) / continuous_denominator
    )

    generated = path.parent / "SingleLIFGradientAuditCompiler_CODE"
    emit("LIF_HOST_RECONSTRUCTION", {
        "log": str(path),
        "log_sha256": digest(path),
        "audit_source_sha256": digest(Path(__file__)),
        "generated_source_sha256": {
            name: digest(generated / name)
            for name in ("runner.cc", "neuronUpdate.cc", "synapseUpdate.cc")
        },
        "dt_ms": dt,
        "host_hidden_spike_time_ms": hidden_step * dt,
        "host_saved_Isyn": saved_current,
        "host_jump_denominator": saved_current - 1.0,
        "continuous_jump_denominator": continuous_denominator,
        "host_raw_jump": raw_jump,
        "host_hidden_gradient": hidden_gradient,
        "genn_hidden_gradient": actual_hidden,
        "hidden_gradient_absolute_difference":
            abs(hidden_gradient - actual_hidden),
        "output_gradient_max_absolute_difference": float(
            np.max(np.abs(output_gradient - actual_output))
        ),
        "score_max_absolute_difference": float(
            np.max(np.abs(scores - actual_scores))
        ),
        "denominator_only_counterfactual_hidden_gradient":
            denominator_counterfactual,
        "continuous_hidden_gradient":
            measured["continuous_hidden_weight_gradient"],
        "counterfactual_is_proposed_fix": False,
    })

    np.testing.assert_equal(
        hidden_step * dt, measured["genn_hidden_spike_time_ms"]
    )
    np.testing.assert_allclose(
        scores, actual_scores, rtol=2e-5, atol=2e-7,
        err_msg="Host forward scores differ from GPU",
    )
    np.testing.assert_allclose(
        abs(saved_current - 1.0), limiter["AuditMinDenom"],
        rtol=2e-5, atol=2e-7,
        err_msg="Host saved threshold state differs from GPU",
    )
    np.testing.assert_allclose(
        abs(raw_jump), limiter["AuditMaxRaw"],
        rtol=3e-5, atol=1e-9,
        err_msg="Host raw adjoint jump differs from GPU",
    )
    np.testing.assert_allclose(
        hidden_gradient, actual_hidden, rtol=3e-5, atol=1e-9,
        err_msg="Host hidden gradient differs from GPU",
    )
    np.testing.assert_allclose(
        output_gradient, actual_output, rtol=2e-5, atol=2e-7,
        err_msg="Host output gradient differs from GPU",
    )

    emit("PASS_LIF_HOST_RECONSTRUCTION", {
        "dt_ms": dt,
        "forward_spike_and_scores_reproduced": True,
        "saved_threshold_state_reproduced": True,
        "raw_jump_and_both_gradients_reproduced": True,
        "continuous_gradient_gate_passed": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.logs:
        audit(path.resolve(strict=True))


if __name__ == "__main__":
    main()