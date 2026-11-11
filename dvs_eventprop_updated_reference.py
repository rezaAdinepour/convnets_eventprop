"""CPU-only continuous gradient audit at the saved post-Adam weights.

No GeNN imports or builds. Creates one exclusive result in a fresh run directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import dvs_eventprop_network_reference as network

data, ref = network.data, network.ref
require, emit = ref.require, ref.emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--updated", type=Path, required=True)
    args = parser.parse_args()
    baseline_path = args.baseline.resolve(strict=True)
    updated_path = args.updated.resolve(strict=True)
    baseline = json.loads(baseline_path.read_text())
    updated = json.loads(updated_path.read_text())
    require(data.file_hash(baseline_path) == updated["baseline_sha256"],
            "Updated artifact is not bound to this baseline")
    require(data.file_hash(Path(__file__).with_name("dvs_eventprop_consistent_probe.py"))
            == updated["diagnostic_sha256"], "Updated artifact producer source changed")
    for artifact in (baseline, updated):
        for name, digest in artifact["identity"]["source_sha256"].items():
            require(data.file_hash(Path(__file__).with_name(name)) == digest,
                    f"Core source changed: {name}")
    for key, value in baseline["condition"].items():
        if key not in {"real_examples", "empty_drains", "optimizer_updates"}:
            require(updated["condition"][key] == value, f"Condition changed: {key}")
    require(updated["sample_metadata"] == baseline["sample_metadata"],
            "Updated input metadata differ")
    require(updated["optimizer_updates"] == 1 and len(updated["updates"]) == 1
            and updated["updates"][0]["adam_step"] == 0,
            "Expected exactly one Adam update")
    require(updated["limiter_effect"]["comparison_weights_and_moments"] == "exact_match"
            and updated["limiter_effect"]["forward_hidden_spike_id_time_parity"] == "exact",
            "Limiter comparison was not controlled")
    require(len(updated["captured_gradients"]) == 2
            and updated["replay_records"][2]["replay_limit"] == [1.0]
            and updated["replay_records"][2]["clipped_jumps"] == 0,
            "Missing unclipped updated-weight control")
    state = [np.asarray(v, dtype=np.float64)
             for v in updated["frozen_comparison_weights_and_moments"]]
    require(len(state) == 6, "Expected weights plus four Adam moment arrays")
    for array, size in zip(state, (400, 4312, 400, 400, 4312, 4312)):
        require(array.shape == (size,) and np.isfinite(array).all(), "Invalid saved state")
    for index, (weight, initial, gradient) in enumerate(zip(
        state[:2], baseline["initial_weights"], baseline["gradients"]
    )):
        g = np.asarray(gradient, dtype=np.float64)
        m, v = 0.1 * g, 0.001 * g * g
        expected = np.asarray(initial) - 0.001 * (m / (1.0 - 0.9)) / (
            np.sqrt(v / (1.0 - 0.999)) + 1e-8
        )
        np.testing.assert_allclose(weight, expected, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(state[2 + 2 * index], m, rtol=2e-10, atol=2e-12)
        np.testing.assert_allclose(state[3 + 2 * index], v, rtol=2e-10, atol=2e-12)
    root = args.root.resolve(strict=True)
    rows = [row for row in data.inventory(root)["train"]
            if row.stable_id == updated["sample_metadata"]["stable_id"]]
    require(len(rows) == 1, "Training parent not uniquely identified")
    payload, label, metadata = data.load_sample(root, rows[0], data.Config(1024))
    require(json.loads(json.dumps(metadata)) == updated["sample_metadata"],
            "Reconstructed input metadata differ")
    ref.self_check()
    provenance = {
        "baseline_sha256": data.file_hash(baseline_path),
        "updated_artifact_sha256": data.file_hash(updated_path),
        "diagnostic_sha256": data.file_hash(Path(__file__)),
        "network_reference_sha256": data.file_hash(Path(network.__file__)),
        "neuron_reference_sha256": data.file_hash(Path(ref.__file__)),
        "updated_weight_sha256": [hashlib.sha256(w.tobytes()).hexdigest() for w in state[:2]],
    }
    emit("UPDATED_REFERENCE_PLAN", {**provenance, "host_first_adam_state": "pass",
         "genn_simulation_performed": False})
    windows = network.input_vectors(payload)
    kernel, dense = state[0].reshape(50, 8), state[1].reshape(392, 11)
    coefficients, counts, dc = network.evaluate(kernel, windows, derivatives=True)
    ce, scores, ds = network.objective(coefficients, dense, int(label))
    drive = dense @ ds
    kg = np.zeros((50, 8))
    for f in range(8):
        kg[:, f] = (dc[f::8] * drive[f::8, None]).sum(axis=0)
    dg = coefficients[:, None] * ds[None, :]
    directions = {}
    for slot in (28, 29):
        direction = np.zeros((50, 8))
        direction[slot, 2] = 1.0
        directions[f"kernel_2_4_{slot - 28}_filter2"] = direction
    directions["coupled_filter2"] = sum(directions.values()) / np.sqrt(2.0)
    fd_checks = []
    for name, direction in directions.items():
        analytic = float(np.sum(kg * direction))
        passed = []
        for epsilon in (3e-6, 1e-6, 3e-7):
            losses, ns = [], []
            for sign in (1.0, -1.0):
                c, n, _ = network.evaluate(
                    kernel + sign * epsilon * direction, windows, filter_only=2,
                    base_coefficients=coefficients, base_counts=counts,
                )
                losses.append(network.objective(c, dense, int(label))[0])
                ns.append(n)
            fd = (losses[0] - losses[1]) / (2.0 * epsilon)
            stable = all(np.array_equal(n, counts) for n in ns)
            passed.append(stable and bool(np.isclose(fd, analytic, rtol=5e-4, atol=1e-9)))
            emit("UPDATED_REFERENCE_FD", {"direction": name, "epsilon": epsilon,
                 "analytic": analytic, "finite_difference": float(fd),
                 "per_neuron_counts_stable": stable, "pass": passed[-1]})
        fd_checks.append(all(passed[-2:]))
    recorded_count = updated["replay_records"][2]["replayed_jumps"]
    emit("UPDATED_REFERENCE_FORWARD", {
        "continuous_ce": ce, "continuous_decision": int(scores.argmax()),
        "continuous_hidden_spikes": int(counts.sum()), "recorded_hidden_spikes": recorded_count,
        "per_neuron_gpu_counts_available": False,
    })
    comparisons = []
    for mode, gradients in zip(("adapted_limited", "bound_one_unclipped"),
                               updated["captured_gradients"]):
        require(len(gradients) == 2, "Expected two captured projections")
        for index, reference in enumerate((kg.reshape(-1), dg.reshape(-1))):
            actual = np.asarray(gradients[index], dtype=np.float64)
            require(actual.shape == reference.shape and np.isfinite(actual).all()
                    and np.linalg.norm(reference) > 0, "Invalid comparison gradients")
            relative = float(np.linalg.norm(actual - reference) / np.linalg.norm(reference))
            row = {"mode": mode, "projection": index,
                   "relative_l2_error_to_continuous": relative,
                   "max_absolute_difference": float(np.max(np.abs(actual - reference))),
                   "within_1pct": relative <= 0.01}
            comparisons.append(row)
            emit("UPDATED_CONTINUOUS_GRADIENT", row)
    report = {**provenance, "sample_metadata": metadata, "continuous_ce": ce,
              "continuous_scores": scores.tolist(), "hidden_counts": counts.tolist(),
              "updated_weights": [w.tolist() for w in state[:2]],
              "gradients": [kg.reshape(-1).tolist(), dg.reshape(-1).tolist()],
              "selected_finite_difference_gates_passed": all(fd_checks),
              "comparisons": comparisons, "full_window_training_ready": False}
    destination = Path.cwd() / "updated_reference_result.json"
    with destination.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    require(all(fd_checks), "Updated continuous derivative reference self-check failed")
    require(int(counts.sum()) == recorded_count, "Updated total hidden spike count differs")
    require(all(row["within_1pct"] for row in comparisons
                if row["mode"] == "bound_one_unclipped"),
            "Unclipped updated-weight gradient exceeds 1 percent reference error")
    emit("PASS_UPDATED_CONTINUOUS_GRADIENT_GATE", {
        "result": str(destination), "optimizer_updates_in_source_artifact": 1,
        "genn_simulation_performed": False, "full_window_training_ready": False,
        "scope": "one saved post-Adam state and one training parent",
    })


if __name__ == "__main__":
    main()
