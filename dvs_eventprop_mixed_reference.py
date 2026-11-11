"""CPU-only continuous audit of saved B8 mixed-lane gradients and real mean."""

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
    parser.add_argument("--mixed", type=Path, required=True)
    args = parser.parse_args()
    baseline_path, mixed_path = [p.resolve(strict=True) for p in (args.baseline, args.mixed)]
    baseline, mixed = [json.loads(p.read_text()) for p in (baseline_path, mixed_path)]
    require(data.file_hash(baseline_path) == mixed["baseline_sha256"], "Baseline binding differs")
    require(data.file_hash(Path(__file__).with_name("dvs_eventprop_mixed_batch_probe.py"))
            == mixed["diagnostic_sha256"], "Mixed producer source changed")
    for result in (baseline, mixed):
        require(result["optimizer_updates"] == 0 and result["weights_and_moments_unchanged"],
                "Expected frozen artifacts")
        for name, digest in result["identity"]["source_sha256"].items():
            require(data.file_hash(Path(__file__).with_name(name)) == digest,
                    f"Core source changed: {name}")
    for key, value in baseline["condition"].items():
        if key not in {"batch_size", "real_examples"}:
            require(mixed["condition"][key] == value, f"Physical condition changed: {key}")
    require(mixed["condition"]["batch_size"] == 8
            and mixed["condition"]["real_examples"] == 6
            and mixed["condition"]["layout"] == [0, 1, 2, 0, 1, 2, 0, 1],
            "Unexpected mixed layout")
    statistics = mixed["lane_replay_statistics"]
    require(len(statistics) == 8 and all(all(row[key] == 0 for key in (
        "clipped_jumps", "invalid_limits", "nonfinite_raw_jumps"
    )) for row in statistics), "Mixed replay was limited or invalid")
    source_a = baseline["continuous_reference_comparison"]["source"]
    a_path = Path(source_a["path"]).resolve(strict=True)
    require(data.file_hash(a_path) == source_a["sha256"], "A reference artifact changed")
    a_reference = json.loads(a_path.read_text())
    require(a_reference["selected_finite_difference_gates_passed"] is True,
            "A reference self-check did not pass")
    require(data.file_hash(Path(network.__file__)) == a_reference["reference_source_sha256"]
            and data.file_hash(Path(ref.__file__)) == a_reference["neuron_reference_sha256"],
            "Continuous reference source changed")
    require(a_reference["sample_metadata"] == baseline["sample_metadata"]
            and a_reference["initial_weight_sha256"] == baseline["initial_weight_sha256"],
            "A reference input or weights differ")
    require(len(mixed["sample_metadata"]) == 2
            and mixed["sample_metadata"][0] == baseline["sample_metadata"], "A provenance differs")
    weights = [np.asarray(w, dtype=np.float64) for w in baseline["initial_weights"]]
    for array, size, digest in zip(weights, (400, 4312), baseline["initial_weight_sha256"]):
        require(array.shape == (size,) and np.isfinite(array).all(), "Invalid weights")
        require(hashlib.sha256(array.tobytes()).hexdigest() == digest, "Weight hash mismatch")
    observed_b = []
    for index, size in enumerate((400, 4312)):
        raw = np.asarray(mixed["raw_lane_gradients"][index], dtype=np.float64)
        require(raw.shape == (8, size) and np.isfinite(raw).all(), "Invalid saved lane gradients")
        for lane in (0, 3, 6):
            np.testing.assert_allclose(raw[lane] * 8, baseline["gradients"][index],
                                       rtol=1e-9, atol=1e-11)
        for lane in (4, 7):
            np.testing.assert_allclose(raw[lane], raw[1], rtol=1e-9, atol=1e-11)
        np.testing.assert_array_equal(raw[[2, 5]], np.zeros((2, size)))
        np.testing.assert_allclose(raw.sum(axis=0), mixed["reduced_gradients"][index],
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(raw.sum(axis=0) * 8 / 6, mixed["real_mean_gradients"][index],
                                   rtol=1e-9, atol=1e-11)
        # Remove the compiler's 1/B factor to obtain the individual B
        # derivative. This is derived from a B8 lane, not a B1 GPU run.
        observed_b.append(raw[1] * 8)
    root = args.root.resolve(strict=True)
    metadata_b = mixed["sample_metadata"][1]
    rows = [row for row in data.inventory(root)["train"] if row.stable_id == metadata_b["stable_id"]]
    require(len(rows) == 1, "B training parent not uniquely identified")
    payload, label, metadata = data.load_sample(root, rows[0], data.Config(1024))
    require(json.loads(json.dumps(metadata)) == metadata_b, "Reconstructed B input differs")
    provenance = {"baseline_sha256": data.file_hash(baseline_path),
        "mixed_artifact_sha256": data.file_hash(mixed_path),
        "A_reference_sha256": data.file_hash(a_path),
        "diagnostic_sha256": data.file_hash(Path(__file__)),
        "network_reference_sha256": data.file_hash(Path(network.__file__)),
        "neuron_reference_sha256": data.file_hash(Path(ref.__file__))}
    emit("MIXED_REFERENCE_PLAN", {**provenance, "parent_B": metadata["parent_id"],
         "B_individual_derivative_source": "saved B8 lane multiplied by 8",
         "genn_simulation_performed": False})
    ref.self_check()
    kernel, dense = weights[0].reshape(50, 8), weights[1].reshape(392, 11)
    windows = network.input_vectors(payload)
    coefficients, counts, dc = network.evaluate(kernel, windows, derivatives=True)
    ce, scores, ds = network.objective(coefficients, dense, int(label))
    drive = dense @ ds
    kg = np.zeros((50, 8))
    for f in range(8):
        kg[:, f] = (dc[f::8] * drive[f::8, None]).sum(axis=0)
    reference_b = [kg.reshape(-1), (coefficients[:, None] * ds[None, :]).reshape(-1)]
    # Exercise large derivatives and the largest observed discrepancy.
    # Selection affects self-check locations, not the full-vector 1% gate.
    indices = []
    candidates = [int(np.argmax(np.abs(reference_b[0]))),
                  int(np.argmax(np.abs(observed_b[0] - reference_b[0])))]
    candidates += [int(i) for i in np.argsort(np.abs(reference_b[0]))[::-1]]
    for i in candidates:
        if i not in indices:
            indices.append(i)
        if len(indices) == 3:
            break
    fd_checks = []
    for flat_index in indices:
        slot, feature = divmod(flat_index, 8)
        direction = np.zeros((50, 8))
        direction[slot, feature] = 1.0
        passed = []
        for epsilon in (3e-6, 1e-6, 3e-7):
            losses, ns = [], []
            for sign in (1.0, -1.0):
                c, n, _ = network.evaluate(kernel + sign * epsilon * direction, windows,
                    filter_only=feature, base_coefficients=coefficients, base_counts=counts)
                losses.append(network.objective(c, dense, int(label))[0])
                ns.append(n)
            fd = (losses[0] - losses[1]) / (2.0 * epsilon)
            analytic = float(reference_b[0][flat_index])
            stable = all(np.array_equal(n, counts) for n in ns)
            passed.append(stable and bool(np.isclose(fd, analytic, rtol=5e-4, atol=1e-9)))
            emit("B_REFERENCE_FD", {"yxpf": [int(v) for v in np.unravel_index(flat_index, (5, 5, 2, 8))],
                 "epsilon": epsilon, "analytic": analytic, "finite_difference": float(fd),
                 "per_neuron_counts_stable": stable, "pass": passed[-1]})
        fd_checks.append(all(passed[-2:]))
    gpu_scores = np.asarray(mixed["scores"], dtype=np.float64)
    require(gpu_scores.shape == (2, 8, 11) and np.isfinite(gpu_scores).all(), "Invalid saved scores")
    emit("B_REFERENCE_FORWARD", {"continuous_ce": ce, "continuous_decision": int(scores.argmax()),
        "gpu_decision": int(gpu_scores[0, 1].argmax()), "continuous_hidden_spikes": int(counts.sum()),
        "recorded_hidden_spikes": statistics[1]["replayed_jumps"],
        "score_max_absolute_difference": float(np.max(np.abs(scores - gpu_scores[0, 1]))),
        "per_neuron_gpu_counts_available": False})
    comparisons = []
    for index, b_reference in enumerate(reference_b):
        a_reference_gradient = np.asarray(a_reference["gradients"][index], dtype=np.float64)
        for mode, actual, expected in (
            ("B_individual", observed_b[index], b_reference),
            ("six_real_example_mean", np.asarray(mixed["real_mean_gradients"][index]),
             (a_reference_gradient + b_reference) / 2.0),
        ):
            require(actual.shape == expected.shape and np.linalg.norm(expected) > 0,
                    "Invalid continuous gradient comparison")
            relative = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            row = {"mode": mode, "projection": index, "relative_l2_error_to_continuous": relative,
                   "max_absolute_difference": float(np.max(np.abs(actual - expected))),
                   "within_1pct": relative <= 0.01}
            comparisons.append(row)
            emit("MIXED_CONTINUOUS_GRADIENT", row)
    destination = Path.cwd() / "mixed_reference_result.json"
    report = {**provenance, "sample_metadata_B": metadata,
        "initial_weight_sha256": baseline["initial_weight_sha256"],
        "B_continuous_scores": scores.tolist(), "B_continuous_ce": ce,
        "B_hidden_counts": counts.tolist(), "B_gradients": [g.tolist() for g in reference_b],
        "selected_finite_difference_gates_passed": all(fd_checks), "comparisons": comparisons,
        "full_window_training_ready": False}
    with destination.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    require(all(fd_checks), "B continuous derivative self-check failed")
    require(int(counts.sum()) == statistics[1]["replayed_jumps"], "B total spike count differs")
    require(all(row["within_1pct"] for row in comparisons),
            "B or real-mean continuous gradient error exceeds 1 percent")
    emit("PASS_MIXED_CONTINUOUS_GRADIENT_GATE", {"result": str(destination),
        "genn_simulation_performed": False, "full_window_training_ready": False,
        "scope": "two frozen training parents in the checked mixed B8 layout"})


if __name__ == "__main__":
    main()
