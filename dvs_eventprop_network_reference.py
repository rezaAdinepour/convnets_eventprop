"""CPU-only continuous network reference for saved frozen 1024ms candidates.

No GeNN imports/builds. Writes one exclusive JSON artifact in the run directory.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import dvs_data as data
import dvs_eventprop_neuron_reference as ref

require, emit = ref.require, ref.emit


def input_vectors(payload):
    ends = payload.end_spikes.astype(np.int64)
    starts = np.r_[0, ends[:-1]]
    windows = []
    for oy in range(7):
        for ox in range(7):
            events = {}
            for ky in range(5):
                for kx in range(5):
                    for p in range(2):
                        pre = (((oy * 4 + ky) * 32 + ox * 4 + kx) * 2 + p)
                        slot = (ky * 5 + kx) * 2 + p
                        for t in payload.spike_times[starts[pre]:ends[pre]]:
                            t = float(t)
                            if t not in events:
                                events[t] = np.zeros(50)
                            events[t][slot] += 1.0
            windows.append(events)
    return windows


def evaluate(kernel, windows, *, derivatives=False, filter_only=None,
             base_coefficients=None, base_counts=None):
    coefficients = (np.zeros(392) if base_coefficients is None
                    else base_coefficients.copy())
    counts = np.zeros(392, dtype=np.int64) if base_counts is None else base_counts.copy()
    gradients = np.zeros((392, 50))
    filters = range(8) if filter_only is None else (filter_only,)
    for position, events in enumerate(windows):
        for feature in filters:
            neuron = position * 8 + feature
            if not events:
                continue
            jumps = {t: float(vector @ kernel[:, feature])
                     for t, vector in events.items()}
            if derivatives:
                times, dt_dw = ref.sensitive_trajectory(jumps, events, 1024.0)
            else:
                spikes, _ = ref.trajectory(jumps, 1024.0)
                times = np.array([s["time_ms"] for s in spikes])
            counts[neuron] = len(times)
            if len(times):
                response = np.array([ref.readout_response(t, 1024.0) for t in times])
                coefficients[neuron] = response[:, 0].sum()
                if derivatives:
                    gradients[neuron] = response[:, 1] @ dt_dw
            else:
                coefficients[neuron] = 0.0
        if derivatives and (position + 1) % 7 == 0:
            emit("REFERENCE_PROGRESS", {"completed_windows": position + 1, "total": 49})
    require(np.isfinite(coefficients).all() and np.isfinite(gradients).all(),
            "Nonfinite continuous reference")
    return coefficients, counts, gradients


def objective(coefficients, dense, label):
    scores = coefficients @ dense
    maximum = float(scores.max())
    exps = np.exp(scores - maximum)
    loss = maximum + np.log(exps.sum()) - scores[label]
    score_gradient = exps / exps.sum()
    score_gradient[label] -= 1.0
    return float(loss), scores, score_gradient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("artifacts", type=Path, nargs=2)
    args = parser.parse_args()
    paths = [p.resolve(strict=True) for p in args.artifacts]
    results = [json.loads(p.read_text()) for p in paths]
    base = results[0]
    ref.self_check()
    for result in results:
        condition = result["condition"]
        for key, expected in {
            "frames": 1024, "physical_horizon_ms": 1024.0,
            "input_hwp": [32, 32, 2], "hidden_hwf": [7, 7, 8],
            "kernel_hwio": [5, 5, 2, 8], "dense_io": [392, 11],
            "conv_stride": [4, 4], "conv_padding": "valid",
            "tau_mem_ms": 20.0, "tau_syn_ms": 5.0, "threshold": 1.0,
            "absolute_reset_voltage": 0.0, "readout": "avg_var_exp_weight",
            "loss": "sparse_categorical_crossentropy", "per_timestep_loss": False,
            "regulariser": None, "scalar_precision": "double", "batch_size": 1,
        }.items():
            require(condition[key] == expected, f"Unsupported condition: {key}")
        require(result["optimizer_updates"] == 0
                and result["weights_and_moments_unchanged"] is True,
                "Artifact was not frozen")
        require(result["sample_metadata"] == base["sample_metadata"]
                and result["initial_weights"] == base["initial_weights"],
                "Input or initial weights differ")
        for name, digest in result["identity"]["source_sha256"].items():
            require(data.file_hash(Path(__file__).with_name(name)) == digest,
                    f"Core source changed: {name}")
        require(all(result["limiter_statistics"][k] == 0 for k in (
            "clipped_jumps", "invalid_limits", "nonfinite_raw_jumps"
        )), "Clipping or invalid replay confounds comparison")
    root = args.root.resolve(strict=True)
    matches = [r for r in data.inventory(root)["train"]
               if r.stable_id == base["sample_metadata"]["stable_id"]]
    require(len(matches) == 1, "Training parent not uniquely identified")
    payload, label, metadata = data.load_sample(root, matches[0], data.Config(1024))
    require(json.loads(json.dumps(metadata)) == base["sample_metadata"],
            "Reconstructed input metadata differ")
    weights = [np.asarray(w, dtype=np.float64) for w in base["initial_weights"]]
    for weight, size, digest in zip(weights, (400, 4312), base["initial_weight_sha256"]):
        require(weight.shape == (size,) and np.isfinite(weight).all(), "Invalid weights")
        require(hashlib.sha256(weight.tobytes()).hexdigest() == digest,
                "Initial weight hash mismatch")
    kernel, dense = weights[0].reshape(50, 8), weights[1].reshape(392, 11)
    emit("REFERENCE_PLAN", {
        "artifact_sha256": [data.file_hash(p) for p in paths],
        "diagnostic_sha256": data.file_hash(Path(__file__)),
        "neuron_reference_sha256": data.file_hash(Path(ref.__file__)),
        "parent_id": metadata["parent_id"], "label": int(label),
        "scope": "one frozen full-window example; continuous dynamics",
        "genn_simulation_performed": False,
    })
    windows = input_vectors(payload)
    coefficients, counts, coefficient_gradients = evaluate(kernel, windows, derivatives=True)
    ce, scores, score_gradient = objective(coefficients, dense, int(label))
    drive = dense @ score_gradient
    kernel_gradient = np.zeros((50, 8))
    for f in range(8):
        kernel_gradient[:, f] = (
            coefficient_gradients[f::8] * drive[f::8, None]
        ).sum(axis=0)
    dense_gradient = coefficients[:, None] * score_gradient[None, :]

    # Full-network CE differences along the sensitive shared filter's two
    # leading entries and their coupled direction. Recompute all 49 neurons
    # affected by each perturbation, retaining the other filters' coefficients.
    directions = {}
    for slot in (28, 29):
        direction = np.zeros((50, 8))
        direction[slot, 2] = 1.0
        directions[f"kernel_2_4_{slot - 28}_filter2"] = direction
    directions["coupled_filter2"] = sum(directions.values()) / np.sqrt(2.0)
    checks = []
    for name, direction in directions.items():
        analytic = float(np.sum(kernel_gradient * direction))
        passed = []
        for epsilon in (3e-6, 1e-6, 3e-7):
            losses, perturb_counts = [], []
            for sign in (1.0, -1.0):
                c, n, _ = evaluate(
                    kernel + sign * epsilon * direction, windows, filter_only=2,
                    base_coefficients=coefficients, base_counts=counts,
                )
                losses.append(objective(c, dense, int(label))[0])
                perturb_counts.append(n)
            fd = (losses[0] - losses[1]) / (2.0 * epsilon)
            stable = all(np.array_equal(n, counts) for n in perturb_counts)
            passed.append(stable and bool(np.isclose(fd, analytic, rtol=5e-4, atol=1e-9)))
            emit("NETWORK_REFERENCE_FD", {
                "direction": name, "epsilon": epsilon, "analytic": analytic,
                "finite_difference": float(fd), "per_neuron_counts_stable": stable,
                "pass": passed[-1],
            })
        checks.append(all(passed[-2:]))

    comparisons = []
    for path, result in zip(paths, results):
        changed = np.flatnonzero(counts != result["hidden_spikes_per_neuron"])
        emit("CONTINUOUS_NETWORK_FORWARD", {
            "artifact": str(path), "dt_ms": result["condition"]["dt_ms"],
            "continuous_hidden_spikes": int(counts.sum()),
            "neurons_with_changed_counts": changed.tolist(),
            "continuous_ce": ce, "continuous_decision": int(scores.argmax()),
            "score_max_absolute_difference": float(np.max(np.abs(
                scores - np.asarray(result["scores"][0])
            ))),
        })
        for projection, reference in enumerate((kernel_gradient, dense_gradient)):
            a = np.asarray(result["gradients"][projection], dtype=np.float64)
            b = reference.reshape(-1)
            require(a.shape == b.shape and np.isfinite(a).all(), "Invalid GPU gradient")
            relative = float(np.linalg.norm(a - b) / np.linalg.norm(b))
            row = {
                "artifact": str(path), "dt_ms": result["condition"]["dt_ms"],
                "projection": projection, "relative_l2_error_to_continuous": relative,
                "max_absolute_difference": float(np.max(np.abs(a - b))),
                "within_1pct": relative <= 0.01,
            }
            comparisons.append(row)
            emit("CONTINUOUS_NETWORK_GRADIENT", row)
    destination = Path.cwd() / "network_reference_result.json"
    report = {
        "artifact_sha256": [data.file_hash(p) for p in paths],
        "reference_source_sha256": data.file_hash(Path(__file__)),
        "neuron_reference_sha256": data.file_hash(Path(ref.__file__)),
        "initial_weight_sha256": base["initial_weight_sha256"],
        "sample_metadata": metadata, "continuous_scores": scores.tolist(),
        "continuous_ce": ce, "hidden_counts": counts.tolist(),
        "gradients": [kernel_gradient.reshape(-1).tolist(), dense_gradient.reshape(-1).tolist()],
        "selected_finite_difference_gates_passed": all(checks),
        "comparisons": comparisons, "full_window_training_ready": False,
    }
    with destination.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    require(all(checks), "Continuous full-network derivative self-check failed")
    emit("PASS_NETWORK_REFERENCE_SELF_CHECKS", {
        "result": str(destination), "selected_shared_kernel_directions": 3,
        "cross_entropy_chain_rule": "included", "genn_simulation_performed": False,
        "full_window_training_ready": False,
        "scope": "one frozen example; all hidden contributions assembled",
    })


if __name__ == "__main__":
    main()
