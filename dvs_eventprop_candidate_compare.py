"""CPU-only timestep comparison of saved frozen candidate JSON artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def emit(name, value):
    print(name, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument(
        "--allow-reviewed-timestep-option-change", action="store_true",
        help="Allow only the source-verified addition of dt=0.00048828125",
    )
    args = parser.parse_args()
    paths = [args.coarse.resolve(strict=True), args.fine.resolve(strict=True)]
    blobs = [path.read_bytes() for path in paths]
    coarse, fine = [json.loads(blob) for blob in blobs]
    if args.allow_reviewed_timestep_option_change:
        old_hash = "6175cd17c4486a79ad9909f6a9eaa4cb5830999fee1eefa3253b0c3b23dc8c7e"
        new_hash = "f1f2ceeab186fec97c6fdfc42a4ca663b87ca9d3ddfc7b3ea3d93272ef0a0b61"
        source_path = Path(__file__).with_name("dvs_eventprop_candidate_probe.py")
        current_source = source_path.read_bytes()
        new_choices = (
            b"        choices=(\n"
            b"            1.0, 0.0078125, 0.00390625, 0.001953125,\n"
            b"            0.0009765625, 0.00048828125,\n"
            b"        ),\n"
        )
        old_choices = (
            b"        choices=(1.0, 0.0078125, 0.00390625, "
            b"0.001953125, 0.0009765625),\n"
        )
        require(hashlib.sha256(current_source).hexdigest() == new_hash,
                "Current probe differs from the reviewed revision")
        require(current_source.count(new_choices) == 1,
                "Cannot reconstruct the prior timestep choices")
        reconstructed = current_source.replace(new_choices, old_choices, 1)
        require(hashlib.sha256(reconstructed).hexdigest() == old_hash,
                "Source difference exceeds the reviewed timestep-option edit")
        identities = [dict(result["identity"]) for result in (coarse, fine)]
        require(identities[0].pop("diagnostic_sha256") == old_hash
                and identities[1].pop("diagnostic_sha256") == new_hash,
                "Unexpected diagnostic source revision pair")
        require(identities[0] == identities[1],
                "Other run source/environment identities differ")
        emit("SOURCE_REVISION_AUDIT", {
            "coarse_diagnostic_sha256": old_hash,
            "fine_diagnostic_sha256": new_hash,
            "only_source_change": "accepted timestep choices",
            "prior_source_reconstructed_and_hash_verified": True,
            "all_other_identity_fields": "exact_match",
        })
    else:
        require(coarse["identity"] == fine["identity"], "Run identities differ")
    left, right = coarse["condition"], fine["condition"]
    require(set(left) == set(right), "Condition schemas differ")
    require(
        {key for key in left if left[key] != right[key]}
        == {"dt_ms", "simulation_timesteps"},
        "Conditions changed beyond timestep and simulation length",
    )
    require(left["dt_ms"] == 2 * right["dt_ms"], "Expected half timestep")
    require(
        coarse["sample_metadata"] == fine["sample_metadata"],
        "Input metadata differ",
    )
    require(
        coarse["initial_values_source"]["sha256"]
        == fine["initial_values_source"]["sha256"],
        "Initializer artifacts differ",
    )
    require(coarse["initial_weights"] == fine["initial_weights"]
            and coarse["initial_weight_sha256"] == fine["initial_weight_sha256"],
            "Initial weights differ")

    counts = []
    for result in (coarse, fine):
        condition = result["condition"]
        require(condition["scalar_precision"] == "double"
                and condition["strict_buffer_checking"] is True,
                "Expected double precision and strict tape checks")
        require(condition["simulation_timesteps"] * condition["dt_ms"]
                == condition["physical_horizon_ms"], "Horizon mismatch")
        require(result["optimizer_updates"] == 0
                and result["weights_and_moments_unchanged"] is True,
                "Run was not frozen")
        stats = result["limiter_statistics"]
        require(all(stats[key] == 0 for key in (
            "clipped_jumps", "invalid_limits", "nonfinite_raw_jumps"
        )), "Clipping or invalid replay values confound refinement")
        require(stats["bound_used_during_replay"] == [1.0],
                "Unexpected replay bound")
        pairs = np.asarray(result["hidden_spike_pairs"], dtype=np.float64)
        require(pairs.ndim == 2 and pairs.shape[1] == 2
                and np.isfinite(pairs).all(), "Invalid spike pairs")
        require(np.all((pairs[:, 0] >= 0) & (pairs[:, 0] < 392))
                and np.all(pairs[:, 0] == np.floor(pairs[:, 0]))
                and np.all((pairs[:, 1] >= 0)
                           & (pairs[:, 1] < condition["physical_horizon_ms"])),
                "Invalid spike IDs or times")
        count = np.bincount(pairs[:, 0].astype(np.int64), minlength=392)
        np.testing.assert_array_equal(count, result["hidden_spikes_per_neuron"])
        require(len(pairs) == result["hidden_spikes"] == stats["replayed_jumps"],
                "Spike/replay count mismatch")
        counts.append(count)

    changed = np.flatnonzero(counts[0] != counts[1])
    emit("SPIKE_COUNT_REFINEMENT", {
        "coarse_total": int(counts[0].sum()),
        "fine_total": int(counts[1].sum()),
        "changed_neurons": [
            {"id": int(i), "coarse": int(counts[0][i]),
             "fine": int(counts[1][i])} for i in changed
        ],
    })
    if len(changed) == 0:
        # Ordinal matching describes timing sensitivity only. It does not
        # prove which neuron caused a shared-weight gradient discrepancy.
        timing_rows = []
        for neuron in range(392):
            trains = [sorted(float(t) for n, t in r["hidden_spike_pairs"]
                             if n == neuron) for r in (coarse, fine)]
            if not trains[0]:
                continue
            differences = np.abs(np.asarray(trains[0]) - trains[1])
            ordinal = int(np.argmax(differences))
            timing_rows.append({
                "neuron": neuron, "filter": neuron % 8,
                "spike_count": len(trains[0]),
                "max_ordinal_time_difference_ms": float(differences[ordinal]),
                "ordinal_at_maximum": ordinal,
                "coarse_time_ms": trains[0][ordinal],
                "fine_time_ms": trains[1][ordinal],
            })
        emit("SPIKE_TIMING_LOCALIZATION", {
            "matching": "sorted spike ordinal within each neuron",
            "largest_timing_differences": sorted(
                timing_rows,
                key=lambda row: row["max_ordinal_time_difference_ms"],
                reverse=True,
            )[:10],
            "causal_gradient_attribution": False,
        })
    scores = [np.asarray(r["scores"], dtype=np.float64) for r in (coarse, fine)]
    require(all(s.shape == (2, 11) and np.isfinite(s).all() for s in scores),
            "Invalid score arrays")
    for score in scores:
        np.testing.assert_allclose(score[1], 0, rtol=0, atol=1e-7)
    emit("SCORE_REFINEMENT", {
        "decisions": [int(np.argmax(s[0])) for s in scores],
        "max_absolute_difference": float(np.max(np.abs(scores[0] - scores[1]))),
    })
    passed = []
    for projection, size in enumerate((400, 4312)):
        a, b = [np.asarray(r["gradients"][projection], dtype=np.float64)
                for r in (coarse, fine)]
        require(a.shape == b.shape == (size,)
                and np.isfinite(a).all() and np.isfinite(b).all(),
                "Invalid gradient arrays")
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        require(na > 0 and nb > 0, "Zero gradient norm")
        relative = float(np.linalg.norm(a - b) / nb)
        passed.append(relative <= 0.01)
        emit("GRADIENT_REFINEMENT", {
            "projection": projection,
            "coarse_l2": na, "fine_l2": nb,
            "relative_l2_change_to_finer": relative,
            "cosine_similarity": float(np.dot(a, b) / (na * nb)),
            "max_absolute_difference": float(np.max(np.abs(a - b))),
            "within_1pct": passed[-1],
        })
        if projection == 0:
            ka, kb = a.reshape(5, 5, 2, 8), b.reshape(5, 5, 2, 8)
            delta = ka - kb
            total_error_squared = float(np.sum(delta * delta))
            emit("CONV_FILTER_LOCALIZATION", {
                "logical_axes": ["kernel_y", "kernel_x", "polarity", "filter"],
                "filters": [
                    {
                        "filter": f,
                        "fine_gradient_l2": float(np.linalg.norm(kb[..., f])),
                        "difference_l2": float(np.linalg.norm(delta[..., f])),
                        "fraction_of_squared_gradient_difference": (
                            float(np.sum(delta[..., f] ** 2)) / total_error_squared
                            if total_error_squared else 0.0
                        ),
                    } for f in range(8)
                ],
                "largest_changed_kernel_entries": [
                    {
                        "yxpf": [int(v) for v in np.unravel_index(int(i), ka.shape)],
                        "coarse": float(a[i]), "fine": float(b[i]),
                        "difference": float(a[i] - b[i]),
                    }
                    for i in np.argsort(np.abs(a - b))[::-1][:10]
                ],
                "causal_neuron_attribution": False,
            })
    emit("REFINEMENT_ASSESSMENT", {
        "input_json_sha256": [hashlib.sha256(blob).hexdigest() for blob in blobs],
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "both_gradients_within_1pct": all(passed),
        "per_neuron_spike_counts_stable": len(changed) == 0,
        "finer_result_is_exact_reference": False,
        "full_window_training_ready": False,
        "scope": "one frozen example; local timestep sensitivity only",
    })
    require(all(passed), "Local gradient-refinement criterion exceeds 1 percent")


if __name__ == "__main__":
    main()
