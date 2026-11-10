"""Frozen forward-response check after changing active predecessors."""

import argparse
import json
from pathlib import Path

import numpy as np

import dvs_eventprop_capacity_probe as capacity
import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_output_gradient_probe as output_probe
from dvs_data import (
    Config as DataConfig,
    file_hash,
    inventory,
    load_sample,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--order", choices=("BAC", "ACB"), default="BAC")
    args = parser.parse_args()

    reference_path = args.reference.resolve(strict=True)
    reference = json.loads(reference_path.read_text())
    current = checkpoint.identity()

    # Explicit cross-revision comparison using unchanged core sources.
    for key, value in current.items():
        if key != "convnets_commit":
            require(
                reference["identity"][key] == value,
                f"Reference source/environment mismatch: {key}",
            )
    require(
        reference["identity"]["capacity_helper_sha256"]
        == file_hash(Path(capacity.__file__).resolve()),
        "Frozen runner changed",
    )
    require(
        reference["identity"]["spike_helper_sha256"]
        == file_hash(Path(output_probe.__file__).resolve()),
        "Spike helper changed",
    )
    require(reference["recording"] == "on", "Expected recording-on reference")
    require(reference["optimizer_updates"] == 0, "Reference was not frozen")

    config = ProbeConfig()
    root = args.root.resolve(strict=True)
    rows = pilot(inventory(root)["train"], 2)[:3]
    loaded = [load_sample(root, row, DataConfig(128)) for row in rows]
    payloads, labels, metadata = zip(*loaded)

    expected_condition = {
        **config.record(),
        "batch_size": 1,
        "real_examples": 3,
        "empty_drains": 1,
        "training_order": [row.stable_id for row in rows],
    }
    require(
        reference["condition"] == normalized(expected_condition),
        "Reference condition/order mismatch",
    )
    require(
        reference["sample_metadata"] == normalized(list(metadata)),
        "Reference input/provenance mismatch",
    )

    initial = [
        np.asarray(value, dtype=np.float32)
        for value in reference["initial_weights"]
    ]
    require(len(initial) == 2, "Expected two projections")
    for value, size, digest in zip(
        initial, (400, 4312), reference["initial_weight_sha256"]
    ):
        require(value.shape == (size,), "Unexpected saved weight shape")
        require(np.isfinite(value).all(), "Nonfinite initial weights")
        require(capacity.array_hash(value) == digest, "Weight hash mismatch")

    order = {"BAC": [1, 0, 2], "ACB": [0, 2, 1]}[args.order]
    emit("PREDECESSOR_PLAN", {
        "current": current,
        "reference_git_head": reference["identity"]["convnets_commit"],
        "reference_json_sha256": file_hash(reference_path),
        "diagnostic_sha256": file_hash(Path(__file__).resolve()),
        "capacity_helper_sha256": file_hash(Path(capacity.__file__).resolve()),
        "reference_order": [0, 1, 2],
        "new_order": order,
        "training_parent_order": [rows[i].stable_id for i in order],
        "optimizer_updates": 0,
        "first_example_replay_gradient_parity_required": args.order == "ACB",
    })

    result = capacity.run(
        config,
        257,
        [to_ml_genn(payloads[i]) for i in order],
        np.asarray([labels[i] for i in order], dtype=np.int64),
        initial_weights=initial,
    )

    baseline_scores = np.asarray(reference["scores"], dtype=np.float64)
    baseline_counts = reference["hidden_spikes_per_presentation"]
    actual_scores = np.asarray(result["scores"], dtype=np.float64)
    require(
        baseline_scores.shape == actual_scores.shape == (4, 11),
        "Unexpected score shape",
    )
    require(len(baseline_counts) == 4, "Unexpected reference spike counts")

    maximum = 0.0
    for presentation, original_index in enumerate(order):
        actual = actual_scores[presentation]
        expected = baseline_scores[original_index]
        difference = float(np.max(np.abs(actual - expected)))
        maximum = max(maximum, difference)
        hidden_count = len(result["hidden"][presentation])

        emit("PREDECESSOR_COMPARISON", {
            "original_example_index": original_index,
            "new_presentation_index": presentation,
            "max_absolute_score_difference": difference,
            "reference_hidden_spikes": baseline_counts[original_index],
            "new_hidden_spikes": hidden_count,
        })
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        require(
            int(np.argmax(actual)) == int(np.argmax(expected)),
            "Decision changed with predecessor",
        )
        require(
            hidden_count == baseline_counts[original_index],
            "Hidden spike count changed with predecessor",
        )

    np.testing.assert_allclose(actual_scores[-1], 0, rtol=0, atol=1e-7)
    require(result["hidden"][-1] == [], "Hidden spikes in empty drain")

    if args.order == "ACB":
        baseline = reference["per_example"][0]
        observed = result["records"][0]
        require(
            baseline["index"] == observed["index"] == 0,
            "Unexpected ownership of A's backward contribution",
        )
        np.testing.assert_array_equal(
            np.asarray(observed["limit"]),
            np.asarray(baseline["limit"]),
            err_msg="A's replay limit changed",
        )
        require(
            len(baseline["gradients"]) == len(observed["gradients"]) == 2,
            "Unexpected gradient projection count",
        )

        differences = []
        for projection, size in enumerate((400, 4312)):
            actual = np.asarray(
                observed["gradients"][projection], dtype=np.float64
            ).reshape(-1)
            expected = np.asarray(
                baseline["gradients"][projection], dtype=np.float64
            ).reshape(-1)
            require(
                actual.shape == expected.shape == (size,),
                "Unexpected gradient shape",
            )
            require(
                np.isfinite(actual).all() and np.isfinite(expected).all(),
                "Nonfinite gradient",
            )
            difference = float(np.max(np.abs(actual - expected)))
            differences.append(difference)
            emit("SUCCESSOR_REPLAY_GRADIENT", {
                "backward_example": "A",
                "reference_forward_successor": "B",
                "new_forward_successor": "C",
                "projection": projection,
                "max_absolute_difference": difference,
                "replay_limit": observed["limit"].tolist(),
            })
            np.testing.assert_allclose(
                actual, expected, rtol=2e-5, atol=2e-7,
                err_msg="A's gradient changed with forward successor",
            )

        emit("PASS_SUCCESSOR_REPLAY_ISOLATION", {
            "backward_example": "A",
            "successors_compared": ["B", "C"],
            "gradient_max_absolute_difference_hidden_output": differences,
            "weights_and_moments_unchanged": True,
            "replay_limits_identical": True,
            "optimizer_updates": 0,
            "parameter_staleness_tested": False,
            "eventprop_gradient_correctness_certified": False,
        })

    emit("PASS_PREDECESSOR_FORWARD_CHECK", {
        "max_absolute_score_difference": maximum,
        "real_examples": 3,
        "optimizer_updates": 0,
        "weights_and_moments_unchanged": True,
        "scope": f"ABC versus {args.order}; frozen B1, dt1ms, 128ms",
        "hidden_spike_counts_match": True,
        "hidden_spike_id_time_parity_certified": False,
        "gradient_order_parity_certified": False,
        "all_predecessor_cases_certified": False,
    })


if __name__ == "__main__":
    main()