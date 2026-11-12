"""Fresh-process weights-only reload of the successful B32 pilot checkpoint.

Reuses the checked frozen B32 evaluator. No learning, validation or test events.
Saved Adam moments are inspected but not restored.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_b32_pilot_block as block_runner

previous = block_runner.previous
bench, pilot, r = previous.bench, previous.pilot, previous.r
require, emit = r.require, r.emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-block", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        path.resolve(strict=True)
        for path in (args.baseline, args.pilot_block, args.result)
    ]
    baseline, metadata_source, source = [
        r.json.loads(path.read_text()) for path in paths
    ]
    identity = r.checkpoint.identity()
    for artifact in (baseline, metadata_source, source):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(
                    artifact["identity"][key] == value,
                    f"Source/environment mismatch: {key}",
                )

    require(
        source["artifact_type"] == block_runner.ARTIFACT
        and source["diagnostic_sha256"]
        == r.file_hash(Path(block_runner.__file__))
        == "62c63efa1fd7f2eabc80b72cf28107bb04c64a7d73c06e177fe5f12f86231ca5",
        "Pilot block producer changed",
    )
    require(
        r.file_hash(Path(bench.__file__))
        == "0242d7ac5138b253baa9dd8d36c0ba0d56c4dcad6622d163036a0c3dce1b6ad8"
        and r.file_hash(Path(previous.__file__))
        == "8574fcaafd3623a8cb5c4c92af94724bad2ccce850b49d3e4e633386d47766bc",
        "Frozen evaluator or metric helper changed",
    )
    require(
        metadata_source["diagnostic_sha256"]
        == r.file_hash(
            Path(__file__).with_name("dvs_eventprop_pilot_block.py")
        )
        and metadata_source["pilot_producer_sha256"]
        == r.file_hash(Path(pilot.__file__)),
        "Metadata producer changed",
    )
    require(
        source["source_block_sha256"] == r.file_hash(paths[1])
        and source["source_condition"] == metadata_source["condition"]
        and source["training_order"] == metadata_source["training_order"],
        "Pilot metadata binding changed",
    )
    require(
        source["baseline_sha256"]
        == metadata_source["baseline_sha256"]
        == r.file_hash(paths[0])
        and baseline["identity"]["diagnostic_sha256"]
        == r.file_hash(Path(pilot.candidate.__file__)),
        "Baseline or precision adapter changed",
    )
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(
            r.file_hash(Path(__file__).with_name(name)) == digest,
            f"Baseline helper changed: {name}",
        )

    require(
        source["pilot_overfit_gate_met"] is True
        and source["stop_reason"] == "pilot_target_reached"
        and source["fixed_bound_clips"] == 0
        and source["batch_size"] == 32
        and source["effective_real_batch_size"] == 22,
        "Expected the successful, unclipped B32 pilot",
    )
    total_updates = source["cumulative_optimizer_updates"]
    expected = source["phase_results"]["fresh_evaluation"]
    checkpoint = source["checkpoint"]
    require(
        expected["scores_at_cumulative_updates"] == total_updates
        and expected["optimizer_updates_in_phase"] == 0
        and expected["score_parameter_state"] == "fixed"
        and expected["following_replay_applies_update"] is False
        and expected["checkpoint"] == checkpoint
        and expected["correct"] >= 21,
        "Final scores/checkpoint do not describe the passed pilot state",
    )
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(
        checkpoint_path.parent == paths[2].parent
        and checkpoint_path.name == f"update_{total_updates:04d}.npz"
        and r.file_hash(checkpoint_path) == checkpoint["sha256"],
        "Checkpoint binding changed",
    )
    require(
        Path.cwd().resolve() != paths[2].parent,
        "Use a new experiment directory",
    )
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(
            set(archive.files) == set(block_runner.STATE_NAMES),
            "Checkpoint schema changed",
        )
        saved = [
            np.array(archive[name], copy=True)
            for name in block_runner.STATE_NAMES
        ]
    for values, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(
            values.dtype == np.float64
            and values.shape == (size,)
            and np.isfinite(values).all(),
            "Invalid checkpoint state",
        )
    require(
        np.all(saved[3] >= 0) and np.all(saved[5] >= 0),
        "Invalid saved Adam variance",
    )

    condition = source["source_condition"]
    require(
        condition["frames"] == 1024
        and condition["dt_ms"] == 0.00006103515625
        and condition["simulation_timesteps"] == 16777216
        and condition["compiler_max_spikes"] == 8193
        and condition["scalar_precision"] == "double"
        and condition["time_precision"] == "float"
        and condition["strict_buffer_checking"] is True,
        "Unexpected physical condition",
    )
    root = args.root.resolve(strict=True)
    by_id = {row.stable_id: row for row in r.inventory(root)["train"]}
    rows = [by_id[key] for key in source["training_order"]]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(
        len(rows) == len(set(source["training_order"])) == 22
        and r.normalize(list(metadata)) == metadata_source["sample_metadata"],
        "Pilot input provenance changed",
    )
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(labels, expected["labels"])

    reference = source["fresh_gradient_reference"]
    counts = np.asarray(reference["hidden_counts"])
    require(
        counts.shape == (22, 392)
        and np.isfinite(counts).all()
        and np.all(counts >= 0),
        "Invalid reference replay counts",
    )
    require(
        len(reference["full_real_mean_gradients"]) == 2,
        "Invalid gradient reference coverage",
    )
    for gradient, size in zip(
        reference["full_real_mean_gradients"], (400, 4312)
    ):
        values = np.asarray(gradient)
        require(
            values.shape == (size,) and np.isfinite(values).all(),
            "Invalid reference gradient",
        )

    provenance = {
        "identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)),
        "source_result_sha256": r.file_hash(paths[2]),
        "source_checkpoint": checkpoint,
        "source_training_updates": total_updates,
        "source_condition": condition,
        "sample_metadata": metadata,
        "helper_sha256": {
            Path(module.__file__).name: r.file_hash(Path(module.__file__))
            for module in (
                block_runner, previous, bench, pilot,
                bench.mixed, pilot.candidate,
            )
        },
        "operation": "weights-only reload; fresh zero Adam moments",
        "cross_revision_basis": "matching checked source and environment",
        "exact_resume": False,
        "validation_performed": False,
        "test_performed": False,
    }
    emit("B32_PILOT_RELOAD_PLAN", provenance)

    # The checked evaluator verifies exact restored weights, fresh zero
    # moments, unchanged final state, zero clipping and metric coverage.
    result = bench.run(
        32, payloads, labels, saved[:2], condition,
        expected["scores"], counts.sum(axis=1), provenance,
    )
    np.testing.assert_array_equal(result["hidden_counts"], counts)
    differences = []
    for projection in range(2):
        actual = np.asarray(result["full_real_mean_gradients"][projection])
        target = np.asarray(reference["full_real_mean_gradients"][projection])
        np.testing.assert_allclose(
            actual, target, rtol=1e-9, atol=1e-11
        )
        differences.append(float(np.max(np.abs(actual - target))))

    scores = np.asarray(result["scores"])
    expected_scores = np.asarray(expected["scores"])
    np.testing.assert_array_equal(
        scores.argmax(axis=1), expected_scores.argmax(axis=1)
    )
    summary = previous.score_summary(scores, labels)
    require(summary["correct"] == expected["correct"], "Decisions changed")
    np.testing.assert_allclose(
        summary["mean_ce"], expected["mean_ce"], rtol=1e-9, atol=1e-11
    )
    passed = {
        "real_examples": 22,
        "correct": summary["correct"],
        "mean_ce": summary["mean_ce"],
        "max_absolute_score_difference": float(
            np.max(np.abs(scores - expected_scores))
        ),
        "gradient_max_absolute_difference_hidden_output": differences,
        "all_decisions_reproduced": True,
        "per_neuron_replay_counts": "exact_match",
        "optimizer_updates": 0,
        "weights_and_fresh_moments_unchanged": True,
        "fixed_bound_clips": 0,
        "pilot_overfit_gate_reproduced": True,
        "full_window_training_ready": False,
        "exact_training_resume_certified": False,
    }
    with (Path.cwd() / "b32_pilot_reload_result.json").open("x") as stream:
        r.json.dump(
            {**provenance, **passed, "scores": scores.tolist()},
            stream, sort_keys=True, allow_nan=False,
        )
    emit("PASS_B32_PILOT_RELOAD", passed)


if __name__ == "__main__":
    main()