"""Offline TASK 11 seed-42 extension plan; no GeNN imports or execution.

Reads existing development artifacts and cache bytes without preprocessing.
Writes one exclusive plan into a user-created, empty experiment directory.
Does not launch training, invoke Git, or read official test event arrays.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


WORKSPACE = Path("/its/home/ra679/Research")
PROJECT = WORKSPACE / "Projects/convnets_eventprop"
ML_PACKAGE = WORKSPACE / "Projects/ml_genn/ml_genn/ml_genn"
EXPERIMENTS = WORKSPACE / "Experiments"
SOURCE = EXPERIMENTS / "task10_full_epoch3_dfd6Fu6N/full_epoch_result.json"
RELOAD = EXPERIMENTS / "task10_selected_reload_nWZhWA2O/selected_validation_reload_result.json"
MANIFEST = EXPERIMENTS / "task10_full_data_audit_z6fVr34l/full_data_manifest.json"
SOURCE_SHA256 = "e8eb07c136472fadadf39ba9c186bba4bc5a45948e931752862d7dcf25b271e9"
MANIFEST_SHA256 = "7e38e35e70799557c9eabc9add60b8b5d5f0f7d84a0510a7a55ea811caa096a8"
CHECKPOINT_SHA256 = "0a0be046241d3a3cc0179fd593d7aea4700a0cf4582dd8f7faaca2a92bb14d3f"
CONFIRMED_HEAD = "ec4accecf9f0552c34e55d290e6a5db37b49f0fa"
ML_HEAD = "af16a5d64c19262743bcc019ff9963f6b03d77c5"
STATE_NAMES = ("hidden", "output", "adam_hidden_m", "adam_hidden_v",
               "adam_output_m", "adam_output_v")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def binding(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = file_hash(path)
    require(expected is None or digest == expected, f"Hash mismatch: {path}")
    return {"path": str(path), "sha256": digest}


def epoch_orders(last):
    rng = np.random.default_rng(42)
    return {epoch: rng.permutation(934).tolist() for epoch in range(1, last + 1)}


def check_state(checkpoint):
    binding(checkpoint["path"], checkpoint["sha256"])
    with np.load(checkpoint["path"], allow_pickle=False) as archive:
        require(set(archive.files) == set(STATE_NAMES), "Checkpoint schema changed")
        for name, size in zip(STATE_NAMES, (400, 4312, 400, 400, 4312, 4312)):
            values = archive[name]
            require(values.dtype == np.float64 and values.shape == (size,)
                    and np.isfinite(values).all(), f"Invalid checkpoint state: {name}")
            if name.endswith("_v"):
                require(np.all(values >= 0), f"Negative Adam variance: {name}")


def check_sources(artifact):
    for name, expected in artifact.get("helper_sha256", {}).items():
        require(Path(name).name == name, "Invalid helper name")
        binding(PROJECT / name, expected)
    for name, expected in artifact["identity"]["source_sha256"].items():
        require(Path(name).name == name, "Invalid source name")
        binding(PROJECT / name, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="Existing empty directory created with mktemp under Experiments")
    args = parser.parse_args()
    output = args.output.resolve(strict=True)
    require(output.is_dir() and output.parent == EXPERIMENTS.resolve()
            and output.name.startswith("task11_extension_plan_")
            and not any(output.iterdir()), "Use a new empty task11_extension_plan_* directory")

    source_binding = binding(SOURCE, SOURCE_SHA256)
    manifest_binding = binding(MANIFEST, MANIFEST_SHA256)
    source, manifest, reload = map(read_json, (SOURCE, MANIFEST, RELOAD))
    reload_binding = binding(RELOAD)
    expected_identity = source["identity"]
    current_identity = {"python": sys.version, "executable": sys.executable,
                        "numpy": np.__version__,
                        "ml_genn": importlib.metadata.version("ml_genn"),
                        "pygenn": importlib.metadata.version("pygenn")}
    for key, actual in current_identity.items():
        require(actual == expected_identity[key], f"Environment changed: {key}")
    require(expected_identity["ml_genn_commit"] == ML_HEAD, "Historical mlGeNN identity changed")
    for artifact in (source, reload):
        check_sources(artifact)
    binding(PROJECT / "dvs_eventprop_full_epoch_continue.py", source["diagnostic_sha256"])
    binding(PROJECT / "dvs_eventprop_selected_validation_reload.py", reload["diagnostic_sha256"])
    binding(PROJECT / "dvs_eventprop_full_data_audit.py", manifest["diagnostic_sha256"])
    binding(PROJECT / "dvs_data.py", manifest["loader_sha256"])
    binding(MANIFEST.parent / "sample_audit.jsonl", manifest["sample_audit_sha256"])
    require(manifest["artifact_type"] == "dvs-development-occ1024-cache-v1"
            and manifest["test_event_arrays_loaded"] == 0
            and manifest["pilot_parent_payloads_reproduced"] == 22,
            "Expected validated development cache")
    require(source["logical_epoch"] == 3 and source["cumulative_optimizer_updates"] == 90
            and source["checkpoint"]["sha256"] == CHECKPOINT_SHA256,
            "Unexpected extension starting state")

    condition = source["condition"]
    expected_condition = {
        "dt_ms": 2 ** -14, "simulation_timesteps": 16777216,
        "input_bin_ms": 1.0, "physical_horizon_ms": 1024.0,
        "profile": "dvs-occ1ms-prefix1024-v1", "batch_size": 32,
        "compiler_max_spikes": 8193, "scalar_precision": "double",
        "time_precision": "float", "strict_buffer_checking": True,
        "seed": 42, "learning_rate": 0.001, "adam_beta1": 0.9,
        "adam_beta2": 0.999, "adam_epsilon": 1e-8, "regulariser": None,
        "tau_mem_ms": 20.0, "tau_syn_ms": 5.0, "threshold": 1.0,
        "absolute_reset_voltage": 0.0, "readout": "avg_var_exp_weight",
        "loss": "sparse_categorical_crossentropy", "per_timestep_loss": False,
        "softmax_temperature": 1.0, "grad_limit": 100.0,
        "limiter_policy": "fixed bound 1; stop on clipping", "spike_recording": False,
    }
    require(condition == expected_condition, "Checked training condition changed")
    orders = epoch_orders(11)
    require(manifest["batch_plan"]["epoch1_training_permutation"] == orders[1],
            "Cache shuffle reference changed")

    # Check the recorded epoch chain, update ownership, and all saved checkpoints.
    chain = []
    epoch_path = SOURCE
    for epoch in (3, 2, 1):
        record = read_json(epoch_path)
        check_sources(record)
        producer = "dvs_eventprop_full_epoch_train.py" if epoch == 1 else "dvs_eventprop_full_epoch_continue.py"
        binding(PROJECT / producer, record["diagnostic_sha256"])
        require(record["logical_epoch"] == epoch and record["full_data_epoch_completed"] is True
                and record["fixed_bound_clips"] == 0 and record["test_performed"] is False
                and record["manifest_sha256"] == MANIFEST_SHA256
                and record["cumulative_optimizer_updates"] == 30 * epoch
                and record["optimizer_updates_this_run"] == 30
                and record["real_training_contributions"] == 934
                and record["training_permutation"] == orders[epoch],
                f"Epoch {epoch} coverage or clock changed")
        updates = record["update_records"]
        require(len(updates) == 30
                and [row["adam_step"] for row in updates] == list(range(30 * (epoch - 1), 30 * epoch))
                and [row["real_examples"] for row in updates] == [32] * 29 + [6]
                and [i for row in updates for i in row["training_indices"]] == orders[epoch],
                f"Epoch {epoch} update ownership changed")
        for row in updates:
            require(row["host_adam_weights_and_moments"] == "pass" and row["clipped_jumps"] == 0
                    and row["gradient_rescale"] == 32 / row["real_examples"]
                    and row["cumulative_optimizer_updates"] == row["adam_step"] + 1,
                    "Invalid historical update check")
            binding(row["checkpoint"]["path"], row["checkpoint"]["sha256"])
        check_state(record["checkpoint"])
        chain.append(binding(epoch_path))
        if epoch > 1:
            epoch_path = Path(record["source_training_result_path"])
            binding(epoch_path, record["source_training_result_sha256"])

    history = source["validation_history"]
    require([row["logical_epoch"] for row in history] == [1, 2, 3]
            and [row["correct"] for row in history] == [69, 84, 85], "Validation history changed")
    for row in history:
        require(row["real_examples"] == 143 and np.isfinite(row["mean_ce"]), "Invalid validation metrics")
        binding(row["checkpoint"]["path"], row["checkpoint"]["sha256"])
        binding(row["reference_arrays"]["path"], row["reference_arrays"]["sha256"])
    best = min(history, key=lambda row: (-row["correct"], row["logical_epoch"]))
    require(best == source["best_validation"] == reload["selected_validation"]
            and reload["source_result_sha256"] == SOURCE_SHA256
            and reload["manifest_sha256"] == MANIFEST_SHA256
            and reload["condition"] == condition and reload["checkpoint"] == source["checkpoint"]
            and reload["ready_for_controlled_epoch_extension"] is True
            and reload["validation_examples"] == 143 and reload["validation_correct"] == 85
            and reload["optimizer_updates"] == 0 and reload["fixed_bound_clips"] == 0
            and reload["all_validation_decisions_reproduced"] is True
            and reload["per_neuron_replay_counts"] == "exact_match"
            and reload["weights_and_fresh_moments_unchanged"] is True
            and reload["test_performed"] is False, "Selected reload evidence changed")
    reference_binding = binding(source["next_epoch_reference"]["path"],
                                source["next_epoch_reference"]["sha256"])
    reference = read_json(reference_binding["path"])
    require(reference["logical_epoch"] == 4 and reference["training_indices"] == orders[4][:32]
            and reference["checkpoint"] == source["checkpoint"], "Epoch-4 next-start reference changed")

    # Hash existing cache files only. No event arrays are loaded or transformed.
    cache_files = 0
    stable_ids = set()
    for split, count, subjects in (("train", 934, set(range(1, 21))),
                                   ("validation", 143, {21, 22, 23})):
        entries = manifest["entries"][split]
        require(len(entries) == count, f"Wrong {split} denominator")
        for index, entry in enumerate(entries):
            metadata = entry["metadata"]
            require(entry["index"] == index and metadata["split"] == split
                    and metadata["subject"] in subjects and 0 <= metadata["label"] <= 10
                    and metadata["stable_id"] not in stable_ids, "Development membership changed")
            stable_ids.add(metadata["stable_id"])
            cache_path = (MANIFEST.parent / entry["cache_path"]).resolve(strict=True)
            require(cache_path.is_relative_to(MANIFEST.parent), "Cache path escapes its directory")
            binding(cache_path, entry["cache_sha256"])
            cache_files += 1
    require(reference["labels"] == [manifest["entries"]["train"][i]["metadata"]["label"]
                                    for i in orders[4][:32]], "Next-start labels changed")
    for name, shape in (("scores", (32, 11)), ("hidden_counts", (32, 392))):
        values = np.asarray(reference[name])
        require(values.shape == shape and np.isfinite(values).all(), f"Invalid next-start {name}")
    for values, size in zip(reference["mean_gradients"], (400, 4312)):
        values = np.asarray(values)
        require(values.shape == (size,) and np.isfinite(values).all(), "Invalid next-start gradient")
    require(len(reference["mean_gradients"]) == 2, "Wrong next-start gradient coverage")

    plan = {
        "artifact_type": "task11-seed42-extension-plan-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "planner": binding(Path(__file__)), "runtime_identity": current_identity,
        "git_metadata": {"evidence_kind": "user-returned metadata before planner source addition; not reprobed",
            "convnets_branch": "feat/dvs-gesture-eventprop", "convnets_head": CONFIRMED_HEAD,
            "origin_branch_head": CONFIRMED_HEAD, "ml_genn_branch": "latency_spike_fixes",
            "ml_genn_head": ML_HEAD, "both_worktrees_clean_at_that_check": True},
        "source_result": source_binding, "selected_reload_result": reload_binding,
        "manifest": manifest_binding, "source_epoch_chain": list(reversed(chain)),
        "starting_checkpoint": source["checkpoint"], "starting_optimizer_updates": 90,
        "epoch4_next_start_reference": reference_binding, "condition": condition,
        "existing_project_source_bindings": [binding(path) for path in sorted(PROJECT.glob("dvs*.py"))],
        "local_ml_genn_source_bindings": [binding(path) for path in sorted(ML_PACKAGE.rglob("*.py"))],
        "provisional_total_epochs": 10, "next_review_block_epochs": [4],
        "launch_authorized_by_plan": False,
        "future_blocks_require_review": True, "additional_seeds_deferred": [43, 44],
        "training_permutations": {str(epoch): orders[epoch] for epoch in range(4, 11)},
        "epoch11_first_reference_order_if_epoch10_completes": orders[11][:32],
        "expected_per_epoch": {"real_training_contributions": 934, "adam_updates": 30,
            "real_batch_sizes": [32] * 29 + [6], "last_gradient_rescale": 32 / 6,
            "frozen_validation_parents": 143, "validation_optimizer_updates": 0,
            "next_start_reference_parents": 32},
        "validation_history_at_start": history, "best_validation_at_start": best,
        "selection_rule": source["selection_rule"],
        "continuation_policy": "Restore weights, Adam moments and metadata clock; fresh model/tapes/seed42. Not exact resume.",
        "scheduling_policy": "Unchanged paired real/empty presentations; same forward/replay weights; learn after real replay only; shared mutable callback context.",
        "metric_policy": "Online training scores use changing weights. Frozen validation uses final epoch weights. Fresh full-training evaluation must be separately identified.",
        "stop_conditions": ["Source/data/environment identity mismatch", "First-batch score/gradient/replay mismatch",
            "Any observed clipping, invalid adjoint, nonfinite state or tape failure",
            "Failed Adam weight/moment, clock, padding or contribution checks", "Sampled free GPU memory fraction below 0.20"],
        "review_each_epoch": ["Complete sample/permutation/update accounting and all runtime guards",
            "Validation correct count/CE, per-class decisions and earliest-tie best selection",
            "Online training CE/correct count labelled as online, activity/filter counts and silent-parent identities",
            "Measured elapsed times and sampled memory; samples do not certify exact peak headroom",
            "Hash-bound final checkpoint, validation reference and next-start reference; retain prior checked updates on failure"],
        "end_of_budget_policy": "Bind best-validation and final checkpoints separately, even when identical; verify fresh reload for both distinct states before any final evaluation request.",
        "resource_context": {"estimated_hours_per_epoch": 3, "estimated_hours_for_seven_more": 21,
            "runtime_guaranteed": False, "historical_process_memory_MiB_approx": 6466,
            "historical_sampled_free_fraction_approx": [0.50, 0.53], "peak_memory_measured": False},
        "known_silent_training_parent": "ibmGestureTrain/user04_fluorescent_led/10.npy; label 10; 399 input spikes; do not remove",
        "comparison_scope": "Genuine EventProp with fixed adjoint bound. Publication surrogate-gradient test accuracy and this development validation accuracy are not a controlled comparison.",
        "publication_differences": ["Binary occupancy rather than count input", "Fixed 1024ms prefix and explicit tails rather than full software gestures",
            "Architecture A with 4712 weights rather than the publication topology", "Filtered continuous LIF with absolute reset and LI weighted-voltage CE readout",
            "EventProp rather than surrogate gradients", "Training-derived subject validation and different budget/precision/hardware measurement conditions",
            "Source-compatible 10ms denoising differs from the paper's stated 1s"],
        "limitations": ["Numerical reference gates cover specific operating points, not every trained state",
            "Fixed-bound gradients are not globally unconstrained exact gradients", "No complete RNG/tape checkpoint or certified exact resume",
            "No established convergence or independent test accuracy"],
        "cache_files_rehashed": cache_files, "preprocessing_performed": False,
        "gpu_execution_performed": False, "test_event_arrays_loaded": 0,
        "held_out_evaluation_requires_fixed_plan_and_explicit_user_authorization": True,
    }
    destination = output / "extension_plan.json"
    with destination.open("x") as stream:
        json.dump(plan, stream, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")
    print("PASS_TASK11_EXTENSION_PLAN", json.dumps({
        "plan": binding(destination), "starting_epoch": 3, "starting_optimizer_updates": 90,
        "next_review_block_epochs": [4], "provisional_total_epochs": 10,
        "cache_files_rehashed": cache_files, "selected_validation_correct": best["correct"],
        "gpu_execution_performed": False, "test_event_arrays_loaded": 0}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
