"""Fresh-process reload of the validation-selected development checkpoint.

No learning or test arrays. Reproduce the selected epoch's complete validation
scores, decisions, per-neuron replay counts and individual gradients.
Readiness is scoped to extending the unchanged checked epoch-block protocol.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_full_epoch_continue as continuation

evaluation = continuation.evaluation
boundary, block, bench, pilot, r = (
    continuation.boundary, continuation.block, continuation.bench,
    continuation.pilot, continuation.r)
cache = continuation.cache
require, emit, pull = r.require, r.emit, r.pull


def select_best(history):
    return min(history, key=lambda row: (-row["correct"], row["logical_epoch"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    args = parser.parse_args()
    manifest_path, source_path = [p.resolve(strict=True) for p in (args.manifest, args.source_result)]
    manifest, source = [r.json.loads(p.read_text()) for p in (manifest_path, source_path)]
    identity = r.checkpoint.identity()
    require(source["artifact_type"] == continuation.ARTIFACT
            and source["diagnostic_sha256"] == r.file_hash(Path(continuation.__file__))
            and source["full_data_epoch_completed"] is True
            and source["logical_epoch"] >= 3 and source["fixed_bound_clips"] == 0
            and source["cumulative_optimizer_updates"] == 30 * source["logical_epoch"]
            and source["test_performed"] is False,
            "Expected at least three completed, checked development epochs")
    for key, value in identity.items():
        if key != "convnets_commit":
            require(source["identity"][key] == value, f"Source/environment changed: {key}")
    for name, digest in source["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Helper changed: {name}")
    require(source["manifest_sha256"] == r.file_hash(manifest_path)
            and manifest["artifact_type"] == cache.FORMAT
            and manifest["diagnostic_sha256"] == r.file_hash(Path(cache.__file__))
            and manifest["loader_sha256"] == identity["source_sha256"]["dvs_data.py"]
            and manifest["config"] == cache.data.Config(1024).record()
            and manifest["test_event_arrays_loaded"] == 0
            and r.file_hash(manifest_path.parent / "sample_audit.jsonl")
            == manifest["sample_audit_sha256"], "Development cache binding changed")
    require(source["selection_rule"] == "highest validation correct count; earliest epoch on ties",
            "Validation-selection policy changed")
    history = source["validation_history"]
    require([row["logical_epoch"] for row in history]
            == list(range(1, source["logical_epoch"] + 1)), "Validation history gaps")
    for row in history:
        require(row["real_examples"] == 143 and 0 <= row["correct"] <= 143
                and np.isfinite(row["mean_ce"])
                and r.file_hash(Path(row["checkpoint"]["path"])) == row["checkpoint"]["sha256"]
                and r.file_hash(Path(row["reference_arrays"]["path"])) == row["reference_arrays"]["sha256"],
                "Historical validation checkpoint/reference changed")
    best = select_best(history)
    require(best == source["best_validation"], "Recorded best checkpoint selection disagrees")
    # These controls exercise the earlier-best and earliest-tie cases that
    # the monotonically improving real history has not exercised.
    earlier = [{"logical_epoch": epoch, "correct": value}
               for epoch, value in enumerate((11, 10, 9), start=1)]
    tied = [{"logical_epoch": epoch, "correct": value}
            for epoch, value in enumerate((9, 11, 11), start=1)]
    require(select_best(earlier)["logical_epoch"] == 1
            and select_best(tied)["logical_epoch"] == 2, "Best-selection control failed")
    emit("BEST_SELECTION_CHECK", {"actual_selected_epoch": best["logical_epoch"],
         "actual_history_recomputed": "pass", "earlier_best_control": "pass",
         "earliest_tie_control": "pass"})
    checkpoint = best["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    reference_path = Path(best["reference_arrays"]["path"]).resolve(strict=True)
    require(checkpoint_path.name == f"update_{30 * best['logical_epoch']:04d}.npz",
            "Selected checkpoint clock/name mismatch")
    require(Path.cwd().resolve() not in (
        source_path.parent, manifest_path.parent, checkpoint_path.parent, reference_path.parent),
        "Use a new experiment directory")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(block.STATE_NAMES), "Checkpoint schema changed")
        saved = [np.array(archive[name], copy=True) for name in block.STATE_NAMES]
    for values, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(values.dtype == np.float64 and values.shape == (size,)
                and np.isfinite(values).all(), "Invalid selected checkpoint state")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Invalid saved Adam variance")
    with np.load(reference_path, allow_pickle=False) as archive:
        names = ("indices", "labels", "scores", "hidden_counts",
                 "individual_hidden_gradients", "individual_output_gradients")
        require(set(archive.files) == set(names), "Validation reference schema changed")
        expected = {name: np.array(archive[name], copy=True) for name in names}
    np.testing.assert_array_equal(expected["indices"], np.arange(143))
    entries = manifest["entries"]["validation"]
    require(len(entries) == 143, "Validation denominator changed")
    for index, entry in enumerate(entries):
        metadata = entry["metadata"]
        require(entry["index"] == index and metadata["split"] == "validation"
                and 21 <= metadata["subject"] <= 23, "Unexpected validation parent")
    np.testing.assert_array_equal(expected["labels"], [e["metadata"]["label"] for e in entries])
    for name, shape in (("scores", (143, 11)), ("hidden_counts", (143, 392)),
                        ("individual_hidden_gradients", (143, 400)),
                        ("individual_output_gradients", (143, 4312))):
        require(expected[name].shape == shape and np.isfinite(expected[name]).all(),
                f"Invalid validation reference: {name}")
    old_metrics = boundary.score_summary(expected["scores"], expected["labels"])
    require(old_metrics["correct"] == best["correct"], "Reference decisions/history disagree")
    np.testing.assert_allclose(old_metrics["mean_ce"], best["mean_ce"], rtol=1e-12, atol=1e-12)
    condition = source["condition"]
    require(condition["dt_ms"] == 0.00006103515625
            and condition["simulation_timesteps"] == 16777216
            and condition["compiler_max_spikes"] == 8193 and condition["batch_size"] == 32
            and condition["physical_horizon_ms"] == 1024.0
            and condition["scalar_precision"] == "double" and condition["time_precision"] == "float"
            and condition["strict_buffer_checking"] is True, "Evaluation condition changed")
    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    network, inp, hid, out = r.build_model(config, 32)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, saved[:2], ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = values.reshape(shape).copy()
    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=16777216, losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=32, rng_seed=42, max_spikes=8193,
        strict_buffer_checking=True, per_timestep_loss=False,
        grad_limit=100.0, softmax_temperature=1.0)
    cn = compiler.compile(network, name="SelectedValidationReload", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)}}, regularisers={})
    context = {"split": "validation", "parents": {
        i: entry["metadata"]["stable_id"] for i, entry in enumerate(entries)}}
    raw = bench.RawCapture(layers, context, 32)
    capture = evaluation.FreezeEvaluation(layers, hid, out, context, 32)
    found = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in found:
            found[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                modified.extend((raw, cb))
            elif cb.name == "GradientLearn":
                modified.append(capture)
            else:
                modified.append(r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False))
        else:
            modified.append(cb)
    require(all(n == 1 for n in found.values()) and len(compiler.instrumentation) == 1,
            "Callback/audit layout changed")
    cn.base_train_callbacks[:] = modified
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "source_result_path": str(source_path), "source_result_sha256": r.file_hash(source_path),
        "manifest_path": str(manifest_path), "manifest_sha256": r.file_hash(manifest_path),
        "selected_validation": best, "checkpoint": checkpoint, "condition": condition,
        "helper_sha256": {Path(module.__file__).name: r.file_hash(Path(module.__file__)) for module in
            (continuation, evaluation, boundary, cache, block, bench, pilot, bench.mixed, pilot.candidate)},
        "operation": "selected weights-only reload; fresh zero Adam moments", "exact_resume": False,
        "selection_rule": source["selection_rule"], "test_performed": False}
    block.write_json("selected_reload_plan.json", provenance)
    emit("SELECTED_VALIDATION_RELOAD_PLAN", provenance)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    metric = bench.mixed.MaskedAccuracy()
    state = {out: metric.create_state()}
    scores = []
    with cn:
        cn.get_readout(out)
        initial = r.checkpoint.training_state(cn, layers)
        require(all(values.dtype == np.float64 for values in initial), "Precision changed")
        for actual, target in zip(initial[:2], saved[:2]):
            np.testing.assert_array_equal(actual.reshape(-1), target)
        require(all(np.all(values == 0) for values in initial[2:]), "Fresh moments not zero")
        observer.snapshot("after_load")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient GPU headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=10, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for start in range(0, 143, 32):
            indices = list(range(start, min(start + 32, 143)))
            count = len(indices)
            loaded = [cache.load_cached_sample(manifest_path.parent, entries[i]) for i in indices]
            targets = np.array([row[1] for row in loaded] + [0] * (32 - count), dtype=np.int64)
            inputs = [r.to_ml_genn(row[0]) for row in loaded] + [empty] * (32 - count)
            context.update({"indices": indices + [-1] * (32 - count), "targets": targets,
                "forward_state": r.checkpoint.training_state(cn, layers)})
            for drain in (False, True):
                metric.real_mask = np.zeros(32, dtype=bool) if drain else np.arange(32) < count
                with observer.phase(f"validation/presentation_{batch}"):
                    cn._train_batch(batch, 0, {inp: [empty] * 32 if drain else inputs},
                        {out: np.zeros(32, dtype=np.int64) if drain else targets}, {out: metric}, state, callbacks)
                    value = np.array(cn.get_readout(out), copy=True)
                require(value.shape == (32, 11) and np.isfinite(value).all(), "Invalid scores")
                if drain:
                    np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)
                else:
                    scores.extend(value[:count].tolist())
                block.same_state(initial, r.checkpoint.training_state(cn, layers))
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                batch += 1
            observer.snapshot(f"after_validation_{start + count}")
            require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient sampled GPU headroom")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        block.same_state(initial, r.checkpoint.training_state(cn, layers))
    require(batch == 10 and len(capture.records) == 5 and state[out].total == 143
            and [i for row in capture.records for i in row["indices"]] == list(range(143)),
            "Validation coverage changed")
    scores = np.asarray(scores)
    np.testing.assert_allclose(scores, expected["scores"], rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(scores.argmax(axis=1), expected["scores"].argmax(axis=1))
    counts = np.concatenate([row["counts"] for row in capture.records])
    np.testing.assert_array_equal(counts, expected["hidden_counts"])
    differences = []
    for projection, name in enumerate(("individual_hidden_gradients", "individual_output_gradients")):
        actual = np.concatenate([row["individual"][projection] for row in capture.records])
        np.testing.assert_allclose(actual, expected[name], rtol=1e-9, atol=1e-11)
        differences.append(float(np.max(np.abs(actual - expected[name]))))
    metrics = boundary.score_summary(scores, expected["labels"])
    require(metrics["correct"] == best["correct"], "Reload accuracy changed")
    np.testing.assert_allclose(metrics["mean_ce"], best["mean_ce"], rtol=1e-9, atol=1e-11)
    passed = {"selected_epoch": best["logical_epoch"], "validation_examples": 143,
        "validation_correct": metrics["correct"], "validation_mean_ce": metrics["mean_ce"],
        "max_absolute_score_difference": float(np.max(np.abs(scores - expected["scores"]))),
        "individual_gradient_max_absolute_difference_hidden_output": differences,
        "all_validation_decisions_reproduced": True, "per_neuron_replay_counts": "exact_match",
        "optimizer_updates": 0, "fixed_bound_clips": 0,
        "weights_and_fresh_moments_unchanged": True, "exact_training_resume_certified": False,
        "ready_for_controlled_epoch_extension": True,
        "readiness_scope": "unchanged checked 1024ms/B32/double/paired/fixed-bound protocol in reviewed epoch blocks",
        "all_training_state_gradient_exactness_certified": False,
        "test_performed": False, "minimum_sampled_free_fraction": min(s["free_fraction"] for s in observer.snapshots)}
    block.write_json("selected_validation_reload_result.json", {
        **provenance, **passed, "scores": scores.tolist(), "labels": expected["labels"].tolist(),
        "memory_snapshots": observer.snapshots, "phases": observer.phases})
    emit("PASS_SELECTED_VALIDATION_RELOAD", passed)


if __name__ == "__main__":
    main()
