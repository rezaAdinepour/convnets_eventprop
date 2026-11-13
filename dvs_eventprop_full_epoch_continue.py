"""One controlled continuation epoch, frozen validation, and next-start reference.

Restores weights, Adam moments and update clock; fresh model/tapes/seed42.
Not exact resume. B32, full 934/143 coverage, fixed bound 1, paired scheduling.
Best validation accuracy wins; ties retain the earliest epoch. No test arrays.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_full_epoch_evaluate as evaluation

training = evaluation.training
boundary, block, bench, pilot, r = (
    training.boundary, training.block, training.bench, training.pilot, training.r)
cache = training.cache
require, emit, pull = r.require, r.emit, r.pull
ARTIFACT = "full-development-controlled-epoch-v1"


def epoch_order(epoch):
    rng = np.random.default_rng(42)
    for _ in range(epoch):
        order = rng.permutation(934).tolist()
    return order


def history_row(epoch, summary):
    require(summary["real_examples"] == 143 and summary["optimizer_updates"] == 0
            and summary["score_parameter_state"] == "fixed", "Expected frozen validation metrics")
    return {"logical_epoch": epoch, **{key: summary[key] for key in (
        "checkpoint", "correct", "mean_ce", "real_examples", "reference_arrays")}}


class LearnOrFreeze(evaluation.FreezeEvaluation):
    def on_batch_end(self, cn, batch, metrics):
        ctx = self.context
        before = r.checkpoint.training_state(cn, self.layers)
        super().on_batch_end(cn, batch, metrics)
        if batch % 2 == 0:
            return
        record = self.records[-1]
        if ctx["mode"] != "training":
            return
        step = ctx["updates"]
        local_step = len(ctx["update_records"])
        require(ctx["step"] == step == ctx["start_updates"] + local_step
                and batch == 2 * local_step + 1, "Learning ownership or Adam clock changed")
        require(record["indices"] == [i for i in ctx["indices"] if i >= 0]
                and record["count"] in (6, 32), "Real contribution coverage changed")
        if local_step == 0:
            expected = ctx["first_reference"]
            np.testing.assert_array_equal(record["counts"], expected["hidden_counts"])
            for actual, target in zip(record["mean"], expected["mean_gradients"]):
                np.testing.assert_allclose(actual, target, rtol=1e-9, atol=1e-11)
            emit("CONTINUATION_FIRST_REPLAY_PARITY", {
                "mean_gradients": "pass", "per_neuron_replay_counts": "exact_match",
                "adam_step": step, "restored_optimizer_moments": "checked",
            })
        for variable, gradient in zip(block.reduced_variables(cn), record["mean"]):
            require(np.isfinite(gradient).all(), "Nonfinite normalized gradient")
            variable.view[...] = gradient.reshape(variable.view.shape)
            variable.push_to_device()
        cn.custom_update("GradientLearn")
        changes = []
        for index, (weight, gradient) in enumerate(zip(before[:2], record["mean"])):
            m, v = ctx["moments"][index]
            m, v = 0.9 * m + 0.1 * gradient, 0.999 * v + 0.001 * gradient ** 2
            expected = weight.reshape(-1) - 0.001 * (m / (1 - 0.9 ** (step + 1))) / (
                np.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
            actual = pull(cn.connection_populations[
                self.layers[index].connection()].vars["weight"]).reshape(-1)
            require(np.isfinite(actual).all(), "Nonfinite updated weights")
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
            matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == gradient.size]
            require(len(matches) == 1, "Ambiguous optimizer moment ownership")
            for key, values in (("M", m), ("V", v)):
                np.testing.assert_allclose(pull(matches[0].vars[key]).reshape(-1),
                                           values, rtol=2e-10, atol=2e-12)
            ctx["moments"][index] = (m, v)
            changes.append(float(np.linalg.norm(actual - weight.reshape(-1))))
        ctx["updates"] = step + 1
        ctx["checkpoint"] = pilot.save_state(cn, self.layers, f"update_{step + 1:04d}.npz")
        row = {"logical_epoch": ctx["epoch"], "adam_step": step,
            "cumulative_optimizer_updates": step + 1, "training_indices": record["indices"],
            "real_examples": record["count"], "gradient_rescale": 32.0 / record["count"],
            "online_metrics": ctx["online_metrics"], "scores_at_optimizer_updates": step,
            "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in record["mean"]],
            "maximum_replayed_spikes_per_neuron": int(record["counts"].max()),
            "fully_silent_hidden_examples": int(np.sum(record["counts"].sum(axis=1) == 0)),
            "replay_spikes_per_filter": record["counts"].reshape(
                record["count"], 7, 7, 8).sum(axis=(0, 1, 2)).astype(int).tolist(),
            "weight_change_l2_hidden_output": changes, "checkpoint": ctx["checkpoint"],
            "forward_replay_state": "exact_match", "host_adam_weights_and_moments": "pass",
            "clipped_jumps": 0}
        block.write_json(f"update_{step + 1:04d}.json", {
            **row, "forward_scores": ctx["forward_scores"].tolist(),
            "labels": ctx["targets"][:record["count"]].tolist(),
            "normalized_mean_gradients": [g.tolist() for g in record["mean"]]})
        ctx["update_records"].append(row)
        ctx["contributions"].extend(record["indices"])
        emit("CONTINUATION_UPDATE", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-training-result", type=Path, required=True)
    parser.add_argument("--source-evaluation-result", type=Path)
    args = parser.parse_args()
    manifest_path, source_path = [p.resolve(strict=True) for p in (
        args.manifest, args.source_training_result)]
    manifest, source = [r.json.loads(p.read_text()) for p in (manifest_path, source_path)]
    identity = r.checkpoint.identity()
    source_type = source["artifact_type"]
    first_source = source_type == "full-development-training-epoch1-v1"
    require(first_source or source_type == ARTIFACT, "Unsupported source epoch artifact")
    producer = Path(training.__file__) if first_source else Path(__file__)
    require(source["diagnostic_sha256"] == r.file_hash(producer), "Source epoch producer changed")
    require(source["full_data_epoch_completed"] is True and source["fixed_bound_clips"] == 0,
            "Expected a passed full training epoch")
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
            and r.file_hash(manifest_path.parent / "sample_audit.jsonl")
            == manifest["sample_audit_sha256"], "Development cache binding changed")
    require(len(manifest["entries"]["train"]) == 934
            and len(manifest["entries"]["validation"]) == 143
            and manifest["test_event_arrays_loaded"] == 0, "Development coverage changed")
    epoch = source["logical_epoch"] + 1
    start_updates = source["cumulative_optimizer_updates"]
    require(epoch >= 2 and start_updates == 30 * (epoch - 1) and source["optimizer_updates_this_run"] == 30
            and source["training_permutation"] == epoch_order(epoch - 1),
            "Source epoch clock or permutation changed")
    order, next_order = epoch_order(epoch), epoch_order(epoch + 1)
    checkpoint = source["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(checkpoint_path.parent == source_path.parent
            and checkpoint_path.name == f"update_{start_updates:04d}.npz"
            and r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint binding changed")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(block.STATE_NAMES), "Checkpoint schema changed")
        saved = [np.array(archive[name], copy=True) for name in block.STATE_NAMES]
    for values, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(values.dtype == np.float64 and values.shape == (size,)
                and np.isfinite(values).all(), "Invalid saved training state")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Invalid Adam variance")
    evaluation_source = None
    if first_source:
        require(args.source_evaluation_result is not None, "Epoch one requires its frozen evaluation")
        evaluation_path = args.source_evaluation_result.resolve(strict=True)
        evaluated = r.json.loads(evaluation_path.read_text())
        require(evaluated["diagnostic_sha256"] == r.file_hash(Path(evaluation.__file__))
                and evaluated["training_result_sha256"] == r.file_hash(source_path)
                and evaluated["manifest_sha256"] == r.file_hash(manifest_path)
                and evaluated["checkpoint"] == checkpoint
                and evaluated["checkpoint_choice"] == "final"
                and evaluated["optimizer_updates"] == 0
                and evaluated["weights_and_fresh_moments_unchanged"] is True
                and evaluated["fixed_bound_clips"] == 0, "Frozen source evaluation binding changed")
        for key, value in identity.items():
            if key != "convnets_commit":
                require(evaluated["identity"][key] == value, f"Evaluation identity changed: {key}")
        for name, digest in evaluated["helper_sha256"].items():
            require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Evaluation helper changed: {name}")
        train_summary = evaluated["phase_results"]["train"]
        reference_path = Path(train_summary["reference_arrays"]["path"]).resolve(strict=True)
        require(reference_path.parent == evaluation_path.parent
                and r.file_hash(reference_path) == train_summary["reference_arrays"]["sha256"]
                and train_summary["checkpoint"] == checkpoint, "Frozen training reference changed")
        with np.load(reference_path, allow_pickle=False) as archive:
            indices = np.asarray(archive["indices"])
            np.testing.assert_array_equal(indices, source["training_permutation"])
            lookup = {int(index): position for position, index in enumerate(indices)}
            positions = [lookup[index] for index in order[:32]]
            first_reference = {
                "training_indices": order[:32], "checkpoint": checkpoint,
                "scores": np.array(archive["scores"][positions], copy=True).tolist(),
                "labels": np.array(archive["labels"][positions], copy=True).tolist(),
                "hidden_counts": np.array(archive["hidden_counts"][positions], copy=True).tolist(),
                "mean_gradients": [np.array(archive[name][positions], copy=True).mean(axis=0).tolist()
                    for name in ("individual_hidden_gradients", "individual_output_gradients")],
            }
        history = [history_row(1, evaluated["phase_results"]["validation"])]
        evaluation_source = {"path": str(evaluation_path), "sha256": r.file_hash(evaluation_path)}
    else:
        require(args.source_evaluation_result is None, "Continuation source already contains validation")
        warm_source = source["next_epoch_reference"]
        warm_path = Path(warm_source["path"]).resolve(strict=True)
        require(warm_path.parent == source_path.parent and r.file_hash(warm_path) == warm_source["sha256"],
                "Next-start reference binding changed")
        first_reference = r.json.loads(warm_path.read_text())
        require(first_reference["logical_epoch"] == epoch, "Wrong next-start epoch")
        history = source["validation_history"]
        require(history[-1] == history_row(epoch - 1, source["phase_results"]["validation"]),
                "Latest validation/history binding changed")
    require(first_reference["training_indices"] == order[:32]
            and first_reference["checkpoint"] == checkpoint, "Wrong first-batch reference")
    require(np.asarray(first_reference["scores"]).shape == (32, 11)
            and np.asarray(first_reference["hidden_counts"]).shape == (32, 392)
            and np.asarray(first_reference["labels"]).shape == (32,)
            and len(first_reference["mean_gradients"]) == 2,
            "Invalid first-batch reference coverage")
    for values, size in zip(first_reference["mean_gradients"], (400, 4312)):
        values = np.asarray(values)
        require(values.shape == (size,) and np.isfinite(values).all(),
                "Invalid first-batch gradient reference")
    require([row["logical_epoch"] for row in history] == list(range(1, epoch)), "Validation history gaps")
    for row in history:
        require(row["real_examples"] == 143 and 0 <= row["correct"] <= 143
                and np.isfinite(row["mean_ce"])
                and r.file_hash(Path(row["checkpoint"]["path"])) == row["checkpoint"]["sha256"]
                and r.file_hash(Path(row["reference_arrays"]["path"])) == row["reference_arrays"]["sha256"],
                "Historical validation checkpoint/reference changed")
    condition = source["condition"]
    require(condition["dt_ms"] == 0.00006103515625
            and condition["simulation_timesteps"] == 16777216
            and condition["compiler_max_spikes"] == 8193 and condition["batch_size"] == 32
            and condition["scalar_precision"] == "double" and condition["time_precision"] == "float"
            and condition["physical_horizon_ms"] == 1024.0
            and condition["strict_buffer_checking"] is True
            and condition["learning_rate"] == 0.001 and condition["regulariser"] is None
            and condition["tau_mem_ms"] == 20.0 and condition["tau_syn_ms"] == 5.0
            and condition["absolute_reset_voltage"] == 0.0 and condition["threshold"] == 1.0
            and condition["adam_beta1"] == 0.9 and condition["adam_beta2"] == 0.999
            and condition["adam_epsilon"] == 1e-8
            and condition["loss"] == "sparse_categorical_crossentropy"
            and condition["per_timestep_loss"] is False and condition["softmax_temperature"] == 1.0
            and condition["readout"] == "avg_var_exp_weight",
            "Training condition changed")
    require(Path.cwd().resolve() not in (manifest_path.parent, source_path.parent), "Use a new experiment directory")
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
        strict_buffer_checking=True, per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0)
    cn = compiler.compile(network, name="ControlledFullDevelopmentEpoch", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)}}, regularisers={})
    context = {"epoch": epoch, "start_updates": start_updates, "updates": start_updates,
        "update_records": [], "contributions": [], "first_reference": first_reference,
        "moments": {0: (saved[2].copy(), saved[3].copy()), 1: (saved[4].copy(), saved[5].copy())}}
    raw = bench.RawCapture(layers, context, 32)
    learn = LearnOrFreeze(layers, hid, out, context, 32)
    found = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in found:
            found[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                modified.extend((raw, cb))
            elif cb.name == "GradientLearn":
                modified.append(learn)
            else:
                modified.append(r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False))
        else:
            modified.append(cb)
    require(all(n == 1 for n in found.values()) and len(compiler.instrumentation) == 1, "Callback layout changed")
    cn.base_train_callbacks[:] = modified
    provenance = {"artifact_type": ARTIFACT, "identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)), "logical_epoch": epoch,
        "source_training_result_path": str(source_path), "source_training_result_sha256": r.file_hash(source_path),
        "source_evaluation": evaluation_source, "source_checkpoint": checkpoint,
        "manifest_path": str(manifest_path), "manifest_sha256": r.file_hash(manifest_path),
        "condition": condition, "training_permutation": order,
        "initial_optimizer_updates": start_updates, "expected_new_updates": 30,
        "helper_sha256": {Path(module.__file__).name: r.file_hash(Path(module.__file__)) for module in
            (training, evaluation, boundary, cache, block, bench, pilot, bench.mixed, pilot.candidate)},
        "state_operation": "restore weights/moments/clock; fresh model/tapes/RNG seed42",
        "exact_resume": False, "test_performed": False,
        "selection_rule": "highest validation correct count; earliest epoch on ties"}
    block.write_json("epoch_plan.json", provenance)
    emit("CONTINUATION_PLAN", {key: value for key, value in provenance.items() if key != "training_permutation"})
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    metric = bench.mixed.MaskedAccuracy()
    phase_results = {}
    batch = 0
    next_reference = None
    with cn:
        for size, m, v in ((400, saved[2], saved[3]), (4312, saved[4], saved[5])):
            matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == size]
            require(len(matches) == 1, "Ambiguous moment restore")
            for key, values in (("M", m), ("V", v)):
                variable = matches[0].vars[key]
                variable.view[...] = values.reshape(variable.view.shape)
                variable.push_to_device()
        block.same_state(r.checkpoint.training_state(cn, layers), saved)
        observer.snapshot("after_restore")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient GPU headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=72, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(epoch - 1)
        for phase, split, indices_order in (
            ("training", "train", order), ("validation", "validation", list(range(143))),
            ("next_start_reference", "train", next_order[:32])):
            entries = manifest["entries"][split]
            context.update({"mode": phase, "split": split,
                "parents": {i: entry["metadata"]["stable_id"] for i, entry in enumerate(entries)}})
            before_phase = r.checkpoint.training_state(cn, layers)
            record_start = len(learn.records)
            state = {out: metric.create_state()}
            scores, labels = [], []
            for start in range(0, len(indices_order), 32):
                indices = indices_order[start:start + 32]
                count = len(indices)
                loaded = [cache.load_cached_sample(manifest_path.parent, entries[i]) for i in indices]
                require(all(row[2]["split"] == split for row in loaded), "Wrong cached partition")
                targets = np.array([row[1] for row in loaded] + [0] * (32 - count), dtype=np.int64)
                inputs = [r.to_ml_genn(row[0]) for row in loaded] + [empty] * (32 - count)
                context.update({"indices": indices + [-1] * (32 - count), "targets": targets,
                    "step": context["updates"], "forward_state": r.checkpoint.training_state(cn, layers)})
                for drain in (False, True):
                    metric.real_mask = np.zeros(32, dtype=bool) if drain else np.arange(32) < count
                    with observer.phase(f"{phase}/presentation_{batch}"):
                        cn._train_batch(batch, context["step"], {inp: [empty] * 32 if drain else inputs},
                            {out: np.zeros(32, dtype=np.int64) if drain else targets}, {out: metric}, state, callbacks)
                        value = np.array(cn.get_readout(out), copy=True)
                    require(value.shape == (32, 11) and np.isfinite(value).all(), "Invalid forward scores")
                    if drain:
                        np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)
                    else:
                        block.same_state(context["forward_state"], r.checkpoint.training_state(cn, layers))
                        context["forward_scores"] = value[:count].copy()
                        context["online_metrics"] = boundary.score_summary(value[:count], targets[:count])
                        scores.extend(value[:count].tolist())
                        labels.extend(targets[:count].tolist())
                        if phase == "training" and start == 0:
                            np.testing.assert_array_equal(targets, first_reference["labels"])
                            np.testing.assert_allclose(value, first_reference["scores"], rtol=1e-9, atol=1e-11)
                            emit("CONTINUATION_FIRST_FORWARD_PARITY", {"all_32_scores": "pass",
                                "weights_and_moments_restored": "exact_match", "next_adam_step": start_updates})
                    pop = cn.neuron_populations[inp.population()]
                    np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                    batch += 1
                observer.snapshot(f"{phase}/after_{start + count}")
                require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient sampled GPU headroom")
            records = learn.records[record_start:]
            require([i for row in records for i in row["indices"]] == indices_order
                    and state[out].total == len(indices_order), "Phase sample/metric coverage changed")
            scores, labels = np.asarray(scores), np.asarray(labels, dtype=np.int64)
            counts = np.concatenate([row["counts"] for row in records]).astype(np.uint32)
            summary = boundary.score_summary(scores, labels)
            summary.update({"phase": phase, "accuracy": summary["correct"] / len(labels),
                "score_parameter_state": "changes between batches" if phase == "training" else "fixed",
                "optimizer_updates": 30 if phase == "training" else 0,
                "maximum_replayed_spikes_per_neuron": int(counts.max()),
                "fully_silent_hidden_examples": int(np.sum(counts.sum(axis=1) == 0)),
                "clipped_jumps": 0, "checkpoint": context["checkpoint"]})
            if phase != "training":
                block.same_state(before_phase, r.checkpoint.training_state(cn, layers))
            if phase == "validation":
                require(np.bincount(labels, minlength=11).tolist() == [13] * 11, "Validation class coverage changed")
                array_path = Path.cwd() / "validation_frozen_reference.npz"
                with array_path.open("xb") as stream:
                    np.savez(stream, indices=np.asarray(indices_order, dtype=np.int64), labels=labels,
                        scores=scores, hidden_counts=counts,
                        individual_hidden_gradients=np.concatenate([row["individual"][0] for row in records]),
                        individual_output_gradients=np.concatenate([row["individual"][1] for row in records]))
                summary["reference_arrays"] = {"path": str(array_path), "sha256": r.file_hash(array_path)}
            elif phase == "next_start_reference":
                require(len(records) == 1, "Unexpected next-start reference coverage")
                next_reference = {"logical_epoch": epoch + 1, "training_indices": indices_order,
                    "checkpoint": context["checkpoint"], "scores": scores.tolist(), "labels": labels.tolist(),
                    "hidden_counts": counts.tolist(), "mean_gradients": [g.tolist() for g in records[0]["mean"]]}
            block.write_json(f"{phase}_result.json", {**summary, "evaluation_order_indices": indices_order,
                "scores": scores.tolist(), "labels": labels.tolist()})
            phase_results[phase] = summary
            emit("CONTINUATION_PHASE_RESULT", summary)
        callbacks.on_epoch_end(epoch - 1, state)
        callbacks.on_train_end(state)
    require(batch == 72 and len(learn.records) == 36
            and context["updates"] == start_updates + 30
            and context["contributions"] == order and len(context["update_records"]) == 30,
            "Epoch update/phase accounting failed")
    history.append(history_row(epoch, phase_results["validation"]))
    best = max(history, key=lambda row: (row["correct"], -row["logical_epoch"]))
    block.write_json("next_epoch_reference.json", next_reference)
    report = {**provenance, "optimizer_updates_this_run": 30,
        "cumulative_optimizer_updates": context["updates"], "real_training_contributions": 934,
        "full_data_epoch_completed": True, "checkpoint": context["checkpoint"],
        "phase_results": phase_results, "validation_history": history, "best_validation": best,
        "update_records": context["update_records"], "fixed_bound_clips": 0,
        "next_epoch_reference": {"path": str(Path.cwd() / "next_epoch_reference.json"),
            "sha256": r.file_hash(Path.cwd() / "next_epoch_reference.json")},
        "validation_performed": True, "final_weight_training_evaluation_performed": False,
        "memory_snapshots": observer.snapshots, "phases": observer.phases,
        "full_window_training_ready": False, "all_trained_states_continuous_gradient_certified": False}
    block.write_json("full_epoch_result.json", report)
    emit("PASS_CONTROLLED_FULL_DEVELOPMENT_EPOCH", {
        "logical_epoch": epoch, "new_optimizer_updates": 30,
        "adam_steps": [start_updates, context["updates"] - 1],
        "real_training_contributions": 934, "validation_examples": 143,
        "fixed_bound_clips": 0, "validation_correct": phase_results["validation"]["correct"],
        "validation_mean_ce": phase_results["validation"]["mean_ce"],
        "validation_ce_change_from_previous_epoch": phase_results["validation"]["mean_ce"] - history[-2]["mean_ce"],
        "best_validation_epoch": best["logical_epoch"], "best_validation_correct": best["correct"],
        "checkpoint": context["checkpoint"], "exact_resume": False,
        "minimum_sampled_free_fraction": min(s["free_fraction"] for s in observer.snapshots),
        "simulation_and_callback_wall_ms": sum(p["profiled_wall_ms"] for p in observer.phases),
        "full_window_training_ready": False, "test_performed": False})


if __name__ == "__main__":
    main()
