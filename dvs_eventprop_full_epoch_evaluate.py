"""Frozen full-development evaluation of an epoch-one checkpoint.

Weights-only fresh-process load. Uses the checked paired training-compiled
forward/replay path, with zero optimizer updates and fixed adjoint bound 1.
No official test arrays. Saved moments are inspected but not restored.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_full_epoch_train as training

boundary, block, bench, pilot, r = (
    training.boundary, training.block, training.bench, training.pilot, training.r)
cache = training.cache
require, emit, pull = r.require, r.emit, r.pull


class FreezeEvaluation(bench.ReducedCapture):
    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            for variable in block.reduced_variables(cn):
                values = pull(variable)
                np.testing.assert_array_equal(values, np.zeros_like(values))
            return
        ctx = self.context
        block.same_state(r.checkpoint.training_state(cn, self.layers), ctx["forward_state"])
        pop = cn.neuron_populations[self.hidden.population()]
        bad = {name: pull(pop.vars[name]).reshape(32, 392)
               for name in ("AuditClips", "AuditNonfinite", "AuditBadLimit")}
        for lane, index in enumerate(ctx["indices"]):
            if any(np.any(values[lane] != 0) for values in bad.values()):
                emit("FROZEN_EVALUATION_INVALID_LANE", {
                    "split": ctx["split"], "lane": lane, "sample_index": index,
                    "stable_id": ctx["parents"].get(index),
                    **{name: int(values[lane].sum()) for name, values in bad.items()},
                })
        super().on_batch_end(cn, batch, metrics)
        block.same_state(r.checkpoint.training_state(cn, self.layers), ctx["forward_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("initial", "final"), default="final")
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    args = parser.parse_args()
    manifest_path, source_path = [p.resolve(strict=True) for p in (
        args.manifest, args.training_result)]
    manifest, source = [r.json.loads(p.read_text()) for p in (manifest_path, source_path)]
    identity = r.checkpoint.identity()
    require(source["artifact_type"] == "full-development-training-epoch1-v1"
            and source["diagnostic_sha256"] == r.file_hash(Path(training.__file__))
            == "49c2f596bc7620cae9b89b1f5f20aadc3dff51c47c84957ed7a3cbf4a0de69b3",
            "Training producer changed")
    require(source["full_data_epoch_completed"] is True
            and source["logical_epoch"] == 1 and source["cumulative_optimizer_updates"] == 30
            and source["real_training_contributions"] == 934 and source["fixed_bound_clips"] == 0
            and [row["adam_step"] for row in source["update_records"]] == list(range(30))
            and [row["real_examples"] for row in source["update_records"]] == [32] * 29 + [6],
            "Expected the completed checked epoch-one result")
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
            == manifest["sample_audit_sha256"], "Data-cache binding changed")
    require(len(manifest["entries"]["train"]) == 934
            and len(manifest["entries"]["validation"]) == 143
            and manifest["test_event_arrays_loaded"] == 0, "Development coverage changed")
    require(source["training_permutation"]
            == manifest["batch_plan"]["epoch1_training_permutation"]
            == np.random.default_rng(42).permutation(934).tolist(), "Training order changed")
    condition = source["condition"]
    require(condition["dt_ms"] == 0.00006103515625
            and condition["simulation_timesteps"] == 16777216
            and condition["physical_horizon_ms"] == 1024.0
            and condition["batch_size"] == 32 and condition["compiler_max_spikes"] == 8193
            and condition["scalar_precision"] == "double" and condition["time_precision"] == "float"
            and condition["strict_buffer_checking"] is True, "Unexpected evaluation condition")
    checkpoint = source["initial_checkpoint"] if args.checkpoint == "initial" else source["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    updates = 0 if args.checkpoint == "initial" else 30
    require(checkpoint_path.parent == source_path.parent
            and checkpoint_path.name == f"update_{updates:04d}.npz"
            and r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint binding changed")
    require(Path.cwd().resolve() not in (source_path.parent, manifest_path.parent),
            "Use a new experiment directory")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(block.STATE_NAMES), "Checkpoint schema changed")
        saved = [np.array(archive[name], copy=True) for name in block.STATE_NAMES]
    for values, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(values.dtype == np.float64 and values.shape == (size,)
                and np.isfinite(values).all(), "Invalid saved checkpoint state")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Invalid saved Adam variance")
    if args.checkpoint == "initial":
        require([r.capacity.array_hash(values) for values in saved[:2]]
                == source["initial_weight_sha256"]
                and all(np.all(values == 0) for values in saved[2:]), "Initial checkpoint changed")
    splits = ("train", "validation") if args.split == "both" else (args.split,)
    orders = {split: source["training_permutation"] if split == "train"
              else list(range(143)) for split in splits}
    for split in splits:
        for index, entry in enumerate(manifest["entries"][split]):
            metadata = entry["metadata"]
            require(entry["index"] == index and metadata["split"] == split
                    and ((split == "train" and 1 <= metadata["subject"] <= 20)
                         or (split == "validation" and 21 <= metadata["subject"] <= 23)),
                    "Sample outside its development partition")
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
        grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="FullDevelopmentFrozenEvaluation", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {}
    raw = bench.RawCapture(layers, context, 32)
    capture = FreezeEvaluation(layers, hid, out, context, 32)
    found = {name: 0 for name in (
        "GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
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
            "Callback or audit layout changed")
    cn.base_train_callbacks[:] = modified
    provenance = {
        "artifact_type": "full-development-frozen-evaluation-v1", "identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)),
        "training_result_path": str(source_path), "training_result_sha256": r.file_hash(source_path),
        "training_producer_sha256": source["diagnostic_sha256"],
        "manifest_path": str(manifest_path), "manifest_sha256": r.file_hash(manifest_path),
        "checkpoint": checkpoint, "checkpoint_choice": args.checkpoint,
        "scores_at_training_optimizer_updates": updates, "condition": condition,
        "splits": list(splits),
        "operation": "weights-only load; fresh zero Adam moments; frozen paired evaluation",
        "helper_sha256": {Path(module.__file__).name: r.file_hash(Path(module.__file__))
                          for module in (training, boundary, cache, block, bench, pilot, bench.mixed, pilot.candidate)},
        "optimizer_updates": 0, "exact_resume": False, "test_performed": False,
    }
    block.write_json("evaluation_plan.json", provenance)
    emit("FULL_FROZEN_EVALUATION_PLAN", provenance)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    metric = bench.mixed.MaskedAccuracy()
    summaries = {}
    total_pairs = sum((len(order) + 31) // 32 for order in orders.values())
    with cn:
        cn.get_readout(out)
        initial = r.checkpoint.training_state(cn, layers)
        require(all(values.dtype == np.float64 for values in initial), "Precision changed")
        for actual, expected in zip(initial[:2], saved[:2]):
            np.testing.assert_array_equal(actual.reshape(-1), expected)
        require(all(np.all(values == 0) for values in initial[2:]), "Fresh moments not zero")
        observer.snapshot("after_load")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient GPU headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=2 * total_pairs, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for split in splits:
            entries, order = manifest["entries"][split], orders[split]
            context.update({"split": split, "parents": {
                index: entry["metadata"]["stable_id"] for index, entry in enumerate(entries)}})
            state = {out: metric.create_state()}
            record_start = len(capture.records)
            scores, labels = [], []
            for start in range(0, len(order), 32):
                indices = order[start:start + 32]
                count = len(indices)
                loaded = [cache.load_cached_sample(manifest_path.parent, entries[index]) for index in indices]
                targets = np.array([row[1] for row in loaded] + [0] * (32 - count), dtype=np.int64)
                inputs = [r.to_ml_genn(row[0]) for row in loaded] + [empty] * (32 - count)
                context.update({"indices": indices + [-1] * (32 - count), "targets": targets,
                    "forward_state": r.checkpoint.training_state(cn, layers)})
                for drain in (False, True):
                    metric.real_mask = np.zeros(32, dtype=bool) if drain else np.arange(32) < count
                    with observer.phase(f"{split}/presentation_{batch}"):
                        cn._train_batch(batch, 0, {inp: [empty] * 32 if drain else inputs},
                            {out: np.zeros(32, dtype=np.int64) if drain else targets},
                            {out: metric}, state, callbacks)
                        value = np.array(cn.get_readout(out), copy=True)
                    require(value.shape == (32, 11) and np.isfinite(value).all(), "Invalid scores")
                    if drain:
                        np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)
                    else:
                        scores.extend(value[:count].tolist())
                        labels.extend(targets[:count].tolist())
                    block.same_state(initial, r.checkpoint.training_state(cn, layers))
                    pop = cn.neuron_populations[inp.population()]
                    np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                    batch += 1
                emit("FULL_FROZEN_EVALUATION_PROGRESS", {"split": split,
                     "completed": start + count, "total": len(order), "optimizer_updates": 0})
                observer.snapshot(f"{split}/after_{start + count}")
                require(observer.snapshots[-1]["free_fraction"] >= 0.20,
                        "Insufficient sampled GPU headroom")
            records = capture.records[record_start:]
            require([i for row in records for i in row["indices"]] == order
                    and state[out].total == len(order), "Frozen sample/metric coverage changed")
            scores, labels = np.asarray(scores), np.asarray(labels, dtype=np.int64)
            counts = np.concatenate([row["counts"] for row in records]).astype(np.uint32)
            require(counts.shape == (len(order), 392), "Replay count coverage changed")
            gradients = [np.concatenate([row["individual"][index] for row in records])
                         for index in range(2)]
            summary = boundary.score_summary(scores, labels)
            class_counts = np.bincount(labels, minlength=11).tolist()
            require(class_counts == manifest["summaries"][split]["class_counts"],
                    "Class denominator coverage changed")
            decisions = scores.argmax(axis=1)
            silent = [position for position in range(len(order)) if counts[position].sum() == 0]
            silent_parents = [{"index": order[pos],
                "stable_id": entries[order[pos]]["metadata"]["stable_id"],
                "label": int(labels[pos]), "decision": int(decisions[pos]),
                "occupancy_spikes": entries[order[pos]]["metadata"]["occupancy_spikes"]}
                for pos in silent]
            reference_path = Path.cwd() / f"{split}_frozen_reference.npz"
            with reference_path.open("xb") as stream:
                np.savez(stream, indices=np.asarray(order, dtype=np.int64), labels=labels,
                         scores=scores, hidden_counts=counts,
                         individual_hidden_gradients=gradients[0],
                         individual_output_gradients=gradients[1])
            summary.update({"split": split, "accuracy": summary["correct"] / len(order),
                "class_counts": class_counts,
                "correct_per_class": np.bincount(labels, weights=(decisions == labels),
                                                  minlength=11).astype(int).tolist(),
                "score_parameter_state": "fixed", "optimizer_updates": 0,
                "scores_at_training_optimizer_updates": updates,
                "maximum_replayed_spikes_per_neuron": int(counts.max()),
                "fully_silent_hidden_examples": len(silent), "silent_parents": silent_parents,
                "replay_spikes_per_filter": counts.reshape(len(order), 7, 7, 8).sum(
                    axis=(0, 1, 2)).astype(int).tolist(),
                "reference_arrays": {"path": str(reference_path),
                    "sha256": r.file_hash(reference_path)},
                "checkpoint": checkpoint, "clipped_jumps": 0,
                "weights_and_fresh_moments_unchanged": True})
            block.write_json(f"{split}_evaluation.json", {
                **provenance, **summary, "evaluation_order_indices": order,
                "scores": scores.tolist(), "labels": labels.tolist()})
            summaries[split] = summary
            emit("FULL_FROZEN_SPLIT_RESULT", summary)
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        block.same_state(initial, r.checkpoint.training_state(cn, layers))
    require(batch == 2 * total_pairs and len(capture.records) == total_pairs,
            "Frozen replay accounting changed")
    report = {**provenance, "phase_results": summaries,
        "weights_and_fresh_moments_unchanged": True, "fixed_bound_clips": 0,
        "validation_performed": "validation" in splits,
        "memory_snapshots": observer.snapshots, "phases": observer.phases,
        "full_window_training_ready": False,
        "continuous_gradient_correctness_of_all_parents_certified": False}
    block.write_json("full_evaluation_result.json", report)
    emit("PASS_FULL_DEVELOPMENT_FROZEN_EVALUATION", {
        "checkpoint_choice": args.checkpoint, "scores_at_training_optimizer_updates": updates,
        "splits": list(splits), "real_examples": sum(len(order) for order in orders.values()),
        "optimizer_updates": 0, "weights_and_fresh_moments_unchanged": True,
        "fixed_bound_clips": 0, "test_performed": False,
        "minimum_sampled_free_fraction": min(s["free_fraction"] for s in observer.snapshots),
        "simulation_and_callback_wall_ms": sum(p["profiled_wall_ms"] for p in observer.phases),
        "full_window_training_ready": False})


if __name__ == "__main__":
    main()
