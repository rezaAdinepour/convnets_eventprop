"""Fresh-process weights-only reload of the one-epoch pilot's final checkpoint.

Uses the training-compiled paired evaluator. No optimizer updates or test data.
Saved moments are inspected but deliberately not restored; no resume claim.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_pilot_train as pilot

r = pilot.r
require, emit, pull = r.require, r.emit, r.pull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source_path, baseline_path = [p.resolve(strict=True) for p in (args.result, args.baseline)]
    source, baseline = [r.json.loads(p.read_text()) for p in (source_path, baseline_path)]
    identity = r.checkpoint.identity()
    for artifact in (source, baseline):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment mismatch: {key}")
    require(source["diagnostic_sha256"] == r.file_hash(Path(pilot.__file__)),
            "Pilot producer source changed")
    require(source["baseline_sha256"] == r.file_hash(baseline_path), "Pilot baseline binding differs")
    require(baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(pilot.candidate.__file__)),
            "Precision adapter source changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Helper source changed: {name}")
    condition = source["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.00006103515625
            and condition["batch_size"] == 8 and condition["compiler_max_spikes"] == 8193
            and condition["scalar_precision"] == "double", "Unexpected pilot condition")
    require(source["optimizer_updates"] == 3 and len(source["update_checkpoints"]) == 3,
            "Expected the completed one-epoch pilot")
    expected = source["phase_results"]["final_evaluation"]
    require(expected["score_parameter_state"] == "fixed"
            and expected["optimizer_updates_in_phase"] == 0 and expected["real_examples"] == 22,
            "Source scores are not a frozen final evaluation")
    checkpoint = source["update_checkpoints"][-1]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(checkpoint_path.is_relative_to(source_path.parent)
            and checkpoint_path.name == "update_0003.npz", "Unexpected final checkpoint selection")
    require(r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint hash mismatch")
    names = ("hidden", "output", "adam_hidden_m", "adam_hidden_v", "adam_output_m", "adam_output_v")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), "Unexpected checkpoint schema")
        saved = [np.array(archive[name], copy=True) for name in names]
    for array, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(array.dtype == np.float64 and array.shape == (size,)
                and np.isfinite(array).all(), "Invalid checkpoint array")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Negative saved Adam variance")
    root = args.root.resolve(strict=True)
    selected = r.pilot(r.inventory(root)["train"], 2)
    order = np.random.default_rng(42).permutation(22)
    require(order.tolist() == source["permutation_of_sorted_pilot"], "Pilot permutation changed")
    rows = [selected[int(i)] for i in order]
    require([row.stable_id for row in rows] == source["training_order"], "Pilot parent order changed")
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(r.normalize(list(metadata)) == source["sample_metadata"], "Pilot input metadata changed")
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(labels, expected["labels"])
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    np.random.seed(42)
    network, inp, hid, out = r.build_model(config, 8)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, saved[:2], ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = values.reshape(shape).copy()
    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=condition["simulation_timesteps"], losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=8, rng_seed=42, max_spikes=8193, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="PilotFinalReload", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"updates": 0, "step": 0, "moments": {}, "replays": [], "checkpoints": [],
               "phase": "final_checkpoint_reload"}
    raw_capture = pilot.CaptureRaw(layers, context)
    frozen = pilot.LearnOrFreeze(layers, hid, out, context)
    counts = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in counts:
            counts[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                modified.extend((raw_capture, cb))
            elif cb.name == "GradientLearn":
                modified.append(frozen)
            else:
                modified.append(r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False))
        else:
            modified.append(cb)
    require(all(n == 1 for n in counts.values()), "Unexpected reload callback layout")
    cn.base_train_callbacks[:] = modified
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "pilot_result_sha256": r.file_hash(source_path), "pilot_producer_sha256": source["diagnostic_sha256"],
        "checkpoint": checkpoint, "source_git_head": source["identity"]["convnets_commit"],
        "cross_revision_basis": "matching producer, adapter, helpers, core source and environment",
        "operation": "final weights only; fresh zero Adam moments", "exact_resume": False}
    emit("PILOT_RELOAD_PLAN", provenance)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    inputs = [r.to_ml_genn(payload) for payload in payloads]
    metric = pilot.PilotMetric(context, labels)
    state = {out: metric.create_state()}
    with cn:
        cn.get_readout(out)
        observer.snapshot("after_load")
        initial = r.checkpoint.training_state(cn, layers)
        require(all(v.dtype == np.float64 for v in initial), "Reload precision changed")
        for actual, weights in zip(initial[:2], saved[:2]):
            np.testing.assert_array_equal(actual.reshape(-1), weights)
        require(all(np.all(v == 0) for v in initial[2:]), "Fresh Adam moments are not zero")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=6, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for start in (0, 8, 16):
            indices = list(range(start, min(start + 8, 22)))
            indices += [-1] * (8 - len(indices))
            targets = np.array([labels[i] if i >= 0 else 0 for i in indices], dtype=np.int64)
            presentations = [inputs[i] if i >= 0 else empty for i in indices]
            for drain in (False, True):
                context["physical_batch"] = batch
                context["metric_indices"] = [-1] * 8 if drain else indices
                if not drain:
                    context["forward_indices"] = indices
                    context["forward_targets"] = targets.copy()
                    context["forward_weights"] = r.checkpoint.training_state(cn, layers)[:2]
                with observer.phase(f"reload/presentation_{batch}"):
                    cn._train_batch(batch, 0, {inp: [empty] * 8 if drain else presentations},
                        {out: np.zeros(8, dtype=np.int64) if drain else targets}, {out: metric}, state, callbacks)
                    cn.get_readout(out)
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                batch += 1
                if drain:
                    observer.snapshot(f"after_reload_pair_{batch // 2}")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        for a, b in zip(initial, r.checkpoint.training_state(cn, layers)):
            np.testing.assert_array_equal(a, b)
    require(context["updates"] == 0 and len(context["replays"]) == 3
            and metric.indices == list(range(22)) and state[out].total == 22, "Reload coverage failed")
    actual_scores = np.asarray(metric.scores)
    expected_scores = np.asarray(expected["scores"])
    require(actual_scores.shape == expected_scores.shape == (22, 11), "Invalid reload score shape")
    difference = float(np.max(np.abs(actual_scores - expected_scores)))
    emit("PILOT_RELOAD_SCORE_COMPARISON", {"max_absolute_difference": difference,
         "correct": int(state[out].correct), "mean_ce": float(np.mean(metric.losses))})
    np.testing.assert_allclose(actual_scores, expected_scores, rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(actual_scores.argmax(axis=1), expected_scores.argmax(axis=1))
    require(int(state[out].correct) == expected["correct"], "Reload decisions changed")
    minimum_free = min(s["free_fraction"] for s in observer.snapshots)
    require(minimum_free >= 0.20, "Less than 20 percent sampled memory headroom")
    report = {**provenance, "scores": metric.scores, "labels": labels.tolist(),
        "correct": int(state[out].correct), "mean_ce": float(np.mean(metric.losses)),
        "max_absolute_score_difference": difference, "optimizer_updates": 0,
        "weights_and_moments_unchanged": True, "minimum_sampled_free_fraction": minimum_free,
        "replay_records": context["replays"], "memory_snapshots": observer.snapshots,
        "phases": observer.phases, "full_window_training_ready": False}
    with (Path.cwd() / "pilot_reload_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_PILOT_FINAL_RELOAD", {"real_examples": 22, "optimizer_updates": 0,
        "max_absolute_score_difference": difference, "all_decisions_reproduced": True,
        "weights_and_moments_unchanged": True, "exact_training_resume_certified": False})


if __name__ == "__main__":
    main()
