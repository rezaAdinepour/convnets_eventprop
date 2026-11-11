"""Three additional pilot epochs from saved weights/moments/Adam step.

Controlled warm start with fresh model/tapes and seed42, not exact resume.
Training-only; paired batches and fresh fixed-weight evaluation each epoch.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_pilot_train as pilot

r = pilot.r
require, emit, pull = r.require, r.emit, r.pull


class WarmStartLearn(pilot.LearnOrFreeze):
    def __init__(self, layers, hidden, output, context, expected_replay):
        super().__init__(layers, hidden, output, context)
        self.expected_replay = expected_replay

    def on_batch_end(self, cn, batch, metrics):
        if batch == 1:
            require(self.context["step"] == 3 and self.context["updates"] == 3,
                    "Warm start must use Adam step3")
            norms = []
            for size in (400, 4312):
                matches = [cu.vars["ReducedGradient"] for name, cu in cn.genn_model.custom_updates.items()
                           if name.startswith("CUBatchReduce") and "ReducedGradient" in cu.vars
                           and cu.vars["ReducedGradient"].view.size == size]
                require(len(matches) == 1, "Ambiguous warm-start gradient")
                norms.append(float(np.linalg.norm(pull(matches[0]))))
            np.testing.assert_allclose(norms, self.expected_replay["gradient_l2_hidden_output"],
                                       rtol=1e-9, atol=1e-11)
            jumps = pull(cn.neuron_populations[self.hidden.population()].vars["AuditJumps"])
            np.testing.assert_array_equal(jumps.reshape(8, 392).sum(axis=1),
                                           self.expected_replay["replayed_jumps_per_real_example"])
            self.context["warm_start_checked"] = True
            emit("WARM_START_FIRST_REPLAY_PARITY", {"adam_step": 3,
                 "gradient_norms_and_replay_counts": "pass"})
        super().on_batch_end(cn, batch, metrics)
        if batch % 2:
            row = self.context["replays"][-1]
            row["logical_epoch"] = self.context["logical_epoch"]
            emit("BLOCK_REPLAY_OWNER", {"logical_epoch": row["logical_epoch"],
                 "phase": row["phase"], "physical_batch": batch,
                 "adam_step": row.get("adam_step"), "real_count": row["real_count"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path, required=True)
    parser.add_argument("--reload-result", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve(strict=True) for p in (args.baseline, args.pilot_result, args.reload_result)]
    baseline, source, reload = [r.json.loads(p.read_text()) for p in paths]
    identity = r.checkpoint.identity()
    for artifact in (baseline, source, reload):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment mismatch: {key}")
    require(source["diagnostic_sha256"] == r.file_hash(Path(pilot.__file__)), "Pilot producer changed")
    require(reload["diagnostic_sha256"] == r.file_hash(Path(__file__).with_name("dvs_eventprop_pilot_reload.py")),
            "Reload producer changed")
    require(source["baseline_sha256"] == r.file_hash(paths[0])
            and reload["pilot_result_sha256"] == r.file_hash(paths[1]), "Artifact binding mismatch")
    require(reload["optimizer_updates"] == 0 and reload["weights_and_moments_unchanged"]
            and reload["max_absolute_score_difference"] == 0.0, "Fresh reload gate has not passed")
    require(baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(pilot.candidate.__file__)),
            "Precision adapter changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Helper changed: {name}")
    require(source["optimizer_updates"] == 3 and len(source["update_checkpoints"]) == 3,
            "Expected post-epoch1 source state")
    checkpoint = source["update_checkpoints"][-1]
    require(checkpoint == reload["checkpoint"], "Reload checked a different checkpoint")
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(checkpoint_path.is_relative_to(paths[1].parent)
            and r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint provenance mismatch")
    names = ("hidden", "output", "adam_hidden_m", "adam_hidden_v", "adam_output_m", "adam_output_v")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), "Unexpected checkpoint schema")
        saved = [np.array(archive[name], copy=True) for name in names]
    for array, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(array.dtype == np.float64 and array.shape == (size,)
                and np.isfinite(array).all(), "Invalid warm-start state")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Invalid saved Adam variance")
    condition = source["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.00006103515625
            and condition["batch_size"] == 8 and condition["compiler_max_spikes"] == 8193
            and condition["scalar_precision"] == "double", "Unexpected pilot configuration")
    root = args.root.resolve(strict=True)
    selected = r.pilot(r.inventory(root)["train"], 2)
    order = np.random.default_rng(42).permutation(22)
    require(order.tolist() == source["permutation_of_sorted_pilot"], "Pilot permutation changed")
    rows = [selected[int(i)] for i in order]
    require([row.stable_id for row in rows] == source["training_order"], "Pilot parent order changed")
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    labels = np.asarray(labels, dtype=np.int64)
    require(r.normalize(list(metadata)) == source["sample_metadata"], "Pilot input metadata changed")
    np.testing.assert_array_equal(labels, reload["labels"])
    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
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
    cn = compiler.compile(network, name="PilotThreeEpochBlock", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"updates": 3, "step": 3, "moments": {
        0: (saved[2].copy(), saved[3].copy()), 1: (saved[4].copy(), saved[5].copy())},
        "replays": [], "checkpoints": [], "warm_start_checked": False}
    raw_capture = pilot.CaptureRaw(layers, context)
    learn = WarmStartLearn(layers, hid, out, context, reload["replay_records"][0])
    counts = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in counts:
            counts[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                modified.extend((raw_capture, cb))
            elif cb.name == "GradientLearn":
                modified.append(learn)
            else:
                modified.append(r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False))
        else:
            modified.append(cb)
    require(all(n == 1 for n in counts.values()), "Unexpected callback layout")
    cn.base_train_callbacks[:] = modified
    block_condition = {**condition, "training_epochs": 3, "training_updates": 9,
        "initial_optimizer_updates": 3, "optimizer_updates": 12,
        "logical_epochs": [2, 3, 4], "order_between_epochs": "fixed recorded seed42 permutation",
        "runner": "controlled warm start; paired training and frozen evaluation each epoch"}
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "pilot_producer_sha256": source["diagnostic_sha256"],
        "baseline_sha256": r.file_hash(paths[0]), "pilot_result_sha256": r.file_hash(paths[1]),
        "reload_result_sha256": r.file_hash(paths[2]), "source_checkpoint": checkpoint,
        "condition": block_condition, "seed": 42, "training_order": source["training_order"],
        "sample_metadata": source["sample_metadata"],
        "restored_state": "weights, Adam M/V, next Adam step3",
        "fresh_state": "model variables, tapes and RNG initialized with seed42",
        "exact_resume": False, "validation_performed": False, "test_performed": False}
    emit("PILOT_BLOCK_PLAN", provenance)
    with (Path.cwd() / "pilot_block_plan.json").open("x") as stream:
        r.json.dump(provenance, stream, sort_keys=True, allow_nan=False)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    inputs = [r.to_ml_genn(p) for p in payloads]
    epoch_results = []
    with cn:
        for size, m, v in ((400, saved[2], saved[3]), (4312, saved[4], saved[5])):
            matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == size]
            require(len(matches) == 1, "Ambiguous saved moment ownership")
            for key, values in (("M", m), ("V", v)):
                variable = matches[0].vars[key]
                variable.view[...] = values.reshape(variable.view.shape)
                variable.push_to_device()
        restored = r.checkpoint.training_state(cn, layers)
        require(len(restored) == len(saved), "Restored state schema differs")
        for a, b in zip(restored, saved):
            np.testing.assert_array_equal(a.reshape(-1), b)
        observer.snapshot("after_load_and_restore")
        emit("WARM_START_STATE_RESTORED", {"weights_and_adam_moments": "exact_match",
             "next_adam_step": 3, "exact_resume": False})
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=36, num_epochs=3)
        callbacks.on_train_begin()
        physical_batch = 0
        for local_epoch, epoch in enumerate((2, 3, 4)):
            callbacks.on_epoch_begin(local_epoch)
            context["logical_epoch"] = epoch
            emit("BLOCK_EPOCH_BEGIN", {"logical_epoch": epoch, "optimizer_updates": context["updates"]})
            summaries = {}
            replay_start = len(context["replays"])
            for phase in ("training_epoch", "fresh_evaluation"):
                context["phase"] = phase
                before_phase = r.checkpoint.training_state(cn, layers)
                updates_before = context["updates"]
                metric = pilot.PilotMetric(context, labels)
                state = {out: metric.create_state()}
                for start in (0, 8, 16):
                    indices = list(range(start, min(start + 8, 22)))
                    indices += [-1] * (8 - len(indices))
                    targets = np.array([labels[i] if i >= 0 else 0 for i in indices], dtype=np.int64)
                    presentations = [inputs[i] if i >= 0 else empty for i in indices]
                    for drain in (False, True):
                        context["step"] = context["updates"]
                        context["physical_batch"] = physical_batch
                        context["metric_indices"] = [-1] * 8 if drain else indices
                        before = r.checkpoint.training_state(cn, layers)
                        if not drain:
                            context["forward_indices"] = indices
                            context["forward_targets"] = targets.copy()
                            context["forward_weights"] = before[:2]
                        with observer.phase(f"epoch_{epoch}/{phase}/presentation_{physical_batch}"):
                            cn._train_batch(physical_batch, context["step"],
                                {inp: [empty] * 8 if drain else presentations},
                                {out: np.zeros(8, dtype=np.int64) if drain else targets},
                                {out: metric}, state, callbacks)
                            cn.get_readout(out)
                        if physical_batch == 0:
                            np.testing.assert_allclose(metric.scores, reload["scores"][:8], rtol=1e-9, atol=1e-11)
                            emit("WARM_START_FIRST_FORWARD_PARITY", {"first_8_scores": "pass"})
                        if not drain:
                            for a, b in zip(before, r.checkpoint.training_state(cn, layers)):
                                np.testing.assert_array_equal(a, b)
                        pop = cn.neuron_populations[inp.population()]
                        np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                        physical_batch += 1
                        if drain:
                            observer.snapshot(f"after_pair_{physical_batch // 2}")
                            require(observer.snapshots[-1]["free_fraction"] >= 0.20,
                                    "Less than 20 percent sampled memory headroom")
                require(metric.indices == list(range(22)) and state[out].total == 22, "Wrong phase coverage")
                if phase == "fresh_evaluation":
                    require(context["updates"] == updates_before, "Evaluation applied a learning update")
                    for a, b in zip(before_phase, r.checkpoint.training_state(cn, layers)):
                        np.testing.assert_array_equal(a, b)
                else:
                    require(context["updates"] - updates_before == 3, "Missing training update")
                summary = {"logical_epoch": epoch, "phase": phase, "real_examples": 22,
                    "correct": int(state[out].correct), "accuracy": float(state[out].result),
                    "mean_ce": float(np.mean(metric.losses)),
                    "optimizer_updates_in_phase": context["updates"] - updates_before,
                    "cumulative_optimizer_updates": context["updates"],
                    "scores": metric.scores, "labels": labels.tolist(),
                    "score_parameter_state": "fixed" if phase == "fresh_evaluation" else "changing during epoch"}
                summaries[phase] = summary
                emit("BLOCK_PHASE_RESULT", {k: v for k, v in summary.items() if k not in {"scores", "labels"}})
            callbacks.on_epoch_end(local_epoch, state)
            result = {"logical_epoch": epoch, "phase_results": summaries,
                      "final_checkpoint": context["checkpoints"][-1],
                      "replay_records": context["replays"][replay_start:]}
            epoch_results.append(result)
            with (Path.cwd() / f"epoch_{epoch:04d}.json").open("x") as stream:
                r.json.dump({**provenance, **result}, stream, sort_keys=True, allow_nan=False)
        callbacks.on_train_end(state)
    training = [row for row in context["replays"] if row["phase"] == "training_epoch"]
    require(context["warm_start_checked"] and physical_batch == 36 and context["updates"] == 12,
            "Wrong warm-start/block coverage")
    require([row["adam_step"] for row in training] == list(range(3, 12))
            and [row["real_count"] for row in training] == [8, 8, 6] * 3, "Wrong learning coverage")
    report = {**provenance, "epochs": epoch_results, "optimizer_updates_this_run": 9,
              "cumulative_optimizer_updates": 12, "checkpoints": context["checkpoints"],
              "memory_snapshots": observer.snapshots, "phases": observer.phases,
              "full_window_training_ready": False}
    with (Path.cwd() / "pilot_block_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_PILOT_THREE_EPOCH_BLOCK", {"logical_epochs": [2, 3, 4],
        "adam_steps": list(range(3, 12)), "optimizer_updates_this_run": 9,
        "cumulative_optimizer_updates": 12, "fixed_bound_clips": 0,
        "fresh_epoch_results": [{k: v for k, v in e["phase_results"]["fresh_evaluation"].items()
                                 if k not in {"scores", "labels"}} for e in epoch_results],
        "exact_resume": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
