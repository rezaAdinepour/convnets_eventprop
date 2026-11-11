"""One checked B32 update (22 real lanes), then frozen full-pilot evaluation.

Controlled warm start from epoch4 weights/moments at Adam step12. Not exact resume.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_batch_compare as bench

pilot, r = bench.pilot, bench.r
require, emit, pull = r.require, r.emit, r.pull


class LearnOne(bench.ReducedCapture):
    def __init__(self, layers, hidden, output, context, reference):
        super().__init__(layers, hidden, output, context, 32)
        self.reference = reference

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return
        before = r.checkpoint.training_state(cn, self.layers)
        for a, b in zip(before, self.context["forward_state"]):
            np.testing.assert_array_equal(a, b)
        super().on_batch_end(cn, batch, metrics)
        record = self.records[-1]
        if self.context["phase"] == "training":
            require(batch == 1 and self.context["updates"] == self.context["step"] == 12,
                    "Unexpected B32 learning ownership or Adam step")
            np.testing.assert_array_equal(record["counts"], self.reference["hidden_counts"])
            for g, expected in zip(record["mean"], self.reference["full_real_mean_gradients"]):
                np.testing.assert_allclose(g, expected, rtol=1e-9, atol=1e-11)
            emit("B32_PREUPDATE_PARITY", {"full_real_mean_gradient": "pass",
                 "per_neuron_replay_counts": "exact_match", "adam_step": 12})
            for size, gradient in zip((400, 4312), record["mean"]):
                matches = [cu.vars["ReducedGradient"] for name, cu in cn.genn_model.custom_updates.items()
                           if name.startswith("CUBatchReduce") and "ReducedGradient" in cu.vars
                           and cu.vars["ReducedGradient"].view.size == size]
                require(len(matches) == 1, "Ambiguous optimizer gradient")
                variable = matches[0]
                variable.view[...] = gradient.reshape(variable.view.shape)
                variable.push_to_device()
            cn.custom_update("GradientLearn")
            changes = []
            for index, (weight, gradient) in enumerate(zip(before[:2], record["mean"])):
                m, v = self.context["moments"][index]
                m, v = 0.9 * m + 0.1 * gradient, 0.999 * v + 0.001 * gradient ** 2
                expected = weight.reshape(-1) - 0.001 * (m / (1 - 0.9 ** 13)) / (
                    np.sqrt(v / (1 - 0.999 ** 13)) + 1e-8)
                actual = pull(cn.connection_populations[self.layers[index].connection()].vars["weight"]).reshape(-1)
                require(np.isfinite(actual).all(), "Nonfinite updated weight")
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
                matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == gradient.size]
                require(len(matches) == 1, "Ambiguous Adam moments")
                np.testing.assert_allclose(pull(matches[0].vars["M"]).reshape(-1), m, rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(pull(matches[0].vars["V"]).reshape(-1), v, rtol=2e-10, atol=2e-12)
                self.context["moments"][index] = (m, v)
                changes.append(float(np.linalg.norm(actual - weight.reshape(-1))))
            require(all(v > 0 for v in changes), "Missing projection update")
            self.context["updates"] = 13
            self.context["checkpoint"] = pilot.save_state(cn, self.layers, "update_0013.npz")
            emit("B32_ADAM_UPDATE", {"adam_step": 12, "cumulative_optimizer_updates": 13,
                 "real_examples": 22, "gradient_rescale": 32.0 / 22.0,
                 "host_adam_weights_and_moments": "pass", "forward_replay_state": "exact_match",
                 "weight_change_l2_hidden_output": changes, "checkpoint": self.context["checkpoint"]})
        else:
            for a, b in zip(before, r.checkpoint.training_state(cn, self.layers)):
                np.testing.assert_array_equal(a, b)


def score_summary(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (22, 11) and np.isfinite(scores).all(), "Invalid pilot scores")
    maximum = scores.max(axis=1)
    ce = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
    ce -= scores[np.arange(22), labels]
    return {"mean_ce": float(ce.mean()), "correct": int(np.sum(scores.argmax(axis=1) == labels)),
            "accuracy": float(np.mean(scores.argmax(axis=1) == labels)), "scores": scores.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--block-result", type=Path, required=True)
    parser.add_argument("--batch32-result", type=Path, required=True)
    args = parser.parse_args()
    paths = [p.resolve(strict=True) for p in (args.baseline, args.block_result, args.batch32_result)]
    baseline, source, reference = [r.json.loads(p.read_text()) for p in paths]
    identity = r.checkpoint.identity()
    for artifact in (baseline, source, reference):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment mismatch: {key}")
    require(source["diagnostic_sha256"] == r.file_hash(Path(__file__).with_name("dvs_eventprop_pilot_block.py"))
            and source["pilot_producer_sha256"] == r.file_hash(Path(pilot.__file__)), "Pilot producer changed")
    require(reference["diagnostic_sha256"] == r.file_hash(Path(bench.__file__))
            and reference["source_block_sha256"] == r.file_hash(paths[1]), "B32 reference source binding changed")
    require(source["baseline_sha256"] == r.file_hash(paths[0])
            and baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(pilot.candidate.__file__)),
            "Baseline or precision adapter changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Helper changed: {name}")
    require(source["cumulative_optimizer_updates"] == 12 and reference["batch_size"] == 32
            and reference["optimizer_updates"] == 0 and reference["weights_and_moments_unchanged"]
            and reference["clipped_jumps"] == 0, "Unexpected source state or B32 control")
    checkpoint = source["epochs"][-1]["final_checkpoint"]
    require(checkpoint == reference["source_checkpoint"], "B32 control used a different checkpoint")
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(checkpoint_path.is_relative_to(paths[1].parent)
            and r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint binding changed")
    names = ("hidden", "output", "adam_hidden_m", "adam_hidden_v", "adam_output_m", "adam_output_v")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(names), "Unexpected checkpoint schema")
        saved = [np.array(archive[name], copy=True) for name in names]
    for v, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(v.dtype == np.float64 and v.shape == (size,) and np.isfinite(v).all(), "Invalid saved state")
    require(np.all(saved[3] >= 0) and np.all(saved[5] >= 0), "Invalid saved Adam variance")
    condition = source["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.00006103515625
            and condition["compiler_max_spikes"] == 8193 and condition["scalar_precision"] == "double",
            "Unexpected physical condition")
    root = args.root.resolve(strict=True)
    by_id = {row.stable_id: row for row in r.inventory(root)["train"]}
    rows = [by_id[key] for key in source["training_order"]]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(len(rows) == 22 and r.normalize(list(metadata)) == source["sample_metadata"]
            == reference["sample_metadata"], "Pilot input provenance changed")
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(labels, source["epochs"][-1]["phase_results"]["fresh_evaluation"]["labels"])
    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    network, inp, hid, out = r.build_model(config, 32)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, saved[:2], ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = values.reshape(shape).copy()
    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=condition["simulation_timesteps"], losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=32, rng_seed=42, max_spikes=8193, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="B32SinglePilotUpdate", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"updates": 12, "step": 12,
               "moments": {0: (saved[2].copy(), saved[3].copy()), 1: (saved[4].copy(), saved[5].copy())}}
    raw = bench.RawCapture(layers, context, 32)
    learn = LearnOne(layers, hid, out, context, reference)
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
    require(all(n == 1 for n in found.values()), "Unexpected callback layout")
    cn.base_train_callbacks[:] = modified
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "source_block_sha256": r.file_hash(paths[1]), "batch32_reference_sha256": r.file_hash(paths[2]),
        "source_checkpoint": checkpoint, "source_condition": condition,
        "batch_size": 32, "effective_real_batch_size": 22, "learning_rate": 0.001,
        "loss_gradient_normalization": "real-example mean; rescale reduced gradient by 32/22",
        "state_operation": "weights/moments/step12 warm start; fresh model/tapes/RNG seed42",
        "exact_resume": False, "validation_performed": False, "test_performed": False}
    emit("B32_UPDATE_PLAN", provenance)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    inputs = [r.to_ml_genn(p) for p in payloads] + [empty] * 10
    context["indices"] = list(range(22)) + [-1] * 10
    context["targets"] = np.r_[labels, np.zeros(10, dtype=np.int64)]
    metric = bench.mixed.MaskedAccuracy()
    summaries = {}
    with cn:
        for size, m, v in ((400, saved[2], saved[3]), (4312, saved[4], saved[5])):
            matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == size]
            require(len(matches) == 1, "Ambiguous moment restore")
            for key, values in (("M", m), ("V", v)):
                variable = matches[0].vars[key]
                variable.view[...] = values.reshape(variable.view.shape)
                variable.push_to_device()
        for a, b in zip(r.checkpoint.training_state(cn, layers), saved):
            np.testing.assert_array_equal(a.reshape(-1), b)
        observer.snapshot("after_load_and_restore")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient memory headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=4, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        for phase, first_batch in (("training", 0), ("fresh_evaluation", 2)):
            context["phase"] = phase
            state = {out: metric.create_state()}
            before_phase = r.checkpoint.training_state(cn, layers)
            scores = None
            for batch in (first_batch, first_batch + 1):
                drain = batch % 2 == 1
                context["step"] = context["updates"]
                metric.real_mask = np.zeros(32, dtype=bool) if drain else np.arange(32) < 22
                if not drain:
                    context["forward_state"] = r.checkpoint.training_state(cn, layers)
                with observer.phase(f"{phase}/presentation_{batch}"):
                    cn._train_batch(batch, context["step"], {inp: [empty] * 32 if drain else inputs},
                        {out: np.zeros(32, dtype=np.int64) if drain else context["targets"]},
                        {out: metric}, state, callbacks)
                    value = np.array(cn.get_readout(out), copy=True)
                if not drain:
                    for a, b in zip(context["forward_state"], r.checkpoint.training_state(cn, layers)):
                        np.testing.assert_array_equal(a, b)
                    scores = value[:22]
                    if phase == "training":
                        np.testing.assert_allclose(scores, reference["scores"], rtol=1e-9, atol=1e-11)
                        emit("B32_PREUPDATE_FORWARD_PARITY", {"all_22_scores": "pass"})
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
            require(state[out].total == 22, "Wrong metric denominator")
            if phase == "fresh_evaluation":
                for a, b in zip(before_phase, r.checkpoint.training_state(cn, layers)):
                    np.testing.assert_array_equal(a, b)
            summary = score_summary(scores, labels)
            summary.update({"phase": phase, "cumulative_optimizer_updates": context["updates"],
                            "scores_at_cumulative_updates": 12 if phase == "training" else 13,
                            "optimizer_updates_in_phase": 1 if phase == "training" else 0})
            summaries[phase] = summary
            emit("B32_PHASE_RESULT", {k: v for k, v in summary.items() if k != "scores"})
            observer.snapshot(f"after_{phase}")
            require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient sampled memory headroom")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
    require(context["updates"] == 13 and len(learn.records) == 2, "Wrong update/replay count")
    change = summaries["fresh_evaluation"]["mean_ce"] - summaries["training"]["mean_ce"]
    report = {**provenance, "phase_results": summaries, "optimizer_updates_this_run": 1,
        "cumulative_optimizer_updates": 13, "checkpoint": context["checkpoint"],
        "fresh_minus_initial_mean_ce": change, "mean_ce_decreased": change < 0,
        "memory_snapshots": observer.snapshots, "phases": observer.phases,
        "full_window_training_ready": False}
    with (Path.cwd() / "b32_update_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_B32_SINGLE_UPDATE", {"adam_step": 12, "optimizer_updates_this_run": 1,
         "cumulative_optimizer_updates": 13, "fixed_bound_clips": 0,
         "fresh_minus_initial_mean_ce": change, "mean_ce_decreased": change < 0,
         "fresh_correct": summaries["fresh_evaluation"]["correct"],
         "full_window_training_ready": False, "exact_resume": False})


if __name__ == "__main__":
    main()
