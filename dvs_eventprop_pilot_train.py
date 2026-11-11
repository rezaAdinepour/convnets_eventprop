"""One-epoch 22-parent training pilot with frozen initial/final evaluations.

Full 1024ms, double scalar, B8, paired real/drain schedule, fixed bound 1.
No validation or official test events. Checkpoints are not exact-resume states.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_genn.metrics.sparse_categorical_accuracy import SparseCategoricalAccuracy

import dvs_eventprop_candidate_probe as candidate

r = candidate.resource
require, emit, pull = r.require, r.emit, r.pull


def save_state(cn, layers, name):
    values = r.checkpoint.training_state(cn, layers)
    require(len(values) == 6 and all(np.isfinite(v).all() for v in values), "Invalid saved state")
    require([v.size for v in values] == [400, 4312, 400, 400, 4312, 4312],
            "Checkpoint weight/moment ownership differs from the checked layout")
    destination = Path.cwd() / name
    with destination.open("xb") as stream:
        np.savez(stream, **dict(zip(
            ("hidden", "output", "adam_hidden_m", "adam_hidden_v", "adam_output_m", "adam_output_v"),
            values,
        )))
    return {"path": str(destination), "sha256": r.file_hash(destination),
            "state_coverage": "weights and Adam moments; not complete training resume"}


class PilotMetric(SparseCategoricalAccuracy):
    def __init__(self, context, labels):
        self.context, self.labels = context, labels
        self.scores, self.indices, self.losses = [], [], []

    def update(self, state, y_true, y_pred, communicator):
        require(y_pred.shape == (8, 11) and np.isfinite(y_pred).all(), "Invalid forward scores")
        indices = np.asarray(self.context["metric_indices"], dtype=np.int64)
        mask = indices >= 0
        np.testing.assert_allclose(y_pred[~mask], 0, rtol=0, atol=1e-7)
        if mask.any():
            targets = np.asarray(y_true)[mask]
            np.testing.assert_array_equal(targets, self.labels[indices[mask]])
            scores = np.asarray(y_pred[mask], dtype=np.float64)
            maximum = scores.max(axis=1)
            losses = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
            losses -= scores[np.arange(len(targets)), targets]
            require(np.isfinite(losses).all(), "Nonfinite cross entropy")
            self.scores.extend(scores.tolist())
            self.indices.extend(indices[mask].tolist())
            self.losses.extend(losses.tolist())
            super().update(state, targets, scores, communicator)
            emit("PILOT_FORWARD", {"phase": self.context["phase"],
                 "physical_batch": self.context["physical_batch"],
                 "real_examples": int(mask.sum()), "mean_ce": float(losses.mean())})


class CaptureRaw(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, context):
        super().__init__("GradientBatchReduce")
        self.layers, self.context = layers, context

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return
        mask = np.asarray(self.context["forward_indices"]) >= 0
        raw = []
        for layer, size in zip(self.layers, (400, 4312)):
            gradient = pull(cn.connection_populations[layer.connection()].vars["weightGradient"])
            require(gradient.dtype == np.float64 and gradient.size == 8 * size
                    and np.isfinite(gradient).all(), "Invalid raw pilot gradients")
            gradient = gradient.reshape(8, size)
            np.testing.assert_array_equal(gradient[~mask], np.zeros_like(gradient[~mask]))
            raw.append(gradient.copy())
        self.context["raw_gradients"] = raw


class LearnOrFreeze(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, hidden, output, context):
        super().__init__("GradientLearn")
        self.layers, self.hidden, self.output, self.context = layers, hidden, output, context

    def on_batch_end(self, cn, batch, metrics):
        reduced_vars = []
        for size in (400, 4312):
            matches = [cu.vars["ReducedGradient"] for name, cu in cn.genn_model.custom_updates.items()
                       if name.startswith("CUBatchReduce") and "ReducedGradient" in cu.vars
                       and cu.vars["ReducedGradient"].view.size == size]
            require(len(matches) == 1, "Ambiguous reduced gradient ownership")
            reduced_vars.append(matches[0])
        if batch % 2 == 0:
            for variable in reduced_vars:
                value = pull(variable)
                np.testing.assert_array_equal(value, np.zeros_like(value))
            return
        ctx = self.context
        indices = np.asarray(ctx["forward_indices"])
        mask = indices >= 0
        count = int(mask.sum())
        require(count in (6, 8), "Unexpected pilot batch size")
        pop = cn.neuron_populations[self.hidden.population()]
        np.testing.assert_array_equal(pull(pop.vars["LambdavLimit"]).reshape(-1), np.ones(8))
        np.testing.assert_array_equal(
            pull(cn.neuron_populations[self.output.population()].vars["YTrueBack"]).reshape(-1),
            ctx["forward_targets"],
        )
        before = r.checkpoint.training_state(cn, self.layers)
        for actual, expected in zip(before[:2], ctx["forward_weights"]):
            np.testing.assert_array_equal(actual, expected)
        telemetry = {name: pull(pop.vars[name]).reshape(8, 392)
                     for name in candidate.audit.COUNTERS + candidate.audit.SCALARS}
        require(all(np.isfinite(v).all() for v in telemetry.values()), "Invalid adjoint telemetry")
        for key in ("AuditClips", "AuditNonfinite", "AuditBadLimit"):
            require(np.all(telemetry[key] == 0), f"Pilot stopped before update: {key} is nonzero")
        require(np.all(telemetry["AuditJumps"][~mask] == 0), "Padding replay emitted jumps")
        gradients = []
        for variable, raw, layer in zip(reduced_vars, ctx["raw_gradients"], self.layers):
            reduced = pull(variable).reshape(-1)
            np.testing.assert_allclose(reduced, raw.sum(axis=0), rtol=1e-9, atol=1e-11)
            gradient = reduced * (8.0 / count)
            require(np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-12,
                    "Invalid or zero aggregate pilot gradient")
            gradients.append(gradient)
            cleared = pull(cn.connection_populations[layer.connection()].vars["weightGradient"])
            np.testing.assert_array_equal(cleared, np.zeros_like(cleared))
        row = {"phase": ctx["phase"], "physical_batch": batch,
               "indices": indices[mask].tolist(), "real_count": count,
               "gradient_rescale": 8.0 / count,
               "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in gradients],
               "replayed_jumps_per_real_example": telemetry["AuditJumps"][mask].sum(axis=1).astype(int).tolist(),
               "maximum_replayed_spikes_per_neuron": int(telemetry["AuditJumps"].max()),
               "clipped_jumps": 0, "forward_replay_weights": "exact_match"}
        if ctx["phase"] == "training_epoch":
            step = ctx["updates"]
            require(ctx["step"] == step, "Adam clock mismatch")
            for variable, gradient in zip(reduced_vars, gradients):
                variable.view[...] = gradient.reshape(variable.view.shape)
                variable.push_to_device()
            cn.custom_update("GradientLearn")
            changes = []
            for index, (weight, gradient) in enumerate(zip(before[:2], gradients)):
                m, v = ctx["moments"].get(index, (np.zeros_like(gradient), np.zeros_like(gradient)))
                m, v = 0.9 * m + 0.1 * gradient, 0.999 * v + 0.001 * gradient ** 2
                expected = weight.reshape(-1) - 0.001 * (m / (1 - 0.9 ** (step + 1))) / (
                    np.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
                group = cn.connection_populations[self.layers[index].connection()]
                actual = pull(group.vars["weight"]).reshape(-1)
                require(np.isfinite(actual).all(), "Nonfinite learned weight")
                np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
                matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == gradient.size]
                require(len(matches) == 1, "Ambiguous Adam moments")
                np.testing.assert_allclose(pull(matches[0].vars["M"]).reshape(-1), m,
                                           rtol=2e-10, atol=2e-12)
                np.testing.assert_allclose(pull(matches[0].vars["V"]).reshape(-1), v,
                                           rtol=2e-10, atol=2e-12)
                ctx["moments"][index] = (m, v)
                changes.append(float(np.linalg.norm(actual - weight.reshape(-1))))
            require(all(v > 0 for v in changes), "A learned projection did not change")
            ctx["updates"] += 1
            checkpoint = save_state(cn, self.layers, f"update_{ctx['updates']:04d}.npz")
            ctx["checkpoints"].append(checkpoint)
            row.update({"adam_step": step, "host_adam_weights_and_moments": "pass",
                        "weight_change_l2_hidden_output": changes, "checkpoint": checkpoint})
        else:
            for actual, expected in zip(r.checkpoint.training_state(cn, self.layers), before):
                np.testing.assert_array_equal(actual, expected)
            row["weights_and_moments_unchanged"] = True
        ctx["replays"].append(row)
        emit("PILOT_REPLAY", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    baseline_path, gate_path = [p.resolve(strict=True) for p in (args.baseline, args.gate)]
    baseline, gate = [r.json.loads(p.read_text()) for p in (baseline_path, gate_path)]
    identity = r.checkpoint.identity()
    for artifact in (baseline, gate):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment mismatch: {key}")
    require(gate["baseline_sha256"] == r.file_hash(baseline_path), "Gate baseline binding changed")
    require(gate["diagnostic_sha256"] == r.file_hash(Path(__file__).with_name("dvs_eventprop_mixed_batch_probe.py")),
            "Gate producer source changed")
    require(baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(candidate.__file__)),
            "Precision adapter source changed")
    require(len(gate["continuous_reference_comparisons"]) == 6
            and all(row["within_1pct"] for row in gate["continuous_reference_comparisons"])
            and gate["continuous_total_counts_match"] is True,
            "Mixed numerical gate has not passed")
    require(gate["optimizer_updates"] == 0 and gate["weights_and_moments_unchanged"],
            "Gate was not frozen")
    condition = gate["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.00006103515625
            and condition["batch_size"] == 8 and condition["compiler_max_spikes"] == 8193
            and condition["scalar_precision"] == "double", "Unexpected passed condition")
    root = args.root.resolve(strict=True)
    selected = r.pilot(r.inventory(root)["train"], 2)
    require(len(selected) == 22, "Expected 22 training parents")
    order = np.random.default_rng(42).permutation(22)
    rows = [selected[int(i)] for i in order]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(np.bincount(labels, minlength=11), np.full(11, 2))
    metadata = r.normalize(list(metadata))
    for expected in gate["sample_metadata"]:
        matches = [m for m in metadata if m["stable_id"] == expected["stable_id"]]
        require(matches == [expected], "Pilot does not reproduce gate input provenance")
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    np.random.seed(42)
    network, inp, hid, out = r.build_model(config, 8)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, baseline["initial_weights"], ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    compiler = candidate.DoubleScalarAuditCompiler(
        example_timesteps=condition["simulation_timesteps"], losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=8, rng_seed=42, max_spikes=8193,
        strict_buffer_checking=True, per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="FullWindowPilot22", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    require(len(compiler.instrumentation) == 1, "Hidden audit missing")
    context = {"updates": 0, "step": 0, "moments": {}, "replays": [], "checkpoints": []}
    raw_capture = CaptureRaw(layers, context)
    learn = LearnOrFreeze(layers, hid, out, context)
    counts = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    callbacks_modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in counts:
            counts[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                callbacks_modified.extend((raw_capture, cb))
            elif cb.name == "GradientLearn":
                callbacks_modified.append(learn)
            else:
                callbacks_modified.append(r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False))
        else:
            callbacks_modified.append(cb)
    require(all(n == 1 for n in counts.values()), "Unexpected training callback layout")
    cn.base_train_callbacks[:] = callbacks_modified
    training_condition = {**condition, "real_examples": 22, "unique_training_parents": 22,
        "training_epochs": 1, "training_updates": 3, "optimizer_updates": 3,
        "real_batch_sizes": [8, 8, 6], "limiter_policy": "fixed bound 1; stop on observed clipping",
        "runner": "paired real-empty; frozen control, training, frozen final evaluation"}
    for key in ("layout", "empty_lanes", "empty_drains"):
        training_condition.pop(key, None)
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "baseline_sha256": r.file_hash(baseline_path), "gate_sha256": r.file_hash(gate_path),
        "condition": training_condition, "seed": 42, "permutation_of_sorted_pilot": order.tolist(),
        "training_order": [row.stable_id for row in rows], "sample_metadata": metadata,
        "validation_performed": False, "test_performed": False}
    emit("PILOT_PLAN", provenance)
    with (Path.cwd() / "pilot_plan.json").open("x") as stream:
        r.json.dump(provenance, stream, sort_keys=True, allow_nan=False)
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    inputs = [r.to_ml_genn(p) for p in payloads]
    phase_results = {}
    with cn:
        cn.get_readout(out)
        observer.snapshot("after_load")
        initial_state = r.checkpoint.training_state(cn, layers)
        require(all(v.dtype == np.float64 for v in initial_state), "Training precision changed")
        require([r.capacity.array_hash(v) for v in initial_state[:2]] == baseline["initial_weight_sha256"],
                "Initial weights differ")
        require(all(np.all(v == 0) for v in initial_state[2:]), "Initial moments are nonzero")
        initial_checkpoint = save_state(cn, layers, "initial_weights.npz")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn, num_batches=18, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        physical_batch = 0
        for phase in ("initial_control", "training_epoch", "final_evaluation"):
            emit("PILOT_PHASE_BEGIN", {"phase": phase, "optimizer_updates": context["updates"]})
            context["phase"] = phase
            metric = PilotMetric(context, labels)
            state = {out: metric.create_state()}
            phase_before = r.checkpoint.training_state(cn, layers)
            updates_before = context["updates"]
            for start in (0, 8, 16):
                indices = list(range(start, min(start + 8, 22)))
                indices += [-1] * (8 - len(indices))
                real_inputs = [inputs[i] if i >= 0 else empty for i in indices]
                targets = np.array([labels[i] if i >= 0 else 0 for i in indices], dtype=np.int64)
                for drain in (False, True):
                    context["physical_batch"] = physical_batch
                    context["step"] = context["updates"]
                    context["metric_indices"] = [-1] * 8 if drain else indices
                    before = r.checkpoint.training_state(cn, layers)
                    if not drain:
                        context["forward_indices"] = indices
                        context["forward_targets"] = targets.copy()
                        context["forward_weights"] = before[:2]
                    with observer.phase(f"{phase}/presentation_{physical_batch}"):
                        cn._train_batch(physical_batch, context["step"],
                            {inp: [empty] * 8 if drain else real_inputs},
                            {out: np.zeros(8, dtype=np.int64) if drain else targets},
                            {out: metric}, state, callbacks)
                        cn.get_readout(out)
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
            require(metric.indices == list(range(22)) and state[out].total == 22, "Wrong phase sample coverage")
            if phase == "initial_control":
                for sample, gate_lane in zip(gate["sample_metadata"], (0, 1)):
                    index = [row.stable_id for row in rows].index(sample["stable_id"])
                    np.testing.assert_allclose(metric.scores[index], gate["scores"][0][gate_lane],
                                               rtol=1e-9, atol=1e-11)
                emit("PILOT_CONTROL_GATE_PARITY", {"A_and_B_scores": "pass",
                     "weights_and_moments": "frozen", "optimizer_updates": context["updates"]})
            if phase != "training_epoch":
                require(context["updates"] == updates_before, "Frozen phase applied an optimizer update")
                for a, b in zip(phase_before, r.checkpoint.training_state(cn, layers)):
                    np.testing.assert_array_equal(a, b)
            else:
                require(context["updates"] - updates_before == 3, "Expected exactly three learning updates")
            summary = {"phase": phase, "real_examples": 22,
                "correct": int(state[out].correct), "accuracy": float(state[out].result),
                "mean_ce": float(np.mean(metric.losses)),
                "optimizer_updates_in_phase": context["updates"] - updates_before,
                "scores": metric.scores, "labels": labels.tolist(),
                "score_parameter_state": "changing during epoch" if phase == "training_epoch" else "fixed"}
            phase_results[phase] = summary
            emit("PILOT_PHASE_RESULT", summary)
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
    require(physical_batch == 18 and context["updates"] == 3, "Wrong presentation/update count")
    updates = [row for row in context["replays"] if row["phase"] == "training_epoch"]
    require([row["adam_step"] for row in updates] == [0, 1, 2]
            and [row["real_count"] for row in updates] == [8, 8, 6], "Wrong learning coverage")
    change = phase_results["final_evaluation"]["mean_ce"] - phase_results["initial_control"]["mean_ce"]
    emit("PILOT_LEARNING_CHANGE", {"fresh_final_minus_initial_mean_ce": change,
         "mean_ce_decreased": change < 0,
         "initial_correct": phase_results["initial_control"]["correct"],
         "fresh_final_correct": phase_results["final_evaluation"]["correct"],
         "convergence_certified": False})
    report = {**provenance, "phase_results": phase_results, "replay_records": context["replays"],
        "initial_checkpoint": initial_checkpoint, "update_checkpoints": context["checkpoints"],
        "optimizer_updates": context["updates"], "memory_snapshots": observer.snapshots,
        "phases": observer.phases, "full_window_training_ready": False,
        "exact_training_resume_supported": False}
    with (Path.cwd() / "pilot_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_PILOT_ONE_EPOCH_ENGINEERING", {"optimizer_updates": 3, "adam_steps": [0, 1, 2],
        "initial_and_final_evaluations_frozen": True, "real_training_examples": 22,
        "fixed_bound_clips": 0, "mean_ce_decreased": change < 0,
        "validation_performed": False, "test_performed": False,
        "convergence_certified": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
