"""Paired real/empty diagnostics: Adam clock and adaptive limiter.

Default: two updates. Optional: one update followed by two frozen limiter
comparisons. This is not a learning-utility experiment.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_candidate_probe as candidate

r = candidate.resource
require, emit, pull = r.require, r.emit, r.pull


class PairLearn(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, hidden, output, context, baseline, compare_limiter=False):
        super().__init__("GradientLearn", lambda batch: batch % 2 == 1)
        self.layers, self.hidden, self.output = layers, hidden, output
        self.context, self.baseline = context, baseline
        self.records, self.moments = [], {}
        self.compare_limiter = compare_limiter
        self.optimizer_updates = 0
        self.captures = []
        self.frozen_state = []

    @property
    def optimizer_updates(self):
        # CallbackList shallow-copies callback objects. Store the counter in
        # the shared context rather than an independently copied integer.
        return self.context["optimizer_updates"]

    @optimizer_updates.setter
    def optimizer_updates(self, value):
        self.context["optimizer_updates"] = value

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            for layer in self.layers:
                gradient = pull(cn.connection_populations[layer.connection()]
                                .vars["weightGradient"])
                np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
            return
        step = self.context["step"]
        require(step == self.optimizer_updates, "Adam step/update count mismatch")
        groups = [cn.connection_populations[layer.connection()] for layer in self.layers]
        weights = [pull(group.vars["weight"]) for group in groups]
        gradients = [pull(group.vars["weightGradient"]) for group in groups]
        for before, actual in zip(self.context["forward_weights"], weights):
            np.testing.assert_array_equal(before, actual)
        np.testing.assert_array_equal(
            pull(cn.neuron_populations[self.output.population()].vars["YTrueBack"]).reshape(-1),
            [self.baseline["sample_metadata"]["label"]],
        )
        pop = cn.neuron_populations[self.hidden.population()]
        limit = pull(pop.vars["LambdavLimit"]).reshape(-1)
        require(np.isfinite(limit).all() and np.all(limit > 0), "Invalid replay bound")
        telemetry = {name: pull(pop.vars[name]).reshape(-1)
                     for name in candidate.audit.COUNTERS + candidate.audit.SCALARS}
        require(all(np.isfinite(v).all() for v in telemetry.values()), "Invalid telemetry")
        require(telemetry["AuditNonfinite"].sum() == 0
                and telemetry["AuditBadLimit"].sum() == 0, "Invalid adjoint replay")
        require(all(np.isfinite(g).all() for g in gradients), "Nonfinite gradient")
        if step == 0:
            for actual, expected in zip(gradients, self.baseline["gradients"]):
                np.testing.assert_allclose(actual.reshape(-1), expected,
                                           rtol=1e-9, atol=1e-11)
        if self.compare_limiter and batch > 1:
            state_before = r.checkpoint.training_state(cn, self.layers)
            if not self.frozen_state:
                self.frozen_state.extend(state_before)
            else:
                for expected, actual in zip(self.frozen_state, state_before):
                    np.testing.assert_array_equal(expected, actual)
            cn.custom_update("ZeroGradient")
            for expected, actual in zip(
                state_before, r.checkpoint.training_state(cn, self.layers)
            ):
                np.testing.assert_array_equal(expected, actual)
            row = {
                "physical_batch": batch, "real_example_index": batch // 2,
                "optimizer_updates": self.optimizer_updates,
                "forward_replay_weights": "exact_match",
                "replay_limit": limit.tolist(),
                "replayed_jumps": int(telemetry["AuditJumps"].sum()),
                "clipped_jumps": int(telemetry["AuditClips"].sum()),
                "max_raw_to_limit_ratio": float(telemetry["AuditMaxRatio"].max()),
                "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in gradients],
                "weights_and_moments_unchanged": True,
            }
            self.captures.append([g.copy() for g in gradients])
            self.records.append(row)
            emit("FROZEN_LIMITER_REPLAY", row)
            return
        super().on_batch_end(cn, batch, metrics)
        changes = []
        for index, (group, weight, gradient) in enumerate(zip(groups, weights, gradients)):
            m, v = self.moments.get(index, (np.zeros_like(gradient), np.zeros_like(gradient)))
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            expected = weight - 0.001 * (m / (1.0 - 0.9 ** (step + 1))) / (
                np.sqrt(v / (1.0 - 0.999 ** (step + 1))) + 1e-8
            )
            actual = pull(group.vars["weight"])
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
            matches = [cu for _, cu in cn.optimisers
                       if cu.vars["M"].view.size == gradient.size]
            require(len(matches) == 1, "Optimizer moment ownership ambiguous")
            np.testing.assert_allclose(pull(matches[0].vars["M"]), m,
                                       rtol=2e-10, atol=2e-12)
            np.testing.assert_allclose(pull(matches[0].vars["V"]), v,
                                       rtol=2e-10, atol=2e-12)
            self.moments[index] = (m, v)
            changes.append(float(np.linalg.norm(actual - weight)))
            np.testing.assert_array_equal(pull(group.vars["weightGradient"]),
                                          np.zeros_like(gradient))
        require(all(change > 0 for change in changes), "Missing projection update")
        row = {
            "physical_batch": batch, "real_example_index": step, "adam_step": step,
            "forward_replay_weights": "exact_match",
            "replay_limit": limit.tolist(),
            "replayed_jumps": int(telemetry["AuditJumps"].sum()),
            "clipped_jumps": int(telemetry["AuditClips"].sum()),
            "max_raw_to_limit_ratio": float(telemetry["AuditMaxRatio"].max()),
            "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in gradients],
            "weight_change_l2_hidden_output": changes,
            "host_adam_weights_and_moments": "pass",
        }
        self.records.append(row)
        self.optimizer_updates += 1
        emit("CONSISTENT_UPDATE", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--compare-limiter", action="store_true")
    args = parser.parse_args()
    path = args.baseline.resolve(strict=True)
    baseline = r.json.loads(path.read_text())
    current = r.checkpoint.identity()
    for key, value in current.items():
        if key != "convnets_commit":
            require(baseline["identity"][key] == value, f"Baseline identity mismatch: {key}")
    require(baseline["identity"]["diagnostic_sha256"]
            == r.file_hash(Path(candidate.__file__)), "Candidate producer source changed")
    condition = baseline["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.0001220703125
            and condition["scalar_precision"] == "double"
            and condition["compiler_max_spikes"] == 8193,
            "Expected the passed full-window candidate condition")
    comparison = baseline["continuous_reference_comparison"]
    require(comparison["unclipped_replay"] and not comparison["neurons_with_changed_counts"]
            and all(p["within_1pct"] for p in comparison["projections"]),
            "Baseline did not pass its continuous reference gate")
    require(baseline["optimizer_updates"] == 0
            and baseline["weights_and_moments_unchanged"], "Baseline was not frozen")
    root = args.root.resolve(strict=True)
    rows = [row for row in r.inventory(root)["train"]
            if row.stable_id == baseline["sample_metadata"]["stable_id"]]
    require(len(rows) == 1, "Training parent not uniquely identified")
    payload, label, metadata = r.load_sample(root, rows[0], r.DataConfig(1024))
    require(r.normalize(metadata) == baseline["sample_metadata"], "Input provenance changed")
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    np.random.seed(config.seed)
    network, inp, hid, out = r.build_model(config, 1)
    layers = [hid, out]
    for layer, values, shape in zip(layers, baseline["initial_weights"],
                                    ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    compiler = candidate.DoubleScalarAuditCompiler(
        example_timesteps=condition["simulation_timesteps"],
        losses="sparse_categorical_crossentropy", dt=config.dt_ms,
        batch_size=1, rng_seed=42, max_spikes=8193, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="ConsistentTwoUpdates", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"step": 0}
    trace = PairLearn(layers, hid, out, context, baseline, args.compare_limiter)
    counts = {name: 0 for name in ("GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    for i, callback in enumerate(cn.base_train_callbacks):
        if isinstance(callback, r.CustomUpdateOnBatchEnd) and callback.name in counts:
            counts[callback.name] += 1
            cn.base_train_callbacks[i] = (trace if callback.name == "GradientLearn" else
                r.CustomUpdateOnBatchEnd(callback.name, lambda batch: batch % 2 == 1))
    require(all(n == 1 for n in counts.values()), "Unexpected compiler callback layout")
    pairs = 3 if args.compare_limiter else 2
    expected_updates = 1 if args.compare_limiter else 2
    training_condition = {
        **condition, "real_examples": pairs, "empty_drains": pairs,
        "optimizer_updates": expected_updates,
        "schedule": "real-empty-real-empty; repeat the same training parent",
        "limiter_policy": "compiler adaptation after real replays only",
        "compare_updated_weight_limiter": args.compare_limiter,
        "manual_bound_one_before_final_pair": args.compare_limiter,
    }
    emit("CONSISTENT_PLAN", {
        "identity": current, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "baseline_sha256": r.file_hash(path), "condition": training_condition,
        "real_presentations": pairs, "empty_presentations": pairs,
        "expected_updates": expected_updates,
        "learning_batches": [1] if args.compare_limiter else [1, 3],
        "limit_adaptation_batches": list(range(1, 2 * pairs, 2)),
        "limiter_policy": "compiler adaptation after real replays only",
    })
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    real = r.to_ml_genn(payload)
    metric = r.RealOnlyAccuracy()
    state = {out: metric.create_state()}
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    with cn:
        require([r.capacity.array_hash(w) for w in r.checkpoint.training_state(cn, layers)[:2]]
                == baseline["initial_weight_sha256"], "Initial weights changed")
        require(all(np.all(v == 0) for v in r.checkpoint.training_state(cn, layers)[2:]),
                "Adam moments were not initially zero")
        callbacks = r.CallbackList(cn.base_train_callbacks + [
            r.SpikeRecorder(inp, key="input"), r.SpikeRecorder(hid, key="hidden"),
        ], compiled_network=cn, num_batches=2 * pairs, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        carried_limit = None
        scores = []
        for batch in range(2 * pairs):
            real_forward = batch % 2 == 0
            context["step"] = trace.optimizer_updates
            if args.compare_limiter and batch == 4:
                variable = cn.neuron_populations[hid.population()].vars["LambdavLimit"]
                old_limit = pull(variable)
                variable.view[...] = 1.0
                variable.push_to_device()
                carried_limit = pull(variable)
                np.testing.assert_array_equal(carried_limit, np.ones_like(carried_limit))
                emit("DECLARED_LIMIT_OVERRIDE", {"physical_batch": batch,
                    "before": old_limit.tolist(), "after": carried_limit.tolist(),
                    "weights_or_moments_modified": False})
            before = r.checkpoint.training_state(cn, layers)
            limit_before = pull(cn.neuron_populations[hid.population()].vars["LambdavLimit"])
            if real_forward:
                context["forward_weights"] = before[:2]
            metric.real_presentation = real_forward
            with observer.phase(f"presentation_{batch}"):
                cn._train_batch(batch, context["step"], {inp: [real if real_forward else empty]},
                                {out: np.array([label if real_forward else 0], dtype=np.int64)},
                                {out: metric}, state, callbacks)
                score = np.asarray(cn.get_readout(out))
                require(np.isfinite(score).all(), "Nonfinite scores")
                scores.append(score.copy())
                if batch == 0:
                    np.testing.assert_allclose(score[0], baseline["scores"][0], rtol=1e-9, atol=1e-11)
            limit_after = pull(cn.neuron_populations[hid.population()].vars["LambdavLimit"])
            if real_forward:
                for a, b in zip(before, r.checkpoint.training_state(cn, layers)):
                    np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(limit_before, limit_after)
                if carried_limit is not None:
                    np.testing.assert_array_equal(limit_after, carried_limit)
            else:
                carried_limit = limit_after.copy()
                require(np.isfinite(carried_limit).all() and np.all(carried_limit > 0),
                        "Real replay produced invalid adapted limit")
            emit("LIMITER_CARRY", {"physical_batch": batch,
                                  "before": limit_before.tolist(), "after": limit_after.tolist()})
            observer.snapshot(f"after_presentation_{batch}")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        recorded = callbacks.get_data()
        for batch in range(2 * pairs):
            require(r.output_probe.spike_pairs(recorded["input"], batch)
                    == r.expected_pairs(real if batch % 2 == 0 else empty), "Input delivery changed")
            hidden = r.output_probe.spike_pairs(recorded["hidden"], batch)
            if batch % 2:
                require(hidden == [], "Hidden spikes in empty drain")
            else:
                require(trace.records[batch // 2]["replayed_jumps"] == len(hidden),
                        "Real forward/replay spike count mismatch")
        if args.compare_limiter:
            require(r.output_probe.spike_pairs(recorded["hidden"], 2)
                    == r.output_probe.spike_pairs(recorded["hidden"], 4),
                    "Limiter comparison forward spike IDs/times differ")
            np.testing.assert_allclose(scores[2], scores[4], rtol=1e-9, atol=1e-11)
    require(len(trace.records) == pairs and state[out].total == pairs
            and trace.optimizer_updates == expected_updates, "Wrong update/metric count")
    require(min(s["free_fraction"] for s in observer.snapshots) >= 0.20,
            "Less than 20 percent sampled memory headroom")
    limiter_effect = None
    if args.compare_limiter:
        require(len(trace.captures) == 2 and trace.records[1]["clipped_jumps"] > 0,
                "Missing limited-gradient observation")
        require(trace.records[2]["replay_limit"] == [1.0]
                and trace.records[2]["clipped_jumps"] == 0,
                "Bound-one comparison was not unclipped")
        projections = []
        for index, (limited, unclipped) in enumerate(zip(*trace.captures)):
            require(np.linalg.norm(unclipped) > 0, "Zero control gradient")
            projections.append({"projection": index,
                "relative_l2_difference_to_unclipped": float(
                    np.linalg.norm(limited - unclipped) / np.linalg.norm(unclipped)),
                "max_absolute_difference": float(np.max(np.abs(limited - unclipped))),
                "cosine_similarity": float(np.sum(limited * unclipped)
                    / (np.linalg.norm(limited) * np.linalg.norm(unclipped)))})
        np.testing.assert_allclose(trace.captures[0][1], trace.captures[1][1],
                                   rtol=1e-9, atol=1e-11)
        limiter_effect = {"projections": projections, "optimizer_updates": 1,
            "forward_hidden_spike_id_time_parity": "exact",
            "comparison_weights_and_moments": "exact_match",
            "scope": "same updated weights and trajectory; limiter bound changed only",
            "unclipped_gradient_accuracy_certified": False}
        emit("UPDATED_WEIGHT_LIMITER_EFFECT", limiter_effect)
    report = {"baseline_sha256": r.file_hash(path), "identity": current,
              "diagnostic_sha256": r.file_hash(Path(__file__)),
              "condition": training_condition, "sample_metadata": metadata,
              "memory_snapshots": observer.snapshots, "phases": observer.phases,
              "updates": [row for row in trace.records if "adam_step" in row],
              "replay_records": trace.records, "optimizer_updates": expected_updates,
              "limiter_effect": limiter_effect,
              "captured_gradients": [[g.tolist() for g in pair] for pair in trace.captures],
              "frozen_comparison_weights_and_moments": [v.tolist() for v in trace.frozen_state],
              "scope": "repeated real examples; consistent weights per pair"}
    with (Path.cwd() / "consistent_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_UPDATED_WEIGHT_LIMITER_EFFECT" if args.compare_limiter else "PASS_CONSISTENT_TWO_UPDATES", {
        "optimizer_updates": expected_updates, "adam_steps": list(range(expected_updates)),
        "first_gradient_matches_frozen_baseline": True,
        "real_metric_examples": pairs, "limiter_adaptation_excludes_empty_replays": True,
        "learning_utility_certified": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
