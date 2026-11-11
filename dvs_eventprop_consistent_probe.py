"""Two real/empty pairs: consistent weights, Adam clock and adaptive limiter.

Repeats one training example twice. This is not a learning-utility experiment.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_candidate_probe as candidate

r = candidate.resource
require, emit, pull = r.require, r.emit, r.pull


class PairLearn(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, hidden, output, context, baseline):
        super().__init__("GradientLearn", lambda batch: batch % 2 == 1)
        self.layers, self.hidden, self.output = layers, hidden, output
        self.context, self.baseline = context, baseline
        self.records, self.moments = [], {}

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            for layer in self.layers:
                gradient = pull(cn.connection_populations[layer.connection()]
                                .vars["weightGradient"])
                np.testing.assert_array_equal(gradient, np.zeros_like(gradient))
            return
        step = self.context["step"]
        require(step == len(self.records), "Adam step/update count mismatch")
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
        emit("CONSISTENT_UPDATE", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
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
    trace = PairLearn(layers, hid, out, context, baseline)
    counts = {name: 0 for name in ("GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    for i, callback in enumerate(cn.base_train_callbacks):
        if isinstance(callback, r.CustomUpdateOnBatchEnd) and callback.name in counts:
            counts[callback.name] += 1
            cn.base_train_callbacks[i] = (trace if callback.name == "GradientLearn" else
                r.CustomUpdateOnBatchEnd(callback.name, lambda batch: batch % 2 == 1))
    require(all(n == 1 for n in counts.values()), "Unexpected compiler callback layout")
    training_condition = {
        **condition, "real_examples": 2, "empty_drains": 2, "optimizer_updates": 2,
        "schedule": "real-empty-real-empty; repeat the same training parent",
        "limiter_policy": "compiler adaptation after real replays only",
    }
    emit("CONSISTENT_PLAN", {
        "identity": current, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "baseline_sha256": r.file_hash(path), "condition": training_condition,
        "real_presentations": 2, "empty_presentations": 2, "expected_updates": 2,
        "learning_and_limit_adaptation_batches": [1, 3],
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
        ], compiled_network=cn, num_batches=4, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        carried_limit = None
        for batch in range(4):
            real_forward = batch % 2 == 0
            context["step"] = len(trace.records)
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
        for batch in range(4):
            require(r.output_probe.spike_pairs(recorded["input"], batch)
                    == r.expected_pairs(real if batch % 2 == 0 else empty), "Input delivery changed")
            hidden = r.output_probe.spike_pairs(recorded["hidden"], batch)
            if batch % 2:
                require(hidden == [], "Hidden spikes in empty drain")
            else:
                require(trace.records[batch // 2]["replayed_jumps"] == len(hidden),
                        "Real forward/replay spike count mismatch")
    require(len(trace.records) == 2 and state[out].total == 2, "Wrong update/metric count")
    require(min(s["free_fraction"] for s in observer.snapshots) >= 0.20,
            "Less than 20 percent sampled memory headroom")
    report = {"baseline_sha256": r.file_hash(path), "identity": current,
              "diagnostic_sha256": r.file_hash(Path(__file__)),
              "condition": training_condition, "sample_metadata": metadata,
              "memory_snapshots": observer.snapshots, "phases": observer.phases,
              "updates": trace.records, "optimizer_updates": 2,
              "scope": "two repeated real examples; consistent weights per pair"}
    with (Path.cwd() / "consistent_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_CONSISTENT_TWO_UPDATES", {"optimizer_updates": 2, "adam_steps": [0, 1],
        "first_gradient_matches_frozen_baseline": True,
        "real_metric_examples": 2, "limiter_adaptation_excludes_empty_replays": True,
        "learning_utility_certified": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
