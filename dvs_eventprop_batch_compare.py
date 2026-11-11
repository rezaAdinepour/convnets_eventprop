"""Frozen full-pilot B8/B32 comparison at the saved epoch4 weights.

Checks individual derivatives, full real-example mean, scores and tape counts.
No optimizer updates. Two models run sequentially, not simultaneously.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_pilot_train as pilot
import dvs_eventprop_mixed_batch_probe as mixed

r = pilot.r
require, emit, pull = r.require, r.emit, r.pull


class RawCapture(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, context, batch_size):
        super().__init__("GradientBatchReduce")
        self.layers, self.context, self.batch_size = layers, context, batch_size

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return
        mask = np.asarray(self.context["indices"]) >= 0
        values = []
        for layer, size in zip(self.layers, (400, 4312)):
            value = pull(cn.connection_populations[layer.connection()].vars["weightGradient"])
            require(value.dtype == np.float64 and value.size == self.batch_size * size
                    and np.isfinite(value).all(), "Invalid raw batch gradients")
            value = value.reshape(self.batch_size, size)
            np.testing.assert_array_equal(value[~mask], np.zeros_like(value[~mask]))
            values.append(value.copy())
        self.context["raw"] = values


class ReducedCapture(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, hidden, output, context, batch_size):
        super().__init__("GradientLearn")
        self.layers, self.hidden, self.output = layers, hidden, output
        self.context, self.batch_size = context, batch_size
        self.records = []

    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return
        indices = np.asarray(self.context["indices"])
        mask = indices >= 0
        count = int(mask.sum())
        pop = cn.neuron_populations[self.hidden.population()]
        np.testing.assert_array_equal(pull(pop.vars["LambdavLimit"]).reshape(-1), np.ones(self.batch_size))
        np.testing.assert_array_equal(
            pull(cn.neuron_populations[self.output.population()].vars["YTrueBack"]).reshape(-1),
            self.context["targets"],
        )
        telemetry = {name: pull(pop.vars[name]).reshape(self.batch_size, 392)
                     for name in pilot.candidate.audit.COUNTERS + pilot.candidate.audit.SCALARS}
        require(all(np.isfinite(v).all() for v in telemetry.values()), "Invalid telemetry")
        require(all(np.all(telemetry[k] == 0) for k in ("AuditClips", "AuditNonfinite", "AuditBadLimit")),
                "Batch comparison replay was limited or invalid")
        np.testing.assert_array_equal(telemetry["AuditJumps"][~mask],
                                      np.zeros_like(telemetry["AuditJumps"][~mask]))
        individual, means = [], []
        for layer, size, raw in zip(self.layers, (400, 4312), self.context["raw"]):
            matches = [cu.vars["ReducedGradient"] for name, cu in cn.genn_model.custom_updates.items()
                       if name.startswith("CUBatchReduce") and "ReducedGradient" in cu.vars
                       and cu.vars["ReducedGradient"].view.size == size]
            require(len(matches) == 1, "Ambiguous reduced gradient")
            reduced = pull(matches[0]).reshape(-1)
            require(reduced.dtype == np.float64, "Reduction precision changed")
            np.testing.assert_allclose(reduced, raw.sum(axis=0), rtol=1e-9, atol=1e-11)
            individual.append(raw[mask] * self.batch_size)
            means.append(reduced * (self.batch_size / count))
            cleared = pull(cn.connection_populations[layer.connection()].vars["weightGradient"])
            np.testing.assert_array_equal(cleared, np.zeros_like(cleared))
        self.records.append({"indices": indices[mask].tolist(), "count": count,
            "individual": individual, "mean": means,
            "counts": telemetry["AuditJumps"][mask].copy()})
        emit("FROZEN_BATCH_CAPTURE", {"batch_size": self.batch_size, "physical_batch": batch,
             "real_count": count, "gradient_rescale": self.batch_size / count,
             "clipped_jumps": 0, "maximum_replayed_spikes_per_neuron": int(telemetry["AuditJumps"].max())})


def run(batch_size, payloads, labels, weights, condition, expected_scores, expected_totals, provenance):
    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    network, inp, hid, out = r.build_model(config, batch_size)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, weights, ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = values.reshape(shape).copy()
    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=condition["simulation_timesteps"], losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=batch_size, rng_seed=42, max_spikes=8193,
        strict_buffer_checking=True, per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name=f"FrozenPilotB{batch_size}", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {}
    raw = RawCapture(layers, context, batch_size)
    capture = ReducedCapture(layers, hid, out, context, batch_size)
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
    require(all(n == 1 for n in found.values()), "Unexpected callback layout")
    cn.base_train_callbacks[:] = modified
    observer = r.ObserveResources()
    observer.snapshot(f"B{batch_size}/baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    inputs = [r.to_ml_genn(p) for p in payloads]
    metric = mixed.MaskedAccuracy()
    state = {out: metric.create_state()}
    scores, ordered_indices = [], []
    pair_count = (22 + batch_size - 1) // batch_size
    emit("FROZEN_BATCH_BEGIN", {"batch_size": batch_size, "real_examples": 22,
         "pairs": pair_count, "optimizer_updates": 0, "limiter_policy": "fixed bound 1"})
    with cn:
        cn.get_readout(out)
        observer.snapshot(f"B{batch_size}/after_load")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient memory headroom after load")
        initial = r.checkpoint.training_state(cn, layers)
        require(all(v.dtype == np.float64 for v in initial), "Training precision changed")
        for a, b in zip(initial[:2], weights):
            np.testing.assert_array_equal(a.reshape(-1), b)
        require(all(np.all(v == 0) for v in initial[2:]), "Fresh optimizer moments not zero")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=2 * pair_count, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for start in range(0, 22, batch_size):
            indices = list(range(start, min(start + batch_size, 22)))
            real_count = len(indices)
            indices += [-1] * (batch_size - real_count)
            context["indices"] = indices
            context["targets"] = np.array([labels[i] if i >= 0 else 0 for i in indices], dtype=np.int64)
            real_inputs = [inputs[i] if i >= 0 else empty for i in indices]
            for drain in (False, True):
                metric.real_mask = np.zeros(batch_size, dtype=bool) if drain else np.array(indices) >= 0
                with observer.phase(f"B{batch_size}/presentation_{batch}"):
                    cn._train_batch(batch, 0, {inp: [empty] * batch_size if drain else real_inputs},
                        {out: np.zeros(batch_size, dtype=np.int64) if drain else context["targets"]},
                        {out: metric}, state, callbacks)
                    value = np.array(cn.get_readout(out), copy=True)
                    require(value.shape == (batch_size, 11) and np.isfinite(value).all(), "Invalid scores")
                if not drain:
                    scores.extend(value[:real_count].tolist())
                    ordered_indices.extend(indices[:real_count])
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                batch += 1
                if drain:
                    observer.snapshot(f"B{batch_size}/after_pair_{batch // 2}")
                    require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient sampled headroom")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        for a, b in zip(initial, r.checkpoint.training_state(cn, layers)):
            np.testing.assert_array_equal(a, b)
    require(ordered_indices == list(range(22)) and state[out].total == 22, "Wrong metric/sample coverage")
    require([i for row in capture.records for i in row["indices"]] == list(range(22)), "Wrong gradient coverage")
    counts = np.concatenate([row["counts"] for row in capture.records])
    np.testing.assert_array_equal(counts.sum(axis=1), expected_totals)
    scores = np.asarray(scores)
    np.testing.assert_allclose(scores, expected_scores, rtol=1e-9, atol=1e-11)
    individual, means = [], []
    for index in range(2):
        individual.append(np.concatenate([row["individual"][index] for row in capture.records]))
        mean = sum(row["count"] * row["mean"][index] for row in capture.records) / 22.0
        np.testing.assert_allclose(mean, individual[index].mean(axis=0), rtol=1e-9, atol=1e-11)
        means.append(mean)
    wall_ms = sum(p["profiled_wall_ms"] for p in observer.phases if "/presentation_" in p["name"])
    report = {**provenance, "batch_size": batch_size, "real_examples": 22,
        "scores": scores.tolist(), "individual_gradients": [g.tolist() for g in individual],
        "full_real_mean_gradients": [g.tolist() for g in means], "hidden_counts": counts.tolist(),
        "real_batch_sizes": [row["count"] for row in capture.records],
        "optimizer_updates": 0, "weights_and_moments_unchanged": True, "clipped_jumps": 0,
        "frozen_22_example_wall_ms": wall_ms, "wall_ms_per_example": wall_ms / 22,
        "memory_snapshots": observer.snapshots, "phases": observer.phases}
    with (Path.cwd() / f"frozen_b{batch_size}_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("FROZEN_BATCH_RESULT", {k: report[k] for k in (
        "batch_size", "real_batch_sizes", "optimizer_updates", "clipped_jumps",
        "frozen_22_example_wall_ms", "wall_ms_per_example")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--block-result", type=Path, required=True)
    args = parser.parse_args()
    baseline_path, source_path = [p.resolve(strict=True) for p in (args.baseline, args.block_result)]
    baseline, source = [r.json.loads(p.read_text()) for p in (baseline_path, source_path)]
    identity = r.checkpoint.identity()
    for artifact in (baseline, source):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment mismatch: {key}")
    require(source["diagnostic_sha256"] == r.file_hash(Path(__file__).with_name("dvs_eventprop_pilot_block.py"))
            and source["pilot_producer_sha256"] == r.file_hash(Path(pilot.__file__)), "Pilot source changed")
    require(source["baseline_sha256"] == r.file_hash(baseline_path), "Baseline binding changed")
    require(baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(pilot.candidate.__file__)),
            "Precision adapter changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest, f"Helper changed: {name}")
    require(source["cumulative_optimizer_updates"] == 12 and source["epochs"][-1]["logical_epoch"] == 4,
            "Expected completed epoch4 block")
    final = source["epochs"][-1]
    expected = final["phase_results"]["fresh_evaluation"]
    require(expected["score_parameter_state"] == "fixed" and expected["optimizer_updates_in_phase"] == 0,
            "Expected frozen source scores")
    checkpoint = final["final_checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(checkpoint_path.is_relative_to(source_path.parent)
            and r.file_hash(checkpoint_path) == checkpoint["sha256"], "Checkpoint binding changed")
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        weights = [np.array(archive[name], copy=True) for name in ("hidden", "output")]
    for weight, size in zip(weights, (400, 4312)):
        require(weight.dtype == np.float64 and weight.shape == (size,) and np.isfinite(weight).all(),
                "Invalid checkpoint weights")
    condition = source["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.00006103515625
            and condition["compiler_max_spikes"] == 8193 and condition["scalar_precision"] == "double",
            "Unexpected physical condition")
    root = args.root.resolve(strict=True)
    by_id = {row.stable_id: row for row in r.inventory(root)["train"]}
    rows = [by_id[key] for key in source["training_order"]]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(len(rows) == 22 and r.normalize(list(metadata)) == source["sample_metadata"], "Input provenance changed")
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(labels, expected["labels"])
    expected_totals = [n for row in final["replay_records"] if row["phase"] == "fresh_evaluation"
                       for n in row["replayed_jumps_per_real_example"]]
    require(len(expected_totals) == 22, "Source replay coverage differs")
    provenance = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "source_block_sha256": r.file_hash(source_path), "source_checkpoint": checkpoint,
        "source_condition": condition, "sample_metadata": metadata,
        "operation": "weights-only frozen comparison; fresh zero Adam moments",
        "helper_sha256": {Path(m.__file__).name: r.file_hash(Path(m.__file__))
                          for m in (pilot, mixed, pilot.candidate)},
        "validation_performed": False, "test_performed": False}
    emit("BATCH_COMPARE_PLAN", provenance)
    b8 = run(8, payloads, labels, weights, condition, expected["scores"], expected_totals, provenance)
    b32 = run(32, payloads, labels, weights, condition, expected["scores"], expected_totals, provenance)
    np.testing.assert_array_equal(b8["hidden_counts"], b32["hidden_counts"])
    comparisons = []
    for projection in range(2):
        for key in ("individual_gradients", "full_real_mean_gradients"):
            a, b = [np.asarray(result[key][projection]) for result in (b8, b32)]
            row = {"projection": projection, "quantity": key,
                   "max_absolute_difference": float(np.max(np.abs(a - b)))}
            emit("B8_B32_GRADIENT_PARITY", row)
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
            comparisons.append(row)
    result = {"gradient_comparisons": comparisons, "hidden_per_neuron_counts": "exact_match",
        "B8_wall_ms": b8["frozen_22_example_wall_ms"], "B32_wall_ms": b32["frozen_22_example_wall_ms"],
        "observed_22_example_time_ratio_B8_over_B32": b8["frozen_22_example_wall_ms"] / b32["frozen_22_example_wall_ms"],
        "optimizer_updates": 0, "full_window_training_ready": False,
        "timing_scope": "single frozen comparison with duplicate evaluation padding; not full-data training throughput"}
    with (Path.cwd() / "batch_compare_result.json").open("x") as stream:
        r.json.dump({**provenance, **result}, stream, sort_keys=True, allow_nan=False)
    emit("PASS_FROZEN_B8_B32_PARITY", result)


if __name__ == "__main__":
    main()
