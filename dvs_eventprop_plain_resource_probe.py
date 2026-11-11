"""Frozen B1 full-window parity/timing with spike recording and audit disabled.

No training updates. Source-checked private training API; fresh run directory.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_candidate_probe as candidate

r = candidate.resource
require, emit, pull = r.require, r.emit, r.pull


class DoublePlainCompiler(candidate.DoubleScalarAuditCompiler):
    def _build_in_hid_neuron_model(self, pop, model, genn_model, compile_state):
        # Retain the validated double scalar/buffer adapter, while bypassing
        # AuditCompiler's hidden-neuron telemetry insertion only.
        return r.EventPropCompiler._build_in_hid_neuron_model(
            self, pop, model, genn_model, compile_state
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    path = args.baseline.resolve(strict=True)
    baseline = r.json.loads(path.read_text())
    identity = r.checkpoint.identity()
    for key, value in identity.items():
        if key != "convnets_commit":
            require(baseline["identity"][key] == value, f"Baseline identity mismatch: {key}")
    require(baseline["identity"]["diagnostic_sha256"]
            == r.file_hash(Path(candidate.__file__)), "Candidate precision adapter source changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest,
                f"Baseline helper changed: {name}")
    condition = baseline["condition"]
    require(condition["frames"] == 1024 and condition["dt_ms"] == 0.0001220703125
            and condition["scalar_precision"] == "double"
            and condition["compiler_max_spikes"] == 8193,
            "Expected passed B1 full-window condition")
    require(baseline["optimizer_updates"] == 0 and baseline["weights_and_moments_unchanged"],
            "Baseline was not frozen")
    require(all(p["within_1pct"] for p in baseline["continuous_reference_comparison"]["projections"]),
            "Baseline did not pass continuous reference gradient gate")
    observer = r.ObserveResources()
    observer.snapshot("baseline")
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
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, baseline["initial_weights"],
                                    ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    compiler = DoublePlainCompiler(
        example_timesteps=condition["simulation_timesteps"],
        losses="sparse_categorical_crossentropy", dt=config.dt_ms, batch_size=1,
        rng_seed=42, max_spikes=8193, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="PlainFrozenB1", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    require(compiler.instrumentation == [], "Limiter telemetry was not disabled")
    capture = r.capacity.CaptureAndClear(layers, hid, out, np.array([label]), 8193)
    found = {name: 0 for name in ("GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    for i, callback in enumerate(cn.base_train_callbacks):
        if isinstance(callback, r.CustomUpdateOnBatchEnd) and callback.name in found:
            found[callback.name] += 1
            cn.base_train_callbacks[i] = (capture if callback.name == "GradientLearn" else
                r.CustomUpdateOnBatchEnd(callback.name, lambda batch: False))
    require(all(n == 1 for n in found.values()), "Unexpected callback structure")
    emit("PLAIN_RESOURCE_PLAN", {"identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)), "baseline_sha256": r.file_hash(path),
        "condition": condition, "spike_recording": False, "limiter_instrumentation": False,
        "limiter_policy": "fixed bound 1", "optimizer_updates": 0,
        "scope": "frozen B1 parity and diagnostic timing; not production training throughput"})
    real = r.to_ml_genn(payload)
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    with observer.phase("build"):
        cn.genn_model.build()
    observer.snapshot("after_build")
    with cn:
        cn.get_readout(out)
        observer.snapshot("after_load")
        initial_state = r.checkpoint.training_state(cn, layers)
        require(all(a.dtype == np.float64 for a in initial_state), "Storage precision changed")
        require([r.capacity.array_hash(w) for w in initial_state[:2]]
                == baseline["initial_weight_sha256"], "Initial weights differ")
        for layer in layers:
            pop = cn.neuron_populations[layer.population()]
            require(pull(pop.vars["v"]).dtype == np.float64
                    and pull(pop.vars["Lambdav"]).dtype == np.float64,
                    "Neuron precision changed")
        require(not any(name in cn.neuron_populations[hid.population()].vars
                        for name in candidate.audit.COUNTERS + candidate.audit.SCALARS),
                "Audit variables remain in the plain model")
        metric = r.RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=2, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []
        for batch, sample in enumerate((real, empty)):
            metric.real_presentation = batch == 0
            with observer.phase(f"presentation_{batch}"):
                cn._train_batch(batch, 0, {inp: [sample]},
                    {out: np.array([label if batch == 0 else 0], dtype=np.int64)},
                    {out: metric}, state, callbacks)
                scores.append(np.array(cn.get_readout(out), copy=True)[0])
            observer.snapshot(f"after_presentation_{batch}")
            pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1 and len(capture.records) == 1, "Wrong metric/replay count")
        for a, b in zip(initial_state, r.checkpoint.training_state(cn, layers)):
            np.testing.assert_array_equal(a, b)
        record = capture.records[0]
        np.testing.assert_array_equal(record["limit"].reshape(-1), [1.0])
        differences = []
        for index, (a, b) in enumerate(zip(record["gradients"], baseline["gradients"])):
            actual, expected = a.reshape(-1), np.asarray(b)
            require(np.isfinite(actual).all(), "Nonfinite gradient")
            row = {"projection": index,
                "relative_l2_difference": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                "max_absolute_difference": float(np.max(np.abs(actual - expected)))}
            emit("PLAIN_GRADIENT_PARITY", row)
            differences.append(row)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(scores, baseline["scores"], rtol=1e-9, atol=1e-11)
    observer.snapshot("after_unload")
    minimum_free = min(s["free_fraction"] for s in observer.snapshots)
    pair_ms = sum(p["profiled_wall_ms"] for p in observer.phases
                  if p["name"].startswith("presentation_"))
    baseline_ms = sum(p["profiled_wall_ms"] for p in baseline["phases"]
                      if p["name"].startswith("presentation_"))
    report = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "baseline_sha256": r.file_hash(path), "condition": condition,
        "sample_metadata": metadata, "spike_recording": False, "limiter_instrumentation": False,
        "scores": np.asarray(scores).tolist(), "gradients": [g.tolist() for g in record["gradients"]],
        "gradient_parity": differences, "phases": observer.phases,
        "memory_snapshots": observer.snapshots, "minimum_sampled_free_fraction": minimum_free,
        "frozen_pair_wall_ms": pair_ms, "historical_recorded_pair_wall_ms": baseline_ms,
        "timing_comparison_is_controlled_benchmark": False, "optimizer_updates": 0,
        "weights_and_moments_unchanged": True, "full_window_training_ready": False}
    with (Path.cwd() / "plain_resource_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    require(minimum_free >= 0.20, "Less than 20 percent sampled memory headroom")
    emit("PASS_PLAIN_B1_RESOURCE_PARITY", {"frozen_pair_wall_ms": pair_ms,
         "historical_recorded_pair_wall_ms": baseline_ms,
         "minimum_sampled_free_fraction": minimum_free, "optimizer_updates": 0,
         "scores_and_gradients_match": True, "weights_and_moments_unchanged": True,
         "production_training_throughput_certified": False})


if __name__ == "__main__":
    main()
