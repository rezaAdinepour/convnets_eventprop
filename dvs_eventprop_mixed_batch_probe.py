"""Frozen mixed B8 first-replay parity with empty lanes and no optimizer updates."""

import argparse
from pathlib import Path

import numpy as np
from ml_genn.metrics.sparse_categorical_accuracy import SparseCategoricalAccuracy

import dvs_eventprop_candidate_probe as candidate
import dvs_eventprop_plain_resource_probe as plain

r = candidate.resource
require, emit, pull = r.require, r.emit, r.pull
LAYOUT = (0, 1, 2, 0, 1, 2, 0, 1)  # A, B, empty, A, B, empty, A, B


class MaskedAccuracy(SparseCategoricalAccuracy):
    def update(self, state, y_true, y_pred, communicator):
        mask = np.asarray(self.real_mask, dtype=bool)
        require(mask.shape == (len(y_true),), "Invalid real-lane mask")
        np.testing.assert_allclose(y_pred[~mask], 0, rtol=0, atol=1e-7)
        if mask.any():
            super().update(state, y_true[mask], y_pred[mask], communicator)


class CaptureMixedLanes(r.CustomUpdateOnBatchEnd):
    def __init__(self, layers, hidden, output, baseline, labels):
        super().__init__("GradientBatchReduce")
        self.layers, self.hidden, self.output = layers, hidden, output
        self.baseline, self.labels = baseline, labels
        self.raw, self.statistics = [], []

    def on_batch_end(self, cn, batch, metrics):
        if batch != 1:
            return
        np.testing.assert_array_equal(
            pull(cn.neuron_populations[self.output.population()].vars["YTrueBack"]).reshape(-1),
            self.labels,
        )
        for index, (layer, size) in enumerate(zip(self.layers, (400, 4312))):
            raw = pull(cn.connection_populations[layer.connection()].vars["weightGradient"])
            require(raw.dtype == np.float64 and raw.size == 8 * size, "Invalid lane storage")
            lanes = raw.reshape(8, size)
            require(np.isfinite(lanes).all(), "Nonfinite lane gradient")
            expected_a = np.asarray(self.baseline["gradients"][index]) / 8.0
            for lane in (0, 3, 6):
                np.testing.assert_allclose(lanes[lane], expected_a, rtol=1e-9, atol=1e-11)
            for lane in (4, 7):
                np.testing.assert_allclose(lanes[lane], lanes[1], rtol=1e-9, atol=1e-11)
            for lane in (2, 5):
                np.testing.assert_array_equal(lanes[lane], np.zeros(size))
            self.raw.append(lanes.copy())
            emit("MIXED_LANE_GRADIENT_PARITY", {
                "projection": index,
                "A_vs_B1_divided_by_8_max_absolute_difference": float(
                    np.max(np.abs(lanes[[0, 3, 6]] - expected_a))),
                "B_cross_lane_max_absolute_difference": float(
                    np.max(np.abs(lanes[[1, 4, 7]] - lanes[1]))),
                "empty_lane_gradients": "exact_zero",
                "independent_B_gradient_correctness_certified": False,
            })
        pop = cn.neuron_populations[self.hidden.population()]
        np.testing.assert_array_equal(pull(pop.vars["LambdavLimit"]).reshape(-1), np.ones(8))
        telemetry = {}
        for name in candidate.audit.COUNTERS + candidate.audit.SCALARS:
            value = pull(pop.vars[name])
            require(value.size == 8 * 392 and np.isfinite(value).all(),
                    f"Unexpected per-lane telemetry: {name}")
            telemetry[name] = value.reshape(8, 392)
        for lane, sample in enumerate(LAYOUT):
            jumps = int(telemetry["AuditJumps"][lane].sum())
            row = {"lane": lane, "sample": ("A", "B", "empty")[sample],
                   "replayed_jumps": jumps,
                   "clipped_jumps": int(telemetry["AuditClips"][lane].sum()),
                   "nonfinite_raw_jumps": int(telemetry["AuditNonfinite"][lane].sum()),
                   "invalid_limits": int(telemetry["AuditBadLimit"][lane].sum())}
            require(all(row[key] == 0 for key in (
                "clipped_jumps", "nonfinite_raw_jumps", "invalid_limits"
            )), "Clipping or invalid adjoint values in a mixed lane")
            if sample == 0:
                require(jumps == self.baseline["hidden_spikes"], "A replay spike count changed")
            elif sample == 2:
                require(jumps == 0, "Empty lane had replay spikes")
            self.statistics.append(row)
            emit("MIXED_LANE_REPLAY", row)
        require(len({self.statistics[i]["replayed_jumps"] for i in (1, 4, 7)}) == 1,
                "B replay counts differ across lanes")


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
    require(baseline["identity"]["diagnostic_sha256"] == r.file_hash(Path(candidate.__file__)),
            "Candidate precision adapter source changed")
    original = baseline["condition"]
    require(original["frames"] == 1024 and original["dt_ms"] == 0.0001220703125
            and original["scalar_precision"] == "double" and original["compiler_max_spikes"] == 8193,
            "Expected passed full-window condition")
    require(baseline["optimizer_updates"] == 0 and baseline["weights_and_moments_unchanged"]
            and all(p["within_1pct"] for p in baseline["continuous_reference_comparison"]["projections"]),
            "Baseline has not passed the frozen continuous gate")
    root = args.root.resolve(strict=True)
    rows = r.pilot(r.inventory(root)["train"], 2)[:2]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(r.normalize(metadata[0]) == baseline["sample_metadata"], "A input provenance changed")
    require(metadata[0]["stable_id"] != metadata[1]["stable_id"], "A and B must be distinct parents")
    config = r.ProbeConfig(frames=1024, dt_ms=original["dt_ms"])
    np.random.seed(config.seed)
    network, inp, hid, out = r.build_model(config, 8)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, baseline["initial_weights"], ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = np.asarray(values, dtype=np.float64).reshape(shape).copy()
    compiler = candidate.DoubleScalarAuditCompiler(
        example_timesteps=original["simulation_timesteps"], losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=8, rng_seed=42, max_spikes=8193,
        strict_buffer_checking=True, per_timestep_loss=False, grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="MixedFrozenB8", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    require(len(compiler.instrumentation) == 1, "Expected one instrumented hidden population")
    targets = np.array([int(labels[i]) if i < 2 else 0 for i in LAYOUT], dtype=np.int64)
    lane_capture = CaptureMixedLanes(layers, hid, out, baseline, targets)
    reduced_capture = plain.CaptureReduced(layers, hid, lane_capture)
    counts = {name: 0 for name in ("GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
    callbacks_modified = []
    for callback in cn.base_train_callbacks:
        if isinstance(callback, r.CustomUpdateOnBatchEnd) and callback.name in counts:
            counts[callback.name] += 1
            if callback.name == "GradientBatchReduce":
                callbacks_modified.extend((lane_capture, callback))
            elif callback.name == "GradientLearn":
                callbacks_modified.append(reduced_capture)
            else:
                callbacks_modified.append(r.CustomUpdateOnBatchEnd(callback.name, lambda batch: False))
        else:
            callbacks_modified.append(callback)
    require(all(v == 1 for v in counts.values()), "Unexpected batch callback layout")
    cn.base_train_callbacks[:] = callbacks_modified
    condition = {**original, "batch_size": 8, "real_examples": 6,
                 "unique_training_parents": 2, "empty_lanes": 2,
                 "limiter_policy": "fixed bound 1", "spike_recording": False,
                 "limiter_instrumentation": True, "layout": list(LAYOUT)}
    emit("MIXED_BATCH_PLAN", {"identity": identity, "condition": condition,
        "diagnostic_sha256": r.file_hash(Path(__file__)), "baseline_sha256": r.file_hash(path),
        "helper_sha256": {Path(m.__file__).name: r.file_hash(Path(m.__file__))
                          for m in (candidate, plain, candidate.audit)},
        "sample_metadata": metadata, "optimizer_updates": 0})
    empty = r.to_ml_genn(r.PreprocessedSpikes(np.zeros(2048, dtype=np.uint32),
                                              np.empty(0, dtype=np.float32)))
    samples = [r.to_ml_genn(p) for p in payloads] + [empty]
    presentations = [[samples[i] for i in LAYOUT], [empty] * 8]
    real_mask = np.array([i < 2 for i in LAYOUT])
    metric = MaskedAccuracy()
    state = {out: metric.create_state()}
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    with cn:
        cn.get_readout(out)
        observer.snapshot("after_load")
        initial_state = r.checkpoint.training_state(cn, layers)
        require(all(v.dtype == np.float64 for v in initial_state), "Training state precision changed")
        require([r.capacity.array_hash(v) for v in initial_state[:2]]
                == baseline["initial_weight_sha256"], "Initial weights differ")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=2, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []
        for batch in range(2):
            metric.real_mask = real_mask if batch == 0 else np.zeros(8, dtype=bool)
            with observer.phase(f"presentation_{batch}"):
                cn._train_batch(batch, 0, {inp: presentations[batch]},
                    {out: targets if batch == 0 else np.zeros(8, dtype=np.int64)},
                    {out: metric}, state, callbacks)
                scores.append(np.array(cn.get_readout(out), copy=True))
            observer.snapshot(f"after_presentation_{batch}")
            pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 6 and len(reduced_capture.records) == 1, "Wrong metric/replay count")
        for a, b in zip(initial_state, r.checkpoint.training_state(cn, layers)):
            np.testing.assert_array_equal(a, b)
        require(all(s.shape == (8, 11) and np.isfinite(s).all() for s in scores), "Invalid scores")
        for lane in (0, 3, 6):
            np.testing.assert_allclose(scores[0][lane], baseline["scores"][0], rtol=1e-9, atol=1e-11)
        for lane in (4, 7):
            np.testing.assert_allclose(scores[0][lane], scores[0][1], rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(scores[0][[2, 5]], 0, rtol=0, atol=1e-7)
        np.testing.assert_allclose(scores[1], 0, rtol=0, atol=1e-7)
        emit("MIXED_FORWARD_PARITY", {"A_vs_B1": "pass", "B_cross_lane": "pass",
             "empty_scores": "zero", "real_metric_examples": 6})
    real_mean_gradients, normalization = [], []
    for index, (raw, reduced) in enumerate(zip(lane_capture.raw, reduced_capture.records[0]["gradients"])):
        scaled = reduced * (8.0 / 6.0)
        expected = (3 * (8 * raw[0]) + 3 * (8 * raw[1])) / 6.0
        np.testing.assert_allclose(scaled, expected, rtol=1e-9, atol=1e-11)
        real_mean_gradients.append(scaled)
        row = {"projection": index, "real_count": 6, "batch_size": 8,
               "scale_for_mean_over_real_examples": 8.0 / 6.0,
               "max_absolute_difference": float(np.max(np.abs(scaled - expected)))}
        normalization.append(row)
        emit("PARTIAL_BATCH_NORMALIZATION", row)
    minimum_free = min(s["free_fraction"] for s in observer.snapshots)
    report = {"identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "baseline_sha256": r.file_hash(path), "condition": condition, "sample_metadata": metadata,
        "scores": np.asarray(scores).tolist(), "raw_lane_gradients": [g.tolist() for g in lane_capture.raw],
        "reduced_gradients": [g.tolist() for g in reduced_capture.records[0]["gradients"]],
        "real_mean_gradients": [g.tolist() for g in real_mean_gradients],
        "lane_replay_statistics": lane_capture.statistics, "normalization": normalization,
        "memory_snapshots": observer.snapshots, "phases": observer.phases,
        "optimizer_updates": 0, "weights_and_moments_unchanged": True}
    with (Path.cwd() / "mixed_batch_result.json").open("x") as stream:
        r.json.dump(report, stream, sort_keys=True, allow_nan=False)
    require(minimum_free >= 0.20, "Less than 20 percent sampled memory headroom")
    emit("PASS_MIXED_B8_PARITY", {"layout": ["A", "B", "empty", "A", "B", "empty", "A", "B"],
        "optimizer_updates": 0, "real_metric_examples": 6,
        "fixed_bound_clips": 0, "minimum_sampled_free_fraction": minimum_free,
        "independent_B_gradient_correctness_certified": False,
        "arbitrary_history_independence_certified": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
