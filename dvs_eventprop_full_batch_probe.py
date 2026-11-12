"""Frozen B32 checks on cached full-data epoch boundary batches.

Original initial weights, no optimizer updates. Check the first full batch,
the final six parents with padding, then those six with active companions.
This is scoped batch/resource evidence, not a continuous derivative gate.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_b32_pilot_block as block
import dvs_eventprop_full_data_audit as cache

bench, pilot, r = block.bench, block.pilot, block.r
require, emit, pull = r.require, r.emit, r.pull


def score_summary(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    require(scores.shape == (len(labels), 11) and len(labels) > 0
            and np.isfinite(scores).all(), "Invalid real-example scores")
    maximum = scores.max(axis=1)
    ce = maximum + np.log(np.exp(scores - maximum[:, None]).sum(axis=1))
    ce -= scores[np.arange(len(labels)), labels]
    require(np.isfinite(ce).all(), "Nonfinite cross entropy")
    return {"mean_ce": float(ce.mean()),
            "correct": int(np.sum(scores.argmax(axis=1) == labels)),
            "real_examples": len(labels)}


class Freeze(bench.ReducedCapture):
    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return
        block.same_state(r.checkpoint.training_state(cn, self.layers),
                         self.context["forward_state"])
        pop = cn.neuron_populations[self.hidden.population()]
        bad = {name: pull(pop.vars[name]).reshape(32, 392)
               for name in ("AuditClips", "AuditNonfinite", "AuditBadLimit")}
        for lane, index in enumerate(self.context["indices"]):
            if any(np.any(values[lane] != 0) for values in bad.values()):
                emit("FULL_BATCH_INVALID_LANE", {
                    "case": self.context["case"], "lane": lane,
                    "training_index": index,
                    "stable_id": self.context["parents"].get(index),
                    **{name: int(values[lane].sum()) for name, values in bad.items()},
                })
        super().on_batch_end(cn, batch, metrics)
        block.same_state(r.checkpoint.training_state(cn, self.layers),
                         self.context["forward_state"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    manifest_path, baseline_path = [
        path.resolve(strict=True) for path in (args.manifest, args.baseline)
    ]
    manifest = r.json.loads(manifest_path.read_text())
    baseline = r.json.loads(baseline_path.read_text())
    identity = r.checkpoint.identity()
    require(manifest["artifact_type"] == cache.FORMAT
            and manifest["diagnostic_sha256"] == r.file_hash(Path(cache.__file__))
            == "69dec1f11e7aa51de7f4e7e302fabec6f5399dbb57c4f82aef41cc3491b6537c",
            "Cache producer changed")
    require(manifest["loader_sha256"] == identity["source_sha256"]["dvs_data.py"]
            and manifest["config"] == cache.data.Config(1024).record()
            and manifest["pilot_parent_payloads_reproduced"] == 22
            and manifest["cross_split_raw_content_duplicates"] == 0
            and manifest["test_event_arrays_loaded"] == 0,
            "Unexpected cache provenance or data condition")
    require(r.file_hash(manifest_path.parent / "sample_audit.jsonl")
            == manifest["sample_audit_sha256"], "Cache audit log changed")
    reload_path = Path(manifest["pilot_reload_path"]).resolve(strict=True)
    require(r.file_hash(reload_path) == manifest["pilot_reload_sha256"],
            "Passed reload result changed")
    reload_result = r.json.loads(reload_path.read_text())
    require(reload_result["diagnostic_sha256"] == r.file_hash(
                Path(__file__).with_name("dvs_eventprop_b32_pilot_reload.py"))
            and reload_result["pilot_overfit_gate_reproduced"] is True,
            "Pilot reload source/gate changed")
    for artifact in (baseline, reload_result):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value,
                        f"Source/environment changed: {key}")
    for name, digest in reload_result["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest,
                f"Pilot helper changed: {name}")
    require(baseline["identity"]["diagnostic_sha256"]
            == r.file_hash(Path(pilot.candidate.__file__)),
            "Precision adapter changed")
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(r.file_hash(Path(__file__).with_name(name)) == digest,
                f"Baseline helper changed: {name}")
    require(baseline["optimizer_updates"] == 0
            and baseline["weights_and_moments_unchanged"] is True
            and all(row["within_1pct"] for row in
                    baseline["continuous_reference_comparison"]["projections"]),
            "Expected the passed frozen initial-weight baseline")

    entries = manifest["entries"]["train"]
    require(len(entries) == 934 and len(manifest["entries"]["validation"]) == 143,
            "Unexpected development coverage")
    order = manifest["batch_plan"]["epoch1_training_permutation"]
    require(order == np.random.default_rng(42).permutation(934).tolist(),
            "Epoch-one permutation changed")
    first, tail = order[:32], order[928:]
    require(len(first) == 32 and len(tail) == 6
            and len(set(first + tail)) == 38, "Boundary batch selection changed")
    layouts = {
        "first_full32": first,
        "last6_padded": tail + [-1] * 26,
        "last6_with_companions": tail + first[:26],
    }
    loaded = {index: cache.load_cached_sample(manifest_path.parent, entries[index])
              for index in first + tail}
    for index, (_, label, metadata) in loaded.items():
        require(entries[index]["index"] == index
                and metadata["split"] == "train"
                and 1 <= metadata["subject"] <= 20
                and int(label) == metadata["label"], "Cache row/label mismatch")
    weights = [np.asarray(values, dtype=np.float64) for values in baseline["initial_weights"]]
    require(len(weights) == 2, "Initial projection coverage changed")
    for values, size, digest in zip(weights, (400, 4312), baseline["initial_weight_sha256"]):
        require(values.shape == (size,) and np.isfinite(values).all()
                and r.capacity.array_hash(values) == digest, "Initial weights changed")
    config = r.ProbeConfig(frames=1024, dt_ms=0.00006103515625)
    np.random.seed(42)
    network, inp, hid, out = r.build_model(config, 32)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(layers, weights, ((5, 5, 2, 8), (392, 11))):
        layer.connection().connectivity.weight = values.reshape(shape).copy()
    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=16777216, losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=32, rng_seed=42, max_spikes=8193,
        strict_buffer_checking=True, per_timestep_loss=False,
        grad_limit=100.0, softmax_temperature=1.0,
    )
    cn = compiler.compile(network, name="CachedFullBatchBoundary", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"parents": {index: values[2]["stable_id"] for index, values in loaded.items()}}
    raw = bench.RawCapture(layers, context, 32)
    capture = Freeze(layers, hid, out, context, 32)
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
    require(all(n == 1 for n in found.values()), "Unexpected callback layout")
    require(len(compiler.instrumentation) == 1, "Hidden audit instrumentation changed")
    cn.base_train_callbacks[:] = modified
    provenance = {
        "identity": identity, "diagnostic_sha256": r.file_hash(Path(__file__)),
        "manifest_path": str(manifest_path), "manifest_sha256": r.file_hash(manifest_path),
        "baseline_path": str(baseline_path), "baseline_sha256": r.file_hash(baseline_path),
        "initial_weight_sha256": baseline["initial_weight_sha256"],
        "helper_sha256": {Path(module.__file__).name: r.file_hash(Path(module.__file__))
                          for module in (cache, block, bench, pilot, bench.mixed, pilot.candidate)},
        "condition": {"profile": "dvs-occ1ms-prefix1024-v1", "input_bin_ms": 1.0,
            "physical_horizon_ms": 1024.0, "dt_ms": config.dt_ms,
            "simulation_timesteps": 16777216, "batch_size": 32,
            "scalar_precision": "double", "time_precision": "float",
            "compiler_max_spikes": 8193, "strict_buffer_checking": True,
            "limiter_policy": "fixed bound 1; stop on clipping",
            "spike_recording": False, "seed": 42},
        "layouts": layouts, "selected_parent_ids": context["parents"],
        "selected_sample_metadata": [loaded[index][2] for index in first + tail],
        "operation": "original initial weights; frozen paired forwards/replays",
        "optimizer_updates": 0, "validation_performed": False, "test_performed": False,
    }
    emit("FULL_BATCH_BOUNDARY_PLAN", {key: value for key, value in provenance.items()
         if key != "selected_sample_metadata"})
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    inputs = {index: r.to_ml_genn(values[0]) for index, values in loaded.items()}
    metric = bench.mixed.MaskedAccuracy()
    results = {}
    with cn:
        cn.get_readout(out)
        initial = r.checkpoint.training_state(cn, layers)
        require(all(values.dtype == np.float64 for values in initial), "Precision changed")
        for actual, target in zip(initial[:2], weights):
            np.testing.assert_array_equal(actual.reshape(-1), target)
        require(all(np.all(values == 0) for values in initial[2:]), "Fresh Adam moments not zero")
        observer.snapshot("after_load")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient GPU headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=6, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for name, indices in layouts.items():
            context.update({"case": name, "indices": indices,
                "targets": np.array([loaded[i][1] if i >= 0 else 0 for i in indices], dtype=np.int64),
                "forward_state": r.checkpoint.training_state(cn, layers)})
            mask = np.array(indices) >= 0
            state = {out: metric.create_state()}
            forward_scores = None
            for drain in (False, True):
                metric.real_mask = np.zeros(32, dtype=bool) if drain else mask
                presentations = [empty] * 32 if drain else [inputs.get(i, empty) for i in indices]
                with observer.phase(f"{name}/presentation_{batch}"):
                    cn._train_batch(batch, 0, {inp: presentations},
                        {out: np.zeros(32, dtype=np.int64) if drain else context["targets"]},
                        {out: metric}, state, callbacks)
                    value = np.array(cn.get_readout(out), copy=True)
                require(value.shape == (32, 11) and np.isfinite(value).all(), "Invalid scores")
                if drain:
                    np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)
                else:
                    forward_scores = value
                block.same_state(initial, r.checkpoint.training_state(cn, layers))
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                batch += 1
            require(state[out].total == int(mask.sum()), "Wrong real-example denominator")
            record = capture.records[-1]
            require(record["indices"] == [i for i in indices if i >= 0], "Gradient parent order changed")
            summary = score_summary(forward_scores[mask], context["targets"][mask])
            summary.update({"case": name, "gradient_rescale": 32.0 / int(mask.sum()),
                "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in record["mean"]],
                "maximum_replayed_spikes_per_neuron": int(record["counts"].max()),
                "fully_silent_hidden_examples": int(np.sum(record["counts"].sum(axis=1) == 0)),
                "clipped_jumps": 0})
            results[name] = {"summary": summary, "scores": forward_scores[mask].tolist(),
                "individual_gradients": [g.tolist() for g in record["individual"]],
                "mean_gradients": [g.tolist() for g in record["mean"]],
                "per_neuron_replay_counts": record["counts"].astype(int).tolist()}
            emit("FULL_BATCH_CASE", summary)
            observer.snapshot(f"after_{name}")
            require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient sampled headroom")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        block.same_state(initial, r.checkpoint.training_state(cn, layers))

    require(len(capture.records) == 3, "Wrong replay count")
    partial, companion = results["last6_padded"], results["last6_with_companions"]
    np.testing.assert_allclose(partial["scores"], companion["scores"][:6], rtol=1e-9, atol=1e-11)
    np.testing.assert_array_equal(partial["per_neuron_replay_counts"],
                                  companion["per_neuron_replay_counts"][:6])
    differences = []
    for projection in range(2):
        a = np.asarray(partial["individual_gradients"][projection])
        b = np.asarray(companion["individual_gradients"][projection])[:6]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(partial["mean_gradients"][projection],
                                   b.mean(axis=0), rtol=1e-9, atol=1e-11)
        differences.append(float(np.max(np.abs(a - b))))
    passed = {"distinct_training_parents": 38, "real_batch_sizes": [32, 6, 32],
        "last6_score_max_absolute_difference": float(np.max(np.abs(
            np.asarray(partial["scores"]) - np.asarray(companion["scores"])[:6]))),
        "last6_individual_gradient_max_absolute_difference_hidden_output": differences,
        "last6_replay_counts": "exact_match", "last6_mean_gradient_normalization": "pass",
        "padding_gradients": "exact_zero", "weights_and_moments_unchanged": True,
        "fixed_bound_clips": 0, "optimizer_updates": 0,
        "minimum_sampled_free_fraction": min(s["free_fraction"] for s in observer.snapshots),
        "full_data_training_executed": False, "full_window_training_ready": False,
        "continuous_gradient_correctness_of_new_parents_certified": False}
    with (Path.cwd() / "full_batch_result.json").open("x") as stream:
        r.json.dump({**provenance, **passed, "cases": results,
            "memory_snapshots": observer.snapshots, "phases": observer.phases},
            stream, sort_keys=True, allow_nan=False)
    emit("PASS_FULL_BATCH_BOUNDARY", passed)


if __name__ == "__main__":
    main()
