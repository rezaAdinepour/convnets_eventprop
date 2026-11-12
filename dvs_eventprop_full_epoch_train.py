"""One training-only epoch over 934 cached development training parents.

Fresh original weights and zero Adam moments; B32, 29 full batches plus six.
Paired real/empty schedule, fixed adjoint bound 1, host-checked Adam updates.
Online scores describe changing weights; no final-weight or validation claim.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_full_batch_probe as boundary

block, bench, pilot, r = boundary.block, boundary.bench, boundary.pilot, boundary.r
cache = boundary.cache
require, emit, pull = r.require, r.emit, r.pull


class LearnEpoch(boundary.Freeze):
    def on_batch_end(self, cn, batch, metrics):
        ctx = self.context
        if batch % 2 == 0:
            # The preceding forward trial was an empty drain, so there must
            # be no pending task gradient on a new real forward presentation.
            for variable in block.reduced_variables(cn):
                values = pull(variable)
                np.testing.assert_array_equal(values, np.zeros_like(values))
            return
        require(ctx["step"] == ctx["updates"] == batch // 2,
                "Adam clock or learning ownership changed")
        before = r.checkpoint.training_state(cn, self.layers)
        super().on_batch_end(cn, batch, metrics)
        record = self.records[-1]
        indices = [index for index in ctx["indices"] if index >= 0]
        require(record["indices"] == indices and record["count"] == len(indices)
                and len(indices) in (6, 32), "Gradient contribution coverage changed")
        step = ctx["updates"]
        if step == 0:
            reference = ctx["reference"]["cases"]["first_full32"]
            np.testing.assert_array_equal(record["counts"], reference["per_neuron_replay_counts"])
            for gradient, expected in zip(record["mean"], reference["mean_gradients"]):
                np.testing.assert_allclose(gradient, expected, rtol=1e-9, atol=1e-11)
            emit("FULL_EPOCH_INITIAL_GRADIENT_PARITY", {
                "mean_gradients": "pass", "per_neuron_replay_counts": "exact_match",
                "adam_step": 0,
            })
        for variable, gradient in zip(block.reduced_variables(cn), record["mean"]):
            require(np.isfinite(gradient).all(), "Nonfinite normalized gradient")
            variable.view[...] = gradient.reshape(variable.view.shape)
            variable.push_to_device()
        cn.custom_update("GradientLearn")
        changes = []
        for index, (weight, gradient) in enumerate(zip(before[:2], record["mean"])):
            m, v = ctx["moments"][index]
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            expected = weight.reshape(-1) - 0.001 * (m / (1 - 0.9 ** (step + 1))) / (
                np.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
            actual = pull(cn.connection_populations[
                self.layers[index].connection()].vars["weight"]).reshape(-1)
            require(np.isfinite(actual).all(), "Nonfinite learned weights")
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
            matches = [cu for _, cu in cn.optimisers if cu.vars["M"].view.size == gradient.size]
            require(len(matches) == 1, "Ambiguous Adam moment ownership")
            for key, values in (("M", m), ("V", v)):
                np.testing.assert_allclose(pull(matches[0].vars[key]).reshape(-1),
                                           values, rtol=2e-10, atol=2e-12)
            ctx["moments"][index] = (m, v)
            changes.append(float(np.linalg.norm(actual - weight.reshape(-1))))
        ctx["updates"] = step + 1
        ctx["checkpoint"] = pilot.save_state(cn, self.layers, f"update_{step + 1:04d}.npz")
        row = {
            "adam_step": step, "cumulative_optimizer_updates": step + 1,
            "training_indices": indices, "real_examples": len(indices),
            "gradient_rescale": 32.0 / len(indices),
            "gradient_l2_hidden_output": [float(np.linalg.norm(g)) for g in record["mean"]],
            "maximum_replayed_spikes_per_neuron": int(record["counts"].max()),
            "fully_silent_hidden_examples": int(np.sum(record["counts"].sum(axis=1) == 0)),
            "replay_spikes_per_filter": record["counts"].reshape(
                len(indices), 7, 7, 8).sum(axis=(0, 1, 2)).astype(int).tolist(),
            "weight_change_l2_hidden_output": changes,
            "host_adam_weights_and_moments": "pass",
            "forward_replay_state": "exact_match", "clipped_jumps": 0,
            "checkpoint": ctx["checkpoint"], "online_metrics": ctx["online_metrics"],
            "scores_at_optimizer_updates": step,
        }
        block.write_json(f"update_{step + 1:04d}.json", {
            **row, "forward_scores": ctx["forward_scores"].tolist(),
            "labels": ctx["targets"][:len(indices)].tolist(),
            "normalized_mean_gradients": [g.tolist() for g in record["mean"]],
        })
        ctx["update_records"].append(row)
        ctx["contributions"].extend(indices)
        emit("FULL_EPOCH_UPDATE", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--boundary-result", type=Path, required=True)
    args = parser.parse_args()
    manifest_path, reference_path = [p.resolve(strict=True) for p in (
        args.manifest, args.boundary_result)]
    manifest, reference = [r.json.loads(p.read_text()) for p in (manifest_path, reference_path)]
    require(reference["diagnostic_sha256"] == r.file_hash(Path(boundary.__file__))
            == "f7091a5a28b288b69daf443ab13142d3251cd6a3e8149f4f6a80a110b2123752",
            "Boundary probe producer changed")
    require(reference["manifest_sha256"] == r.file_hash(manifest_path)
            and reference["optimizer_updates"] == 0
            and reference["weights_and_moments_unchanged"] is True
            and reference["fixed_bound_clips"] == 0
            and reference["last6_mean_gradient_normalization"] == "pass"
            and reference["last6_replay_counts"] == "exact_match"
            and reference["padding_gradients"] == "exact_zero",
            "Expected a passed frozen boundary-batch check")
    require(manifest["artifact_type"] == cache.FORMAT
            and manifest["diagnostic_sha256"] == r.file_hash(Path(cache.__file__))
            and manifest["config"] == cache.data.Config(1024).record()
            and manifest["test_event_arrays_loaded"] == 0
            and r.file_hash(manifest_path.parent / "sample_audit.jsonl")
            == manifest["sample_audit_sha256"], "Cache producer/config/audit changed")
    baseline_path = Path(reference["baseline_path"]).resolve(strict=True)
    require(r.file_hash(baseline_path) == reference["baseline_sha256"], "Initial baseline changed")
    baseline = r.json.loads(baseline_path.read_text())
    identity = r.checkpoint.identity()
    for artifact in (baseline, reference):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(artifact["identity"][key] == value, f"Source/environment changed: {key}")
    for mapping in (reference["helper_sha256"], baseline["identity"]["helper_sha256"]):
        for name, digest in mapping.items():
            require(r.file_hash(Path(__file__).with_name(name)) == digest,
                    f"Helper changed: {name}")
    require(manifest["loader_sha256"] == identity["source_sha256"]["dvs_data.py"],
            "Cache loader identity changed")
    condition = reference["condition"]
    require(condition["dt_ms"] == 0.00006103515625
            and condition["simulation_timesteps"] == 16777216
            and condition["physical_horizon_ms"] == 1024.0
            and condition["batch_size"] == 32 and condition["compiler_max_spikes"] == 8193
            and condition["scalar_precision"] == "double"
            and condition["time_precision"] == "float"
            and condition["strict_buffer_checking"] is True,
            "Unexpected physical training condition")
    entries = manifest["entries"]["train"]
    order = manifest["batch_plan"]["epoch1_training_permutation"]
    require(len(entries) == 934 and len(manifest["entries"]["validation"]) == 143
            and order == np.random.default_rng(42).permutation(934).tolist()
            and reference["layouts"]["first_full32"] == order[:32]
            and reference["layouts"]["last6_padded"] == order[928:] + [-1] * 26,
            "Training coverage/order changed")
    for index, entry in enumerate(entries):
        metadata = entry["metadata"]
        require(entry["index"] == index and metadata["split"] == "train"
                and 1 <= metadata["subject"] <= 20, "Non-training parent in epoch")
    weights = [np.asarray(values, dtype=np.float64) for values in baseline["initial_weights"]]
    require(len(weights) == 2 and baseline["initial_weight_sha256"] == reference["initial_weight_sha256"],
            "Initial weight coverage changed")
    for values, size, digest in zip(weights, (400, 4312), reference["initial_weight_sha256"]):
        require(values.shape == (size,) and np.isfinite(values).all()
                and r.capacity.array_hash(values) == digest, "Initial weight values changed")
    require(Path.cwd().resolve() not in (manifest_path.parent, reference_path.parent),
            "Use a new experiment directory")
    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
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
    cn = compiler.compile(network, name="FullDevelopmentEpoch1", optimisers={
        hid: {"weight": r.Adam(alpha=0.001)}, out: {"weight": r.Adam(alpha=0.001)},
    }, regularisers={})
    context = {"updates": 0, "reference": reference, "update_records": [], "contributions": [],
        "moments": {index: (np.zeros_like(values), np.zeros_like(values))
                    for index, values in enumerate(weights)},
        "parents": {index: entry["metadata"]["stable_id"] for index, entry in enumerate(entries)}}
    raw = bench.RawCapture(layers, context, 32)
    learn = LearnEpoch(layers, hid, out, context, 32)
    found = {name: 0 for name in (
        "GradientBatchReduce", "GradientLearn", "AbsSumReduceBatch", "ReduceAssign")}
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
    require(all(n == 1 for n in found.values()) and len(compiler.instrumentation) == 1,
            "Callback or audit layout changed")
    cn.base_train_callbacks[:] = modified
    provenance = {
        "artifact_type": "full-development-training-epoch1-v1", "identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)),
        "manifest_path": str(manifest_path), "manifest_sha256": r.file_hash(manifest_path),
        "boundary_result_path": str(reference_path), "boundary_result_sha256": r.file_hash(reference_path),
        "baseline_path": str(baseline_path), "baseline_sha256": r.file_hash(baseline_path),
        "initial_weight_sha256": reference["initial_weight_sha256"],
        "helper_sha256": {Path(module.__file__).name: r.file_hash(Path(module.__file__))
                          for module in (boundary, cache, block, bench, pilot, bench.mixed, pilot.candidate)},
        "condition": {**condition, "loss": "sparse_categorical_crossentropy",
            "per_timestep_loss": False, "softmax_temperature": 1.0,
            "learning_rate": 0.001, "adam_beta1": 0.9, "adam_beta2": 0.999,
            "adam_epsilon": 1e-8, "regulariser": None, "grad_limit": 100.0,
            "tau_mem_ms": 20.0, "tau_syn_ms": 5.0, "threshold": 1.0,
            "absolute_reset_voltage": 0.0, "readout": "avg_var_exp_weight"},
        "logical_epoch": 1, "training_permutation": order,
        "expected_real_examples": 934, "expected_optimizer_updates": 30,
        "state_operation": "fresh original initial weights; zero Adam moments; seed42",
        "exact_resume": False, "validation_performed": False, "test_performed": False,
    }
    block.write_json("epoch_plan.json", provenance)
    emit("FULL_EPOCH_PLAN", {key: value for key, value in provenance.items()
         if key != "training_permutation"})
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    empty = r.to_ml_genn(r.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)))
    metric = bench.mixed.MaskedAccuracy()
    state = {out: metric.create_state()}
    online_scores, online_labels = [], []
    with cn:
        cn.get_readout(out)
        initial = r.checkpoint.training_state(cn, layers)
        require(all(values.dtype == np.float64 for values in initial), "Precision changed")
        for actual, expected in zip(initial[:2], weights):
            np.testing.assert_array_equal(actual.reshape(-1), expected)
        require(all(np.all(values == 0) for values in initial[2:]), "Fresh moments not zero")
        initial_checkpoint = pilot.save_state(cn, layers, "update_0000.npz")
        block.write_json("initial_checkpoint.json", initial_checkpoint)
        observer.snapshot("after_load")
        require(observer.snapshots[-1]["free_fraction"] >= 0.20, "Insufficient GPU headroom")
        callbacks = r.CallbackList(cn.base_train_callbacks, compiled_network=cn,
                                   num_batches=60, num_epochs=1)
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        batch = 0
        for start in range(0, 934, 32):
            indices = order[start:start + 32]
            count = len(indices)
            loaded = [cache.load_cached_sample(manifest_path.parent, entries[index]) for index in indices]
            targets = np.array([row[1] for row in loaded] + [0] * (32 - count), dtype=np.int64)
            inputs = [r.to_ml_genn(row[0]) for row in loaded] + [empty] * (32 - count)
            context.update({"indices": indices + [-1] * (32 - count), "targets": targets,
                "case": f"epoch1/batch_{start // 32:02d}", "step": context["updates"],
                "forward_state": r.checkpoint.training_state(cn, layers)})
            for drain in (False, True):
                metric.real_mask = np.zeros(32, dtype=bool) if drain else np.arange(32) < count
                with observer.phase(f"{context['case']}/presentation_{batch}"):
                    cn._train_batch(batch, context["step"], {inp: [empty] * 32 if drain else inputs},
                        {out: np.zeros(32, dtype=np.int64) if drain else targets},
                        {out: metric}, state, callbacks)
                    value = np.array(cn.get_readout(out), copy=True)
                require(value.shape == (32, 11) and np.isfinite(value).all(), "Invalid scores")
                if drain:
                    np.testing.assert_allclose(value, 0, rtol=0, atol=1e-7)
                else:
                    block.same_state(context["forward_state"], r.checkpoint.training_state(cn, layers))
                    context["forward_scores"] = value[:count].copy()
                    context["online_metrics"] = boundary.score_summary(value[:count], targets[:count])
                    online_scores.extend(value[:count].tolist())
                    online_labels.extend(targets[:count].tolist())
                    if start == 0:
                        np.testing.assert_allclose(value, reference["cases"]["first_full32"]["scores"],
                                                   rtol=1e-9, atol=1e-11)
                        emit("FULL_EPOCH_INITIAL_SCORE_PARITY", {"all_32_scores": "pass"})
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(pull(pop.vars["StartSpike"]), pull(pop.vars["EndSpike"]))
                batch += 1
            observer.snapshot(f"after_update_{context['updates']:04d}")
            require(observer.snapshots[-1]["free_fraction"] >= 0.20,
                    "Insufficient sampled GPU headroom; last checked checkpoint was saved")
        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        final_state = r.checkpoint.training_state(cn, layers)
    require(batch == 60 and context["updates"] == len(learn.records) == 30
            and state[out].total == 934 and context["contributions"] == order
            and len(set(context["contributions"])) == 934,
            "Epoch update, replay, sample or metric coverage failed")
    require([row["adam_step"] for row in context["update_records"]] == list(range(30))
            and [row["real_examples"] for row in context["update_records"]] == [32] * 29 + [6],
            "Epoch clock or partial-batch accounting failed")
    summary = boundary.score_summary(online_scores, online_labels)
    summary.update({"score_parameter_state": "changes between batches; pre-update scores",
        "scores_at_optimizer_updates": list(range(30)),
        "fresh_final_weight_metrics": False})
    changes = [float(np.linalg.norm(after.reshape(-1) - before.reshape(-1)))
               for before, after in zip(initial[:2], final_state[:2])]
    require(all(value > 0 for value in changes), "Missing aggregate learned projection change")
    report = {**provenance, "initial_checkpoint": initial_checkpoint,
        "checkpoint": context["checkpoint"], "cumulative_optimizer_updates": 30,
        "optimizer_updates_this_run": 30, "real_training_contributions": 934,
        "online_training_metrics": summary, "online_scores": online_scores,
        "online_labels": online_labels, "update_records": context["update_records"],
        "weight_change_l2_hidden_output": changes,
        "fixed_bound_clips": 0, "full_data_epoch_completed": True,
        "final_weight_training_evaluation_performed": False,
        "memory_snapshots": observer.snapshots, "phases": observer.phases,
        "full_window_training_ready": False,
        "all_trained_states_continuous_gradient_certified": False}
    block.write_json("full_epoch_result.json", report)
    emit("PASS_FULL_DEVELOPMENT_TRAINING_EPOCH", {
        "logical_epoch": 1, "optimizer_updates": 30, "adam_steps": [0, 29],
        "real_training_contributions": 934, "last_real_batch_size": 6,
        "last_gradient_rescale": 32.0 / 6.0, "fixed_bound_clips": 0,
        "host_adam_weights_and_moments": "pass", "online_training_metrics": summary,
        "checkpoint": context["checkpoint"],
        "minimum_sampled_free_fraction": min(s["free_fraction"] for s in observer.snapshots),
        "simulation_and_callback_wall_ms": sum(p["profiled_wall_ms"] for p in observer.phases),
        "validation_performed": False, "full_window_training_ready": False})


if __name__ == "__main__":
    main()
