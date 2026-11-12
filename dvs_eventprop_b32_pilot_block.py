"""Bounded B32 pilot learning block with consistent-weight replay.

22 training parents plus 10 empty lanes. At most 12 new updates.
Controlled weights/moments/clock warm start; not exact training resume.
No validation or official test event arrays.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_b32_update_probe as previous

bench, pilot, r = previous.bench, previous.pilot, previous.r
require, emit, pull = r.require, r.emit, r.pull

ARTIFACT = "b32_pilot_block_v1"
MAX_UPDATES = 12
REVIEW_EVERY = 3
TARGET_CORRECT = 21
STATE_NAMES = (
    "hidden", "output",
    "adam_hidden_m", "adam_hidden_v",
    "adam_output_m", "adam_output_v",
)


def same_state(left, right):
    require(len(left) == len(right) == 6, "Unexpected state coverage")
    for actual, expected in zip(left, right):
        np.testing.assert_array_equal(actual.reshape(-1), expected.reshape(-1))


def write_json(name, value):
    with (Path.cwd() / name).open("x") as stream:
        r.json.dump(value, stream, sort_keys=True, allow_nan=False)


def reduced_variables(cn):
    result = []
    for size in (400, 4312):
        matches = [
            cu.vars["ReducedGradient"]
            for name, cu in cn.genn_model.custom_updates.items()
            if name.startswith("CUBatchReduce")
            and "ReducedGradient" in cu.vars
            and cu.vars["ReducedGradient"].view.size == size
        ]
        require(len(matches) == 1, "Ambiguous reduced-gradient ownership")
        result.append(matches[0])
    return result


class LearnBlock(bench.ReducedCapture):
    def on_batch_end(self, cn, batch, metrics):
        if batch % 2 == 0:
            return

        ctx = self.context
        before = r.checkpoint.training_state(cn, self.layers)
        same_state(before, ctx["forward_state"])
        require(ctx["step"] == ctx["updates"], "Adam clock mismatch")

        # Checks lane reduction, padding, labels, finite telemetry,
        # fixed bound 1, zero clipping, and cleared raw gradients.
        super().on_batch_end(cn, batch, metrics)
        record = self.records[-1]
        require(record["indices"] == list(range(22)), "Replay coverage changed")
        for gradient in record["mean"]:
            require(
                np.isfinite(gradient).all(),
                "Nonfinite normalized mean gradient",
            )

        if len(self.records) == 1:
            reference = ctx["prior_gradient_reference"]
            if reference is not None:
                np.testing.assert_array_equal(
                    record["counts"], reference["hidden_counts"]
                )
                for actual, expected in zip(
                    record["mean"], reference["full_real_mean_gradients"]
                ):
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-9, atol=1e-11
                    )
            emit("B32_FIRST_REPLAY_CHECK", {
                "consistent_forward_replay_state": "exact_match",
                "prior_gradient_reference_available": reference is not None,
                "prior_gradient_parity": (
                    "pass" if reference is not None else "not available"
                ),
                "clipped_jumps": 0,
            })

        if not ctx["learn"]:
            same_state(before, r.checkpoint.training_state(cn, self.layers))
            ctx["final_gradient_reference"] = {
                "full_real_mean_gradients": [
                    gradient.tolist() for gradient in record["mean"]
                ],
                "hidden_counts": record["counts"].astype(int).tolist(),
            }
            return

        step = ctx["updates"]
        for variable, gradient in zip(
            reduced_variables(cn), record["mean"]
        ):
            variable.view[...] = gradient.reshape(variable.view.shape)
            variable.push_to_device()

        cn.custom_update("GradientLearn")
        changes = []
        for index, (weight, gradient) in enumerate(
            zip(before[:2], record["mean"])
        ):
            m, v = ctx["moments"][index]
            m = 0.9 * m + 0.1 * gradient
            v = 0.999 * v + 0.001 * gradient ** 2
            expected = weight.reshape(-1) - (
                0.001 * (m / (1 - 0.9 ** (step + 1)))
                / (np.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
            )
            group = cn.connection_populations[
                self.layers[index].connection()
            ]
            actual = pull(group.vars["weight"]).reshape(-1)
            require(np.isfinite(actual).all(), "Nonfinite updated weights")
            np.testing.assert_allclose(
                actual, expected, rtol=2e-10, atol=2e-12
            )
            matches = [
                cu for _, cu in cn.optimisers
                if cu.vars["M"].view.size == gradient.size
            ]
            require(len(matches) == 1, "Ambiguous Adam moment ownership")
            for key, values in (("M", m), ("V", v)):
                np.testing.assert_allclose(
                    pull(matches[0].vars[key]).reshape(-1),
                    values, rtol=2e-10, atol=2e-12,
                )
            ctx["moments"][index] = (m, v)
            changes.append(float(np.linalg.norm(
                actual - weight.reshape(-1)
            )))

        ctx["updates"] = step + 1
        ctx["checkpoint"] = pilot.save_state(
            cn, self.layers, f"update_{ctx['updates']:04d}.npz"
        )
        row = {
            "adam_step": step,
            "cumulative_optimizer_updates": ctx["updates"],
            "real_examples": 22,
            "gradient_rescale": 32.0 / 22.0,
            "host_adam_weights_and_moments": "pass",
            "forward_replay_state": "exact_match",
            "clipped_jumps": 0,
            "weight_change_l2_hidden_output": changes,
            "checkpoint": ctx["checkpoint"],
        }
        ctx["update_records"].append(row)
        write_json(f"update_{ctx['updates']:04d}.json", row)
        emit("B32_BLOCK_UPDATE", row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pilot-block", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        path.resolve(strict=True)
        for path in (args.baseline, args.pilot_block, args.source_result)
    ]
    baseline, block, source = [
        r.json.loads(path.read_text()) for path in paths
    ]
    identity = r.checkpoint.identity()
    for artifact in (baseline, block, source):
        for key, value in identity.items():
            if key != "convnets_commit":
                require(
                    artifact["identity"][key] == value,
                    f"Source/environment mismatch: {key}",
                )

    require(
        r.file_hash(Path(previous.__file__))
        == "8574fcaafd3623a8cb5c4c92af94724bad2ccce850b49d3e4e633386d47766bc",
        "Single-update helper changed",
    )
    require(
        r.file_hash(Path(bench.__file__))
        == "0242d7ac5138b253baa9dd8d36c0ba0d56c4dcad6622d163036a0c3dce1b6ad8",
        "Batch-capture helper changed",
    )
    require(
        block["diagnostic_sha256"] == r.file_hash(
            Path(__file__).with_name("dvs_eventprop_pilot_block.py")
        )
        and block["pilot_producer_sha256"]
        == r.file_hash(Path(pilot.__file__)),
        "Pilot producer changed",
    )
    require(
        block["baseline_sha256"] == r.file_hash(paths[0])
        and baseline["identity"]["diagnostic_sha256"]
        == r.file_hash(Path(pilot.candidate.__file__)),
        "Baseline or precision adapter changed",
    )
    for name, digest in baseline["identity"]["helper_sha256"].items():
        require(
            r.file_hash(Path(__file__).with_name(name)) == digest,
            f"Baseline helper changed: {name}",
        )

    artifact_type = source.get("artifact_type")
    if artifact_type is None:
        require(
            source["diagnostic_sha256"]
            == r.file_hash(Path(previous.__file__))
            and source["cumulative_optimizer_updates"] == 13
            and source["optimizer_updates_this_run"] == 1,
            "Expected the checked single-update result",
        )
    else:
        require(
            artifact_type == ARTIFACT
            and source["diagnostic_sha256"] == r.file_hash(Path(__file__)),
            "Unsupported or changed block producer",
        )

    require(
        source["source_block_sha256"] == r.file_hash(paths[1])
        and source["source_condition"] == block["condition"],
        "Pilot metadata binding changed",
    )
    require(
        source["batch_size"] == 32
        and source["effective_real_batch_size"] == 22
        and source["learning_rate"] == 0.001
        and source["validation_performed"] is False
        and source["test_performed"] is False,
        "Source execution condition changed",
    )
    start = source["cumulative_optimizer_updates"]
    require(isinstance(start, int) and start >= 13, "Invalid Adam clock")
    expected = source["phase_results"]["fresh_evaluation"]
    require(
        expected["scores_at_cumulative_updates"] == start
        and expected["optimizer_updates_in_phase"] == 0,
        "Source scores do not describe the saved fixed-weight state",
    )

    checkpoint = source["checkpoint"]
    checkpoint_path = Path(checkpoint["path"]).resolve(strict=True)
    require(
        checkpoint_path.parent == paths[2].parent
        and checkpoint_path.name == f"update_{start:04d}.npz"
        and r.file_hash(checkpoint_path) == checkpoint["sha256"],
        "Checkpoint binding changed",
    )
    require(
        Path.cwd().resolve() != paths[2].parent,
        "Use a new experiment directory",
    )
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        require(set(archive.files) == set(STATE_NAMES), "Checkpoint schema changed")
        saved = [np.array(archive[name], copy=True) for name in STATE_NAMES]
    for values, size in zip(saved, (400, 4312, 400, 400, 4312, 4312)):
        require(
            values.dtype == np.float64
            and values.shape == (size,)
            and np.isfinite(values).all(),
            "Invalid saved state",
        )
    require(
        np.all(saved[3] >= 0) and np.all(saved[5] >= 0),
        "Invalid saved Adam variance",
    )

    condition = block["condition"]
    require(
        condition["frames"] == 1024
        and condition["dt_ms"] == 0.00006103515625
        and condition["simulation_timesteps"] == 16777216
        and condition["compiler_max_spikes"] == 8193
        and condition["scalar_precision"] == "double"
        and condition["time_precision"] == "float"
        and condition["strict_buffer_checking"] is True,
        "Unexpected physical condition",
    )
    root = args.root.resolve(strict=True)
    by_id = {row.stable_id: row for row in r.inventory(root)["train"]}
    rows = [by_id[key] for key in block["training_order"]]
    loaded = [r.load_sample(root, row, r.DataConfig(1024)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    require(
        len(rows) == 22
        and len(set(block["training_order"])) == 22
        and r.normalize(list(metadata)) == block["sample_metadata"],
        "Pilot input provenance changed",
    )
    labels = np.asarray(labels, dtype=np.int64)
    np.testing.assert_array_equal(
        labels, block["epochs"][-1]["phase_results"]["fresh_evaluation"]["labels"]
    )
    initial_summary = previous.score_summary(expected["scores"], labels)
    require(
        initial_summary["correct"] == expected["correct"]
        and abs(initial_summary["mean_ce"] - expected["mean_ce"]) < 1e-12,
        "Source metric reconstruction failed",
    )
    require(
        initial_summary["correct"] < TARGET_CORRECT,
        "Source already meets the pilot gate; review before more training",
    )

    np.random.seed(42)
    config = r.ProbeConfig(frames=1024, dt_ms=condition["dt_ms"])
    network, inp, hid, out = r.build_model(config, 32)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False
    for layer, values, shape in zip(
        layers, saved[:2], ((5, 5, 2, 8), (392, 11))
    ):
        layer.connection().connectivity.weight = values.reshape(shape).copy()

    compiler = pilot.candidate.DoubleScalarAuditCompiler(
        example_timesteps=16777216,
        losses="sparse_categorical_crossentropy",
        dt=config.dt_ms, batch_size=32, rng_seed=42,
        max_spikes=8193, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network, name="BoundedB32PilotBlock",
        optimisers={
            hid: {"weight": r.Adam(alpha=0.001)},
            out: {"weight": r.Adam(alpha=0.001)},
        },
        regularisers={},
    )
    context = {
        "updates": start,
        "checkpoint": checkpoint,
        "moments": {
            0: (saved[2].copy(), saved[3].copy()),
            1: (saved[4].copy(), saved[5].copy()),
        },
        "prior_gradient_reference": source.get("fresh_gradient_reference"),
        "update_records": [],
        "indices": list(range(22)) + [-1] * 10,
        "targets": np.r_[labels, np.zeros(10, dtype=np.int64)],
    }
    raw = bench.RawCapture(layers, context, 32)
    learn = LearnBlock(layers, hid, out, context, 32)
    found = {
        name: 0 for name in (
            "GradientBatchReduce", "GradientLearn",
            "AbsSumReduceBatch", "ReduceAssign",
        )
    }
    modified = []
    for cb in cn.base_train_callbacks:
        if isinstance(cb, r.CustomUpdateOnBatchEnd) and cb.name in found:
            found[cb.name] += 1
            if cb.name == "GradientBatchReduce":
                modified.extend((raw, cb))
            elif cb.name == "GradientLearn":
                modified.append(learn)
            else:
                modified.append(
                    r.CustomUpdateOnBatchEnd(cb.name, lambda batch: False)
                )
        else:
            modified.append(cb)
    require(all(count == 1 for count in found.values()), "Callback layout changed")
    cn.base_train_callbacks[:] = modified

    provenance = {
        "artifact_type": ARTIFACT,
        "identity": identity,
        "diagnostic_sha256": r.file_hash(Path(__file__)),
        "source_result_sha256": r.file_hash(paths[2]),
        "source_block_sha256": r.file_hash(paths[1]),
        "baseline_sha256": r.file_hash(paths[0]),
        "source_checkpoint": checkpoint,
        "source_condition": condition,
        "batch_size": 32,
        "effective_real_batch_size": 22,
        "learning_rate": 0.001,
        "maximum_new_updates": MAX_UPDATES,
        "review_every_new_updates": REVIEW_EVERY,
        "target_correct": TARGET_CORRECT,
        "training_order": block["training_order"],
        "loss_gradient_normalization": "real-example mean; 32/22 rescale",
        "limiter_policy": "fixed bound 1; stop before Adam on observed clipping",
        "state_operation": (
            f"weights/moments/step{start} warm start; fresh model/tapes/RNG seed42"
        ),
        "cross_revision_basis": "identical checked core/environment/helper hashes",
        "exact_resume": False,
        "validation_performed": False,
        "test_performed": False,
    }
    write_json("block_plan.json", provenance)
    emit("B32_BLOCK_PLAN", provenance)

    empty = r.to_ml_genn(r.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32), np.empty(0, dtype=np.float32)
    ))
    inputs = [r.to_ml_genn(payload) for payload in payloads] + [empty] * 10
    metric = bench.mixed.MaskedAccuracy()
    observer = r.ObserveResources()
    observer.snapshot("baseline")
    evaluations = []
    physical_batch = 0

    with cn:
        for size, m, v in (
            (400, saved[2], saved[3]), (4312, saved[4], saved[5])
        ):
            matches = [
                cu for _, cu in cn.optimisers
                if cu.vars["M"].view.size == size
            ]
            require(len(matches) == 1, "Ambiguous moment restore")
            for key, values in (("M", m), ("V", v)):
                variable = matches[0].vars[key]
                variable.view[...] = values.reshape(variable.view.shape)
                variable.push_to_device()
        same_state(r.checkpoint.training_state(cn, layers), saved)
        observer.snapshot("after_restore")
        require(
            observer.snapshots[-1]["free_fraction"] >= 0.20,
            "Insufficient sampled GPU headroom",
        )

        callbacks = r.CallbackList(
            cn.base_train_callbacks, compiled_network=cn,
            num_batches=2 * (MAX_UPDATES + 1), num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        while True:
            applied = context["updates"] - start
            require(0 <= applied <= MAX_UPDATES, "Update budget exceeded")
            state = {out: metric.create_state()}
            context["step"] = context["updates"]
            context["forward_state"] = r.checkpoint.training_state(cn, layers)
            metric.real_mask = np.arange(32) < 22

            with observer.phase(f"forward_{context['updates']:04d}"):
                cn._train_batch(
                    physical_batch, context["step"], {inp: inputs},
                    {out: context["targets"]}, {out: metric}, state, callbacks,
                )
                scores = np.array(cn.get_readout(out), copy=True)
            require(
                scores.shape == (32, 11) and np.isfinite(scores).all(),
                "Invalid forward scores",
            )
            same_state(
                context["forward_state"],
                r.checkpoint.training_state(cn, layers),
            )
            input_pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(
                pull(input_pop.vars["StartSpike"]),
                pull(input_pop.vars["EndSpike"]),
            )

            summary = previous.score_summary(scores[:22], labels)
            if applied == 0:
                np.testing.assert_allclose(
                    scores[:22], expected["scores"], rtol=1e-9, atol=1e-11
                )
                emit("B32_BLOCK_SOURCE_PARITY", {
                    "all_22_scores": "pass",
                    "weights_and_moments": "exact_match",
                    "next_adam_step": start,
                })

            review = applied > 0 and applied % REVIEW_EVERY == 0
            met = review and summary["correct"] >= TARGET_CORRECT
            stop = applied == MAX_UPDATES or met
            context["learn"] = not stop

            summary.update({
                "labels": labels.tolist(),
                "scores_at_cumulative_updates": context["updates"],
                "cumulative_optimizer_updates": context["updates"],
                "new_updates_before_scores": applied,
                "score_parameter_state": "fixed",
                "optimizer_updates_in_phase": 0,
                "checkpoint": context["checkpoint"],
                "following_replay_applies_update": not stop,
            })
            if review or stop:
                evaluations.append(summary)
                write_json(
                    f"evaluation_{context['updates']:04d}.json", summary
                )
                emit("B32_BLOCK_EVALUATION", {
                    key: value for key, value in summary.items()
                    if key not in ("scores", "labels")
                })

            physical_batch += 1
            metric.real_mask = np.zeros(32, dtype=bool)
            with observer.phase(f"replay_{context['step']:04d}"):
                cn._train_batch(
                    physical_batch, context["step"], {inp: [empty] * 32},
                    {out: np.zeros(32, dtype=np.int64)},
                    {out: metric}, state, callbacks,
                )
                drain_scores = np.array(cn.get_readout(out), copy=True)
            np.testing.assert_allclose(drain_scores, 0, rtol=0, atol=1e-7)
            np.testing.assert_array_equal(
                pull(input_pop.vars["StartSpike"]),
                pull(input_pop.vars["EndSpike"]),
            )
            require(state[out].total == 22, "Incorrect metric denominator")
            physical_batch += 1
            observer.snapshot(f"after_pair_{physical_batch // 2}")
            require(
                observer.snapshots[-1]["free_fraction"] >= 0.20,
                "Insufficient sampled GPU headroom",
            )
            if stop:
                same_state(
                    context["forward_state"],
                    r.checkpoint.training_state(cn, layers),
                )
                break

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)

    applied = context["updates"] - start
    require(
        applied == len(context["update_records"])
        and len(learn.records) == applied + 1
        and physical_batch == 2 * (applied + 1),
        "Update or replay accounting failed",
    )
    require(evaluations, "Missing reviewed evaluation")
    final = evaluations[-1]
    require(
        final["scores_at_cumulative_updates"] == context["updates"]
        and final["following_replay_applies_update"] is False,
        "Final evaluation/checkpoint mismatch",
    )
    met = final["correct"] >= TARGET_CORRECT
    report = {
        **provenance,
        "initial_fresh_summary": initial_summary,
        "evaluations": evaluations,
        "phase_results": {"fresh_evaluation": final},
        "optimizer_updates_this_run": applied,
        "cumulative_optimizer_updates": context["updates"],
        "checkpoint": context["checkpoint"],
        "update_records": context["update_records"],
        "fresh_gradient_reference": context["final_gradient_reference"],
        "fixed_bound_clips": 0,
        "pilot_overfit_gate_met": met,
        "stop_reason": "pilot_target_reached" if met else "update_budget_reached",
        "fresh_minus_initial_mean_ce": (
            final["mean_ce"] - initial_summary["mean_ce"]
        ),
        "memory_snapshots": observer.snapshots,
        "phases": observer.phases,
        "full_window_training_ready": False,
        "all_trained_states_continuous_gradient_certified": False,
    }
    write_json("b32_pilot_block_result.json", report)
    emit("PASS_B32_PILOT_BLOCK", {
        key: report[key] for key in (
            "optimizer_updates_this_run", "cumulative_optimizer_updates",
            "fixed_bound_clips", "pilot_overfit_gate_met", "stop_reason",
            "fresh_minus_initial_mean_ce", "full_window_training_ready",
        )
    })
    emit("B32_FINAL_FRESH_METRICS", {
        "mean_ce": final["mean_ce"],
        "correct": final["correct"],
        "examples": 22,
        "checkpoint": context["checkpoint"],
    })


if __name__ == "__main__":
    main()