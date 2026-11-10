"""TASK 10: observe adapted limiter activity in a frozen DVS pipeline.

Three fixed training examples plus empty drain.
Control/instrumentation parity, no optimizer updates or test events.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_capacity_probe as capacity
import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_limiter_probe as audit
from dvs_data import (
    Config as DataConfig,
    file_hash,
    inventory,
    load_sample,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_probe import emit, pull


class CapturePipelineLimiter(capacity.CaptureAndClear):
    def on_batch_end(self, cn, batch, metric_state):
        # Captures the per-example gradients and clears only those gradients.
        # This callback still runs before the compiler's bound-adaptation
        # callbacks, so the recorded limit is the one used during replay.
        super().on_batch_end(cn, batch, metric_state)

        pop = cn.neuron_populations[self.hidden.population()]
        if batch == 0:
            for name in audit.COUNTERS:
                require(bool((pull(pop.vars[name]) == 0).all()),
                        f"Warmup unexpectedly changed {name}")
            return

        values = {
            name: pull(pop.vars[name]).reshape(-1)
            for name in audit.COUNTERS + audit.SCALARS
        }
        require(all(np.isfinite(v).all() for v in values.values()),
                "Nonfinite audit telemetry")
        active = values["AuditJumps"] > 0
        jumps = int(values["AuditJumps"].sum())
        clips = int(values["AuditClips"].sum())

        statistics = {
            "backward_example_index": batch - 1,
            "backward_label": int(self.labels[batch - 1]),
            "replayed_jumps": jumps,
            "clipped_jumps": clips,
            "neurons_with_clipping": int(
                np.count_nonzero(values["AuditClips"])
            ),
            "nonfinite_raw_jumps": int(values["AuditNonfinite"].sum()),
            "invalid_limits": int(values["AuditBadLimit"].sum()),
            "max_absolute_raw_jump": float(values["AuditMaxRaw"].max()),
            "max_raw_to_limit_ratio": float(values["AuditMaxRatio"].max()),
            "min_absolute_Isyn_minus_threshold": (
                float(values["AuditMinDenom"][active].min())
                if active.any() else None
            ),
            "bound_used_during_replay":
                self.records[-1]["limit"].tolist(),
        }
        self.records[-1]["limiter_statistics"] = statistics
        emit("PIPELINE_LIMITER_OBSERVATION", statistics)

        require(clips <= jumps, "Clips exceed jump count")
        require(statistics["nonfinite_raw_jumps"] == 0,
                "Nonfinite raw adjoint jumps")
        require(statistics["invalid_limits"] == 0,
                "Invalid adjoint bounds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    root = args.root.resolve(strict=True)
    rows = pilot(inventory(root)["train"], 2)[:3]
    loaded = [
        load_sample(root, row, DataConfig(128)) for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    inputs = [to_ml_genn(p) for p in payloads]
    labels = np.asarray(labels, dtype=np.int64)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "capacity_helper_sha256": file_hash(Path(capacity.__file__)),
        "limiter_helper_sha256": file_hash(Path(audit.__file__)),
    })
    emit("PIPELINE_LIMITER_PLAN", {
        **config.record(),
        "training_order": [row.stable_id for row in rows],
        "real_examples": 3,
        "empty_drains": 1,
        "compiler_tape_capacity": 257,
        "initial_bound_override": None,
        "natural_bound_adaptation": True,
        "optimizer_updates": 0,
    })
    for item in metadata:
        emit("SAMPLE_AUDIT", item)

    control = capacity.run(config, 257, inputs, labels)
    observed = capacity.run(
        config, 257, inputs, labels,
        initial_weights=control["initial_weights"],
        compiler_class=audit.AuditCompiler,
        capture_class=CapturePipelineLimiter,
    )

    require(control["hidden"] == observed["hidden"],
            "Instrumentation changed hidden spike IDs/times")
    np.testing.assert_allclose(
        control["scores"], observed["scores"],
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        control["scores"][:3].argmax(axis=1),
        observed["scores"][:3].argmax(axis=1),
    )

    summaries = []
    for old, new in zip(control["records"], observed["records"]):
        index = old["index"]
        require(index == new["index"], "Backward ownership mismatch")
        np.testing.assert_allclose(
            old["limit"], new["limit"], rtol=2e-5, atol=2e-7
        )
        differences = []
        for a, b in zip(old["gradients"], new["gradients"]):
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)
            differences.append(float(np.max(np.abs(a - b))))

        statistics = new["limiter_statistics"]
        require(
            statistics["replayed_jumps"] == len(observed["hidden"][index]),
            f"Replay/forward spike count mismatch for example {index}",
        )
        summaries.append({
            "stable_id": rows[index].stable_id,
            **statistics,
            "gradient_max_absolute_difference_hidden_output": differences,
        })

    require(len(summaries) == 3, "Expected three replay observations")
    emit("PASS_PIPELINE_LIMITER_OBSERVATION", {
        "instrumentation_parity": "pass",
        "score_max_absolute_difference": float(
            np.max(np.abs(control["scores"] - observed["scores"]))
        ),
        "per_example": summaries,
        "total_replayed_jumps": sum(
            s["replayed_jumps"] for s in summaries
        ),
        "total_clipped_jumps": sum(
            s["clipped_jumps"] for s in summaries
        ),
        "optimizer_updates": 0,
        "scope": "three frozen DVS examples, 128ms, dt1ms, B1",
        "numerical_gradient_correctness_certified": False,
        "training_trajectory_clipping_certified": False,
    })


if __name__ == "__main__":
    main()