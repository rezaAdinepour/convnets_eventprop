"""Measure adaptive-limiter effects on three frozen DVS examples.

Natural bounds versus a fixed replay bound of 1.0.
No optimizer updates, checkpoints, validation or test events.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_capacity_probe as capacity
import dvs_eventprop_limiter_probe as audit
import dvs_eventprop_pipeline_limiter_probe as pipeline
from dvs_eventprop_config import ProbeConfig

emit = capacity.emit
require = capacity.require
pull = capacity.pull


# Distinct compiler names give each condition its own generated directory.
class NaturalLimitCompiler(audit.AuditCompiler):
    pass


class FixedLimitCompiler(audit.AuditCompiler):
    pass


class FixedLimitCapture(pipeline.CapturePipelineLimiter):
    def on_batch_begin(self, cn, batch):
        var = cn.neuron_populations[
            self.hidden.population()
        ].vars["LambdavLimit"]
        var.view[...] = np.float32(1.0)
        var.push_to_device()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    root = args.root.resolve(strict=True)
    rows = capacity.pilot(capacity.inventory(root)["train"], 2)[:3]
    loaded = [
        capacity.load_sample(root, row, capacity.DataConfig(128))
        for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    inputs = [capacity.to_ml_genn(p) for p in payloads]
    labels = np.asarray(labels, dtype=np.int64)

    emit("IDENTITY", {
        **capacity.checkpoint.identity(),
        "diagnostic_sha256": capacity.file_hash(Path(__file__).resolve()),
        "helper_sha256": {
            Path(module.__file__).name:
                capacity.file_hash(Path(module.__file__).resolve())
            for module in (
                capacity, audit, pipeline, capacity.output_probe
            )
        },
        "config": config.record(),
        "fixed_replay_bound": 1.0,
        "training_order": [row.stable_id for row in rows],
        "sample_metadata": list(metadata),
        "optimizer_updates": 0,
    })

    natural = capacity.run(
        config, 257, inputs, labels,
        compiler_class=NaturalLimitCompiler,
        capture_class=pipeline.CapturePipelineLimiter,
    )
    fixed = capacity.run(
        config, 257, inputs, labels,
        initial_weights=natural["initial_weights"],
        compiler_class=FixedLimitCompiler,
        capture_class=FixedLimitCapture,
    )

    require(natural["hidden"] == fixed["hidden"],
            "Limiter intervention changed forward hidden spike IDs/times")
    np.testing.assert_allclose(
        natural["scores"], fixed["scores"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_array_equal(
        natural["scores"][:3].argmax(axis=1),
        fixed["scores"][:3].argmax(axis=1),
    )

    summaries = []
    for old, new in zip(natural["records"], fixed["records"]):
        index = old["index"]
        require(index == new["index"], "Backward ownership mismatch")
        np.testing.assert_array_equal(
            new["limit"], np.ones_like(new["limit"])
        )
        old_stats = old["limiter_statistics"]
        new_stats = new["limiter_statistics"]
        expected_jumps = len(natural["hidden"][index])
        require(
            old_stats["replayed_jumps"]
            == new_stats["replayed_jumps"] == expected_jumps,
            "Replay jump count mismatch",
        )

        comparisons = []
        for projection, (a, b) in enumerate(zip(
            old["gradients"], new["gradients"]
        )):
            a, b = a.astype(np.float64), b.astype(np.float64)
            require(np.isfinite(a).all() and np.isfinite(b).all(),
                    "Nonfinite gradients")
            comparisons.append({
                "projection": projection,
                "natural_gradient_l2": float(np.linalg.norm(a)),
                "fixed_gradient_l2": float(np.linalg.norm(b)),
                "max_absolute_difference": float(np.max(np.abs(a - b))),
                "relative_l2_difference_to_fixed": float(
                    np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)
                ),
                "within_existing_parity_tolerance": bool(
                    np.allclose(a, b, rtol=2e-5, atol=2e-7)
                ),
            })
            # Hidden clipping should not alter the output-weight gradient.
            if projection == 1 or index == 0:
                np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)

        summary = {
            "example_index": index,
            "stable_id": rows[index].stable_id,
            "natural_limit": old["limit"].tolist(),
            "fixed_limit": new["limit"].tolist(),
            "natural_statistics": old_stats,
            "fixed_statistics": new_stats,
            "gradient_comparison": comparisons,
        }
        summaries.append(summary)
        emit("LIMITER_EFFECT", summary)
        require(new_stats["clipped_jumps"] == 0,
                "Fixed bound 1.0 still clips; comparison needs review")

    require(len(summaries) == 3, "Expected three backward comparisons")
    natural_clips = sum(
        s["natural_statistics"]["clipped_jumps"] for s in summaries
    )
    require(natural_clips > 0, "Natural clipping was not reproduced")

    emit("PASS_LIMITER_EFFECT_COMPARISON", {
        "natural_clipped_jumps": natural_clips,
        "fixed_clipped_jumps": 0,
        "forward_hidden_spike_id_time_parity": "pass",
        "optimizer_updates": 0,
        "weights_and_moments_unchanged": True,
        "scope": "three frozen examples; B1, dt1ms, 128ms",
        "fixed_condition": "no clipping observed in these replays",
        "exact_gradient_correctness_certified": False,
        "training_configuration_changed": False,
    })


if __name__ == "__main__":
    main()