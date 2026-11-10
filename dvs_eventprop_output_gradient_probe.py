"""TASK 10: frozen-weight output-gradient directional diagnostic.

One fixed training example, one empty backward drain, no optimizer updates.
No checkpoints or dataset caches are written.
"""

import argparse
from pathlib import Path

import numpy as np

from ml_genn.callbacks import SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.utils.callback_list import CallbackList

import dvs_eventprop_checkpoint_probe as checkpoint
from dvs_data import (
    Config as DataConfig,
    PreprocessedSpikes,
    file_hash,
    inventory,
    load_sample,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_model import compile_model
from dvs_eventprop_probe import emit, pull


class CaptureWithoutLearning(CustomUpdateOnBatchEnd):
    """Capture at the original learning boundary, without calling learning."""

    def __init__(self, original, layers, output, label):
        super().__init__(original.name, original.filter_fn)
        self.layers = layers
        self.output = output
        self.label = label
        self.calls = []
        self.captured_gradients = []

    def on_batch_end(self, cn, batch, metric_state):
        if self.filter_fn is not None and not self.filter_fn(batch):
            return

        require(batch == 1, "Unexpected backward presentation")
        pop = cn.neuron_populations[self.output.population()]
        np.testing.assert_array_equal(
            pull(pop.vars["YTrueBack"]).reshape(-1),
            np.array([self.label]),
        )

        gradients = [
            pull(cn.connection_populations[layer.connection()]
                 .vars["weightGradient"])
            for layer in self.layers
        ]
        require(all(np.isfinite(g).all() for g in gradients),
                "Nonfinite captured gradient")
        self.captured_gradients.append(
            gradients[-1].astype(np.float64)
        )
        self.calls.append(batch)
        emit("CAPTURE", {
            "backward_presentations": self.calls,
            "optimizer_updates": 0,
            "gradient_l2_hidden_output": [
                float(np.linalg.norm(g.astype(np.float64)))
                for g in gradients
            ],
        })
        # Deliberately do not invoke the original GradientLearn callback.


def spike_pairs(data, index=0):
    times, ids = data
    return sorted(zip(
        np.asarray(ids[index]).astype(np.int64).tolist(),
        np.asarray(times[index]).astype(np.float64).tolist(),
    ))


def forward(cn, inp, hid, out, payload, label):
    metric = checkpoint.ScoreMetric()
    state = {out: metric.create_state()}
    callbacks = CallbackList(
        cn.base_validate_callbacks
        + [SpikeRecorder(hid, key="hidden")],
        compiled_network=cn,
        num_batches=1,
        num_epochs=1,
    )
    callbacks.on_test_begin()
    callbacks.on_epoch_begin(0)
    cn._validate_batch(
        0, {inp: [payload]}, {out: label},
        {out: metric}, state, callbacks,
    )
    callbacks.on_epoch_end(0, state)
    callbacks.on_test_end(state)
    require(state[out].total == 1, "Wrong forward denominator")
    return (
        metric.scores[0][0].astype(np.float64),
        spike_pairs(callbacks.get_data()["hidden"]),
    )


def ce(scores, label):
    maximum = scores.max()
    return float(
        maximum + np.log(np.exp(scores - maximum).sum())
        - scores[label]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dt-ms", type=float, choices=(1.0, 0.5, 0.25), default=1.0
    )
    args = parser.parse_args()

    # frames here means simulation steps; physical duration stays 128 ms.
    config = ProbeConfig(
        frames=int(128.0 / args.dt_ms),
        dt_ms=args.dt_ms,
    )
    require(config.frames * config.dt_ms == 128.0,
            "Diagnostic physical horizon changed")
    np.random.seed(config.seed)
    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
    })
    emit("CONFIG", {
        **config.record(),
        "diagnostic": "frozen-output-direction-dt-sensitivity-v2",
        "physical_horizon_ms": 128.0,
        "simulation_timesteps": config.frames,
        "data_occupancy_bins": 128,
        "data_bin_width_ms": 1.0,
        "optimizer_updates": 0,
        "epsilons": [0.01, 0.003, 0.001],
    })

    root = args.root.resolve(strict=True)
    row = pilot(inventory(root)["train"], 2)[0]
    payload, target, metadata = load_sample(
        root, row, DataConfig(128)
    )
    emit("SAMPLE_AUDIT", metadata)
    payload = to_ml_genn(payload)
    label = np.array([target], dtype=np.int64)
    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))

    _, cn, inp, hid, out = compile_model(
        config, 1, learn=True, name="DVSOutputGradientFrozen"
    )
    layers = [hid, out]
    matches = [
        (i, cb) for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    index, original = matches[0]
    capture = CaptureWithoutLearning(original, layers, out, int(target))
    cn.base_train_callbacks[index] = capture

    with cn:
        initial_state = checkpoint.training_state(cn, layers)
        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks
            + [SpikeRecorder(hid, key="hidden")],
            compiled_network=cn,
            num_batches=2,
            num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        warmup_scores = None
        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
            current_label = label if batch == 0 else np.array([0], np.int64)
            cn._train_batch(
                batch, 0, {inp: [sample]}, {out: current_label},
                {out: metric}, state, callbacks,
            )
            if batch == 0:
                warmup_scores = np.array(cn.get_readout(out), copy=True)[0]

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(capture.calls == [1], "Wrong gradient capture count")
        require(state[out].total == 1, "Drain entered metric count")
        recorded = callbacks.get_data()["hidden"]
        expected_hidden = spike_pairs(recorded, 0)
        require(len(expected_hidden) > 0, "Silent diagnostic example")
        require(spike_pairs(recorded, 1) == [], "Hidden spikes in empty drain")

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        baseline, hidden = forward(cn, inp, hid, out, payload, label)
        require(hidden == expected_hidden, "Baseline hidden timing changed")
        np.testing.assert_allclose(
            baseline, warmup_scores, rtol=1e-5, atol=1e-6
        )

        require(len(capture.captured_gradients) == 1,
                "Expected exactly one captured output gradient")
        gradient = capture.captured_gradients[0]
        norm = float(np.linalg.norm(gradient))
        require(np.isfinite(norm) and norm > 1e-8,
                "Output gradient is too small for this diagnostic")
        direction = gradient / norm

        var = cn.connection_populations[out.connection()].vars["weight"]
        original_weight = pull(var)
        require(original_weight.shape == gradient.shape,
                "Weight/gradient storage mismatch")

        def set_weight(value):
            var.view[...] = np.asarray(value, dtype=original_weight.dtype)
            var.push_to_device()

        probability = np.exp(baseline - baseline.max())
        probability /= probability.sum()
        score_derivative = probability.copy()
        score_derivative[int(target)] -= 1.0

        results = []
        try:
            for epsilon in (0.01, 0.003, 0.001):
                scores = []
                for sign in (1.0, -1.0):
                    set_weight(
                        original_weight.astype(np.float64)
                        + sign * epsilon * direction
                    )
                    value, hidden = forward(
                        cn, inp, hid, out, payload, label
                    )
                    require(hidden == expected_hidden,
                            "Output perturbation changed hidden spike timing")
                    scores.append(value)

                plus, minus = scores
                np.testing.assert_allclose(
                    (plus + minus) / 2.0, baseline,
                    rtol=1e-5, atol=1e-6,
                    err_msg="Scores failed expected affine midpoint check",
                )
                finite_difference = (
                    ce(plus, int(target)) - ce(minus, int(target))
                ) / (2.0 * epsilon)
                chain_rule = float(
                    score_derivative @ ((plus - minus) / (2.0 * epsilon))
                )
                scale = max(abs(norm), abs(finite_difference), 1e-12)
                result = {
                    "epsilon": epsilon,
                    "eventprop_directional_derivative": norm,
                    "forward_ce_finite_difference": finite_difference,
                    "forward_score_chain_rule": chain_rule,
                    "relative_error": abs(norm - finite_difference) / scale,
                    "within_1pct_plus_1e6_absolute": bool(
                        abs(norm - finite_difference) <= 0.01 * scale + 1e-6
                    ),
                    "hidden_spike_times_unchanged": True,
                }
                results.append(result)
                emit("OUTPUT_DIRECTION", result)
        finally:
            set_weight(original_weight)

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        derivatives = [
            r["forward_ce_finite_difference"] for r in results
        ]
        fd_scale = max(max(abs(v) for v in derivatives), 1e-12)
        plateau = (
            max(derivatives) - min(derivatives)
            <= 0.005 * fd_scale + 1e-6
        )
        emit("OUTPUT_DIAGNOSTIC", {
            "structural_checks": "pass",
            "optimizer_updates": 0,
            "weights_and_adam_moments_restored_unchanged": True,
            "hidden_spikes": len(expected_hidden),
            "finite_difference_plateau_within_0p5pct": bool(plateau),
            "all_directions_within_1pct": all(
                r["within_1pct_plus_1e6_absolute"] for r in results
            ),
            "scope": (
                f"one output-gradient direction at dt={config.dt_ms}ms"
            ),            "convolution_or_hidden_derivative_certified": False,
        })


if __name__ == "__main__":
    main()
