"""Two-example diagnostic of output-weight staleness during B's replay.

One actual Adam update for A per condition. No update for B.
The saved condition temporarily restores only output weights during B replay.
No checkpoints, validation, test events or longer training.
"""

import argparse
from pathlib import Path

import numpy as np

import dvs_eventprop_capacity_probe as c
import dvs_eventprop_limiter_probe as audit
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_drain_probe import DrainTrace

require = c.require
emit = c.emit
pull = c.pull


def set_output_weight(cn, out, value):
    var = cn.connection_populations[out.connection()].vars["weight"]
    var.view[...] = value.reshape(var.view.shape)
    var.push_to_device()
    np.testing.assert_array_equal(pull(var), value)


class ObserveOneUpdate(c.Callback):
    def __init__(self, original, layers, hid, out, labels, mode, initial):
        self.layers = layers
        self.hid = hid
        self.out = out
        self.labels = labels
        self.mode = mode
        self.initial = initial
        self.observed = {"gradients": []}
        self.learner = DrainTrace(
            original, layers,
            {"step": 0, "epoch": 0, "phase": "diagnostic_A_update"},
            alpha=0.001,
        )

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_begin(self, cn, batch):
        if batch == 1:
            # B must run forward before A's update.
            for layer, expected in zip(self.layers, self.initial[:2]):
                np.testing.assert_array_equal(
                    pull(cn.connection_populations[
                        layer.connection()
                    ].vars["weight"]),
                    expected,
                )
        elif batch == 2:
            require("after_A" in self.observed, "Missing A update")
            if self.mode == "saved":
                set_output_weight(cn, self.out, self.initial[1])
            self.observed["replay_output_weight"] = pull(
                cn.connection_populations[
                    self.out.connection()
                ].vars["weight"]
            )

    def on_batch_end(self, cn, batch, metric_state):
        if batch == 0:
            return
        require(batch in (1, 2), "Unexpected presentation")
        index = batch - 1
        pop = cn.neuron_populations[self.out.population()]
        np.testing.assert_array_equal(
            pull(pop.vars["YTrueBack"]).reshape(-1),
            [int(self.labels[index])],
        )
        gradients = [
            pull(cn.connection_populations[
                layer.connection()
            ].vars["weightGradient"])
            for layer in self.layers
        ]
        require(all(np.isfinite(g).all() for g in gradients),
                "Nonfinite gradient")
        self.observed["gradients"].append(gradients)

        if batch == 1:
            # Executes the original learning callback once, and independently
            # checks its Adam arithmetic through the existing DrainTrace.
            self.learner.on_batch_end(cn, batch, metric_state)
            self.observed["after_A"] = c.checkpoint.training_state(
                cn, self.layers
            )
        else:
            self.observed["replay_limit"] = pull(
                cn.neuron_populations[
                    self.hid.population()
                ].vars["LambdavLimit"]
            )
            if self.mode == "saved":
                set_output_weight(
                    cn, self.out, self.observed["after_A"][1]
                )

        cn.custom_update("ZeroGradient")
        for layer in self.layers:
            require(bool((pull(cn.connection_populations[
                layer.connection()
            ].vars["weightGradient"]) == 0).all()),
                "Gradient clearing failed")


def run(config, inputs, labels, mode, initial_weights=None):
    network, inp, hid, out = c.build_model(config, 1)
    layers = [hid, out]
    if initial_weights is not None:
        for layer, value, shape in zip(
            layers, initial_weights, ((5, 5, 2, 8), (392, 11))
        ):
            layer.connection().connectivity.weight = value.reshape(shape).copy()

    compiler = audit.AuditCompiler(
        example_timesteps=128,
        losses="sparse_categorical_crossentropy",
        dt=1.0, batch_size=1, rng_seed=42,
        max_spikes=257, strict_buffer_checking=True,
        per_timestep_loss=False, grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network, name=f"AdamReplay{mode.title()}",
        optimisers={
            hid: {"weight": c.Adam(alpha=0.001)},
            out: {"weight": c.Adam(alpha=0.001)},
        },
        regularisers={},
    )
    require(len(compiler.instrumentation) == 1,
            "Expected one instrumented hidden population")
    matches = [
        i for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, c.CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")

    empty = c.to_ml_genn(c.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    presentations = inputs + [empty]
    targets = np.r_[labels, np.int64(0)]

    with cn:
        initial = c.checkpoint.training_state(cn, layers)
        if initial_weights is not None:
            for expected, actual in zip(initial_weights, initial[:2]):
                np.testing.assert_array_equal(expected, actual)

        hook = ObserveOneUpdate(
            cn.base_train_callbacks[matches[0]],
            layers, hid, out, labels, mode, initial,
        )
        cn.base_train_callbacks[matches[0]] = hook
        metric = c.RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = c.CallbackList(
            cn.base_train_callbacks + [
                c.SpikeRecorder(inp, key="input"),
                c.SpikeRecorder(hid, key="hidden"),
            ],
            compiled_network=cn, num_batches=3, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []

        for batch, sample in enumerate(presentations):
            metric.real_presentation = batch < 2
            cn._train_batch(
                batch, 0, {inp: [sample]},
                {out: targets[batch:batch + 1]},
                {out: metric}, state, callbacks,
            )
            scores.append(np.array(cn.get_readout(out), copy=True)[0])
            if batch == 0:
                for old, new in zip(
                    initial, c.checkpoint.training_state(cn, layers)
                ):
                    np.testing.assert_array_equal(old, new)

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 2, "Drain entered real metrics")
        require(len(hook.learner.calls) == 1, "Expected exactly one Adam update")
        require(len(hook.observed["gradients"]) == 2,
                "Expected A and B gradient captures")
        for expected, actual in zip(
            hook.observed["after_A"],
            c.checkpoint.training_state(cn, layers),
        ):
            np.testing.assert_array_equal(expected, actual)

        data = callbacks.get_data()
        hidden = []
        for index, sample in enumerate(presentations):
            require(
                c.output_probe.spike_pairs(data["input"], index)
                == c.expected_pairs(sample),
                "Input delivery mismatch",
            )
            hidden.append(c.output_probe.spike_pairs(data["hidden"], index))
        require(hidden[-1] == [], "Hidden spikes in drain")
        scores = np.asarray(scores)
        require(scores.shape == (3, 11) and np.isfinite(scores).all(),
                "Invalid scores")
        np.testing.assert_allclose(scores[-1], 0, rtol=0, atol=1e-7)

        pop = cn.neuron_populations[hid.population()]
        values = {
            name: pull(pop.vars[name]).reshape(-1)
            for name in audit.COUNTERS + audit.SCALARS
        }
        require(
            all(v.shape == (392,) for v in values.values()),
            "Unexpected hidden audit shape",
        )
        require(
            all(np.isfinite(v).all() for v in values.values()),
            "Nonfinite audit telemetry",
        )
        active = values["AuditJumps"] > 0
        statistics = {
            name: int(values[name].astype(np.uint64).sum())
            for name in audit.COUNTERS
        }
        statistics.update({
            "AuditMaxRaw": float(values["AuditMaxRaw"].max()),
            "AuditMaxRatio": float(values["AuditMaxRatio"].max()),
            "AuditMinDenom": (
                float(values["AuditMinDenom"][active].min())
                if active.any() else None
            ),
        })
        emit("B_REPLAY_AUDIT", {
            "mode": mode,
            "forward_hidden_spikes": len(hidden[1]),
            "statistics": statistics,
        })
        require(statistics["AuditJumps"] == len(hidden[1]),
                "B replay jump count mismatch")
        require(statistics["AuditNonfinite"] == 0
                and statistics["AuditBadLimit"] == 0,
                "Invalid replay")
        require(np.isfinite(hook.observed["replay_limit"]).all()
                and (hook.observed["replay_limit"] > 0).all(),
                "Invalid replay limit")

        observed = hook.observed
        emit("ADAM_REPLAY_RUN", {
            "mode": mode,
            "initial_weight_sha256": [
                c.array_hash(w) for w in initial[:2]
            ],
            "hidden_spikes_per_presentation": [len(h) for h in hidden],
            "replay_limit": observed["replay_limit"].tolist(),
            "B_limiter_statistics": statistics,
            "output_forward_to_replay_gap_l2": float(np.linalg.norm(
                observed["replay_output_weight"].astype(np.float64)
                - initial[1].astype(np.float64)
            )),
            "optimizer_updates": 1,
            "updated_example": "A",
            "B_updated": False,
            "final_state_equals_state_after_A_update": True,
        })
        return {
            "initial": initial[:2],
            "after_A": observed["after_A"],
            "scores": scores,
            "hidden": hidden,
            "observed": observed,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = ProbeConfig()
    root = args.root.resolve(strict=True)
    rows = c.pilot(c.inventory(root)["train"], 2)[:2]
    loaded = [c.load_sample(root, row, c.DataConfig(128)) for row in rows]
    payloads, labels, metadata = zip(*loaded)
    inputs = [c.to_ml_genn(p) for p in payloads]
    labels = np.asarray(labels, dtype=np.int64)

    emit("IDENTITY", {
        **c.checkpoint.identity(),
        "diagnostic_sha256": c.file_hash(Path(__file__).resolve()),
        "capacity_helper_sha256": c.file_hash(Path(c.__file__).resolve()),
        "limiter_helper_sha256": c.file_hash(Path(audit.__file__).resolve()),
        "config": config.record(),
        "sample_metadata": list(metadata),
        "scope": "two examples, one Adam update per condition",
    })

    current = run(config, inputs, labels, "current")
    saved = run(config, inputs, labels, "saved", current["initial"])

    require(current["hidden"] == saved["hidden"],
            "Forward hidden spike IDs/times differ between conditions")
    np.testing.assert_allclose(
        current["scores"], saved["scores"], rtol=1e-5, atol=1e-6
    )
    for a, b in zip(current["after_A"], saved["after_A"]):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(
        current["observed"]["replay_limit"],
        saved["observed"]["replay_limit"], rtol=2e-5, atol=1e-9,
    )
    for a, b in zip(
        current["observed"]["gradients"][0],
        saved["observed"]["gradients"][0],
    ):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)

    differences = []
    for projection, (a, b) in enumerate(zip(
        current["observed"]["gradients"][1],
        saved["observed"]["gradients"][1],
    )):
        a, b = a.astype(np.float64), b.astype(np.float64)
        differences.append({
            "projection": projection,
            "max_absolute_difference": float(np.max(np.abs(a - b))),
            "relative_l2_difference_to_saved":
                float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            "within_existing_parity_tolerance":
                bool(np.allclose(a, b, rtol=2e-5, atol=2e-7)),
        })
        if projection == 1:
            np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-7)

    emit("PASS_ADAM_REPLAY_COMPARISON", {
        "B_gradient_comparison": differences,
        "optimizer_updates_per_condition": 1,
        "scope": "output-weight replay substitution only; limited gradients",
        "exact_current_weight_gradient_certified": False,
        "staleness_fixed": False,
    })


if __name__ == "__main__":
    main()