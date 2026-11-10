"""TASK 09: bounded B1 backward-drain and Adam-clock diagnostic.

Uses the inspected local private _train_batch API.
No validation, checkpoints, or scientific accuracy reporting.
"""

import argparse
from pathlib import Path

import numpy as np

from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.metrics.sparse_categorical_accuracy import (
    SparseCategoricalAccuracy,
)
from ml_genn.utils.callback_list import CallbackList

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
from dvs_eventprop_model import compile_model
from dvs_eventprop_probe import Observe, emit, pull


class RealOnlyAccuracy(SparseCategoricalAccuracy):
    """Exclude the empty drain from metric state."""

    def __init__(self):
        self.real_presentation = True

    def update(self, state, y_true, y_pred, communicator):
        if self.real_presentation:
            super().update(state, y_true, y_pred, communicator)
        else:
            # Empty input should produce zero forward scores after reset.
            np.testing.assert_allclose(
                y_pred, 0.0, rtol=0.0, atol=1e-7,
                err_msg="Nonzero forward scores during empty drain",
            )


class DrainTrace(CustomUpdateOnBatchEnd):
    """Trace existing learning calls and check their Adam weight updates."""

    def __init__(self, original, layers, context, alpha):
        super().__init__(original.name, original.filter_fn)
        self.layers = layers
        self.context = context
        self.alpha = alpha
        self.calls = []
        self.moments = {}

    def on_batch_end(self, cn, batch, metric_state):
        enabled = self.filter_fn is None or self.filter_fn(batch)
        if not enabled:
            super().on_batch_end(cn, batch, metric_state)
            return

        step = self.context["step"]
        before, gradients = [], []

        for layer in self.layers:
            group = cn.connection_populations[layer.connection()]
            weight = pull(group.vars["weight"])
            gradient = pull(group.vars["weightGradient"])
            require(weight.shape == gradient.shape,
                    "Weight/gradient storage shapes differ")
            require(bool(np.isfinite(gradient).all()), "Nonfinite gradient")
            before.append(weight)
            gradients.append(gradient)

        # Invoke exactly the original compiler learning callback.
        super().on_batch_end(cn, batch, metric_state)

        norms, changes = [], []
        for index, (layer, weight, gradient) in enumerate(
            zip(self.layers, before, gradients)
        ):
            group = cn.connection_populations[layer.connection()]
            actual = pull(group.vars["weight"])
            require(bool(np.isfinite(actual).all()), "Nonfinite weight")

            # Independent host calculation of configured Adam:
            # beta1=.9, beta2=.999, epsilon=1e-8, zero-based step.
            g = gradient.astype(np.float64)
            m, v = self.moments.get(
                index, (np.zeros_like(g), np.zeros_like(g))
            )
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            predicted = weight.astype(np.float64) - self.alpha * (
                m / (1.0 - 0.9 ** (step + 1))
            ) / (
                np.sqrt(v / (1.0 - 0.999 ** (step + 1))) + 1e-8
            )
            np.testing.assert_allclose(
                actual, predicted, rtol=2e-5, atol=2e-7,
                err_msg=f"Adam weight-update mismatch in projection {index}",
            )
            self.moments[index] = (m, v)

            norms.append(float(np.linalg.norm(g)))
            changes.append(float(np.linalg.norm(
                actual.astype(np.float64) - weight.astype(np.float64)
            )))

        entry = (
            self.context["epoch"],
            batch - 1,
            step,
        )
        self.calls.append(entry)
        emit("DRAIN_LEARN", {
            "epoch": entry[0],
            "backward_sample_index": entry[1],
            "adam_step": entry[2],
            "phase": self.context["phase"],
            "gradient_l2_hidden_output": norms,
            "weight_change_l2_hidden_output": changes,
            "adam_weight_check": "pass",
        })


def weights(cn, layers):
    return [
        pull(cn.connection_populations[layer.connection()].vars["weight"])
        for layer in layers
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", type=int, choices=(1, 3), default=1)
    parser.add_argument("--epochs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()

    config = ProbeConfig()
    np.random.seed(config.seed)
    emit("CONFIG", {
        **config.record(),
        "samples": args.samples,
        "epochs": args.epochs,
        "batch_size": 1,
        "shuffle": False,
        "runner": "explicit-empty-drain-v1",
    })

    directory = Path(__file__).resolve().parent
    for name in (
        "dvs_data.py",
        "dvs_eventprop_config.py",
        "dvs_eventprop_model.py",
        "dvs_eventprop_probe.py",
        "dvs_eventprop_drain_probe.py",
    ):
        path = directory / name
        emit("SOURCE", {"path": str(path), "sha256": file_hash(path)})

    partitions = inventory(args.root)
    rows = pilot(partitions["train"], 2)[:args.samples]
    loaded = [
        load_sample(args.root, row, DataConfig(config.frames))
        for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    labels = np.asarray(labels, dtype=np.int64)
    for item in metadata:
        emit("SAMPLE_AUDIT", item)

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    inputs = [to_ml_genn(p) for p in payloads] + [empty]

    # Dummy label is used only by the unlearned empty forward presentation.
    targets = np.r_[labels, np.int64(0)]

    _, cn, inp, hid, out = compile_model(
        config, 1, learn=True,
        name=f"DVSDrainN{args.samples}E{args.epochs}",
    )
    layers = [hid, out]
    context = {"epoch": 0, "step": 0, "phase": "warmup"}

    matches = [
        (i, callback)
        for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected GradientLearn callback structure")
    index, original = matches[0]
    trace = DrainTrace(original, layers, context, config.learning_rate)
    cn.base_train_callbacks[index] = trace

    emit("REAL_ORDER", [row.stable_id for row in rows])

    metric = RealOnlyAccuracy()
    updates = 0

    # One context preserves Adam moments across the optional second epoch.
    with cn:
        callbacks = CallbackList(
            cn.base_train_callbacks + [Observe(inp, out, targets)],
            compiled_network=cn,
            num_batches=args.samples + 1,
            num_epochs=args.epochs,
        )
        callbacks.on_train_begin()

        for epoch in range(args.epochs):
            context["epoch"] = epoch
            state = {out: metric.create_state()}
            callbacks.on_epoch_begin(epoch)

            for batch in range(args.samples + 1):
                drain = batch == args.samples
                context["phase"] = (
                    "drain" if drain else
                    "warmup" if batch == 0 else "real_forward"
                )
                context["step"] = updates
                metric.real_presentation = not drain

                before = weights(cn, layers) if batch == 0 else None

                # At batch>0, learning belongs to the preceding real sample.
                # Warmup sets the clock but performs no GradientLearn call.
                cn._train_batch(
                    batch=batch,
                    step=updates,
                    x={inp: [inputs[batch]]},
                    y={out: targets[batch:batch + 1]},
                    metrics={out: metric},
                    metric_state=state,
                    callback_list=callbacks,
                )

                if batch == 0:
                    for old, new in zip(before, weights(cn, layers)):
                        require(np.array_equal(old, new),
                                "Warmup unexpectedly changed weights")
                else:
                    updates += 1

            callbacks.on_epoch_end(epoch, state)
            require(state[out].total == args.samples,
                    "Drain contaminated real-example metric count")
            emit("EPOCH_COVERAGE", {
                "epoch": epoch,
                "real_metric_examples": int(state[out].total),
                "real_updates": args.samples,
                "cumulative_updates": updates,
                "drain_metric_examples": 0,
            })

        callbacks.on_train_end(state)

    expected = [
        (epoch, sample, epoch * args.samples + sample)
        for epoch in range(args.epochs)
        for sample in range(args.samples)
    ]
    require(trace.calls == expected, "Missing/duplicate update ownership")
    emit("PASS_DRAIN_CLOCK", {
        "calls_epoch_sample_step": trace.calls,
        "updates": updates,
        "real_samples_per_epoch": args.samples,
        "epochs": args.epochs,
        "validation_performed": False,
    })


if __name__ == "__main__":
    main()
