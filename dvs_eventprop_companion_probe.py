"""Frozen B2 inference lane/companion forward-response diagnostic."""

from pathlib import Path
import argparse

import numpy as np

from ml_genn.callbacks import SpikeRecorder

import dvs_eventprop_inference_parity_probe as base
import dvs_eventprop_output_gradient_probe as spike_helper
from dvs_data import object_hash
from dvs_eventprop_probe import expected_pairs


from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.compilers.compiled_training_network import CompiledTrainingNetwork
from ml_genn.metrics.sparse_categorical_accuracy import SparseCategoricalAccuracy
from ml_genn.optimisers import Adam
from ml_genn.utils.callback_list import CallbackList

require = base.require
emit = base.emit


class RealLaneAccuracy(SparseCategoricalAccuracy):
    def update(self, state, y_true, y_pred, communicator):
        mask = np.asarray(self.real_mask, dtype=bool)
        require(mask.shape == (len(y_true),), "Invalid real-lane mask")
        np.testing.assert_allclose(
            y_pred[~mask], 0, rtol=0, atol=1e-7,
            err_msg="Nonzero scores in an empty lane",
        )
        if mask.any():
            super().update(
                state, y_true[mask], y_pred[mask], communicator
            )


def training_forward(cn, inp, hid, out, presentations, layout, labels):
    learning = [
        i for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(learning) == 1, "Unexpected learning callback structure")
    require(len(cn.optimisers) == 2, "Expected two configured optimizers")
    original = cn.base_train_callbacks[learning[0]]
    cn.base_train_callbacks[learning[0]] = CustomUpdateOnBatchEnd(
        "ZeroGradient", original.filter_fn
    )

    metric = RealLaneAccuracy()
    state = {out: metric.create_state()}
    callbacks = CallbackList(
        cn.base_train_callbacks + [
            SpikeRecorder(inp, key="input"),
            SpikeRecorder(hid, key="hidden"),
        ],
        compiled_network=cn,
        num_batches=len(layout), num_epochs=1,
    )
    callbacks.on_train_begin()
    callbacks.on_epoch_begin(0)
    scores = []

    for batch, indices in enumerate(layout):
        metric.real_mask = np.asarray(
            [index != 3 for index in indices], dtype=bool
        )
        targets = np.asarray(
            [0 if index == 3 else int(labels[index]) for index in indices],
            dtype=np.int64,
        )
        cn._train_batch(
            batch, 0,
            {inp: presentations[2 * batch:2 * batch + 2]},
            {out: targets},
            {out: metric}, state, callbacks,
        )
        value = np.array(cn.get_readout(out), copy=True)
        require(value.shape == (2, 11) and np.isfinite(value).all(),
                "Invalid training-compiled scores")
        scores.append(value)

    callbacks.on_epoch_end(0, state)
    callbacks.on_train_end(state)
    require(state[out].total == 6, "Empty lanes contaminated metric counts")
    return np.concatenate(scores, axis=0), callbacks.get_data()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--compiler", choices=("inference", "eventprop"), default="inference"
    )
    args = parser.parse_args()

    reference_path = args.reference.resolve(strict=True)
    reference = base.json.loads(reference_path.read_text())
    current = base.checkpoint.identity()
    for key, value in current.items():
        if key != "convnets_commit":
            require(reference["identity"][key] == value,
                    f"Reference source/environment mismatch: {key}")
    require(reference["optimizer_updates"] == 0, "Reference was not frozen")
    require(reference["recording"] == "on", "Expected recording-on reference")
    require(reference["hidden_spikes_per_presentation"] == [303, 702, 1994, 0],
            "Unexpected reference counts")

    config = base.ProbeConfig()
    root = args.root.resolve(strict=True)
    rows = base.pilot(base.inventory(root)["train"], 2)[:3]
    loaded = [
        base.load_sample(root, row, base.DataConfig(128)) for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)
    expected_reference_condition = {
        **config.record(),
        "batch_size": 1,
        "real_examples": 3,
        "empty_drains": 1,
        "training_order": [row.stable_id for row in rows],
    }
    require(
        reference["condition"] == base.normalized(expected_reference_condition),
        "Reference condition/order mismatch",
    )
    require(
        reference["sample_metadata"] == base.normalized(list(metadata)),
        "Reference input/provenance mismatch",
    )

    initial = [
        np.asarray(value, dtype=np.float32)
        for value in reference["initial_weights"]
    ]
    require(len(initial) == 2, "Expected two projections")
    for value, size, digest in zip(
        initial, (400, 4312), reference["initial_weight_sha256"]
    ):
        require(value.shape == (size,) and np.isfinite(value).all(),
                "Invalid initial weights")
        require(base.hashlib.sha256(value.tobytes()).hexdigest() == digest,
                "Initial-weight hash mismatch")

    # Reference index 3 denotes the empty presentation.
    layout = [(0, 1), (2, 0), (0, 3), (3, 0), (3, 3)]
    order = [index for batch in layout for index in batch]
    empty = base.to_ml_genn(base.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    examples = [base.to_ml_genn(p) for p in payloads] + [empty]
    presentations = [examples[index] for index in order]

    emit("COMPANION_PLAN", {
        "current": current,
            "compiler_mode": args.compiler,
        "eventprop_compiler_sha256": base.file_hash(
            Path(base.inspect.getfile(EventPropCompiler)).resolve()
        ),
        "training_loop_sha256": base.file_hash(
            Path(base.inspect.getfile(CompiledTrainingNetwork)).resolve()
        ),
        "reference_git_head": reference["identity"]["convnets_commit"],
        "reference_json_sha256": base.file_hash(reference_path),
        "diagnostic_sha256": base.file_hash(Path(__file__).resolve()),
        "inference_helper_sha256": base.file_hash(Path(base.__file__).resolve()),
        "spike_helper_sha256":
            base.file_hash(Path(spike_helper.__file__).resolve()),
        "inference_compiler_sha256": base.file_hash(
            Path(base.inspect.getfile(base.InferenceCompiler)).resolve()
        ),
        "batch_size": 2,
        "layout_reference_indices": layout,
        "empty_reference_index": 3,
        "dt_ms": 1.0,
        "frames": 128,
        "optimizer_updates": 0,
    })

    network, inp, hid, out = base.build_model(config, 2)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = True
    for layer, value, shape in zip(
        layers, initial, ((5, 5, 2, 8), (392, 11))
    ):
        layer.connection().connectivity.weight = value.reshape(shape).copy()

    if args.compiler == "inference":
        cn = base.InferenceCompiler(
            evaluate_timesteps=128,
            dt=1.0, batch_size=2, rng_seed=config.seed,
            reset_time_between_batches=True,
            reset_vars_between_batches=True,
            reset_in_syn_between_batches=True,
        ).compile(network, name="FrozenCompanionB2")
    else:
        cn = EventPropCompiler(
            example_timesteps=128,
            losses="sparse_categorical_crossentropy",
            dt=1.0, batch_size=2, rng_seed=config.seed,
            max_spikes=257, strict_buffer_checking=True,
            per_timestep_loss=False, grad_limit=100.0,
            softmax_temperature=1.0,
        ).compile(
            network, name="FrozenTrainingCompanionB2",
            optimisers={
                hid: {"weight": Adam(alpha=0.001)},
                out: {"weight": Adam(alpha=0.001)},
            },
            regularisers={},
        )

    with cn:
        def read_state():
            if args.compiler == "eventprop":
                return base.checkpoint.training_state(cn, layers)
            return base.weights(cn, layers)

        before = read_state()
        for actual, expected in zip(before[:2], initial):
            np.testing.assert_array_equal(actual.reshape(-1), expected)

        if args.compiler == "eventprop":
            scores, data = training_forward(
                cn, inp, hid, out, presentations, layout, labels
            )
        else:
            predictions, data = cn.predict(
                {inp: presentations}, [out],
                callbacks=[
                    SpikeRecorder(inp, key="input"),
                    SpikeRecorder(hid, key="hidden"),
                ],
            )
            scores = predictions[out]

        scores = np.asarray(scores, dtype=np.float64)
        require(scores.shape == (10, 11) and np.isfinite(scores).all(),
                "Invalid score array")

        after = read_state()
        require(len(before) == len(after), "State layout changed")
        for old, new in zip(before, after):
            np.testing.assert_array_equal(old, new)

        for key in ("input", "hidden"):
            require(len(data[key][0]) == len(data[key][1]) == 10,
                    f"Incorrect {key} recording count")

        hidden = []
        for slot, payload in enumerate(presentations):
            require(
                spike_helper.spike_pairs(data["input"], slot)
                == expected_pairs(payload),
                f"Input delivery mismatch in slot {slot}",
            )
            hidden.append(spike_helper.spike_pairs(data["hidden"], slot))

    reference_scores = np.asarray(reference["scores"], dtype=np.float64)
    require(reference_scores.shape == (4, 11)
            and np.isfinite(reference_scores).all(),
            "Invalid reference scores")
    a_hidden = None
    maximum = 0.0

    for slot, index in enumerate(order):
        difference = float(np.max(np.abs(scores[slot] - reference_scores[index])))
        maximum = max(maximum, difference)
        np.testing.assert_allclose(
            scores[slot], reference_scores[index], rtol=1e-5, atol=1e-6
        )
        require(
            len(hidden[slot]) == reference["hidden_spikes_per_presentation"][index],
            f"Hidden count mismatch in slot {slot}",
        )
        if index == 3:
            require(hidden[slot] == [], "Hidden spikes for empty input")
            np.testing.assert_allclose(scores[slot], 0, rtol=0, atol=1e-7)
        else:
            require(
                int(np.argmax(scores[slot]))
                == int(np.argmax(reference_scores[index])),
                "Real-example decision mismatch",
            )
        if index == 0:
            if a_hidden is None:
                a_hidden = hidden[slot]
            require(hidden[slot] == a_hidden,
                    "A's hidden spike IDs/times depend on lane or companion")

        emit("COMPANION_FORWARD", {
            "batch": slot // 2,
            "lane": slot % 2,
            "reference_example_index": index,
            "max_absolute_score_difference": difference,
            "hidden_spikes": len(hidden[slot]),
            "hidden_spike_pairs_sha256": object_hash(hidden[slot]),
        })

    emit(
        "PASS_TRAINING_COMPANION_FORWARD"
        if args.compiler == "eventprop"
        else "PASS_INFERENCE_COMPANION_PARITY",
        {
            "max_absolute_score_difference": maximum,
            "real_example_presentations": 6,
            "empty_presentations": 4,
            "A_hidden_spike_id_time_parity": "exact",
            "weights_unchanged": True,
            "optimizer_updates": 0,
            "scope": (
                f"checked B2 {args.compiler} forward lanes/companions; "
                "dt1ms, 128ms"
            ),
            "training_compiled_B2_forward_parity_checked":
                args.compiler == "eventprop",
            "adam_moments_unchanged":
                True if args.compiler == "eventprop" else None,
            "backward_batch_independence_certified": False,
            "eventprop_gradient_correctness_certified": False,
        },
    )


if __name__ == "__main__":
    main()