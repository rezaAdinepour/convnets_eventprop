"""Frozen B2 inference lane/companion forward-response diagnostic."""

from pathlib import Path
import argparse

import numpy as np

from ml_genn.callbacks import SpikeRecorder

import dvs_eventprop_inference_parity_probe as base
import dvs_eventprop_output_gradient_probe as spike_helper
from dvs_data import object_hash
from dvs_eventprop_probe import expected_pairs

require = base.require
emit = base.emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
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

    cn = base.InferenceCompiler(
        evaluate_timesteps=128,
        dt=1.0, batch_size=2, rng_seed=config.seed,
        reset_time_between_batches=True,
        reset_vars_between_batches=True,
        reset_in_syn_between_batches=True,
    ).compile(network, name="FrozenCompanionB2")

    with cn:
        before = base.weights(cn, layers)
        for actual, expected in zip(before, initial):
            np.testing.assert_array_equal(actual.reshape(-1), expected)

        predictions, data = cn.predict(
            {inp: presentations}, [out],
            callbacks=[
                SpikeRecorder(inp, key="input"),
                SpikeRecorder(hid, key="hidden"),
            ],
        )
        scores = np.asarray(predictions[out], dtype=np.float64)
        require(scores.shape == (10, 11) and np.isfinite(scores).all(),
                "Invalid score array")
        for old, new in zip(before, base.weights(cn, layers)):
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

    emit("PASS_INFERENCE_COMPANION_PARITY", {
        "max_absolute_score_difference": maximum,
        "real_example_presentations": 6,
        "empty_presentations": 4,
        "A_hidden_spike_id_time_parity": "exact",
        "weights_unchanged": True,
        "optimizer_updates": 0,
        "scope": "checked B2 inference lanes/companions; dt1ms, 128ms",
        "training_compiled_B2_parity_certified": False,
        "backward_batch_independence_certified": False,
        "eventprop_gradient_correctness_certified": False,
    })


if __name__ == "__main__":
    main()