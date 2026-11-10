"""Compare inference scores against saved frozen training forward scores."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from ml_genn.compilers import InferenceCompiler

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
from dvs_eventprop_probe import pull
from dvs_eventprop_model import build_model


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))

def weights(cn, layers):
    result = []
    for layer in layers:
        variables = cn.connection_populations[layer.connection()].vars
        require("g" in variables, "Expected inference weight variable 'g'")
        result.append(pull(variables["g"]))
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    reference_path = args.reference.resolve(strict=True)
    reference = json.loads(reference_path.read_text())
    current = checkpoint.identity()

    # This is a new cross-revision comparison, not a checkpoint reload.
    # Preserve both revisions and require unchanged core source/environment.
    for key, value in current.items():
        if key != "convnets_commit":
            require(
                reference["identity"][key] == value,
                f"Reference source/environment mismatch: {key}",
            )

    require(reference["recording"] == "on", "Expected recording-on reference")
    require(reference["optimizer_updates"] == 0, "Reference was not frozen")
    require(
        reference["hidden_spikes_per_presentation"] == [303, 702, 1994, 0],
        "Unexpected reference sequence",
    )

    config = ProbeConfig()
    rows = pilot(inventory(args.root)["train"], 2)[:3]
    loaded = [
        load_sample(args.root, row, DataConfig(128)) for row in rows
    ]
    payloads, labels, metadata = zip(*loaded)

    expected_condition = {
        **config.record(),
        "batch_size": 1,
        "real_examples": 3,
        "empty_drains": 1,
        "training_order": [row.stable_id for row in rows],
    }
    require(
        reference["condition"] == normalized(expected_condition),
        "Reference condition/order mismatch",
    )
    require(
        reference["sample_metadata"] == normalized(list(metadata)),
        "Reference input/provenance mismatch",
    )

    initial = [
        np.asarray(value, dtype=np.float32)
        for value in reference["initial_weights"]
    ]
    require(len(initial) == 2, "Expected two projections")
    for value, size, expected_hash in zip(
        initial, (400, 4312), reference["initial_weight_sha256"]
    ):
        require(value.shape == (size,), "Unexpected saved weight shape")
        require(np.isfinite(value).all(), "Nonfinite initial weights")
        require(
            hashlib.sha256(value.tobytes()).hexdigest() == expected_hash,
            "Initial-weight hash mismatch",
        )

    emit("INFERENCE_PARITY_IDENTITY", {
        "current": current,
        "reference_git_head": reference["identity"]["convnets_commit"],
        "reference_json_sha256": file_hash(reference_path),
        "diagnostic_sha256": file_hash(Path(__file__).resolve()),
        "inference_compiler_sha256":
            file_hash(Path(inspect.getfile(InferenceCompiler)).resolve()),
        "cross_revision_basis": "identical core source hashes and environment",
        "condition": expected_condition,
        "labels": [int(label) for label in labels],
    })

    network, inp, hid, out = build_model(config, 1)
    layers = [hid, out]
    for layer in (inp, hid):
        layer.population().record_spikes = False

    for layer, value, shape in zip(
        layers, initial, ((5, 5, 2, 8), (392, 11))
    ):
        layer.connection().connectivity.weight = value.reshape(shape).copy()

    cn = InferenceCompiler(
        evaluate_timesteps=128,
        dt=1.0,
        batch_size=1,
        rng_seed=config.seed,
        reset_time_between_batches=True,
        reset_vars_between_batches=True,
        reset_in_syn_between_batches=True,
    ).compile(network, name="FrozenInferenceParity")

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    presentations = [to_ml_genn(p) for p in payloads] + [empty]

    with cn:
        before = weights(cn, layers)
        for actual, expected in zip(before, initial):
            np.testing.assert_array_equal(actual.reshape(-1), expected)

        predictions, _ = cn.predict(
            {inp: presentations}, [out], callbacks=[]
        )
        actual_scores = np.asarray(predictions[out], dtype=np.float64)

        for old, new in zip(before, weights(cn, layers)):
            np.testing.assert_array_equal(old, new)

    expected_scores = np.asarray(reference["scores"], dtype=np.float64)
    require(
        actual_scores.shape == expected_scores.shape == (4, 11),
        "Unexpected score shape",
    )
    require(
        np.isfinite(actual_scores).all()
        and np.isfinite(expected_scores).all(),
        "Nonfinite scores",
    )

    for index, (actual, expected) in enumerate(
        zip(actual_scores, expected_scores)
    ):
        emit("FORWARD_PARITY", {
            "presentation": index,
            "empty": index == 3,
            "max_absolute_score_difference":
                float(np.max(np.abs(actual - expected))),
            "training_scores": expected.tolist(),
            "inference_scores": actual.tolist(),
        })

    np.testing.assert_allclose(
        actual_scores, expected_scores, rtol=1e-5, atol=1e-6,
        err_msg="Training/inference forward-score mismatch",
    )
    np.testing.assert_array_equal(
        np.argmax(actual_scores[:3], axis=1),
        np.argmax(expected_scores[:3], axis=1),
    )
    np.testing.assert_allclose(
        actual_scores[-1], 0, rtol=0, atol=1e-7,
        err_msg="Nonzero inference scores for empty presentation",
    )

    emit("PASS_INFERENCE_FORWARD_PARITY", {
        "real_examples": 3,
        "empty_presentations": 1,
        "max_absolute_score_difference":
            float(np.max(np.abs(actual_scores - expected_scores))),
        "weights_unchanged": True,
        "optimizer_updates": 0,
        "scope": "same frozen weights and ordered inputs; B1, dt1ms, 128ms",
        "all_predecessor_and_batch_companion_cases_certified": False,
        "eventprop_gradient_correctness_certified": False,
    })


if __name__ == "__main__":
    main()