"""TASK 10: observe hidden adjoint jumps and limiter activation.

One fixed training example, empty drain, frozen weights.
A control and instrumented model run sequentially.
No mlGeNN source changes, checkpoints or dataset caches.
"""

import argparse
from pathlib import Path

import numpy as np

from ml_genn.callbacks import Callback, SpikeRecorder
from ml_genn.callbacks.custom_update import CustomUpdateOnBatchEnd
from ml_genn.compilers import EventPropCompiler
from ml_genn.neurons import LeakyIntegrateFire
from ml_genn.neurons.input import Input
from ml_genn.optimisers import Adam
from ml_genn.utils.callback_list import CallbackList

import dvs_eventprop_checkpoint_probe as checkpoint
import dvs_eventprop_output_gradient_probe as output_probe
from dvs_data import (
    Config as DataConfig,
    PreprocessedSpikes,
    file_hash,
    inventory,
    load_sample,
    object_hash,
    pilot,
    require,
    to_ml_genn,
)
from dvs_eventprop_config import ProbeConfig
from dvs_eventprop_drain_probe import RealOnlyAccuracy
from dvs_eventprop_model import build_model
from dvs_eventprop_probe import emit, expected_pairs, pull


COUNTERS = (
    "AuditJumps",
    "AuditClips",
    "AuditNonfinite",
    "AuditBadLimit",
)
SCALARS = (
    "AuditMaxRaw",
    "AuditMaxRatio",
    "AuditMinDenom",
)


class AuditCompiler(EventPropCompiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instrumentation = []

    def _build_in_hid_neuron_model(
        self, pop, model, genn_model, compile_state
    ):
        if isinstance(pop.neuron, Input):
            return super()._build_in_hid_neuron_model(
                pop, model, genn_model, compile_state
            )

        require(isinstance(pop.neuron, LeakyIntegrateFire),
                "This instrumentation supports the checked LIF only")

        # Add before super(): its reset-variable collection must include these.
        for name in COUNTERS:
            genn_model.add_var(name, "unsigned int", 0)
        for name in SCALARS:
            genn_model.add_var(
                name, "scalar",
                1e30 if name == "AuditMinDenom" else 0.0,
            )

        result = super()._build_in_hid_neuron_model(
            pop, model, genn_model, compile_state
        )
        code = result.model["sim_code"]
        prefix = "Lambdav = max(min("
        suffix = ",LambdavLimit),-LambdavLimit);"
        matches = [
            line.strip() for line in code.splitlines()
            if line.strip().startswith(prefix)
            and line.strip().endswith(suffix)
        ]
        require(len(matches) == 1,
                "Expected exactly one checked Lambdav clamp assignment")
        original = matches[0]
        require(code.count(original) == 1, "Ambiguous clamp assignment")
        raw = original[len(prefix):-len(suffix)]
        saved_current = "RingIsyn[ringOffset + RingReadOffset]"
        require(saved_current in raw, "Unexpected saved-current expression")

        observation = f"""
        AuditJumps++;
        const scalar auditRawJump = {raw};
        const scalar auditMaxFinite = 3.4028234663852886e38;
        if (!(auditRawJump >= -auditMaxFinite &&
              auditRawJump <= auditMaxFinite)) {{
            AuditNonfinite++;
        }}
        else {{
            AuditMaxRaw = max(AuditMaxRaw, fabs(auditRawJump));
            AuditMinDenom = min(
                AuditMinDenom, fabs({saved_current} - 1.0));
            if (!(LambdavLimit > 0.0 &&
                  LambdavLimit <= auditMaxFinite)) {{
                AuditBadLimit++;
            }}
            else {{
                AuditMaxRatio = max(
                    AuditMaxRatio, fabs(auditRawJump) / LambdavLimit);
                if (fabs(auditRawJump) > LambdavLimit) {{
                    AuditClips++;
                }}
            }}
        }}
        {original}
        """
        result.model["sim_code"] = code.replace(original, observation)
        self.instrumentation.append({
            "population": pop.name,
            "original_assignment": original,
            "sim_code_json_sha256": object_hash(result.model["sim_code"]),
            "original_assignment_retained": True,
        })
        return result


class CaptureBoth(Callback):
    def __init__(self, layers, output, label):
        self.layers = layers
        self.output = output
        self.label = label
        self.results = []

    def create_state(self, compiled_network, **kwargs):
        return compiled_network

    def on_batch_end(self, cn, batch, metric_state):
        if batch == 0:
            return
        require(batch == 1, "Unexpected backward batch")
        pop = cn.neuron_populations[self.output.population()]
        np.testing.assert_array_equal(
            pull(pop.vars["YTrueBack"]).reshape(-1), [self.label]
        )
        gradients = [
            pull(cn.connection_populations[layer.connection()]
                 .vars["weightGradient"])
            for layer in self.layers
        ]
        require(all(np.isfinite(g).all() for g in gradients),
                "Nonfinite projection gradient")
        self.results.append(gradients)
        # Deliberately do not execute GradientLearn.


def run(
    config, payload, label, instrumented,
    initial_weights=None, forced_limit=None,
):
    network, inp, hid, out = build_model(config, 1)
    layers = [hid, out]
    logical_shapes = [(5, 5, 2, 8), (392, 11)]

    if initial_weights is not None:
        for layer, values, shape in zip(
            layers, initial_weights, logical_shapes
        ):
            layer.connection().connectivity.weight = (
                values.reshape(shape).copy()
            )

    compiler_class = AuditCompiler if instrumented else EventPropCompiler
    compiler = compiler_class(
        example_timesteps=128,
        losses="sparse_categorical_crossentropy",
        dt=1.0,
        batch_size=1,
        rng_seed=config.seed,
        max_spikes=257,
        strict_buffer_checking=True,
        per_timestep_loss=False,
        grad_limit=100.0,
        softmax_temperature=1.0,
    )
    cn = compiler.compile(
        network,
        name="LimiterObserved" if instrumented else "LimiterControl",
        optimisers={
            hid: {"weight": Adam(alpha=config.learning_rate)},
            out: {"weight": Adam(alpha=config.learning_rate)},
        },
        regularisers={},
    )
    if instrumented:
        require(len(compiler.instrumentation) == 1,
                "Expected one instrumented hidden population")
        emit("INSTRUMENTATION", compiler.instrumentation)

    matches = [
        i for i, cb in enumerate(cn.base_train_callbacks)
        if isinstance(cb, CustomUpdateOnBatchEnd)
        and cb.name == "GradientLearn"
    ]
    require(len(matches) == 1, "Unexpected learning callback structure")
    capture = CaptureBoth(layers, out, int(label[0]))
    cn.base_train_callbacks[matches[0]] = capture

    empty = to_ml_genn(PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))

    with cn:
        if forced_limit is not None:
            require(np.isfinite(forced_limit) and forced_limit > 0,
                    "Invalid diagnostic bound")
            limit_var = (
                cn.neuron_populations[hid.population()]
                .vars["LambdavLimit"]
            )
            limit_var.view[...] = np.float32(forced_limit)
            limit_var.push_to_device()
        start_weights = [
            pull(cn.connection_populations[layer.connection()]
                 .vars["weight"])
            for layer in layers
        ]
        if initial_weights is not None:
            for old, new in zip(initial_weights, start_weights):
                np.testing.assert_array_equal(old, new)

        initial_state = checkpoint.training_state(cn, layers)
        metric = RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = CallbackList(
            cn.base_train_callbacks + [
                SpikeRecorder(inp, key="input"),
                SpikeRecorder(hid, key="hidden"),
            ],
            compiled_network=cn,
            num_batches=2,
            num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)

        scores = None
        replay_limit = None
        for batch, sample in enumerate((payload, empty)):
            metric.real_presentation = batch == 0
            if batch == 1:
                replay_limit = pull(
                    cn.neuron_populations[hid.population()]
                    .vars["LambdavLimit"]
                )
            cn._train_batch(
                batch, 0, {inp: [sample]}, {out: label},
                {out: metric}, state, callbacks,
            )
            input_pop = cn.neuron_populations[inp.population()]
            np.testing.assert_array_equal(
                pull(input_pop.vars["StartSpike"]),
                pull(input_pop.vars["EndSpike"]),
            )
            if batch == 0:
                scores = np.array(cn.get_readout(out), copy=True)

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1, "Drain entered real metrics")
        require(len(capture.results) == 1, "Wrong gradient capture count")
        require(scores.shape == (1, 11) and np.isfinite(scores).all(),
                "Invalid scores")

        data = callbacks.get_data()
        require(
            output_probe.spike_pairs(data["input"], 0)
            == expected_pairs(payload),
            "Input delivery mismatch",
        )
        require(output_probe.spike_pairs(data["input"], 1) == [],
                "Input spikes in drain")
        hidden_spikes = output_probe.spike_pairs(data["hidden"], 0)
        require(len(hidden_spikes) > 0, "Silent hidden diagnostic")
        require(output_probe.spike_pairs(data["hidden"], 1) == [],
                "Hidden spikes in empty drain")

        for old, new in zip(
            initial_state, checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        statistics = None
        if instrumented:
            pop = cn.neuron_populations[hid.population()]
            values = {
                name: pull(pop.vars[name]).reshape(-1)
                for name in COUNTERS + SCALARS
            }
            active = values["AuditJumps"] > 0
            jumps = int(values["AuditJumps"].sum())
            clips = int(values["AuditClips"].sum())
            statistics = {
                "replayed_jumps": jumps,
                "recorded_forward_hidden_spikes": len(hidden_spikes),
                "clipped_jumps": clips,
                "nonfinite_raw_jumps": int(values["AuditNonfinite"].sum()),
                "invalid_limits": int(values["AuditBadLimit"].sum()),
                "max_absolute_raw_jump": float(
                    values["AuditMaxRaw"].max()
                ),
                "max_raw_to_limit_ratio": float(
                    values["AuditMaxRatio"].max()
                ),
                "min_absolute_Isyn_minus_threshold": float(
                    values["AuditMinDenom"][active].min()
                ) if active.any() else None,
                "limit_before_replay": replay_limit.tolist(),
                "neurons_with_clipping": int(
                    np.count_nonzero(values["AuditClips"])
                ),
            }
            emit("LIMITER_OBSERVATION", statistics)
            require(jumps == len(hidden_spikes),
                    "Replay jump count differs from forward spike count")
            require(statistics["nonfinite_raw_jumps"] == 0,
                    "Nonfinite raw adjoint jump")
            require(statistics["invalid_limits"] == 0,
                    "Invalid limiter bound")
            require(clips <= jumps, "Invalid clipping count")

        emit("RUN_RESULT", {
            "instrumented": instrumented,
            "scores": scores.tolist(),
            "hidden_spikes": len(hidden_spikes),
            "gradient_l2_hidden_output": [
                float(np.linalg.norm(g.astype(np.float64)))
                for g in capture.results[0]
            ],
            "optimizer_updates": 0,
            "weights_and_moments_unchanged": True,
        })
        return {
            "initial_weights": start_weights,
            "scores": scores.copy(),
            "hidden_spikes": hidden_spikes,
            "gradients": [g.copy() for g in capture.results[0]],
            "statistics": statistics,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--positive-control", action="store_true",
        help="Override the initial hidden adjoint bound to 1e-6",
    )
    args = parser.parse_args()

    forced_limit = 1e-6 if args.positive_control else None

    config = ProbeConfig()
    np.random.seed(config.seed)
    root = args.root.resolve(strict=True)
    row = pilot(inventory(root)["train"], 2)[0]
    payload, target, metadata = load_sample(
        root, row, DataConfig(128)
    )
    payload = to_ml_genn(payload)
    label = np.array([target], dtype=np.int64)

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": file_hash(Path(__file__)),
        "spike_helper_source_sha256": file_hash(
            Path(output_probe.__file__)
        ),
    })
    emit("CONFIG", {
        **config.record(),
        "diagnostic": "hidden-limiter-observation-v2",
        "real_examples": 1,
        "optimizer_updates": 0,
        "positive_control": args.positive_control,
        "initial_adjoint_bound_override": forced_limit,
    })
    emit("SAMPLE_AUDIT", metadata)

    control = run(
        config, payload, label, False,
        forced_limit=forced_limit,
    )
    observed = run(
        config, payload, label, True,
        initial_weights=control["initial_weights"],
        forced_limit=forced_limit,
    )

    require(control["hidden_spikes"] == observed["hidden_spikes"],
            "Instrumentation changed hidden spike timing")
    np.testing.assert_allclose(
        control["scores"], observed["scores"],
        rtol=2e-5, atol=2e-7,
    )
    gradient_differences = []
    for old, new in zip(control["gradients"], observed["gradients"]):
        np.testing.assert_allclose(old, new, rtol=2e-5, atol=2e-7)
        gradient_differences.append(float(np.max(np.abs(old - new))))

    if args.positive_control:
        require(observed["statistics"]["clipped_jumps"] > 0,
                "Positive control detected no clipping")
        require(observed["statistics"]["max_raw_to_limit_ratio"] > 1.0,
                "Positive-control raw jumps never exceeded the bound")

    emit("PASS_LIMITER_OBSERVATION", {
        "instrumentation_parity": "pass",
        "gradient_max_absolute_difference_hidden_output":
            gradient_differences,
        "statistics": observed["statistics"],
        "optimizer_updates": 0,
        "scope": "one frozen training example at dt=1ms",
        "hidden_gradient_correctness_certified": False,
    })


if __name__ == "__main__":
    main()