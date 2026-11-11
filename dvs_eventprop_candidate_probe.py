"""Single-example DVS candidate stability/resource measurement.

128 ms physical horizon; unchanged 1 ms occupancy input representation.
One forward example plus empty backward drain. Zero optimizer updates.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from pygenn.genn_model import UnresolvedType

import dvs_eventprop_resource_probe as resource
import dvs_eventprop_limiter_probe as audit
import dvs_eventprop_pipeline_limiter_probe as pipeline

require = resource.require
emit = resource.emit
pull = resource.pull


class DoubleScalarAuditCompiler(audit.AuditCompiler):
    def pre_compile(self, network, genn_model, **kwargs):
        require(
            not genn_model.neuron_populations
            and not genn_model.synapse_populations,
            "Precision must be selected before populations are built",
        )
        genn_model.precision = UnresolvedType("double")
        # Keep the same clock type. These binary timestep values are exact.
        genn_model.time_precision = UnresolvedType("float")
        return super().pre_compile(network, genn_model, **kwargs)

    def build_neuron_model(self, pop, model, compile_state):
        result = super().build_neuron_model(pop, model, compile_state)
        scratch_names = {"SpikeTimes", "RingSpikeTime", "RingIsyn"}

        for name, value in list(result.egp_vals.items()):
            if isinstance(value, np.ndarray) and np.issubdtype(
                value.dtype, np.floating
            ):
                if name in scratch_names:
                    # These buffers originate from np.empty. Allocate the
                    # correct dtype without reading their undefined contents.
                    result.egp_vals[name] = np.empty(
                        value.shape, dtype=np.float64
                    )
                else:
                    require(np.isfinite(value).all(),
                            f"Nonfinite initialized EGP: {name}")
                    result.egp_vals[name] = value.astype(np.float64)
        return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dt-ms",
        type=float,
        choices=(
            1.0, 0.0078125, 0.00390625, 0.001953125,
            0.0009765625, 0.00048828125,
        ),
        default=0.0078125,
    )
    parser.add_argument(
        "--scalar-precision", choices=("float", "double"), default="float"
    )
    parser.add_argument("--initial-values-from", type=Path)
    parser.add_argument(
        "--allow-initializer-timestep-change", action="store_true"
    )
    parser.add_argument("--tape-capacity", type=int)
    parser.add_argument(
        "--allow-initializer-capacity-change", action="store_true"
    )
    parser.add_argument(
        "--input-frames", type=int, choices=(128, 1024), default=128
    )
    parser.add_argument(
        "--allow-initializer-window-change", action="store_true"
    )
    args = parser.parse_args()
    require(
        not (
            args.allow_initializer_timestep_change
            or args.allow_initializer_capacity_change
            or args.allow_initializer_window_change
        )
        or args.initial_values_from is not None,
        "Initializer change flags require --initial-values-from",
    )
    require(
        args.input_frames == 128 or args.tape_capacity is not None,
        "A 1024ms diagnostic requires an explicit tape capacity",
    )
    require(
        args.scalar_precision != "double"
        or args.initial_values_from is not None,
        "The double precision control requires saved initial weight values",
    )
    scalar_dtype = np.dtype(
        np.float64 if args.scalar_precision == "double" else np.float32
    )

    horizon_ms = float(args.input_frames)
    steps = int(horizon_ms / args.dt_ms)
    require(steps * args.dt_ms == horizon_ms, "Horizon mismatch")
    capacity = (
        2 * steps + 1
        if args.tape_capacity is None else args.tape_capacity
    )
    require(capacity >= 3, "Tape capacity must be at least 3")
    # frames still means input occupancy bins, not simulation timesteps.
    config = resource.ProbeConfig(
        frames=args.input_frames, dt_ms=args.dt_ms
    )
    np.random.seed(config.seed)
    require(
        resource.os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "", "0"),
        "CUDA device mapping needs review",
    )

    observer = resource.ObserveResources()
    observer.snapshot("baseline")

    with observer.phase("data_preparation"):
        root = args.root.resolve(strict=True)
        row = resource.pilot(resource.inventory(root)["train"], 2)[0]
        data_config = resource.DataConfig(args.input_frames)
        payload, label, metadata = resource.load_sample(
            root, row, data_config
        )
        payload = resource.to_ml_genn(payload)
        labels = np.array([label], dtype=np.int64)

    condition = {
        **config.record(),
        "profile": data_config.profile,
        "scalar_precision": args.scalar_precision,
        "time_precision": "float",
        "frames_meaning": "number of 1ms input occupancy bins",
        "input_bin_ms": 1.0,
        "physical_horizon_ms": horizon_ms,
        "simulation_timesteps": steps,
        "compiler_max_spikes": capacity,
        "batch_size": 1,
        "real_examples": 1,
        "empty_drains": 1,
        "optimizer_updates": 0,
    }
    identity = {
        **resource.checkpoint.identity(),
        "diagnostic_sha256": resource.file_hash(Path(__file__).resolve()),
        "helper_sha256": {
            Path(module.__file__).name:
                resource.file_hash(Path(module.__file__).resolve())
            for module in (
                resource, audit, pipeline,
                resource.capacity, resource.output_probe,
            )
        },
        "nvtx_library_sha256": resource.file_hash(resource.NVTX_LIBRARY),
    }
    previous = None
    initial_values_source = None
    if args.initial_values_from is not None:
        source_path = args.initial_values_from.resolve(strict=True)
        previous = resource.json.loads(source_path.read_text())

        # Explicit cross-revision initializer import, not a checkpoint reload.
        for key, value in identity.items():
            if key not in {"convnets_commit", "diagnostic_sha256"}:
                require(previous["identity"][key] == value,
                        f"Initializer source/environment mismatch: {key}")

        source_condition = dict(previous["condition"])
        source_condition.setdefault("scalar_precision", "float")
        source_condition.setdefault("time_precision", "float")
        require(source_condition["scalar_precision"] == "float",
                "This control expects a float32 initializer artifact")
        require(source_condition["time_precision"] == "float",
                "Unexpected source clock precision")
        require(set(source_condition) == set(condition),
                "Initializer condition schema changed")

        allowed_changes = {"scalar_precision"}
        if args.allow_initializer_timestep_change:
            allowed_changes.update({
                "dt_ms", "simulation_timesteps", "compiler_max_spikes"
            })
        if args.allow_initializer_capacity_change:
            allowed_changes.add("compiler_max_spikes")
        if args.allow_initializer_window_change:
            require(
                source_condition["frames"] == 128
                and condition["frames"] == 1024,
                "Only an explicit 128ms-to-1024ms initializer import is allowed",
            )
            allowed_changes.update({
                "frames", "profile", "physical_horizon_ms",
                "simulation_timesteps",
            })
        condition_changes = {
            key: {
                "source": source_condition[key],
                "target": condition[key],
            }
            for key in condition
            if source_condition[key] != condition[key]
        }
        require(
            set(condition_changes).issubset(allowed_changes),
            "Initializer condition changed beyond explicitly allowed fields",
        )
        window_import_audit = None
        if args.allow_initializer_window_change:
            # Reconstruct the original input condition and preserve its prefix.
            source_data_config = resource.DataConfig(source_condition["frames"])
            source_payload, source_label, source_metadata = (
                resource.load_sample(root, row, source_data_config)
            )
            require(
                previous["sample_metadata"] == resource.normalize(source_metadata),
                "Reconstructed initializer source metadata differs",
            )
            require(source_label == label,
                    "Window extension changed the label")
            for key in (
                "archive_sha256", "sample_sha256", "stable_id",
                "parent_id", "origin_us", "loader_source_sha256",
            ):
                require(
                    source_metadata[key] == metadata[key],
                    f"Window extension changed raw provenance: {key}",
                )
            source_payload = resource.to_ml_genn(source_payload)
            source_pairs = resource.expected_pairs(source_payload)
            target_prefix_pairs = [
                pair for pair in resource.expected_pairs(payload)
                if pair[1] < source_condition["physical_horizon_ms"]
            ]
            require(
                target_prefix_pairs == source_pairs,
                "Extended input does not preserve the original 128ms prefix",
            )
            window_import_audit = {
                "source_sample_metadata": resource.normalize(source_metadata),
                "source_prefix_spikes": len(source_pairs),
                "target_prefix_spikes": len(target_prefix_pairs),
                "prefix_id_time_parity": "exact",
                "raw_parent_provenance": "exact_match",
            }
        else:
            require(
                previous["sample_metadata"] == resource.normalize(metadata),
                "Initializer source input/provenance mismatch",
            )
        require(previous["optimizer_updates"] == 0
                and previous["weights_and_moments_unchanged"],
                "Initializer artifact was not frozen")
        require(len(previous["initial_weights"]) == 2
                and len(previous["initial_weight_sha256"]) == 2,
                "Expected two initializer projections")

        for values, size, digest in zip(
            previous["initial_weights"], (400, 4312),
            previous["initial_weight_sha256"],
        ):
            original = np.asarray(values, dtype=np.float64)
            stored = np.asarray(values, dtype=np.float32)
            require(original.shape == stored.shape == (size,)
                    and np.isfinite(original).all(),
                    "Invalid saved initializer")
            np.testing.assert_array_equal(
                original, stored.astype(np.float64)
            )
            require(resource.capacity.array_hash(stored) == digest,
                    "Initializer hash mismatch")

        initial_values_source = {
            "condition_changes": condition_changes,
            "window_import_audit": window_import_audit,
            "path": str(source_path),
            "sha256": resource.file_hash(source_path),
            "reference_git_head": previous["identity"]["convnets_commit"],
            "operation": "import initial numerical values only",
        }
    emit("CANDIDATE_PLAN", {
        "initial_values_source": initial_values_source,
        "identity": identity,
        "condition": condition,
        "data_config": data_config.record(),
        "sample_metadata": metadata,
    })

    with observer.phase("model_definition"):
        network, inp, hid, out = resource.build_model(config, 1)
        layers = [hid, out]
        if previous is not None:
            for layer, values, shape in zip(
                layers, previous["initial_weights"],
                ((5, 5, 2, 8), (392, 11)),
            ):
                layer.connection().connectivity.weight = (
                    np.asarray(values, dtype=scalar_dtype)
                    .reshape(shape).copy()
                )

        compiler_class = (
            DoubleScalarAuditCompiler
            if args.scalar_precision == "double"
            else audit.AuditCompiler
        )
        compiler = compiler_class(
            example_timesteps=steps,
            losses="sparse_categorical_crossentropy",
            dt=args.dt_ms, batch_size=1, rng_seed=config.seed,
            max_spikes=capacity, strict_buffer_checking=True,
            per_timestep_loss=False, grad_limit=100.0,
            softmax_temperature=1.0,
        )
        cn = compiler.compile(
            network, name="SingleDVSCandidate",
            optimisers={
                hid: {"weight": resource.Adam(alpha=config.learning_rate)},
                out: {"weight": resource.Adam(alpha=config.learning_rate)},
            },
            regularisers={},
        )
        require(len(compiler.instrumentation) == 1,
                "Expected one instrumented hidden population")

    learning = [
        i for i, callback in enumerate(cn.base_train_callbacks)
        if isinstance(callback, resource.CustomUpdateOnBatchEnd)
        and callback.name == "GradientLearn"
    ]
    require(len(learning) == 1, "Unexpected learning callback structure")
    capture = pipeline.CapturePipelineLimiter(
        layers, hid, out, labels, capacity
    )
    cn.base_train_callbacks[learning[0]] = capture

    with observer.phase("build"):
        cn.genn_model.build()
    observer.snapshot("after_build")

    empty = resource.to_ml_genn(resource.PreprocessedSpikes(
        np.zeros(2048, dtype=np.uint32),
        np.empty(0, dtype=np.float32),
    ))
    presentations = [payload, empty]
    targets = np.array([label, 0], dtype=np.int64)
    stack = ExitStack()

    try:
        with observer.phase("load_and_initial_sync"):
            stack.enter_context(cn)
            cn.get_readout(out)
        observer.snapshot("after_load")

        initial_state = resource.checkpoint.training_state(cn, layers)
        initial_weights = initial_state[:2]
        require(all(value.dtype == scalar_dtype for value in initial_state),
                "Weight/Adam storage precision differs from requested precision")

        observed_dtypes = {}
        for name, layer in (("hidden", hid), ("output", out)):
            pop = cn.neuron_populations[layer.population()]
            group = cn.connection_populations[layer.connection()]
            for variable in ("v", "Lambdav"):
                value = pull(pop.vars[variable])
                require(value.dtype == scalar_dtype,
                        f"{name}/{variable} precision mismatch")
                observed_dtypes[f"{name}/{variable}"] = str(value.dtype)
            for variable in ("weight", "weightGradient"):
                value = pull(group.vars[variable])
                require(value.dtype == scalar_dtype,
                        f"{name}/{variable} precision mismatch")
                observed_dtypes[f"{name}/{variable}"] = str(value.dtype)

        for name, layer in (("input", inp), ("hidden", hid), ("output", out)):
            pop = cn.neuron_populations[layer.population()]
            for variable, egp in pop.extra_global_params.items():
                dtype = egp.view.dtype
                if np.issubdtype(dtype, np.floating):
                    require(dtype == scalar_dtype,
                            f"{name}/{variable} scalar-buffer precision mismatch")
                    observed_dtypes[f"{name}/egp/{variable}"] = str(dtype)

        if previous is not None:
            for actual, expected in zip(
                initial_weights, previous["initial_weights"]
            ):
                np.testing.assert_array_equal(
                    actual.reshape(-1).astype(np.float64),
                    np.asarray(expected, dtype=np.float64),
                )

        emit("OBSERVED_PRECISION", {
            "requested_scalar_precision": args.scalar_precision,
            "time_precision": "float",
            "array_dtypes": observed_dtypes,
            "initial_values_exactly_restored": previous is not None,
        })
        metric = resource.RealOnlyAccuracy()
        state = {out: metric.create_state()}
        callbacks = resource.CallbackList(
            cn.base_train_callbacks + [
                resource.SpikeRecorder(inp, key="input"),
                resource.SpikeRecorder(hid, key="hidden"),
            ],
            compiled_network=cn, num_batches=2, num_epochs=1,
        )
        callbacks.on_train_begin()
        callbacks.on_epoch_begin(0)
        scores = []

        for batch, sample in enumerate(presentations):
            metric.real_presentation = batch == 0
            with observer.phase(f"presentation_{batch}"):
                cn._train_batch(
                    batch, 0, {inp: [sample]},
                    {out: targets[batch:batch + 1]},
                    {out: metric}, state, callbacks,
                )
                value = np.array(cn.get_readout(out), copy=True)
                require(value.shape == (1, 11)
                        and np.isfinite(value).all(), "Invalid scores")
                scores.append(value[0])
                pop = cn.neuron_populations[inp.population()]
                np.testing.assert_array_equal(
                    pull(pop.vars["StartSpike"]),
                    pull(pop.vars["EndSpike"]),
                )
            observer.snapshot(f"after_presentation_{batch}")

        callbacks.on_epoch_end(0, state)
        callbacks.on_train_end(state)
        require(state[out].total == 1, "Drain entered real metrics")
        require([r["index"] for r in capture.records] == [0],
                "Missing or duplicated backward contribution")
        np.testing.assert_allclose(scores[-1], 0, rtol=0, atol=1e-7)

        for old, new in zip(
            initial_state, resource.checkpoint.training_state(cn, layers)
        ):
            np.testing.assert_array_equal(old, new)

        recorded = callbacks.get_data()
        hidden = []
        for index, sample in enumerate(presentations):
            require(
                resource.output_probe.spike_pairs(recorded["input"], index)
                == resource.expected_pairs(sample),
                "Recorded input mismatch",
            )
            hidden.append(resource.output_probe.spike_pairs(
                recorded["hidden"], index
            ))
        require(len(hidden[0]) > 0, "Entire hidden population is silent")
        require(hidden[1] == [], "Hidden spikes in drain")

        record = capture.records[0]
        statistics = record["limiter_statistics"]
        require(statistics["replayed_jumps"] == len(hidden[0]),
                "Forward/replay jump count mismatch")
        gradient_norms = [
            float(np.linalg.norm(g.astype(np.float64)))
            for g in record["gradients"]
        ]
        require(all(np.isfinite(g) and g > 1e-10 for g in gradient_norms),
                "Invalid or zero aggregate gradient")

        neuron_counts = np.bincount(
            [neuron for neuron, _ in hidden[0]], minlength=392
        )
        filter_counts = neuron_counts.reshape(49, 8).sum(axis=0)
    finally:
        with observer.phase("unload"):
            stack.close()

    observer.snapshot("after_unload")
    minimum_free = min(s["free_fraction"] for s in observer.snapshots)
    result = {
        "initial_values_source": initial_values_source,
        "initial_weight_dtypes": [str(w.dtype) for w in initial_weights],
        "observed_dtypes": observed_dtypes,
        "identity": identity,
        "condition": condition,
        "sample_metadata": resource.normalize(metadata),
        "initial_weights": [w.tolist() for w in initial_weights],
        "initial_weight_sha256": [
            resource.capacity.array_hash(w) for w in initial_weights
        ],
        "scores": np.asarray(scores).tolist(),
        "input_spikes": len(payload.spike_times),
        "hidden_spikes": len(hidden[0]),
        "hidden_spike_pairs": hidden[0],
        "hidden_max_spikes_per_neuron": int(neuron_counts.max()),
        "hidden_spikes_per_filter": filter_counts.tolist(),
        "hidden_spikes_per_neuron": neuron_counts.tolist(),
        "gradient_l2_hidden_output": gradient_norms,
        "gradients": [g.tolist() for g in record["gradients"]],
        "limiter_statistics": statistics,
        "phases": observer.phases,
        "memory_snapshots": observer.snapshots,
        "minimum_sampled_free_fraction": minimum_free,
        "optimizer_updates": 0,
        "weights_and_moments_unchanged": True,
    }
    destination = Path.cwd() / "candidate_result.json"
    with destination.open("x") as stream:
        resource.json.dump(
            result, stream, sort_keys=True, allow_nan=False
        )

    emit("CANDIDATE_RESOURCE_SUMMARY", {
        "result": str(destination),
        "dt_ms": args.dt_ms,
        "physical_horizon_ms": horizon_ms,
        "simulation_timesteps": steps,
        "hidden_spikes": len(hidden[0]),
        "hidden_max_spikes_per_neuron": int(neuron_counts.max()),
        "hidden_spikes_per_filter": filter_counts.tolist(),
        "gradient_l2_hidden_output": gradient_norms,
        "limiter_statistics": statistics,
        "minimum_sampled_free_fraction": minimum_free,
        "optimizer_updates": 0,
    })
    require(minimum_free >= 0.20,
            "Less than 20 percent sampled device memory headroom")
    emit("PASS_SINGLE_DVS_CANDIDATE", {
        "scope": (
            f"one frozen {args.input_frames}ms DVS example "
            "and backward drain"
        ),
        "strict_buffer_checking": True,
        "weights_and_moments_unchanged": True,
        "continuous_device_headroom_certified": False,
        "uninstrumented_training_throughput_certified": False,
        "full_window_training_ready": False,
    })


if __name__ == "__main__":
    main()
