"""Single-example DVS candidate stability/resource measurement.

128 ms physical horizon; unchanged 1 ms occupancy input representation.
One forward example plus empty backward drain. Zero optimizer updates.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import dvs_eventprop_resource_probe as resource
import dvs_eventprop_limiter_probe as audit
import dvs_eventprop_pipeline_limiter_probe as pipeline

require = resource.require
emit = resource.emit
pull = resource.pull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--dt-ms",
        type=float,
        choices=(1.0, 0.0078125, 0.00390625, 0.001953125),
        default=0.0078125,
    )
    args = parser.parse_args()

    horizon_ms = 128.0
    steps = int(horizon_ms / args.dt_ms)
    require(steps * args.dt_ms == horizon_ms, "Horizon mismatch")
    capacity = 2 * steps + 1
    # frames still means input occupancy bins, not simulation timesteps.
    config = resource.ProbeConfig(frames=128, dt_ms=args.dt_ms)
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
        data_config = resource.DataConfig(128)
        payload, label, metadata = resource.load_sample(
            root, row, data_config
        )
        payload = resource.to_ml_genn(payload)
        labels = np.array([label], dtype=np.int64)

    condition = {
        **config.record(),
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
    emit("CANDIDATE_PLAN", {
        "identity": identity,
        "condition": condition,
        "data_config": data_config.record(),
        "sample_metadata": metadata,
    })

    with observer.phase("model_definition"):
        network, inp, hid, out = resource.build_model(config, 1)
        layers = [hid, out]
        compiler = audit.AuditCompiler(
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
        "scope": "one frozen 128ms DVS example and backward drain",
        "strict_buffer_checking": True,
        "weights_and_moments_unchanged": True,
        "continuous_device_headroom_certified": False,
        "uninstrumented_training_throughput_certified": False,
        "full_window_training_ready": False,
    })


if __name__ == "__main__":
    main()