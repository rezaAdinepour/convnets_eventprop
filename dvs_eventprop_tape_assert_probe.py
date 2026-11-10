"""TASK 10: strict EventProp tape-assertion positive control.

Driver uses isolated child processes and exclusive log creation.
Synthetic input only; no optimizer updates or dataset access.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_name(mode):
    return f"StrictTape{mode.title()}"


def child(mode):
    # Prevent an expected child-process abort from creating a core dump.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    import numpy as np

    from ml_genn import InputLayer, Layer, SequentialNetwork
    from ml_genn.callbacks import SpikeRecorder
    from ml_genn.compilers import EventPropCompiler
    from ml_genn.connectivity import Dense
    from ml_genn.neurons import SpikeInput, LeakyIntegrate
    from ml_genn.synapses import Exponential
    from ml_genn.utils.data import PreprocessedSpikes

    import dvs_eventprop_checkpoint_probe as checkpoint

    capacity = 17 if mode == "control" else 2
    payload = PreprocessedSpikes(
        np.array([4], dtype=np.uint32),
        np.array([0, 1, 2, 3], dtype=np.float32),
    )

    emit("IDENTITY", {
        **checkpoint.identity(),
        "diagnostic_source_sha256": digest(Path(__file__)),
    })
    emit("TAPE_CONFIG", {
        "mode": mode,
        "architecture": "SpikeInput(1)->Dense->LI(2)",
        "dt_ms": 1.0,
        "simulation_timesteps": 8,
        "physical_horizon_ms": 8.0,
        "input_upload_capacity": 4,
        "compiler_tape_capacity": capacity,
        "strict_buffer_checking": True,
        "scheduled_spikes": [[0, 0], [0, 1], [0, 2], [0, 3]],
        "optimizer_updates": 0,
    })

    network = SequentialNetwork()
    with network:
        inp = InputLayer(
            SpikeInput(max_spikes=4),
            shape=(1,),
            record_spikes=True,
            name="tape_input",
        )
        out = Layer(
            Dense(np.array([[0.2, -0.1]], np.float32), delay=0),
            LeakyIntegrate(
                v=0.0, bias=0.0, tau_mem=20.0,
                readout="avg_var_exp_weight",
            ),
            shape=(2,),
            synapse=Exponential(tau=5.0),
            name="tape_output",
        )

    cn = EventPropCompiler(
        example_timesteps=8,
        losses="sparse_categorical_crossentropy",
        dt=1.0,
        batch_size=1,
        rng_seed=42,
        max_spikes=capacity,
        strict_buffer_checking=True,
        per_timestep_loss=False,
    ).compile(
        network,
        name=build_name(mode),
        optimisers={},
        regularisers={},
    )
    require(not cn.optimisers, "Unexpected optimizer")

    with cn:
        emit("SIMULATION_BEGIN", {
            "mode": mode,
            "expected_overflow_time_ms": 1.0 if mode == "overflow" else None,
        })
        _, recorded = cn.train(
            {inp: [payload]},
            {out: np.array([0], dtype=np.int64)},
            num_epochs=1,
            shuffle=False,
            callbacks=[SpikeRecorder(inp, key="input")],
        )

        # Pulling results also requires completed device work.
        scores = np.array(cn.get_readout(out), copy=True)
        require(np.isfinite(scores).all(), "Nonfinite scores")

        if mode == "overflow":
            emit("OVERFLOW_SURVIVED", {
                "strict_check_failed_to_terminate": True,
            })
        else:
            times, ids = recorded["input"]
            require(len(times) == len(ids) == 1, "Wrong recording count")
            actual = sorted(zip(
                np.asarray(ids[0]).astype(int).tolist(),
                np.asarray(times[0]).astype(float).tolist(),
            ))
            require(actual == [(0, 0.0), (0, 1.0), (0, 2.0), (0, 3.0)],
                    f"Control input mismatch: {actual}")
            pop = cn.neuron_populations[inp.population()]
            pop.vars["StartSpike"].pull_from_device()
            pop.vars["EndSpike"].pull_from_device()
            np.testing.assert_array_equal(
                pop.vars["StartSpike"].view,
                pop.vars["EndSpike"].view,
            )

    if mode == "overflow":
        raise SystemExit(2)

    emit("PASS_TAPE_CONTROL", {
        "recorded_spikes": 4,
        "compiler_tape_capacity": capacity,
        "optimizer_updates": 0,
    })


def check():
    directory = Path.cwd()
    script = Path(__file__).resolve()

    for mode in ("control", "overflow"):
        log_path = directory / f"{mode}.log"
        emit("CHILD_START", {"mode": mode, "log": str(log_path)})
        with log_path.open("x") as stream:
            process = subprocess.run(
                [sys.executable, "-u", str(script), "--mode", mode],
                cwd=directory,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )

        log = log_path.read_text(errors="replace")
        print(log, end="" if log.endswith("\n") else "\n", flush=True)

        generated = (
            directory / f"{build_name(mode)}EventPropCompiler_CODE"
        )
        source_info = {}
        for name in ("Makefile", "neuronUpdate.cc"):
            path = generated / name
            if path.is_file():
                source_info[name] = {"sha256": digest(path)}
                if name == "Makefile":
                    source_info[name]["mentions_NDEBUG"] = (
                        "NDEBUG" in path.read_text()
                    )
                else:
                    source_info[name]["contains_assert_false"] = (
                        "assert(false);" in path.read_text()
                    )

        emit("CHILD_RESULT", {
            "mode": mode,
            "exit_code": process.returncode,
            "log": str(log_path),
            "log_sha256": digest(log_path),
            "generated_source": source_info,
        })

        if mode == "control":
            require(
                process.returncode == 0 and "PASS_TAPE_CONTROL " in log,
                "Control failed; stop before overflow interpretation",
            )
        else:
            assertion_reported = (
                "device-side assert triggered" in log.lower()
                or "cudaerrorassert" in log.lower()
                or re.search(
                    r"Assertion[^\n]*false[^\n]*failed", log, re.IGNORECASE
                ) is not None
            )
            require("OVERFLOW_SURVIVED " not in log,
                    "Overflow simulation completed despite strict checking")
            require(process.returncode != 0,
                    "Overflow child unexpectedly exited successfully")
            require("ring buffer violation" in log.lower(),
                    "Failure lacks the expected tape-overflow diagnostic")
            require(assertion_reported,
                    "Failure lacks a device-assertion diagnostic")

    emit("PASS_STRICT_TAPE_ASSERTION", {
        "control": "pass",
        "deliberate_overflow": "device assertion detected",
        "separate_child_processes": True,
        "optimizer_updates": 0,
        "scope": "tiny scheduled-input tape path",
        "full_model_capacity_or_resources_certified": False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("check", "control", "overflow"), default="check"
    )
    args = parser.parse_args()
    if args.mode == "check":
        check()
    else:
        child(args.mode)


if __name__ == "__main__":
    main()