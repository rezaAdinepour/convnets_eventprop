"""Audit/cache all development parents at the 1024ms occupancy profile.

CPU preprocessing only: no mlGeNN/PyGeNN imports or simulation.
Test filenames are inventoried by the existing loader; test arrays stay unread.
Outputs are exclusive new files beneath the user-selected working directory.
"""

import argparse
import hashlib
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

import dvs_data as data

require = data.require
FORMAT = "dvs-development-occ1024-cache-v1"


def emit(name, value):
    print(name, data.json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_cached_sample(directory, entry):
    """Read and validate one manifest-bound cache entry for later runners."""
    directory = Path(directory).resolve(strict=True)
    path = (directory / entry["cache_path"]).resolve(strict=True)
    require(path.is_relative_to(directory), "Cache path escapes its run")
    require(data.file_hash(path) == entry["cache_sha256"], "Cache file hash changed")
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"end_spikes", "spike_times", "label"},
                "Cache schema changed")
        end = np.array(archive["end_spikes"], copy=True)
        times = np.array(archive["spike_times"], copy=True)
        label = np.array(archive["label"], copy=True)
    require(label.shape == () and label.dtype == np.int64
            and int(label) == entry["metadata"]["label"], "Cached label changed")
    payload = data.PreprocessedSpikes(end, times)
    data.validate_payload(payload, neurons=2048, frames=1024)
    require(array_hash(end) == entry["end_spikes_sha256"]
            and array_hash(times) == entry["spike_times_sha256"],
            "Cached payload changed")
    return payload, np.int64(label.item()), entry["metadata"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-reload", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    source_path = args.pilot_reload.resolve(strict=True)
    source = data.json.loads(source_path.read_text())
    project = Path(__file__).resolve().parent
    directory = Path.cwd().resolve()
    require(not directory.is_relative_to(root)
            and not directory.is_relative_to(project),
            "Use a new experiment directory outside the dataset and repository")
    require(source["diagnostic_sha256"] == data.file_hash(
                project / "dvs_eventprop_b32_pilot_reload.py")
            == "3d39b45a21814493382ec7727c38b64ba9b9c3ade8ea4ab8fa6896a6f336501f",
            "Passed pilot reload producer changed")
    require(source["pilot_overfit_gate_reproduced"] is True
            and source["correct"] >= 21 and source["real_examples"] == 22
            and source["optimizer_updates"] == 0
            and source["weights_and_fresh_moments_unchanged"] is True
            and source["fixed_bound_clips"] == 0,
            "Expected a passed, frozen pilot reload")
    require(sys.executable == source["identity"]["executable"]
            and sys.version == source["identity"]["python"]
            and np.__version__ == source["identity"]["numpy"],
            "CPU environment changed since the passed pilot")
    require(data.file_hash(Path(data.__file__))
            == source["identity"]["source_sha256"]["dvs_data.py"],
            "Loader source changed")
    for name, digest in source["helper_sha256"].items():
        require(data.file_hash(project / name) == digest, f"Pilot helper changed: {name}")
    checkpoint = source["source_checkpoint"]
    require(data.file_hash(Path(checkpoint["path"])) == checkpoint["sha256"],
            "Passed pilot checkpoint changed")

    config = data.Config(1024)
    pilot_metadata = {row["stable_id"]: row for row in source["sample_metadata"]}
    require(len(pilot_metadata) == 22, "Pilot parent coverage changed")
    require(all(row["config_sha256"] == data.object_hash(config.record())
                and row["split"] == "train" for row in pilot_metadata.values()),
            "Pilot data condition differs")
    partitions = data.inventory(root)
    require(len(partitions["train"]) == 934 and len(partitions["validation"]) == 143,
            "Unexpected development partition sizes")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project, text=True
    ).strip()
    provenance = {
        "artifact_type": FORMAT,
        "diagnostic_sha256": data.file_hash(Path(__file__)),
        "loader_sha256": data.file_hash(Path(data.__file__)),
        "convnets_commit": head,
        "executable": sys.executable, "python": sys.version,
        "numpy": np.__version__, "dataset_root": str(root),
        "config": config.record(), "config_sha256": data.object_hash(config.record()),
        "pilot_reload_path": str(source_path),
        "pilot_reload_sha256": data.file_hash(source_path),
        "passed_pilot_checkpoint": checkpoint,
        "test_filenames_inventoried": len(partitions["test"]),
        "test_event_arrays_loaded": 0,
        "gpu_environment_reprobed": False,
        "archive_files_rehashed": False,
    }
    (directory / "cache").mkdir(exist_ok=False)
    emit("FULL_DATA_AUDIT_PLAN", provenance)
    started = time.perf_counter()
    entries, summaries = {}, {}
    seen_raw = {}
    matched_pilot = set()
    with (directory / "sample_audit.jsonl").open("x") as audit_stream:
        for split in ("train", "validation"):
            split_directory = directory / "cache" / split
            split_directory.mkdir(exist_ok=False)
            entries[split] = []
            for index, row in enumerate(partitions[split]):
                payload, label, metadata = data.load_sample(root, row, config)
                require(metadata["stable_id"] == row.stable_id
                        and metadata["split"] == split
                        and metadata["config_sha256"] == provenance["config_sha256"],
                        "Sample identity/config mismatch")
                digest = metadata["sample_sha256"]
                prior = seen_raw.get(digest)
                require(prior is None or prior[0] == split,
                        f"Duplicate raw content across development splits: {prior} / {row.path}")
                seen_raw[digest] = (split, row.path)
                if row.stable_id in pilot_metadata:
                    require(data.object_hash(metadata)
                            == data.object_hash(pilot_metadata[row.stable_id]),
                            f"Pilot preprocessing changed: {row.path}")
                    matched_pilot.add(row.stable_id)
                data.validate_payload(payload, neurons=2048, frames=1024)
                per_neuron = np.diff(np.r_[0, payload.end_spikes].astype(np.int64))
                require(int(per_neuron.max()) == metadata["max_bins_per_neuron"]
                        and len(payload.spike_times) == metadata["occupancy_spikes"],
                        "Payload/metadata activity mismatch")
                cache_path = split_directory / f"sample_{index:04d}.npz"
                payload_bytes = payload.end_spikes.nbytes + payload.spike_times.nbytes + 8
                require(shutil.disk_usage(directory).free >= payload_bytes + 512 * 1024**2,
                        "Less than 512MiB disk reserve before the next cache file")
                with cache_path.open("xb") as stream:
                    np.savez(stream, end_spikes=payload.end_spikes,
                             spike_times=payload.spike_times,
                             label=np.asarray(label, dtype=np.int64))
                entry = {
                    "index": index, "metadata": metadata,
                    "cache_path": cache_path.relative_to(directory).as_posix(),
                    "cache_sha256": data.file_hash(cache_path),
                    "end_spikes_sha256": array_hash(payload.end_spikes),
                    "spike_times_sha256": array_hash(payload.spike_times),
                    "payload_bytes": payload_bytes,
                    "cache_file_bytes": cache_path.stat().st_size,
                }
                restored, restored_label, _ = load_cached_sample(directory, entry)
                np.testing.assert_array_equal(restored.end_spikes, payload.end_spikes)
                np.testing.assert_array_equal(restored.spike_times, payload.spike_times)
                require(restored_label == label, "Cache label round-trip failed")
                entries[split].append(entry)
                audit_stream.write(data.json.dumps(entry, sort_keys=True, allow_nan=False) + "\n")
                audit_stream.flush()
                if (index + 1) % 50 == 0 or index + 1 == len(partitions[split]):
                    emit("FULL_DATA_PROGRESS", {"split": split, "completed": index + 1,
                         "total": len(partitions[split]),
                         "elapsed_seconds": time.perf_counter() - started})

            rows = entries[split]
            spike_counts = np.array([e["metadata"]["occupancy_spikes"] for e in rows], dtype=np.int64)
            summaries[split] = {
                "parents": len(rows),
                "class_counts": np.bincount(
                    [e["metadata"]["label"] for e in rows], minlength=11).tolist(),
                "subjects": sorted({e["metadata"]["subject"] for e in rows}),
                "empty_prefixes": int(np.count_nonzero(spike_counts == 0)),
                "occupancy_spikes_total": int(spike_counts.sum()),
                "occupancy_spikes_min_median_max": [int(spike_counts.min()),
                    float(np.median(spike_counts)), int(spike_counts.max())],
                "maximum_input_spikes_per_neuron": max(
                    e["metadata"]["max_bins_per_neuron"] for e in rows),
                "payload_bytes_total": sum(e["payload_bytes"] for e in rows),
                "cache_file_bytes_total": sum(e["cache_file_bytes"] for e in rows),
                "cache_round_trip": "exact_match",
            }
            emit("FULL_DATA_SPLIT_SUMMARY", {"split": split, **summaries[split]})

    require(matched_pilot == set(pilot_metadata), "Not all pilot parents matched")
    order = np.random.default_rng(42).permutation(934).tolist()
    train_batch_sizes = [min(32, 934 - start) for start in range(0, 934, 32)]
    validation_batch_sizes = [min(32, 143 - start) for start in range(0, 143, 32)]
    plan = {
        "batch_size": 32, "epoch1_training_permutation_seed": 42,
        "epoch1_training_permutation": order,
        "training_real_batch_sizes": train_batch_sizes,
        "validation_real_batch_sizes": validation_batch_sizes,
        "training_real_contributions_per_epoch": 934,
        "planned_optimizer_updates_per_epoch": len(train_batch_sizes),
        "last_training_batch_real_examples": train_batch_sizes[-1],
        "last_training_gradient_rescale": 32.0 / train_batch_sizes[-1],
        "full_data_training_executed": False,
        "full_data_update_and_padding_semantics_certified": False,
    }
    report = {**provenance, "entries": entries, "summaries": summaries,
              "batch_plan": plan, "pilot_parent_payloads_reproduced": 22,
              "cross_split_raw_content_duplicates": 0,
              "sample_audit_sha256": data.file_hash(directory / "sample_audit.jsonl"),
              "elapsed_seconds": time.perf_counter() - started,
              "process_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
              "full_window_training_ready": False}
    with (directory / "full_data_manifest.json").open("x") as stream:
        data.json.dump(report, stream, sort_keys=True, allow_nan=False)
    emit("PASS_FULL_DEVELOPMENT_DATA_AUDIT", {
        "manifest_path": str(directory / "full_data_manifest.json"),
        "manifest_sha256": data.file_hash(directory / "full_data_manifest.json"),
        "train_parents": 934, "validation_parents": 143,
        "pilot_parent_payloads_reproduced": 22,
        "cache_round_trip": "exact_match", "test_event_arrays_loaded": 0,
        "elapsed_seconds": report["elapsed_seconds"],
        "process_peak_rss_MiB": report["process_peak_rss_MiB"],
        "planned_optimizer_updates_per_epoch": len(train_batch_sizes),
        "last_training_batch_real_examples": train_batch_sizes[-1],
        "full_window_training_ready": False,
    })


if __name__ == "__main__":
    main()
