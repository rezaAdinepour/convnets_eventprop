"""Offline DVS Gesture host loader for TASK 06's base32 occupancy contract.

Dependencies: Python standard library and NumPy.
No downloads, caches, model imports, or filesystem writes.

Production output:
    population shape: (32, 32, 2), HWC
    end_spikes:       (2048,), uint32
    spike_times:     (K,), float32 milliseconds
    labels:          int64, archive IDs 0..10
"""

from __future__ import annotations

import argparse
from collections import Counter, namedtuple
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator, Sequence

import numpy as np


LABELS = (
    "Hand clapping",
    "Right hand wave",
    "Left hand wave",
    "Right arm clockwise",
    "Right arm counterclockwise",
    "Left arm clockwise",
    "Left arm counterclockwise",
    "Arm roll",
    "Air drums",
    "Air guitar",
    "Other gestures",
)

ARCHIVES = {
    "ibmGestureTrain": (
        "7ca8fba75f8bef5f89ab22282c157b93e65826e488bfe4aa2c6040f833e5bbb7"
    ),
    "ibmGestureTest": (
        "c6bfdf5812c33289047e5321b69e12fce1c824c4fd1da081c873143b7db44f57"
    ),
}

MANIFEST_HASHES = {
    "train": "f47f22535fcad6f0c2d607110ce63d99241b9fcf08efbcd2f47f325419af1b12",
    "validation": (
        "397447d1df53dd4dc47db6b82e302a27f5d73cb8a07f548cf37b9cf1ca186d94"
    ),
}

EVENT_DTYPE = np.dtype(
    [("x", np.int16), ("y", np.int16), ("p", np.bool_), ("t", np.int64)]
)

# Structurally matches mlGeNN's inspected namedtuple without importing mlGeNN.
PreprocessedSpikes = namedtuple(
    "PreprocessedSpikes", ("end_spikes", "spike_times")
)

PATH_PATTERN = re.compile(
    r"(ibmGestureTrain|ibmGestureTest)/"
    r"user([0-9]{2})_([^/]+)/([0-9]+)\.npy"
)


def require(condition: bool, message: str) -> None:
    """Validation remains active when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def encoded(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def object_hash(value: object) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class Config:
    frames: int = 1024

    def __post_init__(self) -> None:
        require(
            type(self.frames) is int and self.frames in (128, 1024),
            "frames must be exactly 128 diagnostic or 1024 production",
        )

    @property
    def profile(self) -> str:
        if self.frames == 128:
            return "dvs-occ1ms-prefix128-diagnostic-v1"
        return "dvs-occ1ms-prefix1024-v1"

    def record(self) -> dict:
        return {
            **asdict(self),
            "profile": self.profile,
            "dt_ms": 1,
            "bin_us": 1000,
            "filter_time_us": 10000,
            "sensor_whp": [128, 128, 2],
            "population_hwp": [32, 32, 2],
            "layout": "HWC",
            "border": 0,
            "spatial_rule": "floor(x/4),floor(y/4)",
            "polarity_rule": "numeric 0->channel0,1->channel1;positive spikes",
            "timestamp_rule": "float64(stored_ms)*1000 then int64 truncation",
            "origin_rule": "first retained event; explicit empty fallback",
            "window_rule": "one prefix; left inclusive; right exclusive",
            "occupancy_rule": "one spike per occupied neuron/bin at bin-left ms",
            "tail_rule": "fixed horizon; retain partial observed bins; zero tail",
        }


@dataclass(frozen=True)
class Sample:
    path: str
    label: int
    subject: int
    lighting: str
    official_split: str
    split: str

    def legacy_row(self) -> dict:
        # Exactly the metadata schema used by the TASK 04 runner.
        return {
            "path": self.path,
            "label": self.label,
            "subject": self.subject,
            "lighting": self.lighting,
        }

    @property
    def archive_name(self) -> str:
        return self.path.split("/", 1)[0]

    @property
    def stable_id(self) -> str:
        return f"{ARCHIVES[self.archive_name]}:{self.path}"


def sample_from_path(relative_path: str) -> Sample:
    match = PATH_PATTERN.fullmatch(relative_path)
    require(match is not None, f"Unexpected sample path: {relative_path}")
    folder, subject_text, lighting, label_text = match.groups()
    subject, label = int(subject_text), int(label_text)
    require(label_text == str(label), f"Noncanonical label: {relative_path}")
    require(0 <= label < len(LABELS), f"Invalid class ID: {relative_path}")

    if folder == "ibmGestureTrain":
        require(1 <= subject <= 23, f"Invalid training subject: {relative_path}")
        official, split = "train", "train" if subject <= 20 else "validation"
    else:
        require(24 <= subject <= 29, f"Invalid test subject: {relative_path}")
        official, split = "test", "test"

    return Sample(relative_path, label, subject, lighting, official, split)


def assert_disjoint(partitions: dict[str, Sequence[Sample]]) -> None:
    """Check paths, stable parent IDs, and subjects across partitions."""
    keys = list(partitions)
    for key, rows in partitions.items():
        require(all(row.split == key for row in rows), f"Wrong split in {key}")
        require(
            len({row.path for row in rows}) == len(rows),
            f"Repeated path within {key}",
        )
        require(
            len({row.stable_id for row in rows}) == len(rows),
            f"Repeated parent within {key}",
        )
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            for field in ("path", "stable_id", "subject"):
                a = {getattr(row, field) for row in partitions[left]}
                b = {getattr(row, field) for row in partitions[right]}
                require(not (a & b), f"{field} overlap: {left}/{right}")


def inventory(root: Path) -> dict[str, tuple[Sample, ...]]:
    """Inspect filenames only; validate fixed TASK 04 membership."""
    root = Path(root).resolve(strict=True)
    grouped: dict[str, list[Sample]] = {
        "train": [], "validation": [], "test": []
    }
    resolved_paths: set[Path] = set()

    for folder in ARCHIVES:
        directory = root / folder
        require(directory.is_dir(), f"Missing directory: {directory}")
        for path in sorted(directory.rglob("*.npy")):
            resolved = path.resolve(strict=True)
            require(resolved.is_relative_to(root), f"Path escapes root: {path}")
            require(resolved.is_file(), f"Not a regular sample file: {path}")
            require(
                resolved not in resolved_paths, f"Aliased sample path: {path}"
            )
            resolved_paths.add(resolved)
            row = sample_from_path(path.relative_to(root).as_posix())
            grouped[row.split].append(row)

    result = {
        key: tuple(sorted(rows, key=lambda row: row.path))
        for key, rows in grouped.items()
    }
    assert_disjoint(result)

    expected = {
        "train": (934, [84] + [85] * 10),
        "validation": (143, [13] * 11),
        "test": (264, [24] * 11),
    }
    for split, rows in result.items():
        size, class_counts = expected[split]
        require(len(rows) == size, f"{split}: expected {size}, got {len(rows)}")
        actual = np.bincount([row.label for row in rows], minlength=11)
        require(
            actual.tolist() == class_counts,
            f"{split}: unexpected class counts {actual.tolist()}",
        )
        if split in MANIFEST_HASHES:
            observed = object_hash([row.legacy_row() for row in rows])
            require(
                observed == MANIFEST_HASHES[split],
                f"{split}: metadata membership/hash changed: {observed}",
            )
    return result


def verify_archives(root: Path) -> dict[str, str]:
    """Stream archive bytes; never extract or deserialize them."""
    hashes = {}
    for name, expected in ARCHIVES.items():
        path = Path(root) / f"{name}.tar.gz"
        observed = file_hash(path)
        require(observed == expected, f"Archive hash mismatch: {path}")
        hashes[name] = observed
    return hashes


def pilot(rows: Sequence[Sample], per_class: int) -> tuple[Sample, ...]:
    require(per_class > 0, "per_class must be positive")
    chosen = []
    for label in range(11):
        candidates = sorted(
            (row for row in rows if row.label == label),
            key=lambda row: row.path,
        )
        require(len(candidates) >= per_class, f"Insufficient class {label}")
        chosen.extend(candidates[:per_class])
    return tuple(sorted(chosen, key=lambda row: row.path))


def canonical_events(raw: np.ndarray, chunk_size: int = 262144) -> np.ndarray:
    """Validate Nx4 x,y,p,t_ms and convert without modifying the source.

    Validation uses chunks. The returned structured array contains one complete
    recording; the dataset is never loaded into RAM as a whole.
    """
    require(raw.ndim == 2 and raw.shape[1] == 4, "Expected Nx4 x,y,p,t_ms")
    require(raw.dtype.kind in "iuf", f"Unsupported raw dtype: {raw.dtype}")
    require(chunk_size > 0, "chunk_size must be positive")
    events = np.empty(len(raw), dtype=EVENT_DTYPE)
    previous_time = None

    for start in range(0, len(raw), chunk_size):
        stop = min(start + chunk_size, len(raw))
        values = np.asarray(raw[start:stop], dtype=np.float64)
        require(bool(np.isfinite(values).all()), "Nonfinite raw value")
        xy, polarity = values[:, :2], values[:, 2]
        require(bool((xy == np.floor(xy)).all()), "Fractional coordinate")
        require(bool(((xy >= 0) & (xy < 128)).all()), "Spatial bound violation")
        require(
            bool(((polarity == 0) | (polarity == 1)).all()),
            "Polarity must be exactly 0 or 1",
        )

        with np.errstate(over="ignore", invalid="ignore"):
            scaled = values[:, 3] * 1000.0
        require(bool(np.isfinite(scaled).all()), "Timestamp scaling overflow")
        require(bool((scaled >= 0).all()), "Negative timestamp")
        # Float upper bound is exclusive: float64(int64.max) rounds to 2**63.
        require(
            bool((scaled < float(2**63 - 16384)).all()),
            "Timestamp lacks safe int64/denoiser headroom",
        )
        times = scaled.astype(np.int64)  # Deliberate truncation, never rounding.
        require(bool((times[1:] >= times[:-1]).all()), "Nonmonotonic timestamps")
        if previous_time is not None:
            require(int(times[0]) >= previous_time, "Timestamp chunk disorder")
        previous_time = int(times[-1])

        events["x"][start:stop] = xy[:, 0]
        events["y"][start:stop] = xy[:, 1]
        events["p"][start:stop] = polarity
        events["t"][start:stop] = times

    return events


def denoise(events: np.ndarray, filter_time: int = 10000) -> np.ndarray:
    """Reproduce inspected Tonic 1.5.0 denoise_numpy; bypass raw empties."""
    require(events.dtype == EVENT_DTYPE, "Expected canonical event dtype")
    require(filter_time == 10000, "This profile pins filter_time=10000")
    if len(events) == 0:
        return events.copy()

    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    memory = np.zeros((width, height)) + filter_time
    output = np.zeros_like(events)
    used = 0
    for event in events:
        x, y, t = int(event["x"]), int(event["y"]), event["t"]
        memory[x, y] = t + filter_time
        if (
            (x > 0 and memory[x - 1, y] > t)
            or (x < width - 1 and memory[x + 1, y] > t)
            or (y > 0 and memory[x, y - 1] > t)
            or (y < height - 1 and memory[x, y + 1] > t)
        ):
            output[used] = event
            used += 1
    return output[:used]


def pack_occupancy(
    times_us: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    polarity: np.ndarray,
    *,
    width: int,
    height: int,
    frames: int,
) -> tuple[PreprocessedSpikes, dict]:
    """Pack already-rebased/downsampled fixtures or events; never denoise here.

    Logical occupancy shape is (frames,height,width,2), bool. No dense occupancy
    or count tensor is allocated. Unique int64 keys encode neuron*frames+bin.
    """
    arrays = [np.asarray(a) for a in (times_us, x, y, polarity)]
    require(
        all(a.ndim == 1 and a.dtype.kind in "iub" for a in arrays),
        "Packing inputs must be one-dimensional integer/bool arrays",
    )
    require(len({len(a) for a in arrays}) == 1, "Packing length mismatch")
    require(
        all(type(v) is int and v > 0 for v in (width, height, frames)),
        "Invalid packing dimensions",
    )
    require(frames <= 1024, "Unsupported fixture/production horizon")
    neurons = width * height * 2
    require(neurons * frames < 2**63, "Occupancy key overflow")
    t, x, y, p = arrays
    require(bool((t >= 0).all()), "Negative relative timestamp")
    require(bool((t[1:] >= t[:-1]).all()), "Nonmonotonic relative timestamps")
    require(bool(((x >= 0) & (x < width)).all()), "x out of bounds")
    require(bool(((y >= 0) & (y < height)).all()), "y out of bounds")
    require(bool(((p == 0) | (p == 1)).all()), "Invalid polarity")

    included = t < frames * 1000
    bins = t[included].astype(np.int64) // 1000
    ids = (
        (y[included].astype(np.int64) * width + x[included].astype(np.int64))
        * 2 + p[included].astype(np.int64)
    )
    keys, counts = np.unique(ids * frames + bins, return_counts=True)
    occupied_ids = keys // frames
    scheduled = (keys % frames).astype(np.float32)
    per_neuron = np.bincount(occupied_ids, minlength=neurons)
    cumulative = np.cumsum(per_neuron, dtype=np.int64)
    require(len(keys) <= np.iinfo(np.uint32).max, "uint32 payload overflow")
    payload = PreprocessedSpikes(cumulative.astype(np.uint32), scheduled)
    validate_payload(payload, neurons=neurons, frames=frames)

    included_count = int(included.sum(dtype=np.int64))
    occupied_count = len(keys)
    require(int(counts.sum(dtype=np.int64)) == included_count, "Count mass error")
    audit = {
        "included_events": included_count,
        "prefix_excluded_events": len(t) - included_count,
        "occupancy_spikes": occupied_count,
        "collapsed_excess": included_count - occupied_count,
        "max_cell_count": int(counts.max()) if len(counts) else 0,
        "distinct_neurons": int(np.count_nonzero(per_neuron)),
        "max_bins_per_neuron": int(per_neuron.max()),
        "included_events_by_polarity": np.bincount(
            p[included].astype(np.int64), minlength=2
        ).tolist(),
        "occupancy_spikes_by_polarity": np.bincount(
            occupied_ids % 2, minlength=2
        ).tolist(),
        "empty_prefix": occupied_count == 0,
    }
    return payload, audit


def validate_payload(
    payload: PreprocessedSpikes, *, neurons: int, frames: int
) -> None:
    end, times = payload
    require(end.shape == (neurons,), "Wrong cumulative-index shape")
    require(end.dtype == np.uint32, "Indices must be uint32")
    require(times.ndim == 1 and times.dtype == np.float32, "Wrong time array")
    wide = end.astype(np.int64)
    require(bool((wide[1:] >= wide[:-1]).all()), "Decreasing endpoints")
    require(int(wide[-1]) == len(times), "Final endpoint differs from K")
    require(bool(np.isfinite(times).all()), "Nonfinite scheduled time")
    require(bool(((times >= 0) & (times < frames)).all()), "Time outside trial")
    require(bool((times == np.floor(times)).all()), "Off-grid scheduled time")
    start = 0
    for stop in wide:
        interval = times[start:int(stop)]
        require(
            bool((interval[1:] > interval[:-1]).all()),
            "Duplicate/nonincreasing times within a neuron",
        )
        start = int(stop)


def preprocess(
    events: np.ndarray, config: Config
) -> tuple[PreprocessedSpikes, dict]:
    require(events.dtype == EVENT_DTYPE, "Expected canonical event dtype")
    filtered = denoise(events)
    if len(filtered):
        origin, kind = int(filtered["t"][0]), "first_retained"
    elif len(events):
        origin, kind = int(events["t"][0]), "first_loaded_empty_filtered"
    else:
        origin, kind = None, "empty_no_observed_origin"

    relative = (
        filtered["t"] - origin
        if origin is not None else np.empty(0, dtype=np.int64)
    )
    payload, audit = pack_occupancy(
        relative,
        filtered["x"] // 4,
        filtered["y"] // 4,
        filtered["p"],
        width=32,
        height=32,
        frames=config.frames,
    )
    metadata = {
        "config": config.record(),
        "config_sha256": object_hash(config.record()),
        "origin_us": origin,
        "origin_kind": kind,
        "bin_edges_us": {
            "start": origin,
            "step": 1000,
            "number_of_edges": config.frames + 1,
            "exclusive_end": (
                origin + config.frames * 1000 if origin is not None else None
            ),
        },
        "raw_events": len(events),
        "filtered_events": len(filtered),
        "denoise_removed_events": len(events) - len(filtered),
        "raw_empty": len(events) == 0,
        "filtered_empty": len(filtered) == 0,
        "raw_first_us": int(events["t"][0]) if len(events) else None,
        "raw_last_us": int(events["t"][-1]) if len(events) else None,
        "raw_span_us": (
            int(events["t"][-1]) - int(events["t"][0]) if len(events) else None
        ),
        "filtered_first_us": int(filtered["t"][0]) if len(filtered) else None,
        "filtered_last_us": int(filtered["t"][-1]) if len(filtered) else None,
        "filtered_span_us": (
            int(filtered["t"][-1]) - int(filtered["t"][0])
            if len(filtered) else None
        ),
        "scheduling_rule": "floor(relative_us/1000) ms; advances by 0..999 us",
        **audit,
    }
    require(
        metadata["raw_events"]
        == metadata["denoise_removed_events"] + metadata["filtered_events"],
        "Denoise mass mismatch",
    )
    require(
        metadata["filtered_events"]
        == metadata["included_events"] + metadata["prefix_excluded_events"],
        "Prefix mass mismatch",
    )
    require(
        metadata["included_events"]
        == metadata["occupancy_spikes"] + metadata["collapsed_excess"],
        "Occupancy mass mismatch",
    )
    return payload, metadata


def load_sample(
    root: Path, row: Sample, config: Config
) -> tuple[PreprocessedSpikes, np.int64, dict]:
    """Load one development sample. Test payload loading is deliberately gated."""
    require(row.split != "test", "Reserved test event loading is disabled")
    require(sample_from_path(row.path) == row, "Sample metadata/path mismatch")
    root = Path(root).resolve(strict=True)
    path = (root / row.path).resolve(strict=True)
    require(path.is_relative_to(root), "Sample path escapes dataset root")
    sample_hash = file_hash(path)
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    raw_dtype, raw_shape = str(raw.dtype), list(raw.shape)
    events = canonical_events(raw)
    del raw
    payload, metadata = preprocess(events, config)
    metadata.update({
        **asdict(row),
        "label_name": LABELS[row.label],
        "stable_id": row.stable_id,
        "parent_id": row.stable_id,
        "archive_sha256": ARCHIVES[row.archive_name],
        "sample_sha256": sample_hash,
        "source_path": str(path),
        "raw_dtype": raw_dtype,
        "raw_shape": raw_shape,
        "raw_columns": ["x", "y", "p", "t_ms"],
        "canonical_dtype": EVENT_DTYPE.descr,
        "loader_source_sha256": file_hash(Path(__file__)),
        "denoiser_identity": "local denoise; pinned Tonic1.5.0 semantics",
        "recording_id": None,
        "annotation_row": None,
        "annotated_start_us": None,
        "annotated_end_us": None,
        "physical_polarity_meaning": None,
        "native_orientation": None,
    })
    return payload, np.int64(row.label), metadata


def audit_development_content(
    root: Path, partitions: dict[str, Sequence[Sample]]
) -> dict[str, dict[str, str]]:
    """Hash train/validation files before real pilot preprocessing.

    Duplicate bytes across development partitions are an error. Test sample
    bytes remain unread; test content-duplication audit belongs to evaluation.
    """
    seen: dict[str, tuple[str, str]] = {}
    result: dict[str, dict[str, str]] = {}
    for split in ("train", "validation"):
        result[split] = {}
        for row in partitions[split]:
            require(row.split == split, "Wrong content-audit partition")
            digest = file_hash(Path(root) / row.path)
            previous = seen.get(digest)
            if previous is not None and previous[0] != split:
                raise ValueError(
                    f"Duplicate development content: {previous[1]} / {row.path}"
                )
            seen[digest] = (split, row.path)
            result[split][row.path] = digest
    return result


def iter_batches(
    root: Path,
    rows: Sequence[Sample],
    config: Config,
    *,
    batch_size: int = 1,
    order: Sequence[int] | None = None,
) -> Iterator[tuple[list[PreprocessedSpikes], np.ndarray, list[dict]]]:
    """Bounded host batches; order must be an explicit complete permutation.

    The caller may supply default_rng(42).permutation(len(rows)) and record it.
    This function makes no claim about model partial-batch/update semantics.
    """
    require(type(batch_size) is int and 1 <= batch_size <= 32,
            "batch_size must be 1..32")
    indices = list(range(len(rows))) if order is None else list(order)
    require(
        all(isinstance(i, (int, np.integer)) for i in indices),
        "Noninteger order index",
    )
    require(sorted(indices) == list(range(len(rows))), "Incomplete/duplicate order")
    for start in range(0, len(indices), batch_size):
        loaded = [
            load_sample(root, rows[int(i)], config)
            for i in indices[start:start + batch_size]
        ]
        payloads, labels, metadata = zip(*loaded)
        require(
            sum(len(p.spike_times) for p in payloads)
            <= np.iinfo(np.uint32).max,
            "Batch uint32 offset overflow",
        )
        yield list(payloads), np.asarray(labels, dtype=np.int64), list(metadata)


def to_ml_genn(payload: PreprocessedSpikes):
    """Optional later adapter; not called by CPU tests or manifest inspection."""
    from ml_genn.utils.data import PreprocessedSpikes as MLGeNNSpikes
    return MLGeNNSpikes(payload.end_spikes, payload.spike_times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="Existing .../DVSGesture directory")
    parser.add_argument("--frames", type=int, choices=(128, 1024), default=128)
    args = parser.parse_args()
    config = Config(args.frames)
    archives = verify_archives(args.root)
    partitions = inventory(args.root)

    print("CONFIG", json.dumps(config.record(), sort_keys=True))
    print("CONFIG_SHA256", object_hash(config.record()))
    print("LOADER_SHA256", file_hash(Path(__file__)))
    print("NUMPY_VERSION", np.__version__)
    print("NUMPY_PATH", np.__file__)
    print("ARCHIVES", json.dumps(archives, sort_keys=True))

    for split, rows in partitions.items():
        print("PARTITION", split, len(rows))
        print("METADATA_SHA256", split,
              object_hash([row.legacy_row() for row in rows]))
        print("CLASS_COUNTS", split,
              [sum(row.label == label for row in rows) for label in range(11)])
        print("LIGHTING_COUNTS", split,
              json.dumps(dict(sorted(Counter(r.lighting for r in rows).items()))))
        for subject in sorted({row.subject for row in rows}):
            print("SUBJECT_CLASS_COUNTS", split, subject, [
                sum(row.subject == subject and row.label == label for row in rows)
                for label in range(11)
            ])

    for split, count in (("train", 2), ("validation", 1)):
        chosen = pilot(partitions[split], count)
        print("PILOT_METADATA_SHA256", split,
              object_hash([row.legacy_row() for row in chosen]))
        for row in chosen:
            print("PILOT", split, json.dumps(row.legacy_row(), sort_keys=True))
    print("PASS: archive hashes, fixed manifests, counts, and split disjointness")
    print("No sample arrays opened; reserved test event loading remains disabled.")


if __name__ == "__main__":
    main()