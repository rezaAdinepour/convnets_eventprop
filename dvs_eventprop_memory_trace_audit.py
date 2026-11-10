"""TASK 10: read-only accounting of Nsight CUDA device/array allocations.

No GPU access, GeNN imports, simulation or database modification.
Does not certify total physical GPU usage or device-wide headroom.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sqlite3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def emit(tag, value):
    print(tag, json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    database = args.database.resolve(strict=True)

    uri = database.as_uri() + "?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")

        kinds = [
            dict(row) for row in connection.execute(
                "SELECT id, name, label FROM ENUM_CUDA_MEM_KIND ORDER BY id"
            )
        ]
        operations = {
            row["id"]: row["label"]
            for row in connection.execute(
                "SELECT id, label FROM ENUM_CUDA_DEV_MEM_EVENT_OPER"
            )
        }
        processes = [
            dict(row) for row in connection.execute(
                "SELECT globalPid, pid, name FROM PROCESSES"
            )
        ]
        rows = [
            dict(row) for row in connection.execute("""
                SELECT rowid AS eventRow, *
                FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                ORDER BY start, rowid
            """)
        ]
        diagnostics = [
            dict(row) for row in connection.execute("""
                SELECT d.timestamp, e.label AS severity, d.text
                FROM DIAGNOSTIC_EVENT AS d
                LEFT JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL AS e
                    ON e.id = d.severity
                ORDER BY d.timestamp
            """)
        ]

    require(rows, "No CUDA allocation events were exported")
    emit("MEMORY_TRACE_METADATA", {
        "database": str(database),
        "database_sha256": digest(database),
        "audit_source_sha256": digest(Path(__file__).resolve()),
        "read_only": True,
        "event_count_all_kinds": len(rows),
        "memory_kinds": kinds,
        "operation_labels": operations,
        "processes": processes,
    })
    emit("NSIGHT_DIAGNOSTICS", diagnostics)

    kind_names = {row["id"]: row["name"] for row in kinds}
    summary = defaultdict(lambda: {"events": 0, "reported_bytes": 0})
    for row in rows:
        key = (
            kind_names[row["memKind"]],
            operations[row["memoryOperationType"]],
        )
        summary[key]["events"] += 1
        summary[key]["reported_bytes"] += row["bytes"]
    emit("MEMORY_EVENT_SUMMARY", [
        {"kind": key[0], "operation": key[1], **value}
        for key, value in sorted(summary.items())
    ])

    selected_kinds = {
        row["id"] for row in kinds
        if row["name"] in {
            "CUDA_MEMOPR_MEMORY_KIND_DEVICE",
            "CUDA_MEMOPR_MEMORY_KIND_ARRAY",
        }
    }
    require(len(selected_kinds) == 2, "Unexpected memory-kind enumeration")
    selected = [r for r in rows if r["memKind"] in selected_kinds]
    require(selected, "No device/array allocation events")

    pool_rows = sum(
        bool(r["localMemoryPoolAddress"] or r["importedMemoryPoolAddress"])
        for r in selected
    )
    emit("MEMORY_POOL_CHECK", {"selected_events_with_pool_metadata": pool_rows})
    require(pool_rows == 0,
            "Memory pools present; reserved/utilized accounting needs review")

    # Key allocations by process, device, context, address and memory kind.
    live = {}
    current = Counter()
    peak_net = Counter()
    peak_upper = Counter()
    peak_time = {}
    allocation_rows = []
    zero_size_free_fields = 0

    for timestamp, events in groupby(selected, key=lambda r: r["start"]):
        events = list(events)
        before = current.copy()
        allocated_at_timestamp = Counter()

        for row in events:
            size = int(row["bytes"])
            require(size >= 0, "Negative allocation-event byte count")
            device = int(row["deviceId"])
            key = (
                row["globalPid"], device, row["contextId"],
                row["address"], row["memKind"],
            )
            operation = operations[row["memoryOperationType"]]

            if operation == "Allocation":
                if size == 0:
                    continue
                require(key not in live,
                        f"Duplicate live allocation at event {row['eventRow']}")
                live[key] = size
                current[device] += size
                allocated_at_timestamp[device] += size
                allocation_rows.append(row)

            elif operation == "Deallocation":
                if size == 0 and key not in live and row["address"] == 0:
                    continue
                require(key in live,
                        f"Unmatched deallocation at event {row['eventRow']}")
                previous_size = live.pop(key)
                require(size in (0, previous_size),
                        f"Allocation/free size mismatch at event {row['eventRow']}")
                zero_size_free_fields += int(size == 0)
                current[device] -= previous_size
                require(current[device] >= 0, "Negative live allocation total")
            else:
                raise ValueError(f"Unsupported memory operation: {operation}")

        for device in set(before) | set(current) | set(allocated_at_timestamp):
            # If alloc/free events share a timestamp, their internal ordering
            # may be unresolved. Report both the net-event peak and an upper
            # bound that puts all allocations before frees at that timestamp.
            upper = before[device] + allocated_at_timestamp[device]
            peak_net[device] = max(peak_net[device], current[device])
            if upper > peak_upper[device]:
                peak_upper[device] = upper
                peak_time[device] = timestamp

    require(allocation_rows, "No positive-sized device/array allocations")
    emit("TRACKED_ALLOCATION_PEAKS", [
        {
            "device_id": device,
            "timestamp_net_peak_bytes": peak_net[device],
            "within_timestamp_conservative_peak_bytes": peak_upper[device],
            "within_timestamp_conservative_peak_MiB":
                peak_upper[device] / (1024 ** 2),
            "conservative_peak_timestamp_ns": peak_time.get(device),
            "outstanding_at_trace_end_bytes": current[device],
        }
        for device in sorted(peak_upper)
    ])
    emit("LARGEST_DEVICE_ARRAY_ALLOCATIONS", [
        {
            "timestamp_ns": row["start"],
            "globalPid": row["globalPid"],
            "device_id": row["deviceId"],
            "bytes": row["bytes"],
            "MiB": row["bytes"] / (1024 ** 2),
            "name": row["name"],
        }
        for row in sorted(
            allocation_rows, key=lambda row: row["bytes"], reverse=True
        )[:20]
    ])
    emit("PASS_RECORDED_ALLOCATION_ACCOUNTING", {
        "matched_deallocations": True,
        "zero_size_deallocation_fields_resolved_from_live_allocations":
            zero_size_free_fields,
        "remaining_live_allocation_records": len(live),
        "scope": "recorded CUDA device/array allocations across both runs",
        "managed_and_static_memory_included": False,
        "all_physical_gpu_memory_included": False,
        "headroom_certified": False,
        "collection_diagnostics_require_review": True,
    })


if __name__ == "__main__":
    main()