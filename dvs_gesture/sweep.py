"""Train the runs of a sweep config one after another (all, or the ones named).

    python3 sweep.py configs/sweep_stage_h.json [RUN ...]
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("runs", nargs="*")
    args = parser.parse_args()

    config = args.config.resolve()
    available = list(json.loads(config.read_text()).get("runs", {}))
    if not available:
        parser.error(f"{config} has no runs")
    unknown = [name for name in args.runs if name not in available]
    if unknown:
        parser.error(f"Unknown runs {unknown}; available: {', '.join(available)}")
    names = args.runs or available

    run_sh = Path(__file__).resolve().parent / "run.sh"
    failed = []
    for i, name in enumerate(names, 1):
        print(f"\n=== Sweep run {i}/{len(names)}: {name}", flush=True)
        result = subprocess.run([str(run_sh), "--config", str(config), "--run", name],
                                env={**os.environ, "RUN_TAG": name})
        if result.returncode != 0:
            print(f"=== {name} failed (exit code {result.returncode})", flush=True)
            failed.append(name)

    print(f"\nSweep finished: {len(names) - len(failed)}/{len(names)} runs succeeded")
    if failed:
        print("Failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
