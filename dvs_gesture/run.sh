#!/usr/bin/env bash
# usage: ./run.sh --config configs/base.json [train.py options]
# RUN_TAG=name adds _name to the run directory
set -euo pipefail
cd "$(dirname "$0")"

RESEARCH=$(cd ../../.. && pwd)
DATA_ROOT=${DATA_ROOT:-"$RESEARCH/Experiments/reference_data_verified_vdCGRkGx"}
CACHE_DIR=${CACHE_DIR:-"$RESEARCH/Experiments/dvs_gesture_encoding_cache"}
RUN_DIR="$RESEARCH/Experiments/dvs_gesture_b32_$(date +%Y%m%d_%H%M%S)${RUN_TAG:+_$RUN_TAG}"

source ../.venv/bin/activate
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.5}

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
echo "Run directory: $RUN_DIR"

python -u train.py "$DATA_ROOT" "$RUN_DIR/train" --cache-dir "$CACHE_DIR" "$@" 2>&1 | tee "$RUN_DIR/train.log"
