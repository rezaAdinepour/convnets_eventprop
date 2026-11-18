#!/usr/bin/env bash
# Train in a new, uniquely named experiment directory and keep the log there.
# Extra arguments go to train.py, e.g.
#   ./run.sh --encoding evdownsampling --epochs 5
set -euo pipefail
cd "$(dirname "$0")"

RESEARCH=$(cd ../../.. && pwd)
DATA_ROOT=${DATA_ROOT:-"$RESEARCH/Experiments/reference_data_verified_vdCGRkGx"}
RUN_DIR="$RESEARCH/Experiments/dvs_gesture_b32_$(date +%Y%m%d_%H%M%S)"

source ../.venv/bin/activate
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.5}

mkdir -p "$RUN_DIR"
git rev-parse HEAD > "$RUN_DIR/git_commit.txt"
git status --short > "$RUN_DIR/git_status.txt"
echo "Run directory: $RUN_DIR"

python -u train.py "$DATA_ROOT" "$RUN_DIR/train" "$@" 2>&1 | tee "$RUN_DIR/train.log"
