#!/usr/bin/env bash
# Whole-gesture EvDownsampling training and selected-checkpoint validation.
# Environment of the recorded run: PyGeNN 5.5.0 and mlGeNN 3 at a00d91b.
# DATA_ROOT contains the extracted DVSGesture directory and its official archives.
# Example on another machine: DATA_ROOT=/data PYTHON=/path/to/venv/bin/python ./dvs_gesture/run.sh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CORE_DIR=$(dirname "$SCRIPT_DIR")
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-"$(dirname "$(dirname "$CORE_DIR")")/Experiments"}
DATA_ROOT=${DATA_ROOT:-"$EXPERIMENTS_DIR/reference_data_verified_vdCGRkGx"}
PYTHON=${PYTHON:-"$CORE_DIR/.venv/bin/python"}
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.5}
NUM_EPOCHS=${NUM_EPOCHS:-100}
PREPROCESSING=${PREPROCESSING:-evdownsampling}
# On the original machine use the same five initial arrays as the paired run.
# Set INITIAL_FROM="" for fresh initialisation or supply another training run.
REFERENCE_RUN="$EXPERIMENTS_DIR/task12_whole_100epochs_Qcu3KQxq/train"
if [[ ! -d "$REFERENCE_RUN/checkpoints" ]]; then
    REFERENCE_RUN=""
fi
INITIAL_FROM=${INITIAL_FROM-"$REFERENCE_RUN"}

export CUDA_PATH
export PATH="$(dirname "$PYTHON"):$CUDA_PATH/bin:$PATH"
mkdir -p "$EXPERIMENTS_DIR"
JOB=$(mktemp -d "$EXPERIMENTS_DIR/dvs_gesture_XXXXXXXX")
printf 'Run directory: %s\nInitial weights: %s\n' "$JOB" "${INITIAL_FROM:-fresh initialisation}"
INITIAL_ARGS=()
if [[ -n "$INITIAL_FROM" ]]; then
    INITIAL_ARGS=(--initial-from "$INITIAL_FROM")
fi
(cd "$CORE_DIR" && sha256sum dvs_gesture/{data,evdownsampling,model,train}.py dvs_gesture/run.sh) > "$JOB/source.sha256"
"$PYTHON" -E -s -B -u "$SCRIPT_DIR/train.py" "$DATA_ROOT" --run-dir "$JOB/train" \
    --whole-gesture --full-epochs "$NUM_EPOCHS" --preprocessing "$PREPROCESSING" \
    "${INITIAL_ARGS[@]}" 2>&1 | tee "$JOB/train.log"
"$PYTHON" -E -s -B -u "$SCRIPT_DIR/train.py" "$DATA_ROOT" --run-dir "$JOB/reload" \
    --validate-full "$JOB/train" 2>&1 | tee "$JOB/reload.log"
