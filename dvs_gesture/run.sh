#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS="$PWD/../../Experiments"
DATA=${DATA_ROOT:-"$EXPERIMENTS/reference_data_verified_vdCGRkGx"}
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.5}
export PATH="$PWD/.venv/bin:$CUDA_PATH/bin:$PATH"

REFERENCE="$EXPERIMENTS/task12_whole_100epochs_Qcu3KQxq/train"
INITIAL=()
if [[ -d "$REFERENCE/checkpoints" ]]; then
    INITIAL=(--initial-from "$REFERENCE")
fi

mkdir -p "$EXPERIMENTS"
RUN=$(mktemp -d "$EXPERIMENTS/dvs_gesture_XXXXXXXX")
echo "Run directory: $RUN"
sha256sum dvs_gesture/{data,evdownsampling,model,train}.py dvs_gesture/run.sh > "$RUN/source.sha256"

TRAIN=(python3 -E -s -B -u dvs_gesture/train.py "$DATA")
"${TRAIN[@]}" --run-dir "$RUN/train" --whole-gesture --full-epochs "${NUM_EPOCHS:-100}" \
    --preprocessing evdownsampling "${INITIAL[@]}" 2>&1 | tee "$RUN/train.log"
"${TRAIN[@]}" --run-dir "$RUN/reload" --validate-full "$RUN/train" 2>&1 | tee "$RUN/reload.log"
