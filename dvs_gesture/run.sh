#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXPERIMENTS="$PWD/../../Experiments"
DATA=${DATA_ROOT:-"$EXPERIMENTS/reference_data_verified_vdCGRkGx"}
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda-13.5}
export PATH="$PWD/.venv/bin:$CUDA_PATH/bin:$PATH"

case "${1:-train}" in
    smoke) MODE=() ;;
    whole-smoke) MODE=(--whole-gesture) ;;
    activity) MODE=(--whole-gesture --activity-check) ;;
    train) MODE=(--whole-gesture --full-epochs "${NUM_EPOCHS:-100}") ;;
    *) echo "Usage: $0 [smoke|whole-smoke|activity|train]" >&2; exit 2 ;;
esac
if [[ -n ${GRAD_LIMIT:-} ]]; then
    MODE+=(--grad-limit "$GRAD_LIMIT")
fi
if [[ -n ${BATCH_SIZE:-} ]]; then
    MODE+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n ${RING_SLOTS:-} ]]; then
    MODE+=(--ring-slots "$RING_SLOTS")
fi
REFERENCE="$EXPERIMENTS/task12_whole_100epochs_Qcu3KQxq/train"
INITIAL=()
if [[ -d "$REFERENCE/checkpoints" ]]; then
    INITIAL=(--initial-from "$REFERENCE" --hidden-weight-scale "${HIDDEN_WEIGHT_SCALE:-1}")
elif [[ -n ${HIDDEN_WEIGHT_SCALE:-} ]]; then
    echo "HIDDEN_WEIGHT_SCALE requires $REFERENCE/checkpoints" >&2
    exit 1
fi

mkdir -p "$EXPERIMENTS"
RUN=$(mktemp -d "$EXPERIMENTS/dvs_gesture_XXXXXXXX")
echo "Run directory: $RUN"
sha256sum dvs_gesture/{data,evdownsampling,model,train,training_log}.py dvs_gesture/run.sh > "$RUN/source.sha256"

TRAIN=(python3 -E -s -B -u dvs_gesture/train.py "$DATA")
"${TRAIN[@]}" --run-dir "$RUN/train" "${MODE[@]}" --preprocessing "${PREPROCESSING:-evdownsampling}" \
    "${INITIAL[@]}" 2>&1 | tee "$RUN/train.log"
if [[ ${1:-train} == train ]]; then
    "${TRAIN[@]}" --run-dir "$RUN/reload" --validate-full "$RUN/train" 2>&1 | tee "$RUN/reload.log"
fi
