#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

LOGS=(baseline.log evdownsampling.log)
if [[ ! -f baseline.log || ! -f evdownsampling.log ]]; then
    EXPERIMENTS=${EXPERIMENTS_DIR:-../../Experiments}
    LOGS=("$EXPERIMENTS/task12_whole_100epochs_Qcu3KQxq/train.log"
          "$EXPERIMENTS/task12_evdown_100epochs_31dX3nFu/train.log")
fi

export PATH="$PWD/../convnets_eventprop/.venv/bin:$PATH"
MPLBACKEND=Agg exec "${PLOT_PYTHON:-python3}" dvs_gesture/plot_training.py "${LOGS[@]}" \
    --output-dir dvs_gesture/figures "$@"
