#!/usr/bin/env bash
# Run from the checkout or the extracted review ZIP. For training curves: plot.sh --split train
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(dirname "$SCRIPT_DIR")
PROJECTS=$(dirname "$ROOT")
EXPERIMENTS_DIR=${EXPERIMENTS_DIR:-"$(dirname "$PROJECTS")/Experiments"}

if [[ -f "$ROOT/baseline.log" && -f "$ROOT/evdownsampling.log" ]]; then
    BASELINE_LOG="$ROOT/baseline.log"
    EVDOWN_LOG="$ROOT/evdownsampling.log"
else
    BASELINE_LOG="$EXPERIMENTS_DIR/task12_whole_100epochs_Qcu3KQxq/train.log"
    EVDOWN_LOG="$EXPERIMENTS_DIR/task12_evdown_100epochs_31dX3nFu/train.log"
fi

LOCAL_PYTHON="$PROJECTS/convnets_eventprop/.venv/bin/python"
if [[ -x "$LOCAL_PYTHON" ]]; then
    PYTHON=${PLOT_PYTHON:-"$LOCAL_PYTHON"}
else
    PYTHON=${PLOT_PYTHON:-python3}
fi

export MPLBACKEND=Agg
exec "$PYTHON" "$SCRIPT_DIR/plot_training.py" "$BASELINE_LOG" "$EVDOWN_LOG" \
    --output-dir "$SCRIPT_DIR/figures" "$@"
