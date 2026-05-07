#!/usr/bin/env bash
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARENT_REPO="$(cd "$REPO/.." && pwd)"
EXP="$REPO/speculative_decoding/Experiment_Backend"
RESULTS="$EXP/results"

mkdir -p "$RESULTS"

cd "$PARENT_REPO"

echo "===== [$(date '+%F %T')] Exp 1 start ====="
python -u "$EXP/run_batch_sweep.py" 2>&1 | tee "$RESULTS/exp1.log"
echo "===== [$(date '+%F %T')] Exp 1 done  rc=$? ====="

echo "===== [$(date '+%F %T')] Exp 2 start ====="
python -u "$EXP/run_k_batch_dataset_sweep.py" 2>&1 | tee "$RESULTS/exp2.log"
echo "===== [$(date '+%F %T')] Exp 2 done  rc=$? ====="

echo "===== [$(date '+%F %T')] aggregate ====="
python "$EXP/aggregate_sweep.py" 2>&1 | tee "$RESULTS/aggregate.log"
echo "===== [$(date '+%F %T')] all done ====="
