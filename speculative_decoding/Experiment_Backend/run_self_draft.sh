#!/usr/bin/env bash
set -euo pipefail
set -o errtrace

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
INNER_REPO="$REPO_ROOT"
cd "$REPO_ROOT"

TARGET="${TARGET:-inference/model/SDAR-8B-Chat}"
DRAFTS="${DRAFTS:-SDAR-4B-Chat}"
DATASETS="${DATASETS:-gsm8k humaneval mbpp}"
BRANCH="${BRANCH:-greedy_match}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
NUM_BLOCKS="${NUM_BLOCKS:-32}"
BLOCK_LENGTH="${BLOCK_LENGTH:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-4}"
K="${K:-4}"
SEED="${SEED:-42}"

if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-sdar}"
fi

echo "[driver] REPO_ROOT=$REPO_ROOT"
echo "[driver] TARGET=$TARGET"
echo "[driver] DRAFTS=$DRAFTS"
echo "[driver] DATASETS=$DATASETS"
echo "[driver] BRANCH=$BRANCH NUM_SAMPLES=$NUM_SAMPLES NUM_BLOCKS=$NUM_BLOCKS K=$K"
echo

SCRIPT_PY="$INNER_REPO/speculative_decoding/Experiment_Backend/self_draft_compare.py"

for draft_name in $DRAFTS; do
  draft_path="inference/model/${draft_name}"
  for ds in $DATASETS; do
    echo "====================================================================="
    echo " cross=${draft_name}   dataset=${ds}"
    echo "====================================================================="

    python -u "$SCRIPT_PY" \
      --target_model "$TARGET" \
      --cross_draft_model "$draft_path" \
      --target_device cuda:0 \
      --cross_draft_device cuda:0 \
      --dataset "$ds" \
      --num_samples "$NUM_SAMPLES" \
      --num_blocks "$NUM_BLOCKS" \
      --block_length "$BLOCK_LENGTH" \
      --denoising_steps "$DENOISING_STEPS" \
      --K "$K" \
      --seed "$SEED" \
      --branch "$BRANCH"
    echo
  done
done

echo "[driver] all runs complete."
