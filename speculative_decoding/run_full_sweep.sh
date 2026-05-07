#!/usr/bin/env bash
# =============================================================================
# run_full_sweep.sh  speculative decoding  sweep driver
# =============================================================================
#  sweep
# `speculative_decoding/results/{DATE}/sweeps/` 2026-04-13/224216/run.sh
# tee  + CUDA_VISIBLE_DEVICES  +  run  log
#  stage
#   (1) FORWARD       × batch -forward  []
#       sweep_forward.py
#        speedup(K)  t_draft / t_verify
#        micro-benchmark draft_devicecuda:0
#   (2) MODULE        (1)  model_forward_ms  q/k/v/oattn_core
#                      gate/up/downmlp_elemnorms_other HBM roofline []
#       sweep_module_breakdown.py
#       """"
#        draft_device
#   (3) REGIME       batch × prompt_len × params  regime []
#       sweep_regime.py
#        BW-bound / launch-bound / compute-bound
#        draft_device
#   (4) ACCEPT       {1.7B, 4B, 8B}  × block_length  α
#                      [ or ]
#       sweep_accept.py
#        speedup(K)  α = p
#        draft_device=cuda:0 target_device=cuda:1
#             8B + 4B
#       GSM8K
#   (5) E2E          native (target-only AR) vs speculative
#                      [****]
#       bench/run_benchmark.py --compare both
#         - speculative  native
#         - dual-GPU  single-GPU e2e
#         -  e2e latency
#        configs/bench_dual_gpu.yamldraft cuda:0, target cuda:1
#              configs/bench_fixed_blocks.yaml
#       bench_latency_{tag}.json  hf_native / hf_speculative
#             end_to_end_mscomponent_wall_msspeedup_native_over_spec_ms_ratio
#  /
#   speedup(K) ≈ ((1 − α^K)/(1 − α))  T_native / (K  t_draft + t_verify(K))
# (1)–(4) (5)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
#   #  sweepdraft+target  cuda:0
#   bash speculative_decoding/run_full_sweep.sh
#   # ****draft  0target  1
#   GPUS="0,1" bash speculative_decoding/run_full_sweep.sh
#   #  +  e2e dual-GPU
#   GPUS="0,1" STAGES="e2e" bash speculative_decoding/run_full_sweep.sh
#   #  +  accept  e2e α  speedup
#   GPUS="0,1" STAGES="accept e2e" bash speculative_decoding/run_full_sweep.sh
#   #  grid
#   QUICK=1 bash speculative_decoding/run_full_sweep.sh
#   #  sweep  vs  e2e
#   GPUS="0"   STAGES="e2e" SUBTAG=single bash speculative_decoding/run_full_sweep.sh
#   GPUS="0,1" STAGES="e2e" SUBTAG=dual   bash speculative_decoding/run_full_sweep.sh
#   #  HBM H100 ~3350 GB/s
#   HBM_BW=3350 bash speculative_decoding/run_full_sweep.sh
#   #  DATE
#   DATE=2026-04-20 bash speculative_decoding/run_full_sweep.sh
#   GPUS           "0" "0,1"                "0"
#   DATE           results/{DATE}                              $(date +%F)
#   STAGES          of "forward module regime accept e2e"
#   SUBTAG          results/{DATE}/sweeps{_SUBTAG}/
#   HBM_BW         HBM  GB/sA800=2039H100=3350                   2039
#   QUICK          =1   sweep  grid                 0
#   NUM_SAMPLES    sweep_accept / bench                            20
#   BENCH_RUNTIME  e2e hf | jetengine                                hf
#   REPO_ROOT
# -----------------------------------------------------------------------------
# DATE=2026-04-14GPUS="0,1"
# -----------------------------------------------------------------------------
#   speculative_decoding/results/2026-04-14/sweeps/
#   ├── driver.log
#   ├── run.sh                                    #  CLI + grid
#   ├── forward/
#   │   └── forward_{MODEL}_bs{N}.{json,log}
#   ├── module_breakdown/
#   │   ├── module_breakdown.{json,csv}
#   │   ├── module_breakdown_bs{b}_pl{pl}.png
#   │   └── roofline.png
#   ├── regime/
#   │   ├── regime_records.{json,csv}
#   │   └── regime_vs_{batch,prompt_len,params}.png
#   ├── accept/
#   │   └── accept_{DRAFT}_to_{TARGET}.{json,log}
#   └── e2e/
#       ├── bench_{runtime}_{gpu_mode}_nb{K}_bl{BL}_ds{DS}.json   #  JSON
#       ├── bench_{runtime}_{gpu_mode}_nb{K}_bl{BL}_ds{DS}.log    #
#       └── gpu_mode.txt                                           #  single / dual
# =============================================================================

set -euo pipefail

# ---- Resolve repo root -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$REPO_ROOT"

# ---- Config knobs ------------------------------------------------------------
GPUS="${GPUS:-0}"                     # "0"  / "0,1"
DATE="${DATE:-$(date +%F)}"
STAGES="${STAGES:-forward module regime accept e2e}"
SUBTAG="${SUBTAG:-}"
HBM_BW="${HBM_BW:-2039}"
QUICK="${QUICK:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
BENCH_RUNTIME="${BENCH_RUNTIME:-hf}"

# ---- Parse GPU layout --------------------------------------------------------
# CUDA_VISIBLE_DEVICES  id cuda:0/cuda:1
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
N_GPUS="${#GPU_LIST[@]}"

if [[ "$N_GPUS" == "1" ]]; then
  GPU_MODE="single"
  DRAFT_DEVICE="cuda:0"
  TARGET_DEVICE="cuda:0"
  BENCH_CONFIG="speculative_decoding/configs/bench_fixed_blocks.yaml"
elif [[ "$N_GPUS" == "2" ]]; then
  GPU_MODE="dual"
  DRAFT_DEVICE="cuda:0"     #  cuda:0 =  ${GPU_LIST[0]}
  TARGET_DEVICE="cuda:1"    #  cuda:1 =  ${GPU_LIST[1]}
  BENCH_CONFIG="speculative_decoding/configs/bench_dual_gpu.yaml"
else
  echo "[driver] ERROR: GPUS must be 1 or 2 comma-separated ids (got '$GPUS')" >&2
  exit 2
fi

# Models   SDAR 1.7B
MODELS=(
  "SDAR-1_7B-Chat:inference/model/SDAR-1_7B-Chat"
  "SDAR-4B-Chat:inference/model/SDAR-4B-Chat"
  "SDAR-8B-Chat:inference/model/SDAR-8B-Chat"
)

# Grids quick
if [[ "$QUICK" == "1" ]]; then
  BATCHES=(1 4)
  PROMPT_LENS=(64)
  BLOCK_LENGTHS=(4 8)
  REGIME_BATCHES=(1 4)
  REGIME_PROMPT_LENS=(64 256)
  E2E_NUM_BLOCKS=(4)
  E2E_BLOCK_LENGTHS=(4)
else
  BATCHES=(1 2 4 8 16 32 64 128)
  PROMPT_LENS=(64 256)
  BLOCK_LENGTHS=(1 2 4 6 8 12 16)
  REGIME_BATCHES=(1 2 4 8 16)
  REGIME_PROMPT_LENS=(64 256 1024)
  E2E_NUM_BLOCKS=(2 4 8)           # num_blocks per request bench  K
  E2E_BLOCK_LENGTHS=(4 8)
fi
FIXED_DS=4
NUM_BLOCKS=8
ITERS=5
WARMUP=1

# ---- Output layout -----------------------------------------------------------
SWEEPS_NAME="sweeps"
if [[ -n "$SUBTAG" ]]; then
  SWEEPS_NAME="sweeps_${SUBTAG}"
fi
OUT_BASE="speculative_decoding/results/$DATE/$SWEEPS_NAME"
FWD_DIR="$OUT_BASE/forward"
MOD_DIR="$OUT_BASE/module_breakdown"
REG_DIR="$OUT_BASE/regime"
ACC_DIR="$OUT_BASE/accept"
E2E_DIR="$OUT_BASE/e2e"
mkdir -p "$FWD_DIR" "$MOD_DIR" "$REG_DIR" "$ACC_DIR" "$E2E_DIR"
echo "$GPU_MODE (GPUS=$GPUS, draft=$DRAFT_DEVICE, target=$TARGET_DEVICE)" > "$E2E_DIR/gpu_mode.txt"

DRIVER_LOG="$OUT_BASE/driver.log"
RUN_SNAPSHOT="$OUT_BASE/run.sh"

#  config
{
  echo "#!/usr/bin/env bash"
  echo "# Snapshot of the invocation that produced this sweeps/ directory"
  echo "# Generated at $(date -Iseconds)"
  echo "GPUS=\"$GPUS\" DATE=$DATE STAGES=\"$STAGES\" SUBTAG=\"$SUBTAG\" \\"
  echo "  HBM_BW=$HBM_BW QUICK=$QUICK NUM_SAMPLES=$NUM_SAMPLES \\"
  echo "  BENCH_RUNTIME=$BENCH_RUNTIME \\"
  echo "  bash speculative_decoding/run_full_sweep.sh"
  echo
  echo "# Resolved layout:"
  echo "#   GPU_MODE       = $GPU_MODE"
  echo "#   DRAFT_DEVICE   = $DRAFT_DEVICE"
  echo "#   TARGET_DEVICE  = $TARGET_DEVICE"
  echo "#   BENCH_CONFIG   = $BENCH_CONFIG"
  echo
  echo "# Grids used:"
  echo "#   BATCHES=${BATCHES[*]}"
  echo "#   PROMPT_LENS=${PROMPT_LENS[*]}"
  echo "#   BLOCK_LENGTHS=${BLOCK_LENGTHS[*]}"
  echo "#   REGIME_BATCHES=${REGIME_BATCHES[*]}"
  echo "#   REGIME_PROMPT_LENS=${REGIME_PROMPT_LENS[*]}"
  echo "#   E2E_NUM_BLOCKS=${E2E_NUM_BLOCKS[*]}"
  echo "#   E2E_BLOCK_LENGTHS=${E2E_BLOCK_LENGTHS[*]}"
  echo "#   FIXED_DS=$FIXED_DS  NUM_BLOCKS=$NUM_BLOCKS  ITERS=$ITERS  WARMUP=$WARMUP"
} > "$RUN_SNAPSHOT"
chmod +x "$RUN_SNAPSHOT"

#  stdout  tee  driver.log
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "=============================================================="
echo "[driver] speculative_decoding full sweep"
echo "[driver] REPO_ROOT      = $REPO_ROOT"
echo "[driver] DATE           = $DATE"
echo "[driver] GPUS (physical)= $GPUS   (mode=$GPU_MODE)"
echo "[driver] DRAFT_DEVICE   = $DRAFT_DEVICE"
echo "[driver] TARGET_DEVICE  = $TARGET_DEVICE"
echo "[driver] STAGES         = $STAGES"
echo "[driver] OUT_BASE       = $OUT_BASE"
echo "[driver] BENCH_CONFIG   = $BENCH_CONFIG (runtime=$BENCH_RUNTIME)"
echo "[driver] QUICK          = $QUICK"
echo "=============================================================="

#  cuda:0 / cuda:1  id
export CUDA_VISIBLE_DEVICES="$GPUS"

has_stage() {
  [[ " $STAGES " == *" $1 "* ]]
}

# =============================================================================
# (1) FORWARD  model × batch  forward  [ $DRAFT_DEVICE]
# =============================================================================
if has_stage forward; then
  echo
  echo "---- [1/5] sweep_forward: model × batch × block_length ----"
  for entry in "${MODELS[@]}"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    for bs in "${BATCHES[@]}"; do
      tag="${name}_bs${bs}"
      echo "[forward] $tag"
      python -u speculative_decoding/sweep_forward.py \
        --model "$path" \
        --device "$DRAFT_DEVICE" \
        --batch "$bs" \
        --prompt_len 69 \
        --block_lengths "${BLOCK_LENGTHS[@]}" \
        --fixed_steps "$FIXED_DS" \
        --num_blocks "$NUM_BLOCKS" \
        --iters "$ITERS" --warmup "$WARMUP" \
        --output "$FWD_DIR/forward_${tag}.json" \
        2>&1 | tee "$FWD_DIR/forward_${tag}.log"
    done
  done
fi

# =============================================================================
# (2) MODULE BREAKDOWN   q/k/v/o / attn_core / gate/up/down / mlp_elem
#      run  PNG  = HBM roofline
# =============================================================================
if has_stage module; then
  echo
  echo "---- [2/5] sweep_module_breakdown: submodule  + HBM roofline ----"
  python -u speculative_decoding/sweep_module_breakdown.py \
    --models "${MODELS[@]#*:}" \
    --device "$DRAFT_DEVICE" \
    --batches "${BATCHES[@]:0:3}" \
    --prompt_lens "${PROMPT_LENS[@]}" \
    --block_length 4 \
    --denoising_steps "$FIXED_DS" \
    --num_blocks "$NUM_BLOCKS" \
    --iters 3 --warmup "$WARMUP" \
    --hbm_bw_gb_s "$HBM_BW" --bytes_per_param 2 \
    --output_dir "$MOD_DIR" \
    2>&1 | tee "$MOD_DIR/module_breakdown.log"
fi

# =============================================================================
# (3) REGIME  batch × prompt_len × params  regime
# =============================================================================
if has_stage regime; then
  echo
  echo "---- [3/5] sweep_regime: BW/launch/compute  ----"
  python -u speculative_decoding/sweep_regime.py \
    --models "${MODELS[@]#*:}" \
    --device "$DRAFT_DEVICE" \
    --batches "${REGIME_BATCHES[@]}" \
    --prompt_lens "${REGIME_PROMPT_LENS[@]}" \
    --block_length 4 --denoising_steps "$FIXED_DS" \
    --num_blocks "$NUM_BLOCKS" \
    --iters 3 --warmup "$WARMUP" \
    --output_dir "$REG_DIR" \
    2>&1 | tee "$REG_DIR/regime.log"
fi

# =============================================================================
# (4) ACCEPT   α
#   draft/target  cuda:0
#   draft cuda:0 / target cuda:1
# =============================================================================
if has_stage accept; then
  echo
  echo "---- [4/5] sweep_accept: α(draft, target, block_length) [$GPU_MODE] ----"
  PAIRS=(
    "SDAR-4B-Chat:SDAR-8B-Chat"
    "SDAR-1_7B-Chat:SDAR-8B-Chat"
    "SDAR-1_7B-Chat:SDAR-4B-Chat"
  )
  for pair in "${PAIRS[@]}"; do
    draft_name="${pair%%:*}"
    target_name="${pair#*:}"
    tag="${draft_name}_to_${target_name}_${GPU_MODE}"
    echo "[accept] $tag   draft=$DRAFT_DEVICE  target=$TARGET_DEVICE"
    python -u speculative_decoding/sweep_accept.py \
      --draft  "inference/model/$draft_name" \
      --target "inference/model/$target_name" \
      --draft_device "$DRAFT_DEVICE" \
      --target_device "$TARGET_DEVICE" \
      --block_lengths "${BLOCK_LENGTHS[@]}" --fixed_steps "$FIXED_DS" \
      --num_samples "$NUM_SAMPLES" --warmup "$WARMUP" \
      --num_blocks "$NUM_BLOCKS" \
      --output "$ACC_DIR/accept_${tag}.json" \
      2>&1 | tee "$ACC_DIR/accept_${tag}.log"
  done
fi

# =============================================================================
# (5) E2E  native (target-only AR) vs speculative
#     - speculative  native        [speedup_native_over_spec_ms_ratio > 1]
#     - dual-GPU                     [ single / dual  tag]
#     -  e2e latency          [end_to_end_ms.mean_ms]
#    --compare both  run  hf_native / hf_speculative
#    JSON  comparison.speedup_native_over_spec_ms_ratio  speedup
#    E2E_NUM_BLOCKS × E2E_BLOCK_LENGTHS  bench_fixed_blocks.yaml
#    bench_dual_gpu.yaml draft cuda:0 / target cuda:1
# =============================================================================
if has_stage e2e; then
  echo
  echo "---- [5/5] bench/run_benchmark: native vs speculative e2e [$GPU_MODE] ----"
  echo "[e2e] config=$BENCH_CONFIG runtime=$BENCH_RUNTIME draft=$DRAFT_DEVICE target=$TARGET_DEVICE"
  for nb in "${E2E_NUM_BLOCKS[@]}"; do
    for bl in "${E2E_BLOCK_LENGTHS[@]}"; do
      tag="bench_${BENCH_RUNTIME}_${GPU_MODE}_nb${nb}_bl${bl}_ds${FIXED_DS}"
      echo "[e2e] $tag"
      python -u speculative_decoding/bench/run_benchmark.py \
        --config "$BENCH_CONFIG" \
        --runtime "$BENCH_RUNTIME" \
        --compare both --no_eos_stop \
        --draft_device "$DRAFT_DEVICE" \
        --target_device "$TARGET_DEVICE" \
        --num_blocks "$nb" \
        --block_length "$bl" \
        --denoising_steps "$FIXED_DS" \
        --num_samples "$NUM_SAMPLES" \
        --warmup 2 \
        --output "$E2E_DIR/${tag}.json" \
        2>&1 | tee "$E2E_DIR/${tag}.log"
    done
  done
fi

echo
echo "=============================================================="
echo "[driver] done  all stages complete ($GPU_MODE mode)"
echo "[driver] outputs: $OUT_BASE"
echo "[driver] next steps:"
echo "  - e2e/:  bench_*.json  comparison.speedup_native_over_spec_ms_ratio"
echo "           (GPUS=0 vs GPUS=0,1, SUBTAG )  dual-GPU "
echo "  - accept/ + forward/:  speedup(K)  K"
echo "           e2e/ "
echo "=============================================================="
