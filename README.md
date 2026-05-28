# Speculative Decoding for SDAR Block Diffusion

Speculative decoding implementation for SDAR (Synergy of Diffusion and AutoRegression) models.
Combines a small draft model (block diffusion denoising) with a larger target model (multi-block
causal verification + Modified Rejection Sampling) to accelerate generation.

## Layout

The repo is grouped into three top-level entry-point folders plus the core SD library:

```
.
├── sdar/                            # Shared SDAR support (used by everything)
│   ├── modeling_sdar.py             #   vendored HF SDAR modeling code
│   ├── new_attn.py                  #   single-block causal mask helper
│   ├── new_attn_multi_block.py      #   multi-block causal mask construction
│   └── run_native_tp2.py            #   patch_stock_rms_norm + TP=2 native runner
│
├── main_table/                      # Main-table runners (Vanilla, Vanilla+CG, SimSD)
│   ├── run_vanilla_tp2_cache.py     #   Vanilla baseline (TP=2, multinomial)
│   ├── run_native_tp2_cache.py      #   Vanilla + CUDA-Graph baseline (TP=2)
│   └── quality_compare.py           #   quality eval driver
│                                    #   (SimSD speed: speculative_decoding/bench/run_benchmark.py)
│                                    #   (SimSD entry: speculative_decoding/speculative_decode.py)
│
├── ablation/                        # Ablation studies (no duplicates)
│   ├── draft_probs_ablation_v1_5*.py    # v1.5 (base / batch / fused / graph) ablations
│   ├── draft_probs_ablation_v2.py       # v2: exact 4D-mask rules on draft side
│   ├── draft_probs_ablation_v3.py       # v3: single-replace per-block probs
│   ├── verify_layout_ablation.py        # verify-layout ablation
│   ├── microbench_graph_vs_eager.py     # cuda-graph vs eager micro-bench
│   ├── bench/                            # per-ablation latency runners (bench_v1_5*.py)
│   └── sweep/                            # sweep drivers + common util
│
└── speculative_decoding/            # Core SimSD library (entry point + internals)
    ├── speculative_decode.py        #   Entry point: draft → verify → MRS loop
    ├── draft.py                     #   draft_one_block(): block diffusion denoising
    ├── verify.py                    #   multi-block causal mask + target verify
    ├── mrs.py                       #   Modified Rejection Sampling
    ├── cache_aware.py               #   KV-cache pipelined SD path
    ├── config.py                    #   SpecConfig + YAML loader
    ├── configs/                     #   YAML experiment configs
    ├── bench/                       #   SimSD latency benchmark harness (used by main_table)
    └── Experiment_Backend/          #   Shared quality library (self_draft_compare)
```

### Why these three top-level groups

- **`sdar/`** — every runner (vanilla, vanilla+CG, SimSD, every ablation) imports the
  same SDAR modeling code, the same attention-mask helpers, and `patch_stock_rms_norm`.
  Pulling them into one folder makes the shared dependency explicit.
- **`main_table/`** — the scripts that produce the main-table numbers (speed + quality)
  for Vanilla, Vanilla+CG, and SimSD.
- **`ablation/`** — every ablation variant lives here so the main table is not mixed
  with research experiments. Each ablation pairs a monkey-patch script with its
  matching `bench/bench_v1_5_*.py` runner.

The legacy `draft_probs_ablation.py` (v1, superseded by v1.5) has been removed.

## Setup

1. Install Python dependencies (PyTorch, transformers, datasets, etc.):
   ```bash
   pip install -r speculative_decoding/requirements_speculative.txt
   ```
2. Download SDAR checkpoints from HuggingFace (or point `--target_model` /
   `--draft_model` to your local path) and place them under
   `inference/model/SDAR-{1.7B,4B,8B}-Chat/` (or pass paths directly). The
   vendored `sdar/modeling_sdar.py` matches the 8B-Chat / `-bN` variants; if
   you instead load 1.7B/4B without `-bN` suffix, HuggingFace's
   `trust_remote_code=True` will pull the corresponding modeling file from
   the checkpoint directory.

## Quick start

SimSD speculative decoding (alignment test on a single block):
```bash
python speculative_decoding/speculative_decode.py \
  --config speculative_decoding/configs/single_block_test.yaml
```

SimSD latency benchmark (native vs speculative):
```bash
python speculative_decoding/bench/run_benchmark.py \
  --config speculative_decoding/configs/bench_fixed_blocks.yaml \
  --compare both --no_eos_stop
```

Vanilla + CG baseline (TP=2 native, KV cache + cuda_graph):
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29560 \
  main_table/run_native_tp2_cache.py \
  --target_model <hf-id-or-path-to-SDAR-target> \
  --output_dir runs/native_tp2 \
  --use_cuda_graph
```

Vanilla baseline (TP=2 multinomial, no cuda_graph):
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29560 \
  main_table/run_vanilla_tp2_cache.py \
  --target_model <hf-id-or-path-to-SDAR-target> \
  --output_dir runs/vanilla_tp2
```

Quality comparison:
```bash
python main_table/quality_compare.py --help
```

## Notes

- Model weights, datasets, and run outputs are intentionally not part of this repo
  (see `.gitignore`). Download them separately.
- The IFEval scorer in `speculative_decoding/Experiment_Backend/self_draft_compare.py`
  requires OpenCompass on `sys.path`. Set `OPENCOMPASS_PATH=/path/to/opencompass`
  before running if you want IFEval scoring; otherwise the loader is a no-op.
- Conda activation is opt-in via `CONDA_SH` / `CONDA_ENV` environment variables
  (see `ablation/sweep/run_self_draft.sh`).
