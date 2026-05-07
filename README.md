# Speculative Decoding for SDAR Block Diffusion

Speculative decoding implementation for SDAR (Synergy of Diffusion and AutoRegression) models.
Combines a small draft model (block diffusion denoising) with a larger target model (multi-block
causal verification + Modified Rejection Sampling) to accelerate generation.

## Layout

```
.
├── speculative_decoding/         # Main SD module
│   ├── speculative_decode.py     # Entry point: draft -> verify -> MRS loop
│   ├── draft.py                  # draft_one_block(): block diffusion denoising
│   ├── verify.py                 # Multi-block causal mask + target verify
│   ├── mrs.py                    # Modified Rejection Sampling
│   ├── cache_aware.py            # KV-cache pipelined SD path
│   ├── config.py                 # SpecConfig + YAML loader
│   ├── configs/                  # YAML experiment configs
│   ├── bench/                    # Latency benchmark harness
│   └── Experiment_Backend/       # Quality / acceptance / sweep drivers
├── new_attn_multi_block.py       # Multi-block causal mask construction
├── new_attn.py                   # Single-block mask helper
├── run_native_tp2_cache.py       # TP=2 native baseline (KV cache + cuda_graph)
├── run_vanilla_tp2_cache.py      # TP=2 vanilla baseline (multinomial sampling)
└── run_native_tp2.py             # Provides patch_stock_rms_norm
```

## Setup

1. Install Python dependencies (PyTorch, transformers, datasets, etc.):
   ```bash
   pip install -r speculative_decoding/requirements_speculative.txt
   ```
2. Download SDAR checkpoints from HuggingFace (or point `--target_model` /
   `--draft_model` to your local path) and place them under
   `inference/model/SDAR-{1.7B,4B,8B}-Chat/` (or pass paths directly).

## Quick start

Speculative decoding (alignment test on a single block):
```bash
python speculative_decoding/speculative_decode.py \
  --config speculative_decoding/configs/single_block_test.yaml
```

Latency benchmark (native vs speculative):
```bash
python speculative_decoding/bench/run_benchmark.py \
  --config speculative_decoding/configs/bench_fixed_blocks.yaml \
  --compare both --no_eos_stop
```

TP=2 native baseline:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29560 \
  run_native_tp2_cache.py \
  --target_model <hf-id-or-path-to-SDAR-target> \
  --output_dir runs/native_tp2 \
  --use_cuda_graph
```

## Notes

- Model weights, datasets, and run outputs are intentionally not part of this repo
  (see `.gitignore`). Download them separately.
- The IFEval scorer in `Experiment_Backend/self_draft_compare.py` requires
  OpenCompass on `sys.path`. Set `OPENCOMPASS_PATH=/path/to/opencompass` before
  running if you want IFEval scoring; otherwise the loader is a no-op.
- Conda activation is opt-in via `CONDA_SH` / `CONDA_ENV` environment variables
  (see `Experiment_Backend/run_self_draft.sh`).
