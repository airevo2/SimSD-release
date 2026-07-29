# SimSD — Speculative Decoding for SDAR Block Diffusion

Speculative decoding for SDAR (Synergy of Diffusion and AutoRegression)
models. A small draft model (SDAR-1.7B-Chat) runs block-diffusion denoising;
a larger target model (SDAR-8B-Chat) verifies via multi-block causal attention
in a single forward pass, with greedy-match acceptance + variable-length
truncate commit. The default pipeline runs draft on one GPU and target on
another with full stream overlap (incl. speculative target K/V extend).

Compared to the TP=2 vanilla baseline at ~9.6 tok/s, the optimized `ours` config
reaches **63–74 tok/s** on a single 96 GB Blackwell pair, lossless w.r.t.
argmax-decoded target (token-by-token identical output).

---

## 1. Install

Tested on Python 3.10, CUDA 12.8 (NVIDIA RTX PRO 6000 Blackwell). The repo has
no Python package of its own — it is run from the source tree.

```bash
# 1) Create env (conda or venv). Match the python / cuda versions if you can.
conda create -n simsd python=3.10 -y && conda activate simsd

# 2) Install PyTorch matching your CUDA toolkit (cu128 example below):
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# 3) The rest of the deps:
pip install -r requirements.txt
```

`flash-attn` is required by the vendored SDAR modeling code (top-level
`from flash_attn.ops.triton.layer_norm import rms_norm_fn`). If the prebuilt
wheel does not match your torch/CUDA, build it from source:

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

### Model weights

Download from HuggingFace (or point `--draft_model` / `--target_model` to your
local copies):

- `JetLM/SDAR-1.7B-Chat` (draft)
- `JetLM/SDAR-8B-Chat`   (target)

```bash
huggingface-cli download JetLM/SDAR-1.7B-Chat
huggingface-cli download JetLM/SDAR-8B-Chat
```

---

## 2. One-click reproduce main_table

```bash
python main.py
```

That single command runs all three methods (`vanilla` / `vanilla_cg` / `ours`)
× {`latency`, `quality`} over four datasets (`gsm8k`, `mbpp`, `triviaqa`,
`mmlu`) at N=200, writes per-job `SUMMARY.json` plus a top-level
`UNIFIED.json`, and prints a unified results table at the end. Throughput is
computed from `torch.cuda.Event` (not wall clock).

The settings come from `configs/experiments/sdar.yaml`, which `main.py` reads by
default — HuggingFace checkpoints (`JetLM/SDAR-8B-Chat` +
`JetLM/SDAR-1.7B-Chat`), GPUs `0,1`, and the optimized `ours` config (fused
denoise + speculative target extend + truncate + greedy-match). Every value is
overridable on the CLI:

```bash
python main.py -e configs/experiments/sdar.yaml --num_samples 20 --datasets gsm8k
python main.py -e <file> --print_config      # resolved settings + the exact commands
```

## 3. Other model families

Family-specific knowledge lives in `speculative_decoding/adapters/`; the adapter
is chosen by inspecting the loaded checkpoint, not by a flag. Adding a family
means adding one module there.

```bash
# LLaDA2.0-mini against itself — correctness harness for the pipeline
python main.py -e configs/experiments/llada2.yaml

# mini drafts, flash verifies (flash sharded over 4 GPUs)
python main.py -e configs/experiments/llada2_flash.yaml

# upstream generate() vs SimSD native vs self-draft, in one process
python scripts/llada2_selfdraft_check.py --dtype float32
```

Two LLaDA2 constraints are encoded in the configs rather than left to be
rediscovered:

- **Eager only.** `LLaDA2MoeSparseMoeBlock.moe_infer` does
  `tokens_per_expert.cpu().numpy()` and loops over 256 experts in Python, so
  nothing can be captured into a cuda_graph. The adapter raises with that
  explanation instead of failing inside capture.
- **Sharded target.** LLaDA2.0-flash is 191.6 GiB in bf16, so the TP=2 baselines
  would need 95.8 GiB/rank against 95.6 GiB of usable HBM — it misses by a hair.
  `runtime.target_gpus` spreads it over several cards instead
  with `device_map="auto"`. That is *naive* model parallelism, not tensor
  parallelism and not pipeline parallelism either — layers are placed across
  devices but a single input runs through them strictly in order, so one GPU is
  busy at a time. It buys capacity, not speed. Compare against
  `native_sharded`, which places the target identically, rather than against a
  TP run; otherwise a placement difference gets attributed to speculation.

---

## Notes

- Model weights, datasets and run outputs are not part of this repo
  (see `.gitignore`). Download them separately.
- The IFEval scorer in `speculative_decoding/Experiment_Backend/self_draft_compare.py`
  requires OpenCompass on `sys.path`. Set `OPENCOMPASS_PATH=/path/to/opencompass`
  before running if you want IFEval scoring; otherwise the loader is a no-op.
- `vanilla` / `vanilla_cg` runners use `torchrun --nproc_per_node=2`. `ours`
  is single-process dual-GPU (draft on `cuda:0`, target on `cuda:1`). The 4 GPU
  IDs you give via `--gpus` are split: first pair for vanilla/vanilla_cg, the
  same pair re-used (one process at a time) for ours.
- vanilla_cg can hang on `dist.destroy_process_group()` after writing
  `SUMMARY.json` (NCCL + cuda_graph teardown deadlock); `main.py` watchdog
  detects this and force-kills after `--cleanup_grace_s` (default 90s).
