# Quick start

Four calibrated experiments, one entry point. Every setting lives in a YAML; the
entry point holds no implicit defaults, so "what was this number run with" always
has exactly one source.

```bash
python run_protocol.py --list
```

```
available experiments (configs/protocol/):
  llada2_latency   llada2  latency  bl=[4, 8, 16, 32]  arms=['ours', 'target']
  llada2_quality   llada2  quality  bl=[4, 8, 16, 32]  arms=['ours', 'target']
  sdar_latency     sdar    latency  bl=[4, 8]  arms=['ours', 'nat1', 'tp2cg', 'tp2van']
      historical breakdown protocol (repo prompts, gsm8k, N=20) + full kernel stack; not comparable across arms of other files
  sdar_quality     sdar    quality  bl=[4, 8]  arms=['ours', 'target']
```

## Run one

```bash
# Always dry-run first: prints the exact command without touching a GPU.
python run_protocol.py llada2_quality --arm ours --dry_run

# One arm.
python run_protocol.py llada2_quality --arm ours

# Both arms, serially (one 8-GPU node cannot hold two LLaDA2 arms at once).
python run_protocol.py llada2_quality --arm both

# SDAR: --bl picks the checkpoint pair (bl=4 and bl=8 are different models).
python run_protocol.py sdar_quality --arm both --bl 8

# Smoke test anything in ~2 min.
python run_protocol.py sdar_quality --arm both --set num_samples=6 n_warmup=1 datasets='[mmlu]'
```

## The block-length sweep

The YAMLs pin `block_length: 16` as the working point, but the main results come
from sweeping it. `--bl` selects the point:

```bash
for BL in 4 8 16 32; do
  python run_protocol.py llada2_quality --arm both --bl $BL
done
python scripts/52_bl_sweep_table.py --root runs/protocol/llada2_quality
```

`--list` shows which values each experiment was swept at. The two families differ
in what `--bl` can be:

| | swept at | outside that set |
|---|---|---|
| LLaDA2 | 4, 8, 16, 32 | allowed with a warning — one model pair covers every bl |
| SDAR | 4, 8 | **refused** — each bl needs its own checkpoint pair, and only these exist |

`--bl` also moves `denoising_steps` (the `ds = bl` protocol: one position revealed
per step) and, where the YAML uses `fixed_tokens`, rescales `num_blocks`. In the
latency arm `gen_length` must stay divisible by `bl`; 128 works for all four.

What the sweep shows, briefly: quality is monotone in bl for `ours`
(0.818 / 0.807 / 0.787 / 0.764 at bl=4/8/16/32), speedup rises with bl
(1.64x / 2.14x / 2.50x / 2.61x), and bl=32 is a clear loss — its extra 0.11x costs
8.5pp on gsm8k. bl=16 is the working point because bl=4/8/16 are statistically
indistinguishable on quality while bl=16 is the fastest of the three. Note the
baseline also moves with bl, so always report both columns.

Results land in `runs/protocol/<experiment>/<arm>_bl<N>/SUMMARY.json`, with
per-sample records in the sibling `*.jsonl`.

## What to report

| Arm | Metric | Field |
|---|---|---|
| `*_quality` | pass@1 | `pass_at_1` |
| `*_latency` | throughput | `tokens_per_second_paper` (prefill excluded) |

Two traps:

- **`pass_at_1` from a `*_latency` run is garbage.** Fixed length truncates
  answers mid-sentence — LLaDA2 target gsm8k reads 0.22–0.25 there vs 0.925 in the
  quality arm.
- **Wall-clock in a `*_quality` run is not comparable across arms.** The arms
  generate different lengths (226 vs 204 tok/sample). Throughput comes from the
  latency arm only.

`tokens_per_second` (without `_paper`) includes prefill and reads lower. Don't
use it.

## GPUs per arm

| Family | ours | target |
|---|---|---|
| LLaDA2 | 5 (draft 1 + target sharded over 4) | 5 |
| SDAR | 2 (one each) | 1 |

LLaDA2's target needs 4 cards because `LLaDA2.0-flash` is 191.6 GiB in bf16
against 95 GiB of usable HBM. Both arms get 5 so the hardware budget is matched —
and because sharding is naive model parallelism, **only one card computes at any
instant in either arm**, so the comparison is bandwidth-matched too.

## Changing a setting

| To change | Edit | How |
|---|---|---|
| Something temporarily | command line | `--set num_samples=20 datasets='[gsm8k]'` |
| Block length | command line | `--bl 8` — also moves `denoising_steps` and `num_blocks` |
| A calibrated setting | YAML `args:` | commit it; that file is the provenance |
| One arm's own switch | YAML `arms.<arm>.args:` | e.g. `fused_linear: draft` for `ours` only |
| GPU placement | YAML `arms.<arm>` | `draft_device` / `target_device` / `target_gpus` |
| Runner | YAML `arms.<arm>.runner` | add `torchrun: 2` to launch under `torch.distributed.run` |
| Add an experiment | new YAML in `configs/protocol/` | `--list` finds it automatically |
| Environment | `ENV_DEFAULTS` in `run_protocol.py` | or just `export` — real env wins |
| Decode algorithm | `speculative_decoding/cache_aware.py` | then run `scripts/50_d2_mixed_graph_check.py` |
| New model family | `speculative_decoding/adapters/` | add one module; nothing above changes |

Args merge in three layers, later winning: shared `args:` → `arms.<arm>.args:` →
`--set`.

The entry point refuses three mistakes before spending GPU time: a flag the
chosen runner does not accept, a `--bl` with no SDAR checkpoint, and a
`denoising_steps` that has drifted away from `block_length`.

## Environment

`run_protocol.py` sets these itself (each via `setdefault`, so an existing export
wins):

```
SIMSD_ENV=/mnt/home/haotian.ye/envs/simsd     # its bin/python becomes the interpreter
HF_HUB_CACHE=/mnt/home/haotian.ye/hf_cache
HF_MODULES_CACHE=/mnt/home/haotian.ye/hf_modules
PYTHONNOUSERSITE=1
TOKENIZERS_PARALLELISM=false
```

Moving machines needs no file edit — `export SIMSD_ENV=<prefix>` is enough.
`HF_MODULES_CACHE` should stay pinned: `trust_remote_code` modules are cached per
repo+revision, so a different directory means a re-fetch and no longer provably
the same modeling code.

## Repository layout

```
run_protocol.py              entry point for the four calibrated experiments
QUICKSTART.md                this file
configs/
  protocol/                  the four calibrated settings (single source of truth)
    llada2_quality.yaml      EOS on   -> pass@1
    llada2_latency.yaml      EOS off  -> tok/s
    sdar_quality.yaml        truncate + argmax, aligned with LLaDA2
    sdar_latency.yaml        historical breakdown protocol + full kernel stack
  experiments/               older YAML schema for main.py (method x mode matrix)

main_table/                  runners (each is a __main__ script)
  run_paper_protocol.py        llada2_* both arms, sdar_quality
  run_ours_dual_gpu.py         sdar_latency arm=ours
  run_native_tp2_cache.py      sdar_latency arm=nat1 / tp2cg   (under torchrun)
  run_vanilla_tp2_cache.py     sdar_latency arm=tp2van         (under torchrun)
  run_native_sharded.py        target-only reference for a sharded target

speculative_decoding/
  cache_aware.py             decode core: native / speculative / pipelined
                             + resolve_graph_roles (per-role cuda_graph)
  adapters/                  the only layer that knows architecture names
    base.py                    attention forward factories
    llada2.py                  fused QKV, query_layernorm, dense, kwargs plumbing
    sdar.py                    q/k/v_proj, o_proj, CausalLM cache passthrough
  verify.py  mrs.py          acceptance rules (greedy_match / MRS)
  prompts_opencompass.py     OpenCompass templates, sampling, scorers

kernels/
  moe_dispatch_fused.py      grouped-GEMM expert dispatch (LLaDA2 MoE)
  thin_linear_fused.py       split-K GEMV for M <= 16
  fused_toggle.py            install/remove the kernels on a loaded model
  workspace/<op>/            correctness gate only (CONTRACT, build_oracle,
                             duipai, oracle/manifest.jsonl) -- the 44 GB of
                             oracle/*.pt is regenerable, see kernels/README.md

main.py                      older entry point: method x mode matrix
sdar/                        SDAR-specific helpers

docs/  models/  scripts/  runs/          local only, not tracked
```

## The two entry points

`run_protocol.py` runs the **four calibrated experiments** — the settings that
paper numbers come from. `main.py` drives a **method × mode matrix** from
`configs/experiments/` and is for exploration, ablations, and bring-up. They share
the runners and the decode core but not the config schema. Reach for
`run_protocol.py` unless you are exploring.
