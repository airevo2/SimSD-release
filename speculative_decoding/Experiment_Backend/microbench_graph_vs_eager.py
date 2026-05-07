"""Microbench: per-forward latency for {1.7B, 8B} × {eager, CUDA graph} × bs ∈ {1,2,4,8,16}.

Goal: answer two questions
  1. With CUDA graph, does the per-forward latency gap between 1.7B and 8B
     widen (because graph removes launch overhead  param-count differences
     become more visible)?
  2. At what batch size does each model cross from BW-bound into
     compute-bound on H100 SXM (HBM 3350 GB/s, bf16 dense 989 TFLOPs/s)?

Methodology
  - Pads prompt left to padded_len = prompt_len + num_blocks*block_length so seq_len
    stays constant across the warmup loop (same as draft.py's CUDA graph cache).
  - Builds the same block-causal attention mask draft.py uses.
  - For graph mode: captures once via _get_draft_forward_graph_batch (or
    _get_draft_forward_graph at bs=1) and calls g.replay() inside the timed loop.
  - For eager: builds attention_mask once, runs model() directly (no graph).
  - All timings via cuda.Event around the model() / replay() call only.
  - Excludes mask construction, sampling, etc.  pure model forward time.

Outputs
  - Prints a markdown-friendly table to stdout
  - Dumps results JSON to --output for downstream plotting
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Match run_benchmark.py: disable torch.compile before importing torch so that
# the transformers custom modeling code (which imports LossKwargs only when
# compile is active) doesn't trip on missing utilities.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch._dynamo

# Shim: transformers/utils removed LossKwargs after v4.57; the cached SDAR
# modeling code still imports it. Provide an empty stub so the import doesn't
# explode. Functionally a no-op in inference (it's only used as a TypedDict
# base for KwargsForCausalLM).
import transformers.utils as _tu
if not hasattr(_tu, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False): pass
    _tu.LossKwargs = LossKwargs

from transformers import AutoModelForCausalLM

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding.draft import (  # noqa: E402
    _build_block_causal_attn,
    _get_draft_forward_graph,
    _get_draft_forward_graph_batch,
    patch_sdpa_eval_attention,
    clear_draft_graph_cache,
)


def time_n(fn, n: int, device: torch.device) -> list[float]:
    """Time fn n times via CUDA events. Returns list of ms per call."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    torch.cuda.synchronize(device)
    for i in range(n):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize(device)
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def bench_one(model_path: str, batch: int, prompt_len: int, block_length: int,
              device: torch.device, dtype: torch.dtype, mode: str,
              warmup: int, iters: int, mask_token_id: int = 151669) -> dict:
    """Bench one (model, batch, mode) point. mode in {'eager','graph'}."""
    print(f"  [load] {model_path}  {device}", flush=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        )
        .to(device)
        .eval()
    )
    patch_sdpa_eval_attention(model)
    seq_len = prompt_len + block_length

    # Build inputs (same for both modes)
    full_ids = torch.full(
        (batch, seq_len), mask_token_id, dtype=torch.long, device=device,
    )
    attn = _build_block_causal_attn(seq_len, block_length, device)

    if mode == "graph":
        if batch == 1:
            g, static_ids, static_mask, static_logits = _get_draft_forward_graph(
                model, seq_len, block_length, device, mask_token_id,
            )
        else:
            g, static_ids, static_mask, static_logits = (
                _get_draft_forward_graph_batch(
                    model, batch, seq_len, block_length, device, mask_token_id,
                )
            )
        # Seed values into static buffers
        static_ids.copy_(full_ids)

        # Warmup replays
        with torch.no_grad():
            for _ in range(warmup):
                g.replay()

        # Timed replays
        ts = time_n(lambda: g.replay(), iters, device)
    else:  # eager
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids=full_ids, attention_mask=attn,
                          use_cache=False, return_dict=True)

        def _fn():
            with torch.no_grad():
                model(input_ids=full_ids, attention_mask=attn,
                      use_cache=False, return_dict=True)

        ts = time_n(_fn, iters, device)

    # Cleanup
    del model
    if mode == "graph":
        clear_draft_graph_cache()
    torch.cuda.empty_cache()

    mean = sum(ts) / len(ts)
    p50 = sorted(ts)[len(ts) // 2]
    return {
        "mean_ms": mean,
        "p50_ms": p50,
        "min_ms": min(ts),
        "max_ms": max(ts),
        "iters": iters,
        "warmup": warmup,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["inference/model/SDAR-1_7B-Chat",
                            "inference/model/SDAR-8B-Chat"])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--prompt_len", type=int, default=68)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--output", default=None)
    p.add_argument("--hbm_bw_gb_s", type=float, default=3350.0,
                   help="H100 SXM ≈ 3350 GB/s")
    p.add_argument("--peak_tflops", type=float, default=989.0,
                   help="H100 SXM bf16 dense ≈ 989 TFLOPs/s")
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    # Resolve model paths
    REPO_ROOT = str(REPO)
    resolved = []
    for m in args.models:
        if os.path.isdir(m):
            resolved.append(m)
        else:
            cand = os.path.join(REPO_ROOT, m)
            resolved.append(cand if os.path.isdir(cand) else m)

    # Param sizes (bf16 GB) and approximate FLOPs/forward formula
    PARAM_BYTES = {"SDAR-1_7B-Chat": 3.4e9, "SDAR-4B-Chat": 8e9,
                   "SDAR-8B-Chat": 16e9}
    PARAM_COUNT = {"SDAR-1_7B-Chat": 1.7e9, "SDAR-4B-Chat": 4e9,
                   "SDAR-8B-Chat": 8e9}

    results = []
    for model_path in resolved:
        mname = os.path.basename(model_path)
        param_gb = PARAM_BYTES.get(mname, 0) / 1e9
        param_count = PARAM_COUNT.get(mname, 0)
        bw_floor_ms = (param_gb / args.hbm_bw_gb_s) * 1000

        for batch in args.batches:
            seq_len = args.prompt_len + args.block_length
            # Forward FLOPs ≈ 2 × params × seq_len (FMA accounting)
            flops = 2 * param_count * batch * seq_len
            compute_floor_ms = (flops / (args.peak_tflops * 1e12)) * 1000

            for mode in ["eager", "graph"]:
                print(f"\n[bench] {mname} bs={batch} mode={mode}", flush=True)
                try:
                    r = bench_one(
                        model_path, batch, args.prompt_len, args.block_length,
                        device, dtype, mode, args.warmup, args.iters,
                    )
                    r.update({
                        "model": mname,
                        "batch": batch,
                        "mode": mode,
                        "seq_len": seq_len,
                        "bw_floor_ms": bw_floor_ms,
                        "compute_floor_ms": compute_floor_ms,
                        "ratio_to_floor": r["mean_ms"] / max(bw_floor_ms, compute_floor_ms),
                        "regime_floor": "bw" if bw_floor_ms > compute_floor_ms else "compute",
                    })
                    results.append(r)
                    print(f"   mean={r['mean_ms']:.2f} ms  p50={r['p50_ms']:.2f} "
                          f"min={r['min_ms']:.2f}  max={r['max_ms']:.2f}  "
                          f"floor={max(bw_floor_ms, compute_floor_ms):.2f} ms  "
                          f"× {r['ratio_to_floor']:.1f}",
                          flush=True)
                except Exception as e:
                    print(f"  FAILED: {e}", flush=True)

    # Build table
    print("\n\n========== RESULTS ==========\n")
    print(f"{'model':<18} {'bs':>3} {'mode':<6} "
          f"{'T_fwd (ms)':>11} {'BW floor':>10} {'Comp floor':>11} "
          f"{'/floor':>8} {'regime':<8}")
    print("-" * 88)
    for r in results:
        print(f"{r['model']:<18} {r['batch']:>3} {r['mode']:<6} "
              f"{r['mean_ms']:>11.2f} {r['bw_floor_ms']:>10.2f} "
              f"{r['compute_floor_ms']:>11.2f} {r['ratio_to_floor']:>8.1f} "
              f"{r['regime_floor']:<8}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\n[wrote] {args.output}")


if __name__ == "__main__":
    main()
