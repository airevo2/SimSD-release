#!/usr/bin/env python3
"""End-to-end gate: does swapping in the fused MoE change what the model says?

Per-element tolerances on one operator are a proxy. What we actually care about
is whether the full model's next-token choice moves. Runs one real LLaDA2.0-mini
forward with the stock ``moe_infer`` and with the fused kernel, and compares
logits + argmax.

fp32 is the gate: bf16 is known to flip ~1 token in 21 from routing sensitivity
alone (see docs/llada2-plan.md), so a bf16 disagreement proves nothing either way.

    python kernels/workspace/moe_dispatch/verify_model.py --dtype float32
    python kernels/workspace/moe_dispatch/verify_model.py --dtype bfloat16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inclusionAI/LLaDA2.0-mini")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seq", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    from speculative_decoding.cache_aware import (
        patch_sdpa_with_cache, patch_family_plumbing)
    from kernels import fused_toggle

    dt = getattr(torch, args.dtype)
    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, trust_remote_code=True,
        device_map={"": args.device}).eval()
    patch_sdpa_with_cache(m)
    patch_family_plumbing(m)

    dev = torch.device(args.device)
    V, L = m.config.vocab_size, args.seq
    torch.manual_seed(0)
    ids = torch.randint(0, V, (1, L), device=dev)
    ids[:, L // 2:] = 156895
    kw = dict(input_ids=ids,
              attention_mask=torch.ones(1, 1, L, L, dtype=torch.bool, device=dev),
              position_ids=torch.arange(L, device=dev).unsqueeze(0),
              past_key_values=DynamicCache(), use_cache=True, store_kv=False)

    with torch.inference_mode():
        stock = m(**kw).logits.float().clone()

    # 顺序是有意的：stock 先跑，因为 stacking 会释放 per-expert 权重
    # （否则 fp32 下 mini 放不下）。
    n = fused_toggle.apply_moe_to_model(m, True)
    print(f"[fused] {n} MoE blocks switched to grouped GEMM")
    with torch.inference_mode():
        fused = m(**kw).logits.float().clone()

    d = (stock - fused).abs()
    a_s, a_f = stock.argmax(-1)[0], fused.argmax(-1)[0]
    agree = int((a_s == a_f).sum())
    scale = stock.abs().max().item()
    print(f"\ndtype={args.dtype}  seq={L}  logit scale={scale:.2f}")
    print(f"  max|Δ|   = {d.max().item():.3e}   ({d.max().item()/scale:.2e} 相对)")
    print(f"  mean|Δ|  = {d.mean().item():.3e}")
    print(f"  argmax   = {agree}/{L} 一致")
    ok = agree == L
    print(f"\n{'PASS' if ok else 'FAIL'}"
          + ("" if ok else "  —— fp32 下 argmax 应当全一致" if args.dtype == "float32"
             else "  (bf16 下不一致属预期，见 docs/llada2-plan.md)"))
    return 0 if (ok or args.dtype == "bfloat16") else 1


if __name__ == "__main__":
    sys.exit(main())
