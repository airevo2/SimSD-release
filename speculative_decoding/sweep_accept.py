#!/usr/bin/env python3
"""Acceptance rate sweep for ONE (draft, target) pair across (block_length, denoising_steps).

Loads (draft, target) ONCE, then sweeps a grid of (block_length, denoising_steps).
For each grid point, runs num_samples gsm8k prompts through speculative_generate()
(from speculative_decode.py) and aggregates the MRS acceptance stats.

Output JSON:
  {
    "draft_model": "...",
    "target_model": "...",
    "config": {...},
    "results": {
       "bl=B_ds=S": {
          "block_length": B,
          "denoising_steps": S,
          "num_samples": N,
          "total_committed_blocks": K,
          "token_acc_rate": α,              # p in user's formula
          "avg_accepted_per_block": μ,      # n in user's formula (truncate-on-reject)
          "block_full_accept_rate": f,
       }, ...
    }
  }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    os.path.join(os.path.expanduser("~"), ".flashinfer_local"),
)

import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

from speculative_decoding.config import SpecConfig  # noqa: E402
from speculative_decoding.speculative_decode import speculative_generate  # noqa: E402
from speculative_decoding.verify import patch_multi_block_mask_fn  # noqa: E402


def _resolve(path: str) -> str:
    if os.path.isdir(path):
        return os.path.abspath(path)
    cand = os.path.join(_ROOT, path)
    return os.path.abspath(cand) if os.path.isdir(cand) else path


def load_gsm8k_prompts(tokenizer, n: int, split: str = "train") -> List[List[int]]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    ds = ds.select(range(min(n, len(ds))))
    prompts = []
    for idx in range(len(ds)):
        msgs = [{"role": "user", "content": ds[idx]["question"]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(tokenizer.encode(prompt, add_special_tokens=False))
    return prompts


def run_grid_point(draft, target, prompts, cfg, pad_token_id, padded_len, eos_ids):
    n_accepted_all: List[int] = []
    full_flags: List[int] = []
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_bonus_tokens = 0
    total_committed_blocks = 0

    for p in prompts:
        torch._dynamo.reset()
        gen, stats = speculative_generate(
            draft, target, p, cfg, pad_token_id, padded_len, eos_ids,
        )
        per_blk = stats.get("per_block_accepted", [])
        n_accepted_all.extend(per_blk)
        full_flags.extend(1 if n == cfg.block_length else 0 for n in per_blk)
        total_draft_tokens += stats.get("total_draft_tokens", 0)
        total_accepted_tokens += stats.get("total_accepted_tokens", 0)
        total_bonus_tokens += stats.get("total_bonus_tokens", 0)
        total_committed_blocks += len(per_blk)

    n_blocks = max(1, total_committed_blocks)
    bl = cfg.block_length
    return {
        "num_samples": len(prompts),
        "block_length": bl,
        "denoising_steps": cfg.denoising_steps,
        "total_committed_blocks": total_committed_blocks,
        "total_draft_tokens": total_draft_tokens,
        "total_accepted_tokens": total_accepted_tokens,
        "total_bonus_tokens": total_bonus_tokens,
        "token_acc_rate": total_accepted_tokens / max(1, total_committed_blocks * bl),
        "avg_accepted_per_block": float(sum(n_accepted_all) / n_blocks),
        "block_full_accept_rate": float(sum(full_flags) / n_blocks),
    }


def load_pair(draft_path: str, target_path: str, draft_device: str, target_device: str):
    tokenizer = AutoTokenizer.from_pretrained(draft_path, trust_remote_code=True)
    print(f"[sweep-accept] loading draft {draft_path} -> {draft_device}", flush=True)
    draft = AutoModelForCausalLM.from_pretrained(
        draft_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(draft_device).eval()
    print(f"[sweep-accept] loading target {target_path} -> {target_device}", flush=True)
    target = AutoModelForCausalLM.from_pretrained(
        target_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(target_device).eval()
    patch_multi_block_mask_fn(target, block_causal_prompt=True)
    return tokenizer, draft, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--draft_device", default="cuda:0")
    p.add_argument("--target_device", default="cuda:0")
    p.add_argument("--block_lengths", type=int, nargs="+",
                   default=[1, 2, 4, 6, 8, 12, 16])
    p.add_argument("--fixed_steps", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    args.draft = _resolve(args.draft)
    args.target = _resolve(args.target)

    tokenizer, draft, target = load_pair(
        args.draft, args.target, args.draft_device, args.target_device
    )

    all_prompts = load_gsm8k_prompts(tokenizer, args.num_samples + args.warmup, args.dataset_split)
    warmup_prompts = all_prompts[: args.warmup]
    eval_prompts = all_prompts[args.warmup :]

    pad_token_id = tokenizer.pad_token_id or 0
    eos_ids = None  # no_eos_stop  keep runs fixed length for fair comparison

    grid: List[tuple] = []
    for bl in args.block_lengths:
        for S in {args.fixed_steps, bl}:
            if (bl, S) not in grid:
                grid.append((bl, S))

    max_bl = max(args.block_lengths)
    max_plen = max(len(x) for x in all_prompts)
    padded_len = (((max_plen + (args.num_blocks + 1) * 2 * max_bl) + 127) // 128) * 128

    results: Dict[str, Any] = {}
    for bl, S in grid:
        tag = f"bl={bl}_ds={S}"
        cfg = SpecConfig(
            draft_model=args.draft,
            target_model=args.target,
            block_length=bl,
            denoising_steps=S,
            block_size=S,
            num_blocks=args.num_blocks,
            num_samples=args.num_samples,
            seed=args.seed,
            draft_device=args.draft_device,
            target_device=args.target_device,
        )
        torch.manual_seed(args.seed)

        # Warmup
        for pr in warmup_prompts:
            torch._dynamo.reset()
            speculative_generate(draft, target, pr, cfg, pad_token_id, padded_len, eos_ids)

        print(f"[{tag}] running {len(eval_prompts)} samples", flush=True)
        row = run_grid_point(draft, target, eval_prompts, cfg, pad_token_id,
                             padded_len, eos_ids)
        print(
            f"  token_acc={row['token_acc_rate']:.4f}  "
            f"avg_accepted={row['avg_accepted_per_block']:.3f}/{bl}  "
            f"full_blk={row['block_full_accept_rate']:.3f}",
            flush=True,
        )
        results[tag] = row

    report = {
        "draft_model": args.draft,
        "target_model": args.target,
        "config": {k: v for k, v in vars(args).items() if k != "output"},
        "padded_len": padded_len,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
