#!/usr/bin/env python3
"""
Native dLLM TPS benchmark for LLaDA / SDAR models.

Measures end-to-end throughput (tokens/sec) of standard block diffusion
inference (multi-step denoising) without speculative decoding.

This is the baseline for comparing against SSD and other speculative methods.

Timing: cuda.synchronize() brackets around each sample; TPS = generated_tokens / wall_time.

Usage (from dllm_causal_train/):
  # SDAR-8B native
  python speculative_decoding/bench/llada_native_tps.py \
    --model_path inference/model/SDAR-8B-Chat \
    --model_type sdar --block_length 4 --denoising_steps 4 \
    --num_blocks 8 --num_samples 20

  # LLaDA-8B-Instruct native
  python speculative_decoding/bench/llada_native_tps.py \
    --model_path /path/to/LLaDA-8B-Instruct \
    --model_type llada --block_length 8 --denoising_steps 8 \
    --num_blocks 4 --num_samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SDAR_MASK_TOKEN_ID = 151669
LLADA_MASK_TOKEN_ID = 156895


def _sdar_native_generate(
    model, prompt_ids: List[int], num_blocks: int,
    block_length: int, denoising_steps: int,
    mask_token_id: int, device: torch.device,
) -> Tuple[List[int], float]:
    """SDAR native: reuse draft_one_block loop. Returns (gen_ids, wall_seconds)."""
    from speculative_decoding.bench.native_generate import native_multi_block_generate
    from speculative_decoding.bench.timing import cuda_sync_if_available

    cuda_sync_if_available()
    t0 = time.perf_counter()
    with torch.cuda.device(device):
        gen, _ = native_multi_block_generate(
            model, prompt_ids, num_blocks, block_length,
            denoising_steps, mask_token_id,
            eos_ids=None, return_timings=False, use_cuda_graph=False,
        )
    cuda_sync_if_available()
    t1 = time.perf_counter()
    return gen, t1 - t0


def _llada_native_generate(
    model, prompt_ids: List[int], num_blocks: int,
    block_length: int, denoising_steps: int,
    mask_token_id: int, device: torch.device,
    threshold: float = 0.95,
) -> Tuple[List[int], float]:
    """LLaDA native: multi-block sequential generation."""

    def _get_schedule(bl, ds):
        base = bl // ds
        rem = bl % ds
        return [base + (1 if i < rem else 0) for i in range(ds)]

    all_gen = []
    context_ids = list(prompt_ids)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for blk in range(num_blocks):
        input_t = torch.tensor(
            [context_ids + [mask_token_id] * block_length],
            dtype=torch.long, device=device,
        )
        seq_len = input_t.shape[1]
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.bfloat16)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        ctx_len = len(context_ids)

        schedule = _get_schedule(block_length, denoising_steps)
        for step_idx, n_unmask in enumerate(schedule):
            resp = input_t[0, ctx_len:]
            is_masked = resp == mask_token_id
            num_masked = is_masked.sum().item()
            if num_masked == 0:
                break

            with torch.no_grad():
                out = model(input_ids=input_t, attention_mask=attn_mask,
                            position_ids=position_ids)
            logits = out.logits[0, ctx_len:]
            probs = F.softmax(logits.float(), dim=-1)
            pred = probs.argmax(dim=-1)
            conf = probs.gather(1, pred.unsqueeze(-1)).squeeze(-1)

            conf_masked = torch.where(is_masked, conf,
                                      torch.tensor(-torch.inf, device=device))
            high = conf_masked > threshold
            n_high = high.sum().item()
            if n_high >= n_unmask:
                transfer = high
            else:
                _, tidx = torch.topk(conf_masked, k=min(n_unmask, num_masked))
                transfer = torch.zeros(block_length, dtype=torch.bool, device=device)
                transfer[tidx] = True

            for pos in transfer.nonzero(as_tuple=True)[0]:
                input_t[0, ctx_len + pos] = pred[pos]

        block_ids = input_t[0, ctx_len:].tolist()
        all_gen.extend(block_ids)
        context_ids.extend(block_ids)

    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return all_gen, t1 - t0


def main():
    parser = argparse.ArgumentParser(description="Native dLLM TPS benchmark")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["sdar", "llada"], default="sdar")
    parser.add_argument("--block_length", type=int, default=4)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    mask_token_id = SDAR_MASK_TOKEN_ID if args.model_type == "sdar" else LLADA_MASK_TOKEN_ID

    model_path = args.model_path
    if not os.path.isdir(model_path):
        cand = os.path.join(_ROOT, model_path)
        if os.path.isdir(cand):
            model_path = cand

    print(f"[native_tps] model={model_path} type={args.model_type}")
    print(f"  bl={args.block_length} ds={args.denoising_steps} nb={args.num_blocks}")
    print(f"  samples={args.num_samples} warmup={args.warmup} device={args.device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    if args.model_type == "sdar":
        from speculative_decoding.draft import patch_sdpa_eval_attention
        patch_sdpa_eval_attention(model)

    print(f"Loading dataset ({args.dataset})...")
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompt_key = "question"
    else:
        ds = load_dataset(args.dataset, split="test" if args.dataset in ("humaneval", "mbpp") else "train")
        prompt_key = ds.column_names[0]

    total_needed = args.num_samples + args.warmup
    if len(ds) > total_needed:
        ds = ds.select(range(total_needed))

    all_prompt_ids = []
    for i in range(len(ds)):
        text = ds[i][prompt_key]
        messages = [{"role": "user", "content": text}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = text
        all_prompt_ids.append(tokenizer.encode(prompt, add_special_tokens=False))

    gen_fn = _sdar_native_generate if args.model_type == "sdar" else _llada_native_generate

    e2e_times = []
    token_counts = []
    tps_list = []
    rows = []

    for i, pids in enumerate(all_prompt_ids):
        gen, wall_s = gen_fn(
            model, pids, args.num_blocks, args.block_length,
            args.denoising_steps, mask_token_id, device,
        )
        if i < args.warmup:
            print(f"  warmup {i+1}/{args.warmup}: {wall_s*1000:.0f}ms")
            continue

        n_tok = len(gen)
        tps = n_tok / wall_s if wall_s > 0 else 0
        e2e_times.append(wall_s)
        token_counts.append(n_tok)
        tps_list.append(tps)
        rows.append({
            "idx": i, "prompt_len": len(pids), "gen_len": n_tok,
            "e2e_s": wall_s, "tps": tps,
        })

        if (i - args.warmup + 1) % 5 == 0:
            import statistics
            print(f"  [{i - args.warmup + 1}/{args.num_samples}] "
                  f"mean_tps={statistics.mean(tps_list):.1f} "
                  f"last_tps={tps:.1f}")

    if not tps_list:
        print("No valid results.")
        sys.exit(1)

    import statistics
    mean_tps = statistics.mean(tps_list)
    p50_tps = sorted(tps_list)[len(tps_list) // 2]
    mean_ms = statistics.mean(e2e_times) * 1000
    p50_ms = sorted(e2e_times)[len(e2e_times) // 2] * 1000

    report = {
        "model": model_path,
        "model_type": args.model_type,
        "config": {
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "num_blocks": args.num_blocks,
            "num_samples": args.num_samples,
            "warmup": args.warmup,
            "dataset": args.dataset,
            "device": args.device,
            "seed": args.seed,
            "mask_token_id": mask_token_id,
        },
        "results": {
            "mean_tps": mean_tps,
            "p50_tps": p50_tps,
            "std_tps": statistics.pstdev(tps_list),
            "mean_e2e_ms": mean_ms,
            "p50_e2e_ms": p50_ms,
            "total_tokens": sum(token_counts),
            "n": len(tps_list),
        },
        "samples": rows,
    }

    print(f"\n{'='*60}")
    print(f"NATIVE TPS RESULTS: {args.model_type}")
    print(f"  Mean TPS:  {mean_tps:.2f} tok/s")
    print(f"  P50 TPS:   {p50_tps:.2f} tok/s")
    print(f"  Mean E2E:  {mean_ms:.0f} ms")
    print(f"  P50 E2E:   {p50_ms:.0f} ms")
    print(f"{'='*60}")

    text = json.dumps(report, indent=2, ensure_ascii=False)

    out_path = args.output
    if not out_path:
        from speculative_decoding.report_paths import report_date_str
        out_path = os.path.join(
            _ROOT, "speculative_decoding", "results", report_date_str(),
            f"native_tps_{args.model_type}_bl{args.block_length}_ds{args.denoising_steps}"
            f"_nb{args.num_blocks}.json",
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
