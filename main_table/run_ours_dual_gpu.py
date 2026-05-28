"""SimSD speculative (draft + target + MRS) on dual GPUs, KV cache + cuda_graph.

Single-process: draft on cuda:0, target on cuda:1. No tensor parallel — each
model lives fully on one card. Mirrors run_native_tp2_cache.py's output schema
so main.py can aggregate all three methods (vanilla / vanilla+CG / ours)
uniformly.

Run:
  CUDA_VISIBLE_DEVICES=<a>,<b> python main_table/run_ours_dual_gpu.py \
      --draft_model <DRAFT> --target_model <TARGET> \
      --output_dir <path> [--use_cuda_graph]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding.speculative_decode import load_prompts
from speculative_decoding.Experiment_Backend.self_draft_compare import SCORERS
from speculative_decoding.cache_aware import (
    speculative_generate_with_kv_cache,
    speculative_generate_with_kv_cache_pipelined,
)
from sdar.run_native_tp2 import patch_stock_rms_norm


def _load(model_path: str, device: torch.device):
    m = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    n_rms = patch_stock_rms_norm(m)
    return m, n_rms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_model", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--draft_device", default="cuda:0")
    p.add_argument("--target_device", default="cuda:1")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--num_blocks", type=int, default=32)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--no_eos_stop", action="store_true", default=False)
    p.add_argument("--use_cuda_graph", action="store_true", default=False)
    p.add_argument("--kv_cache_max_len", type=int, default=1024)
    p.add_argument("--draft_sampling", default="argmax",
                   choices=["argmax", "multinomial"])
    p.add_argument("--speculative_branch", default="greedy_match",
                   choices=["greedy_match", "mrs"],
                   help="greedy_match: argmax-equivalent lossless rule. "
                        "mrs: stochastic Modified Rejection Sampling.")
    p.add_argument("--mrs_verify_order", default="position",
                   choices=["position", "step_map"])
    p.add_argument("--partial_block_fill", default="target_argmax",
                   choices=["truncate", "truncate_no_bonus", "draft_argmax",
                            "target_argmax", "target_argmax_all"],
                   help="Default 'target_argmax' per plan/NAMING.md "
                        "(post 2026-05-06).")
    p.add_argument("--remasking_strategy", default="low_confidence_static",
                   choices=["low_confidence_static", "low_confidence_dynamic"])
    p.add_argument("--confidence_threshold", type=float, default=0.9)
    p.add_argument("--datasets", nargs="+",
                   default=["gsm8k", "humaneval", "mbpp"])
    p.add_argument("--pipeline", action="store_true", default=False,
                   help="Use speculative_generate_with_kv_cache_pipelined "
                        "(draft on cuda:0 + target on cuda:1 overlap). "
                        "Requires use_cuda_graph=True and distinct devices.")
    p.add_argument("--return_timings", action="store_true", default=False,
                   help="Capture per-stage timing breakdown (draft denoise / "
                        "verify / extend / MRS-commit GPU + wall) from gen_fn "
                        "and aggregate per dataset into SUMMARY.json.")
    p.add_argument("--fused_denoise", action="store_true", default=False,
                   help="Fuse all denoising_steps denoise iterations into a "
                        "single cuda_graph (saves 3x dispatch + Python loop "
                        "overhead). Requires argmax + low_confidence_static "
                        "+ no early-stop. Cached per n_known.")
    p.add_argument("--speculative_target_extend", action="store_true", default=False,
                   help="Mirror the optimistic draft extend on the target "
                        "cache: speculatively extend target K/V before MRS "
                        "so target stream stays continuous (no idle wait "
                        "for MRS). Rolls back on reject like draft side.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_device = torch.device(args.draft_device)
    target_device = torch.device(args.target_device)

    if args.pipeline:
        assert args.use_cuda_graph, "--pipeline requires --use_cuda_graph"
        assert args.draft_device != args.target_device, \
            "--pipeline requires draft_device != target_device"
    gen_fn = (speculative_generate_with_kv_cache_pipelined if args.pipeline
              else speculative_generate_with_kv_cache)
    print(f"[init] draft={args.draft_device}  target={args.target_device}  "
          f"use_cuda_graph={args.use_cuda_graph}  pipeline={args.pipeline}  "
          f"entry={gen_fn.__name__}", flush=True)
    torch.manual_seed(args.seed)
    t0 = time.time()
    print(f"[load] draft  {args.draft_model}", flush=True)
    draft_model, n_d = _load(args.draft_model, draft_device)
    print(f"[load] target {args.target_model}", flush=True)
    target_model, n_t = _load(args.target_model, target_device)
    print(f"[load] done in {time.time()-t0:.1f}s  "
          f"(RMSNorm patched: draft={n_d}  target={n_t})", flush=True)

    for m in (draft_model, target_model):
        if not hasattr(m.config, "block_size"):
            m.config.block_size = args.block_length

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or 0
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id

    cfg = SimpleNamespace(
        K=1,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        num_blocks=args.num_blocks,
        mask_token_id=args.mask_token_id,
        draft_sampling=args.draft_sampling,
        speculative_branch=args.speculative_branch,
        mrs_verify_order=args.mrs_verify_order,
        partial_block_fill=args.partial_block_fill,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        use_cuda_graph=args.use_cuda_graph,
        kv_cache_max_len=args.kv_cache_max_len,
        fused_denoise=args.fused_denoise,
        speculative_target_extend=args.speculative_target_extend,
    )

    summary = {"runs": []}
    for dataset in args.datasets:
        ds_cfg = SimpleNamespace(
            dataset=dataset, dataset_split="test", num_samples=args.num_samples,
        )
        ds, prompt_ids = load_prompts(tokenizer, ds_cfg)
        print(f"\n[{dataset}] loaded {len(prompt_ids)} prompts", flush=True)

        results = []
        for i, pids in enumerate(prompt_ids):
            plen = len(pids)
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(target_device)
            t_start = time.time()
            ev_start.record(stream=torch.cuda.current_stream(target_device))
            eos_ids_arg = [eos_id] if eos_id is not None else None
            ret = gen_fn(
                draft_model, target_model, pids, cfg,
                pad_token_id=pad_id, padded_len=plen + args.num_blocks * args.block_length,
                eos_ids=eos_ids_arg, return_timings=args.return_timings,
            )
            if args.return_timings:
                gen_ids, stats, timing = ret
            else:
                gen_ids, stats = ret
                timing = None
            ev_end.record(stream=torch.cuda.current_stream(target_device))
            ev_end.synchronize()
            gpu_ms = ev_start.elapsed_time(ev_end)
            dt = time.time() - t_start

            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            a = stats.get("total_accepted_tokens", 0) or 0
            d = stats.get("total_draft_tokens", 1) or 1
            alpha = a / d if d else 0.0
            rec = {
                "idx": i, "prompt_len": plen, "n_tokens": len(gen_ids),
                "end_to_end_s": dt, "gpu_event_ms": gpu_ms, "text": txt,
                "spec_acceptance": {
                    "total_accepted_tokens": int(a),
                    "total_draft_tokens": int(d),
                    "total_bonus_tokens": int(stats.get("total_bonus_tokens", 0) or 0),
                    "accept_rate": alpha,
                },
            }
            if timing is not None:
                rec["timing"] = {k: v for k, v in timing.items() if k != "per_block"}
            results.append(rec)
            preview = txt.replace("\n", "⏎")[:60]
            print(f"  [{dataset} {i+1:>3}/{len(prompt_ids)}] {len(gen_ids):>3}t  "
                  f"{dt:.2f}s  gpu={gpu_ms:.0f}ms  α={alpha:.2f}  {preview!r}",
                  flush=True)

        # Throughput uses gpu_event_ms (cuda.Event); drop first sample as warmup.
        timed = results[1:] if len(results) > 1 else results
        total_gpu_ms = sum(r["gpu_event_ms"] for r in timed)
        total_tok = sum(r["n_tokens"] for r in timed)
        ms_per_tok = total_gpu_ms / total_tok if total_tok else None
        tok_per_s = total_tok * 1000.0 / total_gpu_ms if total_gpu_ms else None

        scorer = SCORERS.get(dataset)
        n_pass = 0
        for r, ref in zip(results, ds):
            if scorer is not None:
                ok_, _ = scorer(r["text"], ref)
                n_pass += int(ok_)
        pass_at_1 = n_pass / len(results) if results else None

        accept_rates = [r["spec_acceptance"]["accept_rate"] for r in results]
        mean_alpha = sum(accept_rates) / len(accept_rates) if accept_rates else None

        run = {
            "dataset": dataset, "n": len(results), "n_timed": len(timed),
            "total_gpu_ms": total_gpu_ms, "total_tokens": total_tok,
            "ms_per_token": ms_per_tok, "tokens_per_second": tok_per_s,
            "pass_at_1": pass_at_1, "mean_accept_rate": mean_alpha,
        }
        if args.return_timings and timed and timed[0].get("timing"):
            stage_keys = [k for k in timed[0]["timing"].keys()
                          if isinstance(timed[0]["timing"][k], (int, float))]
            stage_totals = {}
            for k in stage_keys:
                tot = sum(r["timing"].get(k, 0.0) for r in timed)
                stage_totals[f"sum_{k}"] = tot
                stage_totals[f"mean_per_sample_{k}"] = tot / len(timed)
                if total_tok:
                    unit_to_ms = 1000.0 if k.endswith("_s") else 1.0
                    stage_totals[f"ms_per_token_{k}"] = tot * unit_to_ms / total_tok
            run["timing_breakdown"] = stage_totals
        summary["runs"].append(run)
        (out_dir / f"ours_dual_gpu_{dataset}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n"
        )
        print(f"\n  [{dataset}] ms/tok={ms_per_tok:.2f}  tok/s={tok_per_s:.1f}  "
              f"pass@1={pass_at_1}  α={mean_alpha:.3f}", flush=True)

    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary at {out_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
