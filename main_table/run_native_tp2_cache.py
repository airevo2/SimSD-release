"""TP=2 native 8B with KV cache (+ optional cuda_graph) on gsm8k / humaneval / mbpp.

Reuses cache_aware.native_generate_with_kv_cache for the per-block denoise +
extend logic. Each rank runs the same code path with tp_plan='auto' sharding;
since the model's lm_head output is replicated (regular Tensor, not DTensor),
argmax sampling is deterministic across ranks  no broadcast of input_ids/
sampled tokens needed. Only EOS termination is broadcast.

Run:
  CUDA_VISIBLE_DEVICES=<a>,<b> torchrun --nproc_per_node=2 --master_port=29560 \
      run_native_tp2_cache.py --output_dir <path> [--use_cuda_graph]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding.speculative_decode import load_prompts
from speculative_decoding.Experiment_Backend.self_draft_compare import SCORERS
from speculative_decoding.cache_aware import (
    native_generate_with_kv_cache,
    patch_sdpa_with_cache,
    patch_sdpa_static_cache,
    patch_causallm_pass_cache,
)
from sdar.run_native_tp2 import patch_stock_rms_norm


def _bcast_long(val: int, device, src: int = 0) -> int:
    t = torch.tensor([val], dtype=torch.long, device=device)
    dist.broadcast(t, src=src)
    return int(t.item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", required=True,
                   help="HF model id or local path to SDAR target checkpoint.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--num_blocks", type=int, default=32)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--no_eos_stop", action="store_true", default=False)
    p.add_argument("--use_cuda_graph", action="store_true", default=False,
                   help="Wrap denoise/extend forwards in cuda_graph; needs "
                        "NCCL-in-graph support (torch 2.7+ + nccl-cu12 >= 2.27).")
    p.add_argument("--breakdown", action="store_true", default=False,
                   help="Per-sample cuda.Event GPU wall + per-block "
                        "denoise/extend breakdown via cache_aware return_timings.")
    p.add_argument("--kv_cache_max_len", type=int, default=1024,
                   help="StaticBlockCache permanent slot size; smaller cuts "
                        "redundant K/V scan (caps prompt + gen length).")
    p.add_argument("--datasets", nargs="+",
                   default=["gsm8k", "humaneval", "mbpp"])
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    is_main = (rank == 0)
    device = torch.device(f"cuda:{rank}")

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    def log(s):
        if is_main:
            print(s, flush=True)

    log(f"[init] world={world}  cuda:{rank}  use_cuda_graph={args.use_cuda_graph}")
    log(f"[load] {args.target_model} with tp_plan='auto'")
    t0 = time.time()
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model, tp_plan="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).eval()
    log(f"[load] done in {time.time()-t0:.1f}s")

    n_rms = patch_stock_rms_norm(model)
    log(f"[patch] {n_rms} SDARRMSNorm replaced with stock impl")

    if args.use_cuda_graph:
        ok = patch_sdpa_static_cache(model)
        log(f"[patch] patch_sdpa_static_cache: {ok}")
    else:
        ok = patch_sdpa_with_cache(model)
        log(f"[patch] patch_sdpa_with_cache: {ok}")
    pass_ok = patch_causallm_pass_cache(model)
    log(f"[patch] patch_causallm_pass_cache: {pass_ok}")

    if not hasattr(model.config, "block_size"):
        model.config.block_size = args.block_length

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id

    # Sharded n_kv_heads / n_q_heads under tp_plan="auto": each rank holds
    # heads // world. StaticBlockCache (post 2026-05-05) stores n_q_heads
    # (GQA-expanded layout) so the SDPA call can pick EFFICIENT_ATTENTION
    # instead of MATH-fallback (~1.8× faster). Without explicit override it
    # would allocate the FULL head count on every rank  src/dst dim mismatch.
    n_kv_full = getattr(model.config, "num_key_value_heads",
                        model.config.num_attention_heads)
    n_q_full = model.config.num_attention_heads
    n_kv_sharded = n_kv_full // world
    n_q_sharded = n_q_full // world
    log(f"[tp] n_kv_heads={n_kv_full}/{n_kv_sharded}  n_q_heads={n_q_full}/{n_q_sharded} (full/per-rank)")

    # cfg compatible with native_generate_with_kv_cache
    cfg = SimpleNamespace(
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        num_blocks=args.num_blocks,
        mask_token_id=args.mask_token_id,
        draft_sampling="argmax",          # argmax  deterministic across ranks
        use_cuda_graph=args.use_cuda_graph,
        kv_cache_max_len=args.kv_cache_max_len,
        n_kv_heads_override=n_kv_sharded,
        n_q_heads_override=n_q_sharded,
    )

    summary = {"runs": []}
    for dataset in args.datasets:
        if is_main:
            ds_cfg = SimpleNamespace(
                dataset=dataset, dataset_split="test",
                num_samples=args.num_samples,
            )
            ds, prompt_ids = load_prompts(tokenizer, ds_cfg)
            log(f"\n[{dataset}] loaded {len(prompt_ids)} prompts")
        else:
            ds, prompt_ids = None, None

        n_prompts = _bcast_long(len(prompt_ids) if is_main else 0, device, src=0)

        results = []
        for i in range(n_prompts):
            # Sync prompt to all ranks (so each rank runs identical code path).
            if is_main:
                pids = prompt_ids[i]
                plen = len(pids)
            else:
                pids = None
                plen = 0
            plen = _bcast_long(plen, device, src=0)
            if is_main:
                pid_t = torch.tensor(pids, dtype=torch.long, device=device)
            else:
                pid_t = torch.zeros(plen, dtype=torch.long, device=device)
            dist.broadcast(pid_t, src=0)
            pids_local = pid_t.tolist()

            # cuda.Event measures pure GPU wall (no implicit CPU↔GPU sync
            # between forwards). cuda.synchronize() before t_start drains
            # any pending work so the timing window starts clean.
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t_start = time.time()
            ev_start.record()
            eos_ids_arg = [eos_id] if eos_id is not None else None
            ret = native_generate_with_kv_cache(
                model, pids_local, cfg, eos_ids=eos_ids_arg,
                return_timings=args.breakdown,
            )
            if args.breakdown:
                gen_ids, timing = ret
            else:
                gen_ids, timing = ret, None
            ev_end.record()
            ev_end.synchronize()
            gpu_ms = ev_start.elapsed_time(ev_end)
            dt = time.time() - t_start

            if is_main:
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
                row = {"idx": i, "prompt_len": plen, "n_tokens": len(gen_ids),
                       "end_to_end_s": dt, "gpu_event_ms": gpu_ms, "text": txt}
                if timing is not None:
                    row["breakdown"] = {
                        "prefill_s": timing.get("prefill_s", 0.0),
                        "total_denoise_s": timing.get("total_denoise_s", 0.0),
                        "total_commit_extend_s": timing.get("total_commit_extend_s", 0.0),
                        "total_block_wall_s": timing.get("total_block_wall_s", 0.0),
                    }
                results.append(row)
                preview = txt.replace("\n", "⏎")[:60]
                bd_str = ""
                if timing is not None:
                    bd = row["breakdown"]
                    bd_str = (f"  prefill={bd['prefill_s']*1000:.0f}ms"
                              f"  denoise={bd['total_denoise_s']*1000:.0f}ms"
                              f"  extend={bd['total_commit_extend_s']*1000:.0f}ms")
                print(f"  [{dataset} {i+1:>3}/{n_prompts}] {len(gen_ids):>3}t  "
                      f"{dt:.2f}s  gpu={gpu_ms:.0f}ms{bd_str}  {preview!r}", flush=True)

        if is_main:
            # Throughput uses gpu_event_ms (cuda.Event) — pure GPU wall, no
            # CPU↔GPU sync noise. Drop first sample as warmup.
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

            run = {"dataset": dataset, "n": len(results),
                   "n_timed": len(timed),
                   "total_gpu_ms": total_gpu_ms, "total_tokens": total_tok,
                   "ms_per_token": ms_per_tok, "tokens_per_second": tok_per_s,
                   "pass_at_1": pass_at_1}
            summary["runs"].append(run)
            (out_dir / f"native_tp2_cache_{dataset}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in results) + "\n"
            )
            print(f"\n  [{dataset}] ms/tok={ms_per_tok:.2f}  tok/s={tok_per_s:.1f}  pass@1={pass_at_1}", flush=True)

    if is_main:
        (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[done] summary at {out_dir / 'SUMMARY.json'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
