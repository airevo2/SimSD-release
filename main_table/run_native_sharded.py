"""Target-only block diffusion with the target sharded across GPUs.

The baseline for `ours` when the target does not fit on one card. The existing
baselines (run_vanilla_tp2_cache.py / run_native_tp2_cache.py) hardcode
``--nproc_per_node=2`` and load with ``tp_plan="auto"``. LLaDA2.0-flash is
102.9 B params = 191.6 GiB in bf16, so TP=2 is 95.8 GiB per rank against 95.6
GiB of usable HBM — it misses by a hair, before any K/V or activations.
TP=4 would fit at ~48 GiB per rank, but the launcher does not expose the
world size yet.

The comparison this enables is deliberately placement-matched: `ours` runs its
target under naive model parallelism (layers split across devices, one GPU busy
at a time, accelerate moving activations across the boundary), so the reference
has to be the same model placed the same way. Timing a naively-sharded target
against a tensor-parallel one would mix the speculative-decoding effect with a
placement effect and attribute both to speculation.

Single process, no torchrun, no NCCL. Reuses
``cache_aware.native_generate_with_kv_cache``, so the per-block denoise, the
mask construction, the unmask schedule and the K/V cache are literally the same
code `ours` runs — only the draft and the acceptance rule are absent.

Run:
  CUDA_VISIBLE_DEVICES=<...> python main_table/run_native_sharded.py \
      --target_model <TARGET> --target_gpus 0,1,2,3 --output_dir <path>
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
from speculative_decoding.cache_aware import native_generate_with_kv_cache
from main_table.run_ours_dual_gpu import _load, _parse_gpu_list, _placement


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", required=True)
    p.add_argument("--target_device", default="cuda:0",
                   help="Ignored when --target_gpus is given.")
    p.add_argument("--target_gpus", default=None,
                   help="Shard across these LOCAL GPU indices, e.g. '0,1,2,3'. "
                        "Unset = load whole onto --target_device.")
    p.add_argument("--target_max_memory", default="88GiB")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--denoising_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--no_eos_stop", action="store_true", default=False)
    p.add_argument("--kv_cache_max_len", type=int, default=4096)
    p.add_argument("--draft_sampling", default="argmax",
                   choices=["argmax", "multinomial"])
    p.add_argument("--remasking_strategy", default="low_confidence_static",
                   choices=["low_confidence_static", "low_confidence_dynamic"])
    p.add_argument("--confidence_threshold", type=float, default=0.9)
    p.add_argument("--return_timings", action="store_true", default=False)
    p.add_argument("--datasets", nargs="+", default=["gsm8k"])
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_gpus = _parse_gpu_list(args.target_gpus)

    print(f"[init] target={target_gpus if target_gpus else args.target_device}",
          flush=True)
    torch.manual_seed(args.seed)
    t0 = time.time()
    model, n_rms = _load(args.target_model, torch.device(args.target_device),
                         shard_gpus=target_gpus,
                         max_memory_per_gpu=args.target_max_memory)
    print(f"[load] done in {time.time()-t0:.1f}s (RMSNorm patched: {n_rms})",
          flush=True)
    print(f"[place] target {_placement(model)}", flush=True)

    device = next(model.parameters()).device
    sharded = bool(getattr(model, "hf_device_map", None)) and len(
        {str(v) for v in model.hf_device_map.values()}) > 1

    if not hasattr(model.config, "block_size"):
        model.config.block_size = args.block_length

    tokenizer = AutoTokenizer.from_pretrained(args.target_model,
                                              trust_remote_code=True)
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id
    eos_ids = [eos_id] if eos_id is not None else None

    cfg = SimpleNamespace(
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        num_blocks=args.num_blocks,
        mask_token_id=args.mask_token_id,
        draft_sampling=args.draft_sampling,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        use_cuda_graph=False,          # sharded: capture cannot span devices
        kv_cache_max_len=args.kv_cache_max_len,
    )

    summary = {"runs": []}
    for dataset in args.datasets:
        ds_cfg = SimpleNamespace(dataset=dataset, dataset_split="test",
                                 num_samples=args.num_samples)
        ds, prompt_ids = load_prompts(tokenizer, ds_cfg)
        print(f"\n[{dataset}] loaded {len(prompt_ids)} prompts", flush=True)

        results = []
        for i, pids in enumerate(prompt_ids):
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            # Sharded work spans devices, so bracket it with a full sync at both
            # ends or the event pair on the entry device under-measures.
            if sharded:
                torch.cuda.synchronize()
            else:
                torch.cuda.synchronize(device)
            t_start = time.time()
            ev_start.record(stream=torch.cuda.current_stream(device))
            ret = native_generate_with_kv_cache(
                model, pids, cfg, eos_ids=eos_ids,
                return_timings=args.return_timings)
            gen_ids, timing = ret if args.return_timings else (ret, None)
            if sharded:
                torch.cuda.synchronize()
            ev_end.record(stream=torch.cuda.current_stream(device))
            ev_end.synchronize()
            gpu_ms = ev_start.elapsed_time(ev_end)
            dt = time.time() - t_start

            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            rec = {"idx": i, "prompt_len": len(pids), "n_tokens": len(gen_ids),
                   "end_to_end_s": dt, "gpu_event_ms": gpu_ms, "text": txt}
            if timing is not None:
                rec["timing"] = {k: v for k, v in timing.items()
                                 if k != "per_block"}
            results.append(rec)
            preview = txt.replace("\n", "⏎")[:60]
            print(f"  [{dataset} {i+1:>3}/{len(prompt_ids)}] "
                  f"{len(gen_ids):>3}t  {dt:.2f}s  gpu={gpu_ms:.0f}ms  "
                  f"{preview!r}", flush=True)

        # Throughput from cuda.Event; drop the first sample as warmup, matching
        # run_ours_dual_gpu.py so the two are directly comparable.
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

        summary["runs"].append({
            "dataset": dataset, "n": len(results), "n_timed": len(timed),
            "total_gpu_ms": total_gpu_ms, "total_tokens": total_tok,
            "ms_per_token": ms_per_tok, "tokens_per_second": tok_per_s,
            "pass_at_1": pass_at_1, "mean_accept_rate": None,
        })
        (out_dir / f"native_sharded_{dataset}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n")
        print(f"\n  [{dataset}] ms/tok={ms_per_tok:.2f}  tok/s={tok_per_s:.1f}  "
              f"pass@1={pass_at_1}", flush=True)

    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary at {out_dir / 'SUMMARY.json'}", flush=True)


if __name__ == "__main__":
    main()
