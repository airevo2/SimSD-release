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


def _parse_gpu_list(spec):
    """"1,2,3" -> [1, 2, 3]. Indices are LOCAL, i.e. after CUDA_VISIBLE_DEVICES."""
    if not spec:
        return None
    return [int(x) for x in str(spec).replace(" ", "").split(",") if x != ""]


def _load(model_path: str, device: torch.device, shard_gpus=None,
          max_memory_per_gpu: str = "88GiB"):
    """Load one model, either whole onto ``device`` or sharded across GPUs.

    Sharding exists for targets that do not fit on one card — LLaDA2.0-flash is
    191.6 GiB in bf16 against 95.6 GiB usable. ``device_map="auto"`` lays the layers
    out sequentially and accelerate hooks each one to move activations across
    the boundary.

    This is *naive* model parallelism, not tensor parallelism — and not pipeline
    parallelism either, despite the layer-wise split. Real PP splits the batch
    into micro-batches so several stages run at once; here a single input walks
    the layers strictly in order and exactly one GPU is busy at a time. At bs=1
    that costs nothing (there is nothing to pipeline), but it does mean the
    placement buys capacity, not speed: only TP parallelises within a layer.

    ``max_memory`` is restricted to ``shard_gpus`` so the planner cannot spill
    onto the draft's card. Layers stay whole — the checkpoint declares
    ``_no_split_modules = ["LLaDA2MoeDecoderLayer"]``.
    """
    kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=True)
    if shard_gpus:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {g: max_memory_per_gpu for g in shard_gpus}
        m = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(
            model_path, **kwargs).to(device).eval()
    n_rms = patch_stock_rms_norm(m)
    return m, n_rms


def _placement(model) -> str:
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return str(next(model.parameters()).device)
    per = {}
    for name, dev in dm.items():
        per.setdefault(str(dev), []).append(name)
    return "  ".join(f"cuda:{d}={len(v)}mod" for d, v in sorted(per.items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_model", required=True)
    p.add_argument("--target_model", required=True)
    p.add_argument("--draft_device", default="cuda:0")
    p.add_argument("--target_device", default="cuda:1",
                   help="Ignored when --target_gpus is given: the entry device "
                        "is then whatever accelerate puts the embedding on.")
    p.add_argument("--target_gpus", default=None,
                   help="Shard the target across these LOCAL GPU indices (after "
                        "CUDA_VISIBLE_DEVICES), e.g. '1,2,3'. For targets too "
                        "big for one card — LLaDA2.0-flash is 191.6 GiB bf16. "
                        "Unset = load the target whole onto --target_device.")
    p.add_argument("--target_max_memory", default="88GiB",
                   help="Per-GPU cap handed to accelerate's planner when "
                        "--target_gpus is set. Leave headroom for K/V + "
                        "activations; 88GiB of a 96 GB card is a safe default.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--num_blocks", type=int, default=32)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--no_eos_stop", action="store_true", default=False)
    p.add_argument("--use_cuda_graph", action="store_true", default=False)
    p.add_argument("--kv_cache_max_len", type=int, default=1024,
                   help="0 = auto: size to max_prompt_len + gen_length (rounded "
                        "to 64) per dataset. Every forward scans the WHOLE "
                        "buffer, so the fixed 1024 wastes bandwidth on short "
                        "prompts: draft denoise measures 35.7 ms/block at 1024 "
                        "vs 31.2 ms at 256 (scripts/33_extend_probe.py).")
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
    p.add_argument("--fold_draft_extend", action="store_true", default=False,
                   help="Fold the draft-side cache extend into the next fused "
                        "denoise graph (one fewer draft forward per block). "
                        "Requires --fused_denoise. See docs/optimize-extend.md.")
    p.add_argument("--speculative_target_extend", action="store_true", default=False,
                   help="Mirror the optimistic draft extend on the target "
                        "cache: speculatively extend target K/V before MRS "
                        "so target stream stays continuous (no idle wait "
                        "for MRS). Rolls back on reject like draft side.")
    p.add_argument("--fused_linear", default="off",
                   choices=("off", "draft", "target", "both"),
                   help="Route nn.Linear through the fused split-K GEMV kernel "
                        "(kernels/thin_linear_fused.py). At M=batch*block_length=4 "
                        "cuBLAS leaves most SMs idle; see "
                        "docs/kernel-optimization.md. Draft-side only by default "
                        "because the target model is already near DRAM peak. "
                        "NOT byte-identical to cuBLAS -- gate with "
                        "scripts/37_fused_linear_quality.sh.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    draft_device = torch.device(args.draft_device)
    target_device = torch.device(args.target_device)
    target_gpus = _parse_gpu_list(args.target_gpus)

    if target_gpus:
        if draft_device.index in target_gpus:
            raise SystemExit(
                f"--draft_device {args.draft_device} is inside --target_gpus "
                f"{target_gpus}; the target would evict the draft. Give the "
                f"draft a card of its own."
            )
        if args.use_cuda_graph:
            raise SystemExit(
                "--target_gpus (sharded target) cannot be combined with "
                "--use_cuda_graph: capture would have to span accelerate's "
                "cross-device copies, and StaticBlockCache puts every layer's "
                "K/V on a single device."
            )

    if args.pipeline:
        assert args.use_cuda_graph, "--pipeline requires --use_cuda_graph"
        assert args.draft_device != args.target_device, \
            "--pipeline requires draft_device != target_device"
    gen_fn = (speculative_generate_with_kv_cache_pipelined if args.pipeline
              else speculative_generate_with_kv_cache)
    print(f"[init] draft={args.draft_device}  "
          f"target={target_gpus if target_gpus else args.target_device}  "
          f"use_cuda_graph={args.use_cuda_graph}  pipeline={args.pipeline}  "
          f"entry={gen_fn.__name__}", flush=True)
    torch.manual_seed(args.seed)
    t0 = time.time()
    print(f"[load] draft  {args.draft_model}", flush=True)
    draft_model, n_d = _load(args.draft_model, draft_device)
    print(f"[load] target {args.target_model}"
          f"{f' sharded over {target_gpus} @ {args.target_max_memory}' if target_gpus else ''}",
          flush=True)
    target_model, n_t = _load(args.target_model, target_device,
                              shard_gpus=target_gpus,
                              max_memory_per_gpu=args.target_max_memory)
    print(f"[load] done in {time.time()-t0:.1f}s  "
          f"(RMSNorm patched: draft={n_d}  target={n_t})", flush=True)
    print(f"[place] draft  {_placement(draft_model)}", flush=True)
    print(f"[place] target {_placement(target_model)}", flush=True)

    # A sharded target's entry device is wherever accelerate put the embedding,
    # not what --target_device said. Everything downstream derives devices from
    # the model itself, so keep this in sync for the timing calls below.
    target_device = next(target_model.parameters()).device
    sharded = bool(getattr(target_model, "hf_device_map", None)) and len(
        {str(v) for v in target_model.hf_device_map.values()}) > 1

    if args.fused_linear != "off":
        # Must happen before any CUDA graph capture: capture freezes whichever
        # forward is installed at capture time.
        from kernels import fused_toggle
        for tag, m in (("draft", draft_model), ("target", target_model)):
            if args.fused_linear in (tag, "both"):
                n = fused_toggle.apply_to_model(m, True)
                fused_toggle.warmup(m, block_length=args.block_length)
                print(f"[fused_linear] {tag}: {n} nn.Linear -> split-K GEMV",
                      flush=True)

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
        fold_draft_extend=args.fold_draft_extend,
    )

    summary = {"runs": []}
    for dataset in args.datasets:
        ds_cfg = SimpleNamespace(
            dataset=dataset, dataset_split="test", num_samples=args.num_samples,
        )
        ds, prompt_ids = load_prompts(tokenizer, ds_cfg)
        if args.kv_cache_max_len == 0:
            need = max(len(x) for x in prompt_ids) + args.num_blocks * args.block_length
            cfg.kv_cache_max_len = -(-need // 64) * 64
        print(f"\n[{dataset}] loaded {len(prompt_ids)} prompts  "
              f"kv_cache_max_len={cfg.kv_cache_max_len}", flush=True)

        results = []
        for i, pids in enumerate(prompt_ids):
            plen = len(pids)
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            # Sharded: the work spans several devices, so an event pair on the
            # entry device only brackets it if every device is idle at both
            # ends. Sync all of them rather than just the entry one.
            if sharded:
                torch.cuda.synchronize()
            else:
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
            if sharded:
                torch.cuda.synchronize()
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
