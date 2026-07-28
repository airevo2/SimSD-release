#!/usr/bin/env python3
"""Throughput measurement under the *paper's* protocol (arXiv 2606.02544v1 §4.1).

Why this file exists
--------------------
`run_ours_dual_gpu.py` / `run_native_tp2_cache.py` time the whole ``gen_fn`` call
with one cuda.Event pair, and ``gen_fn`` prefills the prompt internally — so their
tok/s *includes* prompt prefill. The paper reports
    "Throughput is measured as decoded tokens per second, **excluding prompt
     prefilling**, with **3 warmup runs**" (§4.1),
with L = 512 (max gen length, EOS may stop earlier), denoising steps = response
length, batch size 1, KV cache, CUDA Graph on, 200 examples per dataset.

Measuring the same way here made the reproduction land ~10% low (see
rebuttal_plan.md §Phase 0). This runner implements the paper's protocol so new
rows (single-GPU ablation, other model families) can sit in the paper's tables.

How prefill is excluded without touching the engine
---------------------------------------------------
``cache_aware._prefill_prompt_static`` (and the eager ``_prefill_prompt``) are
wrapped so each call stamps a cuda.Event on the *clock device* right after it
returns. Prefill is the first thing every generate path does, so
    prefill_ms = ev_start .. (last prefill stamp)
    decode_ms  = total_gpu_ms - prefill_ms
Nothing in speculative_decoding/ is modified; the wrapper only records events
(no torch.cuda.synchronize), so cross-GPU pipeline overlap is preserved.

Methods (all single-process; no tensor parallel — TP baselines keep using
run_vanilla_tp2_cache.py / run_native_tp2_cache.py):
  native_single : target only, 1 GPU                      (ablation row a)
  simsd_single  : draft + target on the SAME GPU, no pipe  (ablation row b)
  simsd_dual    : draft + target on 2 GPUs, no pipe        (ablation row c)
  simsd_pipe    : draft + target on 2 GPUs, pipelined      (ablation row d = "ours")

Both tok/s variants are always written to SUMMARY.json
(`tokens_per_second` incl. prefill, `tokens_per_second_paper` excl. prefill) so
no number is silently redefined.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding import cache_aware
from speculative_decoding.cache_aware import (
    native_generate_with_kv_cache,
    speculative_generate_with_kv_cache,
    speculative_generate_with_kv_cache_pipelined,
)
from speculative_decoding.speculative_decode import load_prompts
from speculative_decoding.prompts_opencompass import (
    MAX_OUT_LEN as OC_MAX_OUT_LEN, SCORERS_OC, load_prompts_oc,
)
from speculative_decoding.Experiment_Backend.self_draft_compare import SCORERS
from sdar.run_native_tp2 import patch_stock_rms_norm

METHODS = ("native_single", "simsd_single", "simsd_dual", "simsd_pipe")

# ── prefill instrumentation ───────────────────────────────────────────────
# Set by the runner before each timed generate; the wrappers append one
# cuda.Event per prefill call (draft + target for spec paths).
_PREFILL_EVENTS: List[torch.cuda.Event] = []
_CLOCK_DEVICE: Optional[torch.device] = None


def _install_prefill_probe() -> None:
    """Wrap the prefill entry points so they stamp a cuda.Event on return.

    Idempotent. Records only — never synchronizes, so the pipelined path's
    draft/target overlap is unaffected.
    """
    if getattr(cache_aware, "_prefill_probe_installed", False):
        return
    for name in ("_prefill_prompt_static", "_prefill_prompt"):
        orig = getattr(cache_aware, name, None)
        if orig is None:
            continue

        def make(fn):
            def wrapped(*a, **kw):
                out = fn(*a, **kw)
                if _CLOCK_DEVICE is not None:
                    ev = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.device(_CLOCK_DEVICE):
                        ev.record(stream=torch.cuda.current_stream(_CLOCK_DEVICE))
                    _PREFILL_EVENTS.append(ev)
                return out
            return wrapped

        setattr(cache_aware, name, make(orig))
    cache_aware._prefill_probe_installed = True


def _load(path: str, device: torch.device, dtype: torch.dtype,
          shard_gpus=None, max_memory_per_gpu: str = "88GiB",
          fused_moe: bool = False):
    """Load one model, whole onto ``device`` or sharded across ``shard_gpus``.

    Sharding is required for LLaDA2.0-flash (191.6 GiB bf16 vs 95.6 GiB/card);
    see main_table/run_ours_dual_gpu.py for the placement notes. ``fused_moe``
    installs the grouped-GEMM expert dispatch, which is both ~2.5x faster and
    the precondition for any cuda_graph use on LLaDA2.
    """
    kwargs = dict(torch_dtype=dtype, trust_remote_code=True)
    if shard_gpus:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {g: max_memory_per_gpu for g in shard_gpus}
        m = AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(path, **kwargs).to(device).eval()
    patch_stock_rms_norm(m)
    if fused_moe:
        from kernels import fused_toggle
        n = fused_toggle.apply_moe_to_model(m, True)
        if n:
            print(f"[fused_moe] {n} MoE blocks -> grouped GEMM", flush=True)
    return m


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--target_model", default="JetLM/SDAR-8B-Chat")
    p.add_argument("--draft_model", default=None,
                   help="Required for every simsd_* method.")
    p.add_argument("--draft_device", default="cuda:0")
    p.add_argument("--target_device", default="cuda:1",
                   help="For simsd_single set this equal to --draft_device.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--datasets", nargs="+",
                   default=["gsm8k", "mbpp", "triviaqa", "mmlu"])
    p.add_argument("--num_samples", type=int, default=200,
                   help="Paper §4.1: 200 examples per dataset for speed.")
    # ── paper protocol knobs ──
    p.add_argument("--gen_length", type=int, default=512,
                   help="Paper §4.1: L=512 MAX gen length (EOS may stop earlier). "
                        "Use 0 with --prompts opencompass to take OpenCompass's "
                        "per-dataset max_out_len (gsm8k/mbpp 512, triviaqa 50).")
    p.add_argument("--prompts", default="repo", choices=["repo", "opencompass"],
                   help="'repo': speculative_decode.load_prompts (hand-written "
                        "0-shot). 'opencompass': the paper's stated setup — "
                        "4-shot gsm8k / 3-shot mbpp / OC's triviaqa+mmlu "
                        "templates, with OC-format scorers.")
    p.add_argument("--shuffle_seed", type=int, default=42,
                   help="Shuffle the dataset before taking --num_samples "
                        "(OpenCompass prompts only). 0 = the repo's original "
                        "first-N slice, which for MMLU means 100 "
                        "abstract_algebra + 100 anatomy out of 57 subjects.")
    p.add_argument("--mmlu_style", default="simple_evals",
                   choices=["simple_evals", "5shot"],
                   help="OpenCompass MMLU variant: 'simple_evals' is OC's "
                        "current default (0-shot CoT, 'ANSWER: $LETTER'); "
                        "'5shot' is the classic mmlu_gen_4d595a short answer.")
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4,
                   help="Paper: denoising steps = response length, i.e. ds=bl "
                        "(one position per step).")
    p.add_argument("--n_warmup", type=int, default=3,
                   help="Paper §4.1: 3 warmup runs -> drop first 3 samples from "
                        "the throughput aggregate (repo default elsewhere is 1).")
    p.add_argument("--no_eos_stop", action="store_true", default=False,
                   help="Off by default: L is a MAXIMUM in the paper, so EOS "
                        "stopping stays enabled. Turn on for fixed-length runs.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="Repo runners hardcode bfloat16; the paper text says FP32. "
                        "Exposed so the discrepancy can be measured, not guessed.")
    # ── method knobs (defaults = what the release's main.py uses for 'ours') ──
    p.add_argument("--draft_sampling", default="argmax",
                   choices=["argmax", "multinomial"])
    p.add_argument("--speculative_branch", default="greedy_match",
                   choices=["greedy_match", "mrs"],
                   help="Release default is greedy_match+argmax. The paper's Eq.(2) "
                        "is MRS, which requires multinomial drafting; pass "
                        "--speculative_branch mrs --draft_sampling multinomial "
                        "--temperature 1.0 for the paper's sampling setup.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--partial_block_fill", default="truncate",
                   choices=["truncate", "truncate_no_bonus", "draft_argmax",
                            "target_argmax", "target_argmax_all"])
    p.add_argument("--remasking_strategy", default="low_confidence_static",
                   choices=["low_confidence_static", "low_confidence_dynamic"])
    p.add_argument("--confidence_threshold", type=float, default=0.9)
    p.add_argument("--draft_steps_per_block", type=int, default=-1,
                   help="DRAFT EARLY-STOP -- this is NOT the repeat-until-block-"
                        "full gamma. The draft stops after k denoising steps and "
                        "the remaining bl-k positions are taken from the "
                        "target's argmax in the single verify forward and "
                        "auto-accepted, so the block still closes in ONE "
                        "speculation round. -1 (default) = no early stop, which "
                        "is what every run up to 2026-07-28 used. Known unfair "
                        "for throughput (4.66x at k=4): the target resolves "
                        "bl-k positions in one forward while the baseline is "
                        "forced to reveal one per forward. For the gamma that "
                        "re-drafts in gamma-wide chunks until the block is "
                        "full, see --spec_gamma.")
    p.add_argument("--fused_denoise", action="store_true", default=False)
    p.add_argument("--speculative_target_extend", action="store_true", default=False)
    p.add_argument("--fold_draft_extend", action="store_true", default=False,
                   help="Fold the draft-side cache extend into the fused denoise "
                        "graph (one fewer draft forward per block). "
                        "See docs/optimize-extend.md.")
    p.add_argument("--eos_min_prob", type=float, default=0.0,
                   help="Ignore an EOS that arrived as the MRS/greedy bonus token "
                        "unless the target puts at least this much probability on "
                        "EOS at that position. 0 = released behaviour. Fixes "
                        "premature termination that costs 12-17 accuracy points "
                        "on OpenCompass MMLU (see rebuttal_plan.md).")
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--kv_cache_max_len", type=int, default=0,
                   help="0 = auto: size the static cache to "
                        "max_prompt_len + gen_length (rounded to 64) per dataset. "
                        "Every forward scans the WHOLE buffer (the bool mask does "
                        "not skip slots), so the repo's fixed 1024 wastes "
                        "bandwidth on short prompts and cannot fit few-shot ones. "
                        "Pass a value to pin it (recorded in SUMMARY.json).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target_gpus", default=None,
                   help="Shard the target across these LOCAL GPU indices, e.g. "
                        "'1,2,3'. Required for LLaDA2.0-flash (191.6 GiB bf16).")
    p.add_argument("--target_max_memory", default="88GiB")
    p.add_argument("--no_cuda_graph", action="store_true", default=False,
                   help="Run eager. Mandatory when the target is sharded "
                        "(StaticBlockCache is single-device) and for LLaDA2 "
                        "unless --fused_moe is on.")
    p.add_argument("--fused_moe", action="store_true", default=False,
                   help="Install the grouped-GEMM MoE dispatch "
                        "(kernels/moe_dispatch_fused.py). LLaDA2 only; no-op "
                        "elsewhere. ~2.5x on the forward and removes the host "
                        "sync that blocks cuda_graph.")
    args = p.parse_args()

    if args.method.startswith("simsd") and not args.draft_model:
        p.error(f"--draft_model is required for {args.method}")
    if args.method == "simsd_single":
        args.target_device = args.draft_device
    if args.method == "native_single":
        args.draft_device = args.target_device

    if args.gen_length and args.gen_length % args.block_length:
        p.error(f"gen_length {args.gen_length} must be divisible by "
                f"block_length {args.block_length}")
    if not args.gen_length and args.prompts != "opencompass":
        p.error("--gen_length 0 (auto) only makes sense with "
                "--prompts opencompass")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    draft_device = torch.device(args.draft_device)
    target_device = torch.device(args.target_device)

    global _CLOCK_DEVICE
    _CLOCK_DEVICE = target_device
    _install_prefill_probe()

    print(f"[protocol] paper §4.1: L={args.gen_length} (max, EOS "
          f"{'off' if args.no_eos_stop else 'on'}), bl={args.block_length}, "
          f"ds={args.denoising_steps}, drop {args.n_warmup} warmup, "
          f"N={args.num_samples}/dataset, prefill EXCLUDED from tok/s", flush=True)
    print(f"[method] {args.method}  draft={args.draft_device} "
          f"target={args.target_device}  dtype={args.dtype}  "
          f"branch={args.speculative_branch}/{args.draft_sampling}", flush=True)

    torch.manual_seed(args.seed)
    t0 = time.time()
    tgt_gpus = ([int(x) for x in args.target_gpus.replace(" ", "").split(",") if x]
                if args.target_gpus else None)
    target_model = _load(args.target_model, target_device, dtype,
                         shard_gpus=tgt_gpus,
                         max_memory_per_gpu=args.target_max_memory,
                         fused_moe=args.fused_moe)
    if args.method == "native_single":
        draft_model = None
    elif args.draft_model == args.target_model:
        draft_model = target_model if draft_device == target_device else \
            _load(args.draft_model, draft_device, dtype, fused_moe=args.fused_moe)
    else:
        draft_model = _load(args.draft_model, draft_device, dtype,
                            fused_moe=args.fused_moe)
    if tgt_gpus:
        # accelerate decides where the embedding lands; everything downstream
        # derives devices from the model, so keep the clock device in sync.
        target_device = next(target_model.parameters()).device
        _CLOCK_DEVICE = target_device
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    for m in (target_model, draft_model):
        if m is not None and not hasattr(m.config, "block_size"):
            m.config.block_size = args.block_length

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    pad_id = tokenizer.pad_token_id or 0
    eos_ids = None if args.no_eos_stop else [tokenizer.eos_token_id]

    # num_blocks / kv_cache_max_len are set per dataset below (OpenCompass gives
    # each dataset its own max_out_len, and the static cache should be sized to
    # the actual prompts: every forward scans the WHOLE buffer, so an oversized
    # kv_cache_max_len costs bandwidth on every step).
    cfg = SimpleNamespace(
        K=1,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        num_blocks=0,
        mask_token_id=args.mask_token_id,
        draft_sampling=args.draft_sampling,
        speculative_branch=args.speculative_branch,
        mrs_verify_order="position",
        partial_block_fill=args.partial_block_fill,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_cuda_graph=not args.no_cuda_graph,
        kv_cache_max_len=args.kv_cache_max_len,
        fused_denoise=args.fused_denoise,
        speculative_target_extend=args.speculative_target_extend,
        eos_min_prob=args.eos_min_prob,
        fold_draft_extend=args.fold_draft_extend,
    )

    def generate(pids: List[int]):
        if args.method == "native_single":
            return native_generate_with_kv_cache(
                target_model, pids, cfg, eos_ids=eos_ids), None
        fn = (speculative_generate_with_kv_cache_pipelined
              if args.method == "simsd_pipe" else
              speculative_generate_with_kv_cache)
        gen_ids, stats = fn(
            draft_model, target_model, pids, cfg,
            pad_token_id=pad_id,
            padded_len=len(pids) + cfg.num_blocks * args.block_length,
            eos_ids=eos_ids,
        )
        return gen_ids, stats

    summary = {"protocol": "arXiv 2606.02544v1 §4.1",
               "prompts": args.prompts, "config": vars(args), "runs": []}
    for dataset in args.datasets:
        ds_cfg = SimpleNamespace(dataset=dataset, dataset_split="test",
                                 num_samples=args.num_samples,
                                 mmlu_style=args.mmlu_style,
                                 shuffle_seed=args.shuffle_seed)
        if args.prompts == "opencompass":
            ds, prompt_ids = load_prompts_oc(tokenizer, ds_cfg)
            key = "mmlu_5shot" if (dataset == "mmlu" and
                                   args.mmlu_style == "5shot") else dataset
            gen_len = args.gen_length or OC_MAX_OUT_LEN[key]
            scorer = SCORERS_OC.get(dataset)
        else:
            ds, prompt_ids = load_prompts(tokenizer, ds_cfg)
            gen_len = args.gen_length
            scorer = SCORERS.get(dataset)

        bl = args.block_length
        cfg.num_blocks = -(-gen_len // bl)          # ceil: OC's 50 -> 13 blocks
        max_prompt = max(len(x) for x in prompt_ids)
        need = max_prompt + cfg.num_blocks * bl
        cfg.kv_cache_max_len = (args.kv_cache_max_len if args.kv_cache_max_len
                                else -(-need // 64) * 64)
        if cfg.kv_cache_max_len < need:
            raise SystemExit(f"--kv_cache_max_len {cfg.kv_cache_max_len} < "
                             f"needed {need} for {dataset}")
        print(f"\n[{dataset}] {len(prompt_ids)} prompts "
              f"(prompt_len max={max_prompt}, mean="
              f"{sum(len(x) for x in prompt_ids)/len(prompt_ids):.0f})  "
              f"gen<={cfg.num_blocks * bl} ({cfg.num_blocks} blocks)  "
              f"kv_cache_max_len={cfg.kv_cache_max_len}  prompts={args.prompts}",
              flush=True)

        results = []
        for i, pids in enumerate(prompt_ids):
            _PREFILL_EVENTS.clear()
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(target_device)
            with torch.cuda.device(target_device):
                ev_start.record(stream=torch.cuda.current_stream(target_device))
            wall0 = time.time()
            gen_ids, stats = generate(pids)
            with torch.cuda.device(target_device):
                ev_end.record(stream=torch.cuda.current_stream(target_device))
            ev_end.synchronize()
            wall = time.time() - wall0

            total_ms = ev_start.elapsed_time(ev_end)
            # Prefill window = start .. last prefill stamp (prefill is the first
            # thing every generate path does; spec paths stamp draft then target).
            prefill_ms = (ev_start.elapsed_time(_PREFILL_EVENTS[-1])
                          if _PREFILL_EVENTS else 0.0)
            decode_ms = total_ms - prefill_ms

            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            rec = {"idx": i, "prompt_len": len(pids), "n_tokens": len(gen_ids),
                   "gpu_event_ms": total_ms, "prefill_gpu_ms": prefill_ms,
                   "decode_gpu_ms": decode_ms, "end_to_end_s": wall, "text": txt}
            if stats is not None:
                a = stats.get("total_accepted_tokens", 0) or 0
                d = stats.get("total_draft_tokens", 1) or 1
                rec["accept_rate"] = a / d if d else 0.0
                rec["total_accepted_tokens"] = int(a)
                rec["total_draft_tokens"] = int(d)
                rec["total_bonus_tokens"] = int(stats.get("total_bonus_tokens", 0) or 0)
                for k in ("eos_stop_from_bonus", "eos_stop_from_accepted",
                          "dropped_eos_bonus"):
                    if stats.get(k):
                        rec[k] = int(stats[k])
                if stats.get("eos_stop_q"):
                    rec["eos_stop_q"] = [round(x, 4) for x in stats["eos_stop_q"]]
            results.append(rec)
            tag = f" α={rec['accept_rate']:.3f}" if "accept_rate" in rec else ""
            print(f"  [{dataset} {i+1:>3}/{len(prompt_ids)}] {len(gen_ids):>3}t  "
                  f"decode={decode_ms:6.0f}ms  prefill={prefill_ms:5.0f}ms"
                  f"{tag}  {txt[:44]!r}".replace("\n", "⏎"), flush=True)

        timed = results[args.n_warmup:] if len(results) > args.n_warmup else results
        tok = sum(r["n_tokens"] for r in timed)
        dec_ms = sum(r["decode_gpu_ms"] for r in timed)
        tot_ms = sum(r["gpu_event_ms"] for r in timed)
        pre_ms = sum(r["prefill_gpu_ms"] for r in timed)

        n_pass = sum(1 for r, ref in zip(results, ds)
                     if scorer is not None and scorer(r["text"], ref)[0])
        alphas = [r["accept_rate"] for r in results if "accept_rate" in r]

        run = {
            "dataset": dataset, "n": len(results), "n_timed": len(timed),
            "prompts": args.prompts, "gen_length": cfg.num_blocks * bl,
            "num_blocks": cfg.num_blocks,
            "kv_cache_max_len": cfg.kv_cache_max_len,
            "max_prompt_len": max_prompt,
            "n_warmup_dropped": min(args.n_warmup, len(results)),
            "total_tokens": tok,
            "total_decode_gpu_ms": dec_ms, "total_gpu_ms": tot_ms,
            "total_prefill_gpu_ms": pre_ms,
            "prefill_share": pre_ms / tot_ms if tot_ms else None,
            # Paper protocol: prefill excluded.
            "tokens_per_second_paper": tok * 1000.0 / dec_ms if dec_ms else None,
            "ms_per_token_paper": dec_ms / tok if tok else None,
            # Repo-legacy definition, kept so old numbers stay comparable.
            "tokens_per_second": tok * 1000.0 / tot_ms if tot_ms else None,
            "ms_per_token": tot_ms / tok if tok else None,
            "pass_at_1": n_pass / len(results) if results else None,
            "mean_accept_rate": sum(alphas) / len(alphas) if alphas else None,
        }
        summary["runs"].append(run)
        (out_dir / f"{args.method}_{dataset}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results) + "\n")
        print(f"\n  [{dataset}] tok/s(paper, excl prefill)="
              f"{run['tokens_per_second_paper']:.1f}  "
              f"tok/s(incl prefill)={run['tokens_per_second']:.1f}  "
              f"prefill={run['prefill_share']*100:.1f}%  "
              f"pass@1={run['pass_at_1']}  α={run['mean_accept_rate']}", flush=True)

    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] {out_dir / 'SUMMARY.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
