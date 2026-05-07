#!/usr/bin/env python3
"""
Multi-block Speculative (draft+target+MRS) vs Native


  -  Target Native/Speculative  target  checkpoint
  -  speculative_decode.speculative_generate(return_timings=True)  draft_one_block

 dllm_causal_train :

  python speculative_decoding/bench/run_benchmark.py \\
    --compare both --num_samples 20 --num_blocks 8 \\
    --draft_model inference/model/SDAR-4B-Chat \\
    --target_model inference/model/SDAR-8B-Chat \\
    --draft_device cuda:1 --target_device cuda:1 \\
    #  speculative_decoding/results/{}/bench_latency_{tag}.json

 Target draft:

  python speculative_decoding/bench/run_benchmark.py --compare native \\
    --target_model /path/to/SDAR-XXB-Chat --num_blocks 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

#  speculative_decode  dllm_causal_train
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch._dynamo

# Shim: transformers/utils removed LossKwargs after recent installs/edits;
# the cached SDAR modeling code still imports it. Provide an empty TypedDict
# stub so the import doesn't explode. Functionally a no-op for inference.
import transformers.utils as _tu
if not hasattr(_tu, "LossKwargs"):
    from typing import TypedDict
    class LossKwargs(TypedDict, total=False): pass
    _tu.LossKwargs = LossKwargs

from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative_decoding.config import REPO_ROOT, SpecConfig, default_config, load_config


def _resolve_local_model_path(path: str) -> str:
    """YAML  inference/model/...  dllm_causal_train (REPO_ROOT)"""
    if not path:
        return path
    if os.path.isdir(path):
        return os.path.abspath(path)
    cand = os.path.join(REPO_ROOT, path)
    if os.path.isdir(cand):
        return os.path.abspath(cand)
    return path
from speculative_decoding.report_paths import report_date_str
from speculative_decoding.speculative_decode import load_prompts
from speculative_decoding.verify import patch_multi_block_mask_fn

from speculative_decoding.bench.backends import (
    HFNativeBackend,
    HFSpeculativeBackend,
    JetEngineNativeBackend,
    compute_verify_padded_len,
)
from speculative_decoding.bench.timing import summarize_latency_ms, summarize_ms, summarize_scalar


def _load_models(args, dtype: torch.dtype, load_draft: bool):
    """load_draft=False  Targetnative-only tokenizer  target """
    tok_path = args.draft_model if load_draft else args.target_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    draft_dev = torch.device(args.draft_device)
    target_dev = torch.device(args.target_device)

    draft = None
    if load_draft:
        draft = AutoModelForCausalLM.from_pretrained(
            args.draft_model, trust_remote_code=True, torch_dtype=dtype,
        ).to(draft_dev).eval()

    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, trust_remote_code=True, torch_dtype=dtype,
    ).to(target_dev).eval()
    #  patchNative baseline  create_multi_block_causal_mask Target
    # Speculative  patch main

    # Public JetLM/SDAR-*-Chat config.json does not ship `block_size`, but
    # speculative_decoding/verify.py expects model.config.block_size to exist
    # (saves/restores it around the multi-block forward). Seed it with
    # denoising_steps. See plan/10 §1 setup gotchas.
    ds = getattr(args, "denoising_steps", None) or 4
    if not hasattr(target.config, "block_size"):
        target.config.block_size = ds
    if draft is not None and not hasattr(draft.config, "block_size"):
        draft.config.block_size = ds
    return tokenizer, draft, target


def _run_backend_on_prompts(
    backend,
    all_prompt_ids: List[List[int]],
    warmup: int,
    batch: int = 1,
) -> Dict[str, Any]:
    """ +

    batch=1 (default): per-prompt loop, calls ``backend.generate``.
    batch>1: chunks ``all_prompt_ids`` into groups of ``batch`` and calls
    ``backend.generate_batch``. The first chunk is used as warmup; subsequent
    chunks are timed. ``end_to_end_s`` is the **batch wall-clock** shared by
    all rows in a chunk (so per-row tok/s reflects shared latency, not
    individual-row latency  that's the point of batching).
    """
    e2e_times: List[float] = []
    token_counts: List[int] = []
    draft_totals: List[float] = []
    verify_totals: List[float] = []
    rows: List[Dict[str, Any]] = []
    #  ( speculative ) res.stats
    #  res.stats  backend.generate()  end_to_end_s  (t1 = perf_counter())
    #  post-hoc  Python
    # end_to_end_ms / throughput  GPU sync
    all_per_block_accepted: List[int] = []  # flatten  block
    total_draft_tokens = 0
    total_accepted_tokens = 0
    total_bonus_tokens = 0

    if batch <= 1:
        iter_units = [(i, [p]) for i, p in enumerate(all_prompt_ids)]
    else:
        iter_units = []
        for chunk_start in range(0, len(all_prompt_ids), batch):
            chunk = all_prompt_ids[chunk_start:chunk_start + batch]
            iter_units.append((chunk_start, chunk))

    for unit_idx, (start_idx, chunk) in enumerate(iter_units):
        backend.reset_runtime_state()
        is_warmup = unit_idx < warmup

        if batch <= 1:
            p = chunk[0]
            if is_warmup:
                backend.generate(p, return_timings=True)
                continue
            res_list = [backend.generate(p, return_timings=True)]
            chunk_indices = [start_idx]
        else:
            if is_warmup:
                backend.generate_batch(chunk, return_timings=True)
                continue
            res_list = backend.generate_batch(chunk, return_timings=True)
            chunk_indices = list(range(start_idx, start_idx + len(chunk)))

        # Aggregate timing across the chunk. For batch>1 every row in a chunk
        # shares the same end_to_end_s (batch wall-clock); we record it once
        # per chunk into e2e_times, then per-row into rows[] for downstream
        # ms/tok analysis.
        chunk_walls_seen = set()
        for res, idx_in_all in zip(res_list, chunk_indices):
            assert res.end_to_end_s is not None
            wall = float(res.end_to_end_s)
            if wall not in chunk_walls_seen:
                e2e_times.append(wall)
                chunk_walls_seen.add(wall)
            token_counts.append(len(res.generated_ids))

            p = all_prompt_ids[idx_in_all]
            row: Dict[str, Any] = {
                "idx": idx_in_all,
                "prompt_len": len(p),
                "gen_len": len(res.generated_ids),
                "end_to_end_s": res.end_to_end_s,
                "batch": batch,
            }
            if res.timing:
                row["timing"] = res.timing
                if backend.mode == "speculative":
                    draft_totals.append(res.timing.get("total_draft_s", 0.0))
                    verify_totals.append(res.timing.get("total_target_verify_s", 0.0))
                elif backend.mode == "native":
                    draft_totals.append(res.timing.get("total_block_wall_s", 0.0))

            # Acceptance  stats timing
            if backend.mode == "speculative" and res.stats:
                pb = res.stats.get("per_block_accepted", []) or []
                td = int(res.stats.get("total_draft_tokens", 0) or 0)
                ta = int(res.stats.get("total_accepted_tokens", 0) or 0)
                tb = int(res.stats.get("total_bonus_tokens", 0) or 0)
                all_per_block_accepted.extend(pb)
                total_draft_tokens += td
                total_accepted_tokens += ta
                total_bonus_tokens += tb
                row["accept"] = {
                    "per_block_accepted": pb,
                    "total_draft_tokens": td,
                    "total_accepted_tokens": ta,
                    "total_bonus_tokens": tb,
                    "accept_rate": (ta / td) if td > 0 else 0.0,
                }
            rows.append(row)

    # Throughput: per-row tokens / per-row wall (= batch wall, all rows in a
    # chunk share). With batch=1, this matches the old per-prompt tps.
    tps_per_row = []
    for row in rows:
        gl = row.get("gen_len", 0)
        wall = row.get("end_to_end_s", 0.0) or 0.0
        if wall > 0 and gl > 0:
            tps_per_row.append(gl / wall)
    # Batch-aggregate throughput: sum of tokens generated in a chunk divided
    # by chunk wall-clock. For batch=1 equals per-row tps.
    tps_batch = []
    chunk_size = batch if batch > 1 else 1
    for chunk_start in range(0, len(rows), chunk_size):
        chunk_rows = rows[chunk_start:chunk_start + chunk_size]
        if not chunk_rows:
            continue
        wall = chunk_rows[0].get("end_to_end_s", 0.0) or 0.0
        toks = sum(r.get("gen_len", 0) for r in chunk_rows)
        if wall > 0 and toks > 0:
            tps_batch.append(toks / wall)
    out: Dict[str, Any] = {
        "backend": backend.name,
        "batch": batch,
        "end_to_end_ms": summarize_latency_ms(e2e_times),
        "throughput_tokens_per_sec": summarize_scalar(tps_per_row, suffix="_tok_per_s"),
        "throughput_batch_tokens_per_sec": summarize_scalar(tps_batch, suffix="_tok_per_s"),
        "samples": rows,
    }
    if draft_totals:
        out["component_wall_ms"] = {
            "draft_or_native_block": summarize_latency_ms(draft_totals),
        }
    if verify_totals:
        out["component_wall_ms"] = out.get("component_wall_ms", {})
        out["component_wall_ms"]["target_verify"] = summarize_latency_ms(verify_totals)

    #  EOS / gen_len  mean
    ms_per_block: List[float] = []
    ms_per_out_tok: List[float] = []
    for row in rows:
        t_s = row["end_to_end_s"]
        n_blk = len(row.get("timing", {}).get("per_block", []))
        if n_blk > 0:
            ms_per_block.append(t_s * 1000.0 / n_blk)
        gl = row.get("gen_len", 0)
        if gl and gl > 0:
            ms_per_out_tok.append(t_s * 1000.0 / gl)
    out["normalized_fairness_metrics"] = {
        "ms_per_block": summarize_ms(ms_per_block),
        "ms_per_output_token": summarize_ms(ms_per_out_tok),
        "note": "end_to_end_s /  block   / gen_len --no_eos_stop ",
    }

    #  ( speculative) res.stats timing
    if backend.mode == "speculative" and total_draft_tokens > 0:
        out["acceptance"] = {
            "total_draft_tokens": total_draft_tokens,
            "total_accepted_tokens": total_accepted_tokens,
            "total_bonus_tokens": total_bonus_tokens,
            "accept_rate": total_accepted_tokens / total_draft_tokens,
            "bonus_rate": total_bonus_tokens / total_draft_tokens,
            "per_block_accepted": summarize_ms(
                [float(x) for x in all_per_block_accepted]
            ),
            "note": (
                "accept_rate = total_accepted / total_draft (MRS  first-reject "
                " draft token  bonus)bonus_rate =  reject  block "
                "per_block_accepted  mean/p50/p95 ∈[0, block_length]"
            ),
        }

    return out


def main():
    parser = argparse.ArgumentParser(description="SDAR multi-block latency bench")
    parser.add_argument("--config", type=str, default=None, help=" YAML SpecConfig")
    parser.add_argument(
        "--runtime",
        type=str,
        choices=["hf", "jetengine"],
        default="hf",
        help=" runtime: hf = HuggingFace +  draft/verify; "
             "jetengine = JetEngine  ( native )",
    )
    parser.add_argument("--compare", type=str, choices=["both", "native", "speculative"], default="both")
    parser.add_argument("--draft_model", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--draft_device", type=str, default="cuda:1")
    parser.add_argument("--target_device", type=str, default="cuda:1")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_samples", type=int, default=None, help=" 20 --config  YAML ")
    parser.add_argument("--num_blocks", type=int, default=None, help=" 8 --config  YAML ")
    parser.add_argument("--block_length", type=int, default=None, help=" 4 --config  YAML ")
    parser.add_argument("--denoising_steps", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--warmup", type=int, default=2, help=" warmup")
    parser.add_argument(
        "--no_eos_stop",
        action="store_true",
        help=" EOS  num_blocks Native/Spec ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=" JSON results/{}/bench_latency_{tag}.json",
    )
    parser.add_argument("--mask_token_id", type=int, default=151669)
    parser.add_argument(
        "--target_eval_sdpa",
        action="store_true",
        help="Target verify  eval + externally-built 4D bool mask  SDPA ( "
             "sweep_forward) patch_multi_block_mask_fnCLI > YAML",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=" pipelined speculative: verify_N  draft_{N+1}  GPU  "
             " dual-GPU (draft_device != target_device) ",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help=" prefix-only KV cache (à la generate.py:block_diffusion_generate)"
             "scaffolding only  speculative_generate  use_kv_cache=True  "
             "raise NotImplementedError,  1 ( SpecConfig )",
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="draft  denoising forward  CUDA graph replay (per seq_len )"
             " launch-bound  (1.7B/4B)  forward ~4-5x; "
             " seq_len  ~3  forward  captureSDPA "
             "graph capture  attention  sync ",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help=" backend.generate  prompt >1  (batch, padded_len) "
             " capture  graph batch ",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="speculative  draft  block  / target  verify  block "
             " num_blocksK=1  speculative",
    )
    args = parser.parse_args()

    overrides = {k: getattr(args, k) for k in (
        "draft_model", "target_model", "draft_device", "target_device",
        "num_samples", "num_blocks", "block_length", "dataset", "dataset_split",
        "mask_token_id", "seed",
    ) if getattr(args, k) is not None}
    if args.denoising_steps is not None:
        overrides["denoising_steps"] = args.denoising_steps
    if args.batch is not None:
        overrides["batch"] = args.batch
    if args.K is not None:
        overrides["K"] = args.K
    if args.target_eval_sdpa:
        overrides["target_eval_sdpa"] = True
    if args.pipeline:
        overrides["pipeline"] = True
    if args.use_cuda_graph:
        overrides["use_cuda_graph"] = True
    if args.use_kv_cache:
        overrides["use_kv_cache"] = True

    if args.config:
        cfg = load_config(args.config, overrides)
    else:
        if "num_blocks" not in overrides:
            overrides["num_blocks"] = 8
        if "num_samples" not in overrides:
            overrides["num_samples"] = 20
        if "block_length" not in overrides:
            overrides["block_length"] = 4
        cfg = default_config(overrides)

    cfg.draft_model = _resolve_local_model_path(cfg.draft_model)
    cfg.target_model = _resolve_local_model_path(cfg.target_model)

    torch.manual_seed(cfg.seed)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    if args.runtime == "jetengine" and args.compare in ("both", "speculative"):
        raise NotImplementedError(
            "JetEngine runtime  native mode --compare native"
            " --runtime hf  speculativespeculative+jetengine "
        )

    # batch>1: backend.generate_batch path. _run_backend_on_prompts collects
    # cfg.batch prompts at a time and calls generate_batch (HF native + spec
    # backends have batched implementations; jet falls back to per-prompt).

    load_draft = args.compare in ("both", "speculative")
    jet_llm = None
    if args.runtime == "hf":
        tokenizer, draft, target = _load_models(
            argparse.Namespace(
                draft_model=cfg.draft_model,
                target_model=cfg.target_model,
                draft_device=cfg.draft_device,
                target_device=cfg.target_device,
            ),
            torch_dtype,
            load_draft=load_draft,
        )
    else:  # jetengine
        from transformers import AutoTokenizer
        from jetengine import LLM
        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model, trust_remote_code=True)
        # JetEngine  runner/KV cache,  CUDA_VISIBLE_DEVICES
        #  .to(device) cuda:0
        print(f"[run_benchmark] loading JetEngine LLM from {cfg.target_model}", flush=True)
        jet_llm = LLM(
            cfg.target_model,
            max_num_seqs=64,
            block_length=cfg.block_length,
        )
        draft = None
        target = None

    ds, all_prompt_ids = load_prompts(tokenizer, cfg)
    max_plen = max(len(p) for p in all_prompt_ids)
    padded_len = compute_verify_padded_len(max_plen, cfg.num_blocks, cfg.block_length, K=cfg.K)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id

    # CUDA graph cache in draft is keyed by seq_len (= prompt_len + k*block_length
    # at block k). Variable prompt lengths across samples  every sample recaptures
    # num_blocks graphs  2-3x slowdown over eager. Left-pad every prompt to
    # max_plen so seq_len is sample-invariant, then the cache is populated once
    # by the warmup sample and all timed samples replay.
    if getattr(cfg, "use_cuda_graph", False):
        all_prompt_ids = [
            [pad_token_id] * (max_plen - len(p)) + list(p)
            for p in all_prompt_ids
        ]

    report: Dict[str, Any] = {
        "report_date": report_date_str(),
        "baseline_and_methodology": {
            "native_dllm": (
                "Target  draft_one_blockforward  eval  SDARForCausalLM  "
                "model(input_ids, attention_mask=1, position_ids) token_labels/block_ids"
                " create_multi_block_causal_mask / build_verify_sequence"
                " HF  dLLM  MASK"
                " speculative  target verify causal layout"
            ),
            "speculative_target_verify": (
                "target_verify_forward  token_labels+block_ids  verify "
                " patch  create_multi_block_causal_mask eval_multi_block "
            ),
            "native_runs_before_target_patch": (
                "compare=both  Native Target  patch_multi_block_mask_fn Speculative"
            ),
        },
        "config": {
            "draft_model": cfg.draft_model,
            "target_model": cfg.target_model,
            "draft_device": cfg.draft_device,
            "target_device": cfg.target_device,
            "num_samples": len(all_prompt_ids),
            "num_blocks": cfg.num_blocks,
            "block_length": cfg.block_length,
            "denoising_steps": cfg.denoising_steps,
            "K": getattr(cfg, "K", 1),
            "batch": getattr(cfg, "batch", 1),
            "dataset": cfg.dataset,
            "padded_len": padded_len,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "no_eos_stop": args.no_eos_stop,
            "target_eval_sdpa": getattr(cfg, "target_eval_sdpa", False),
            "pipeline": getattr(cfg, "pipeline", False),
            "use_cuda_graph": getattr(cfg, "use_cuda_graph", False),
        },
        "results": {},
    }

    report["config"]["runtime"] = args.runtime

    if args.compare in ("both", "native"):
        if args.runtime == "hf":
            native_be = HFNativeBackend(
                model=target,
                num_blocks=cfg.num_blocks,
                block_length=cfg.block_length,
                denoising_steps=cfg.denoising_steps,
                mask_token_id=cfg.mask_token_id,
                eos_token_id=eos_id,
                pad_token_id=pad_token_id,
                use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            )
        else:  # jetengine
            native_be = JetEngineNativeBackend(
                llm=jet_llm,
                max_tokens=cfg.num_blocks * cfg.block_length,
                block_length=cfg.block_length,
                denoising_steps=cfg.denoising_steps,
                ignore_eos=args.no_eos_stop,
            )
        report["results"][native_be.name] = _run_backend_on_prompts(
            native_be, all_prompt_ids, args.warmup,
            batch=getattr(cfg, "batch", 1),
        )

    if args.compare in ("both", "speculative"):
        if draft is None:
            raise RuntimeError("speculative  draft  --compare both  speculative  draft_model")
        # verify-side cuda graph is only captured on the SDPA-eval branch
        # (training branch goes through flex_attention which has dynamic
        # shape logic). Auto-enable the flag when use_cuda_graph is on.
        if getattr(cfg, "use_cuda_graph", False) and not getattr(cfg, "target_eval_sdpa", False):
            cfg.target_eval_sdpa = True
            report["config"]["target_eval_sdpa"] = True
            print("[run_benchmark] use_cuda_graph=True  forcing target_eval_sdpa=True so verify is graph-safe")
        # eval_sdpa  verify  4D mask model.eval() + SDPA;
        # patch create_multi_block_causal_mask
        # training branch
        if not getattr(cfg, "target_eval_sdpa", False):
            patch_multi_block_mask_fn(target, block_causal_prompt=cfg.block_causal_prompt)
        else:
            # ── pipeline  ──
            # SDARAttention.forward  `if torch.all(attention_mask):` ,
            # 0-dim bool tensor  Python __bool__  GPUCPU sync,  (×36)
            #  reduce kernel  dispatch .  verify forward
            #  ~36  sync × ~1ms = ~36ms CPU dispatch wall,
            # pipeline  "verify_N  draft_{N+1} overlap" .
            # patch_sdpa_eval_attention  SDPA  forward,
            #  sync.  idempotent, draft  use_cuda_graph
            #  patch  attn class;  target  from_pretrained(trust_remote_code)
            #  modeling_sdar ,  attn class  draft ,
            #  patch .
            from speculative_decoding.draft import patch_sdpa_eval_attention
            patch_sdpa_eval_attention(target)
            if draft is not None:
                # pipeline  draft  dispatch-async  overlap.
                patch_sdpa_eval_attention(draft)
        spec_be = HFSpeculativeBackend(
            draft_model=draft,
            target_model=target,
            cfg=cfg,
            pad_token_id=pad_token_id,
            padded_len=padded_len,
            eos_token_id=eos_id,
        )
        report["results"][spec_be.name] = _run_backend_on_prompts(
            spec_be, all_prompt_ids, args.warmup,
            batch=getattr(cfg, "batch", 1),
        )

    # comparison native vs speculativeruntime
    native_key = f"{args.runtime}_native"
    spec_key = "hf_speculative"
    if args.compare == "both" and native_key in report["results"] and spec_key in report["results"]:
        n_mean = report["results"][native_key]["end_to_end_ms"]["mean_ms"]
        s_mean = report["results"][spec_key]["end_to_end_ms"]["mean_ms"]
        if s_mean > 0:
            report["comparison"] = {
                "native_key": native_key,
                "spec_key": spec_key,
                "speedup_native_over_spec_ms_ratio": n_mean / s_mean,
                "note": ">1  native  speculative <1  speculative ",
            }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)

    out_path = args.output
    if not out_path:
        out_path = os.path.join(
            cfg.output_dir,
            report_date_str(),
            f"bench_latency_{cfg.tag}.json",
        )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
