#!/usr/bin/env python3
"""Regime attribution sweep.

Sweeps along three axes  models × batches × prompt_len  and produces three
diagnostic curves for `per_forward model_forward_ms` (pure GPU time, CUDA
event), for both draft and target roles:

  regime_vs_batch.png        one subplot per prompt_len, x=batch (log2)
  regime_vs_prompt_len.png   one subplot per batch,      x=prompt_len
  regime_vs_params.png       one subplot per role,       x=params (B)

Interpretation rules of thumb:
  - Flat curves (along batch or tokens)  BW-bound or launch/CPU-bound
  - Near-linear growth  compute-bound
  - To disambiguate BW-bound vs launch-bound, compare against the bf16
    weight-read lower bound (~ 2 * params / HBM_BW) and check whether the
    observed latency scales with parameter size.

This script is a **diagnostic / bottleneck attribution** tool, not a serving
latency benchmark  it inherits the `sync_after=True` stage boundaries from
sweep_forward, which serializes async CPU↔GPU overlap.

Reuses helpers from sweep_forward, so the timing recipe matches exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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
from transformers import AutoModelForCausalLM  # noqa: E402

from speculative_decoding.sweep_forward import (  # noqa: E402
    MASK_TOKEN_ID,
    HFForwardTimer,
    StageAcc,
    hf_block_causal_run,
    hf_verify_run,
)
from speculative_decoding.verify import patch_multi_block_mask_fn  # noqa: E402


def count_params_b(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


def padded_len_for(prompt_len: int, num_blocks: int, block_length: int,
                   floor: int = 128) -> int:
    tight = ((prompt_len + num_blocks * 2 * block_length + 127) // 128) * 128
    return max(floor, tight)


def measure_cell(model, timer, role: str, *, batch: int, prompt_len: int,
                 block_length: int, denoising_steps: int, num_blocks: int,
                 padded_len: int, warmup: int, iters: int) -> Dict[str, Any]:
    """Mirror sweep_forward's per-cell timing: model_forward_ms via CUDA events,
    end_to_end_stage_ms via wall-clock with cuda sync at the boundaries."""
    if role == "draft":
        def run_once():
            hf_block_causal_run(
                model, batch, prompt_len, block_length, denoising_steps,
                num_blocks, MASK_TOKEN_ID, acc=StageAcc(),
            )
        n_forwards_per_run = num_blocks * denoising_steps
    elif role == "target":
        def run_once():
            hf_verify_run(
                model, batch, prompt_len, block_length, num_blocks,
                denoising_steps, padded_len, MASK_TOKEN_ID, acc=StageAcc(),
            )
        n_forwards_per_run = num_blocks
    else:
        raise ValueError(role)

    for _ in range(warmup):
        run_once()

    e2e_ms: List[float] = []
    fwd_sum_ms: List[float] = []
    for _ in range(iters):
        timer.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)
        fwd_sum_ms.append(sum(timer.collect()))

    mean_fwd_run = sum(fwd_sum_ms) / len(fwd_sum_ms)
    mean_e2e = sum(e2e_ms) / len(e2e_ms)
    return {
        "end_to_end_stage_ms_mean": round(mean_e2e, 4),
        "model_forward_ms_per_run_mean": round(mean_fwd_run, 4),
        "model_forward_ms_per_forward_mean": round(mean_fwd_run / max(1, n_forwards_per_run), 4),
        "n_forwards_per_run": n_forwards_per_run,
    }


def run(args) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for model_path in args.models:
        full_path = model_path if os.path.isdir(model_path) else os.path.join(_ROOT, model_path)
        model_name = os.path.basename(full_path.rstrip("/"))
        print(f"[regime] loading {full_path} on {args.device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            full_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(args.device).eval()
        params_b = count_params_b(model)
        print(f"  params = {params_b:.3f} B", flush=True)

        timer = HFForwardTimer(model)
        verify_patched = False
        try:
            for batch in args.batches:
                for prompt_len in args.prompt_lens:
                    padded_len = padded_len_for(
                        prompt_len, args.num_blocks, args.block_length
                    )
                    tag = (f"{model_name} bs={batch} pl={prompt_len} "
                           f"bl={args.block_length} S={args.denoising_steps}")

                    print(f"  [{tag}] draft", flush=True)
                    d = measure_cell(
                        model, timer, "draft",
                        batch=batch, prompt_len=prompt_len,
                        block_length=args.block_length,
                        denoising_steps=args.denoising_steps,
                        num_blocks=args.num_blocks,
                        padded_len=padded_len,
                        warmup=args.warmup, iters=args.iters,
                    )

                    if not verify_patched:
                        patch_multi_block_mask_fn(model, block_causal_prompt=True)
                        verify_patched = True

                    print(f"  [{tag}] target", flush=True)
                    v = measure_cell(
                        model, timer, "target",
                        batch=batch, prompt_len=prompt_len,
                        block_length=args.block_length,
                        denoising_steps=args.denoising_steps,
                        num_blocks=args.num_blocks,
                        padded_len=padded_len,
                        warmup=args.warmup, iters=args.iters,
                    )

                    base = {
                        "model": model_name,
                        "params_b": round(params_b, 3),
                        "batch": batch,
                        "prompt_len": prompt_len,
                        "block_length": args.block_length,
                        "denoising_steps": args.denoising_steps,
                        "num_blocks": args.num_blocks,
                        "padded_len": padded_len,
                    }
                    records.append({**base, "role": "draft", **d})
                    records.append({**base, "role": "target", **v})
                    print(f"    draft  per_fwd={d['model_forward_ms_per_forward_mean']:.3f} ms"
                          f"   target per_fwd={v['model_forward_ms_per_forward_mean']:.3f} ms",
                          flush=True)
        finally:
            timer.close()
            del model
            torch.cuda.empty_cache()
    return records


def write_json_csv(records: List[Dict[str, Any]], out_dir: str) -> None:
    with open(os.path.join(out_dir, "regime_records.json"), "w") as f:
        json.dump(records, f, indent=2)
    fieldnames = sorted({k for r in records for k in r.keys()})
    with open(os.path.join(out_dir, "regime_records.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)


def _filter(records, **kw):
    return [r for r in records if all(r.get(k) == v for k, v in kw.items())]


def plot_curves(records: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[regime] matplotlib not available, skipping plots: {e}")
        return

    y_key = "model_forward_ms_per_forward_mean"
    models = sorted({r["model"] for r in records}, key=lambda m: next(
        r["params_b"] for r in records if r["model"] == m))
    roles = ["draft", "target"]
    prompt_lens = sorted({r["prompt_len"] for r in records})
    batches = sorted({r["batch"] for r in records})

    # ------- 1) latency vs batch, subplot per prompt_len -------
    fig, axes = plt.subplots(1, len(prompt_lens),
                             figsize=(5 * len(prompt_lens), 4), squeeze=False)
    for ax, pl in zip(axes[0], prompt_lens):
        for m in models:
            for role in roles:
                sub = sorted(_filter(records, model=m, role=role, prompt_len=pl),
                             key=lambda r: r["batch"])
                if not sub:
                    continue
                xs = [r["batch"] for r in sub]
                ys = [r[y_key] for r in sub]
                style = "-" if role == "target" else "--"
                ax.plot(xs, ys, marker="o", linestyle=style, label=f"{m} {role}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch size")
        ax.set_ylabel("model_forward_ms / forward")
        ax.set_title(f"prompt_len={pl}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Regime diagnostic: forward latency vs batch size "
                 "(flat  BW/launch-bound, linear  compute-bound)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_vs_batch.png"), dpi=150)
    plt.close(fig)

    # ------- 2) latency vs prompt_len, subplot per batch -------
    fig, axes = plt.subplots(1, len(batches),
                             figsize=(5 * len(batches), 4), squeeze=False)
    for ax, b in zip(axes[0], batches):
        for m in models:
            for role in roles:
                sub = sorted(_filter(records, model=m, role=role, batch=b),
                             key=lambda r: r["prompt_len"])
                if not sub:
                    continue
                xs = [r["prompt_len"] for r in sub]
                ys = [r[y_key] for r in sub]
                style = "-" if role == "target" else "--"
                ax.plot(xs, ys, marker="o", linestyle=style, label=f"{m} {role}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("prompt_len")
        ax.set_ylabel("model_forward_ms / forward")
        ax.set_title(f"batch={b}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Regime diagnostic: forward latency vs prompt_len "
                 "(flat  BW/launch-bound, >linear  attention compute)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_vs_prompt_len.png"), dpi=150)
    plt.close(fig)

    # ------- 3) latency vs params, subplot per role -------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, role in zip(axes[0], roles):
        for b in batches:
            for pl in prompt_lens:
                sub = sorted(_filter(records, role=role, batch=b, prompt_len=pl),
                             key=lambda r: r["params_b"])
                if len(sub) < 2:
                    continue
                xs = [r["params_b"] for r in sub]
                ys = [r[y_key] for r in sub]
                ax.plot(xs, ys, marker="o", label=f"bs={b} pl={pl}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("params (B)")
        ax.set_ylabel("model_forward_ms / forward")
        ax.set_title(f"{role} role")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle("Regime diagnostic: forward latency vs params "
                 "(slope < params ratio  small model is launch-bound)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_vs_params.png"), dpi=150)
    plt.close(fig)

    print(f"[regime] wrote plots to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="model paths (relative to repo root allowed)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--prompt_lens", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = run(args)
    write_json_csv(records, args.output_dir)
    plot_curves(records, args.output_dir)
    print(f"[regime] done  {len(records)} records in {args.output_dir}")


if __name__ == "__main__":
    main()
