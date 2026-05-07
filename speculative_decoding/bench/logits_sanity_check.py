#!/usr/bin/env python3
"""
Logits Sanity Check: Native Block Diffusion vs Causal-Mask Single Forward.

Compares two inference paths on the same model and same inputs:
  Phase 1 (Native):   multi-step block diffusion with standard attention
  Phase 2 (New Mask): single forward with causal mask (create_causal_mask_from_labels)

Metrics per position:
  - Top-1 token agreement
  - Top-K overlap (K=5, 10)
  - Logits cosine similarity
  - KL divergence (both directions)
  - JS divergence
  - Max probability difference
  - Gold token logprob difference
  - Per-position CE loss

Outputs:
  - per_position.csv           row-level detail for every (sample, position)
  - summary.json               aggregate stats by overall / step / prompt_len
  - summary.md                 human-readable summary
  - divergence_vs_position.png (optional, if matplotlib available)
  - cosine_vs_position.png
  - top1_agreement_vs_position.png

Supports both SDAR and LLaDA models via --model_type flag.

Usage (from dllm_causal_train/):
  python speculative_decoding/bench/logits_sanity_check.py \
    --model_path inference/model/SDAR-8B-Chat \
    --model_type sdar --block_length 4 --denoising_steps 4 \
    --num_samples 20 --mode new
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from new_attn import create_causal_mask_from_labels


SDAR_MASK_TOKEN_ID = 151669
LLADA_MASK_TOKEN_ID = 156895


def _get_unmask_schedule_sdar(block_length: int, denoising_steps: int) -> List[int]:
    """SDAR-style unmask: ceil(remaining / remaining_steps) per step."""
    schedule = []
    remaining = block_length
    for step in range(1, denoising_steps + 1):
        rem_steps = denoising_steps - step + 1
        n = min(-(-remaining // rem_steps), remaining)
        schedule.append(n)
        remaining -= n
    if remaining > 0:
        schedule[-1] += remaining
    return schedule


def _get_unmask_schedule_llada(block_length: int, denoising_steps: int) -> List[int]:
    """LLaDA-style: evenly distribute, remainder goes to early steps."""
    if denoising_steps == 0:
        return []
    base = block_length // denoising_steps
    rem = block_length % denoising_steps
    schedule = [base + (1 if i < rem else 0) for i in range(denoising_steps)]
    return schedule


def run_native_inference(
    model, input_ids: torch.Tensor, prompt_len: int,
    block_length: int, denoising_steps: int,
    mask_token_id: int, model_type: str,
    threshold: float = 0.95,
) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 1: standard multi-step block diffusion.

    Returns (response_ids, step_map, logits_at_unmask, probs_at_unmask).
    logits_at_unmask: (block_length, vocab_size) raw logits when each pos was unmasked.
    probs_at_unmask:  (block_length, vocab_size) softmax probs at unmask time.
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    step_map = torch.zeros(block_length, dtype=torch.long)
    all_logits = torch.zeros(block_length, vocab_size)
    all_probs = torch.zeros(block_length, vocab_size)

    if model_type == "llada":
        schedule = _get_unmask_schedule_llada(block_length, denoising_steps)
    else:
        schedule = _get_unmask_schedule_sdar(block_length, denoising_steps)

    seq_len = input_ids.shape[1]
    if model_type == "sdar":
        from speculative_decoding.draft import _build_block_causal_attn
        attn_mask = _build_block_causal_attn(seq_len, block_length, device)
    else:
        attn_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=torch.bfloat16)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    for step_idx, n_unmask in enumerate(schedule):
        step = step_idx + 1
        resp = input_ids[0, prompt_len:]
        is_masked = (resp == mask_token_id).cpu()
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        with torch.no_grad():
            if model_type == "sdar":
                outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                                use_cache=False, return_dict=True)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                                position_ids=position_ids)

        block_logits_cpu = outputs.logits[0, prompt_len:].float().cpu()
        del outputs
        probs = F.softmax(block_logits_cpu, dim=-1)
        pred_ids = probs.argmax(dim=-1)
        confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

        conf_masked = torch.where(is_masked, confidence,
                                  torch.tensor(-torch.inf))

        if model_type == "llada":
            high_conf = conf_masked > threshold
            n_high = high_conf.sum().item()
            if n_high >= n_unmask:
                transfer_mask = high_conf
            else:
                _, topk_idx = torch.topk(conf_masked, k=min(n_unmask, num_masked))
                transfer_mask = torch.zeros(block_length, dtype=torch.bool)
                transfer_mask[topk_idx] = True
        else:
            _, topk_idx = torch.topk(conf_masked, k=min(n_unmask, num_masked))
            transfer_mask = torch.zeros(block_length, dtype=torch.bool)
            transfer_mask[topk_idx] = True

        for pos in transfer_mask.nonzero(as_tuple=True)[0]:
            input_ids[0, prompt_len + pos] = pred_ids[pos].to(device)
            step_map[pos] = step
            all_logits[pos] = block_logits_cpu[pos]
            all_probs[pos] = probs[pos]

    response_ids = input_ids[0, prompt_len:].tolist()
    return response_ids, step_map, all_logits, all_probs


def run_causal_mask_forward(
    model, prompt_ids: List[int], response_ids: List[int],
    step_map: torch.Tensor, prompt_len: int,
    block_length: int, denoising_steps: int,
    block_causal_prompt: bool, mask_token_id: int,
    model_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """Phase 2: single forward with causal mask.

    Returns (logits, probs, per_pos_ce_loss).
    logits: (block_length, vocab_size) raw logits at mask positions.
    probs:  (block_length, vocab_size) softmax probs.
    """
    device = next(model.parameters()).device
    block_size = denoising_steps
    mask_label = block_size + 1

    causal_input = torch.tensor(
        [prompt_ids + response_ids + [mask_token_id] * block_length],
        dtype=torch.long, device=device,
    )
    causal_seq_len = causal_input.shape[1]
    mask_start = prompt_len + block_length

    tl = [0] * prompt_len + step_map.tolist() + [mask_label] * block_length
    token_labels = torch.tensor([tl], dtype=torch.long, device=device)

    bool_mask = create_causal_mask_from_labels(
        token_labels, block_size, block_causal_prompt=block_causal_prompt,
    )

    if model_type == "sdar":
        attn_mask = bool_mask.squeeze(1)
    else:
        attn_mask = torch.where(
            bool_mask, 0.0, torch.tensor(float("-inf"))
        ).to(dtype=torch.bfloat16, device=device)

    tl_cpu = token_labels[0].cpu()
    clean_idx_cpu = torch.nonzero(
        (tl_cpu >= 1) & (tl_cpu <= block_size), as_tuple=True
    )[0]
    mask_idx_cpu = torch.nonzero(
        tl_cpu == mask_label, as_tuple=True
    )[0]

    if model_type == "llada":
        _orig = model.model.rotary_emb.forward
        def _patched(x, position_ids):
            cos, sin = _orig(x, position_ids)
            rope_dev = cos.device
            ci = clean_idx_cpu.to(rope_dev)
            mi = mask_idx_cpu.to(rope_dev)
            if mi.numel() > 0 and ci.numel() == mi.numel():
                cos[0, mi] = cos[0, ci]
                sin[0, mi] = sin[0, ci]
            return cos, sin
        model.model.rotary_emb.forward = _patched

    causal_pos = torch.arange(causal_seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        if model_type == "sdar":
            outputs = model(input_ids=causal_input, attention_mask=attn_mask,
                            use_cache=False, return_dict=True)
        else:
            outputs = model(input_ids=causal_input, attention_mask=attn_mask,
                            position_ids=causal_pos)

    if model_type == "llada":
        model.model.rotary_emb.forward = _orig

    mask_logits = outputs.logits[0, mask_start:mask_start + block_length].float().cpu()
    mask_probs = F.softmax(mask_logits, dim=-1)

    gt_tensor = torch.tensor(response_ids, device=mask_logits.device)
    per_pos_loss = F.cross_entropy(mask_logits, gt_tensor, reduction="none").cpu().tolist()

    return mask_logits.cpu(), mask_probs.cpu(), per_pos_loss


def compute_metrics(
    p1_logits: torch.Tensor, p1_probs: torch.Tensor,
    p2_logits: torch.Tensor, p2_probs: torch.Tensor,
    response_ids: List[int], block_length: int,
) -> List[Dict[str, Any]]:
    """Compute per-position comparison metrics between Phase 1 and Phase 2."""
    rows = []
    eps = 1e-10

    p1_logprobs = torch.log(p1_probs + eps)
    p2_logprobs = torch.log(p2_probs + eps)

    for i in range(block_length):
        lp1 = p1_logits[i]
        lp2 = p2_logits[i]
        pp1 = p1_probs[i]
        pp2 = p2_probs[i]

        top1_p1 = pp1.argmax().item()
        top1_p2 = pp2.argmax().item()
        top1_match = int(top1_p1 == top1_p2)

        _, topk5_p1 = pp1.topk(5)
        _, topk5_p2 = pp2.topk(5)
        topk5_overlap = len(set(topk5_p1.tolist()) & set(topk5_p2.tolist()))

        _, topk10_p1 = pp1.topk(10)
        _, topk10_p2 = pp2.topk(10)
        topk10_overlap = len(set(topk10_p1.tolist()) & set(topk10_p2.tolist()))

        cos_sim = F.cosine_similarity(lp1.unsqueeze(0), lp2.unsqueeze(0)).item()

        kl_fwd = F.kl_div(p2_logprobs[i], pp1, reduction="sum", log_target=False).item()
        kl_rev = F.kl_div(p1_logprobs[i], pp2, reduction="sum", log_target=False).item()

        m = 0.5 * (pp1 + pp2)
        m_logprobs = torch.log(m + eps)
        js_fwd = F.kl_div(m_logprobs, pp1, reduction="sum", log_target=False).item()
        js_rev = F.kl_div(m_logprobs, pp2, reduction="sum", log_target=False).item()
        js_div = 0.5 * (js_fwd + js_rev)

        max_prob_diff = (pp1.max().item() - pp2.max().item())

        gt_id = response_ids[i]
        gold_logprob_p1 = p1_logprobs[i, gt_id].item()
        gold_logprob_p2 = p2_logprobs[i, gt_id].item()
        gold_logprob_diff = gold_logprob_p1 - gold_logprob_p2

        rows.append({
            "position": i,
            "top1_match": top1_match,
            "top1_native": top1_p1,
            "top1_newmask": top1_p2,
            "topk5_overlap": topk5_overlap,
            "topk10_overlap": topk10_overlap,
            "cosine_sim": cos_sim,
            "kl_native_to_newmask": kl_fwd,
            "kl_newmask_to_native": kl_rev,
            "js_divergence": js_div,
            "max_prob_diff": max_prob_diff,
            "gold_logprob_native": gold_logprob_p1,
            "gold_logprob_newmask": gold_logprob_p2,
            "gold_logprob_diff": gold_logprob_diff,
            "gold_token_id": gt_id,
        })
    return rows


def aggregate_stats(all_rows: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from all per-position rows."""
    import statistics

    def _summarize(vals):
        if not vals:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "p50": float("nan"), "p5": float("nan"), "p95": float("nan")}
        n = len(vals)
        s = sorted(vals)
        return {
            "n": n,
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals),
            "p5": s[max(0, int(n * 0.05))],
            "p50": s[n // 2],
            "p95": s[min(n - 1, int(n * 0.95))],
        }

    keys = ["cosine_sim", "kl_native_to_newmask", "kl_newmask_to_native",
            "js_divergence", "max_prob_diff", "gold_logprob_diff",
            "topk5_overlap", "topk10_overlap"]

    overall = {}
    for k in keys:
        overall[k] = _summarize([r[k] for r in all_rows])

    top1_matches = [r["top1_match"] for r in all_rows]
    overall["top1_agreement_rate"] = sum(top1_matches) / max(len(top1_matches), 1)
    overall["total_positions"] = len(all_rows)

    by_position = defaultdict(list)
    for r in all_rows:
        by_position[r["position"]].append(r)

    position_summary = {}
    for pos in sorted(by_position.keys()):
        prows = by_position[pos]
        psummary = {}
        for k in keys:
            psummary[k] = _summarize([r[k] for r in prows])
        psummary["top1_agreement_rate"] = sum(r["top1_match"] for r in prows) / len(prows)
        psummary["n"] = len(prows)
        position_summary[pos] = psummary

    return {"overall": overall, "by_position": position_summary}


def write_csv(rows: List[Dict], path: str, sample_indices: List[int],
              step_maps: List[List[int]], prompt_lens: List[int]):
    """Write per-position CSV with sample metadata."""
    if not rows:
        return
    fieldnames = (
        ["sample_idx", "prompt_len", "step", "position"]
        + [k for k in rows[0].keys() if k != "position"]
    )
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        offset = 0
        for si, (idx, smap, plen) in enumerate(zip(sample_indices, step_maps, prompt_lens)):
            bl = len(smap)
            for pi in range(bl):
                r = dict(rows[offset + pi])
                r["sample_idx"] = idx
                r["prompt_len"] = plen
                r["step"] = smap[pi]
                w.writerow(r)
            offset += bl


def write_summary_md(stats: Dict, config: Dict, path: str):
    """Write human-readable markdown summary."""
    o = stats["overall"]
    with open(path, "w") as f:
        f.write("# Logits Sanity Check: Native vs New Mask\n\n")
        f.write(f"**Model**: `{config['model_path']}`\n")
        f.write(f"**Type**: {config['model_type']}\n")
        f.write(f"**Block length**: {config['block_length']}, "
                f"**Denoising steps**: {config['denoising_steps']}\n")
        f.write(f"**Mode**: {config['mode']} (block_causal_prompt="
                f"{'True' if config['mode'] == 'new' else 'False'})\n")
        f.write(f"**Samples**: {config['num_samples']}, "
                f"**Total positions**: {o['total_positions']}\n\n")

        f.write("## Overall\n\n")
        f.write("| Metric | Mean | Std | P5 | P50 | P95 |\n")
        f.write("|--------|------|-----|-----|-----|-----|\n")
        f.write(f"| Top-1 Agreement | {o['top1_agreement_rate']:.4f} | - | - | - | - |\n")
        for k in ["cosine_sim", "kl_native_to_newmask", "kl_newmask_to_native",
                   "js_divergence", "max_prob_diff", "gold_logprob_diff",
                   "topk5_overlap", "topk10_overlap"]:
            s = o[k]
            f.write(f"| {k} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['p5']:.4f} | {s['p50']:.4f} | {s['p95']:.4f} |\n")

        f.write("\n## By Position\n\n")
        f.write("| Pos | N | Top1Agree | CosSim | KL(NM) | JS | TopK5 | TopK10 |\n")
        f.write("|-----|---|-----------|--------|---------|-----|-------|--------|\n")
        bp = stats["by_position"]
        for pos in sorted(bp.keys()):
            ps = bp[pos]
            f.write(f"| {pos} | {ps['n']} | {ps['top1_agreement_rate']:.3f} | "
                    f"{ps['cosine_sim']['mean']:.4f} | "
                    f"{ps['kl_native_to_newmask']['mean']:.4f} | "
                    f"{ps['js_divergence']['mean']:.4f} | "
                    f"{ps['topk5_overlap']['mean']:.1f} | "
                    f"{ps['topk10_overlap']['mean']:.1f} |\n")

        f.write("\n## Interpretation\n\n")
        top1 = o["top1_agreement_rate"]
        cos_mean = o["cosine_sim"]["mean"]
        js_mean = o["js_divergence"]["mean"]
        if top1 > 0.98 and cos_mean > 0.99:
            f.write("Native and new_mask logits are **highly aligned**. "
                    "The causal mask faithfully reproduces the native distribution.\n")
        elif top1 > 0.95:
            f.write("Native and new_mask logits are **mostly aligned** but show "
                    "minor divergence at some positions. Investigate per-position trends.\n")
        else:
            f.write("**Significant divergence** detected between native and new_mask. "
                    "This may indicate a mask design issue, padding/position bug, "
                    "or RoPE alignment problem.\n")


def try_plot(stats: Dict, output_dir: str):
    """Attempt to generate diagnostic plots; skip silently if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[logits_sanity_check] matplotlib not available, skipping plots")
        return

    bp = stats["by_position"]
    positions = sorted(bp.keys())
    if len(positions) < 2:
        return

    def _plot_metric(metric_key: str, ylabel: str, filename: str, ylim=None):
        means = [bp[p][metric_key]["mean"] for p in positions]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(positions, means, "o-", markersize=3)
        ax.set_xlabel("Token Position in Block")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Position")
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    _plot_metric("js_divergence", "JS Divergence", "js_divergence_vs_position.png")
    _plot_metric("cosine_sim", "Cosine Similarity", "cosine_sim_vs_position.png")
    _plot_metric("kl_native_to_newmask", "KL(Native||NewMask)", "kl_fwd_vs_position.png")

    top1_rates = [bp[p]["top1_agreement_rate"] for p in positions]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(positions, top1_rates, width=0.8)
    ax.set_xlabel("Token Position in Block")
    ax.set_ylabel("Top-1 Agreement Rate")
    ax.set_title("Top-1 Agreement vs Position")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "top1_agreement_vs_position.png"), dpi=150)
    plt.close(fig)

    print(f"[logits_sanity_check] plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Logits sanity check: native vs new_mask")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["sdar", "llada"], default="sdar")
    parser.add_argument("--block_length", type=int, default=4)
    parser.add_argument("--denoising_steps", type=int, default=4)
    parser.add_argument("--mode", type=str, choices=["new", "legacy"], default="new",
                        help="new=block_causal_prompt=True, legacy=False")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Confidence threshold for LLaDA unmask (ignored for SDAR)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    mask_token_id = SDAR_MASK_TOKEN_ID if args.model_type == "sdar" else LLADA_MASK_TOKEN_ID
    block_causal_prompt = (args.mode == "new")

    if args.output_dir is None:
        from speculative_decoding.report_paths import report_date_str
        args.output_dir = os.path.join(
            _ROOT, "speculative_decoding", "results", report_date_str(),
            f"logits_sanity_{args.model_type}_bl{args.block_length}_ds{args.denoising_steps}_{args.mode}",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[logits_sanity_check] model={args.model_path} type={args.model_type}")
    print(f"  block_length={args.block_length} ds={args.denoising_steps} mode={args.mode}")
    print(f"  num_samples={args.num_samples} device={args.device}")
    print(f"  output={args.output_dir}")

    model_path = args.model_path
    if not os.path.isdir(model_path):
        cand = os.path.join(_ROOT, model_path)
        if os.path.isdir(cand):
            model_path = cand

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        args.device = str(next(model.parameters()).device)
        print(f"  device_map=auto, primary device={args.device}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(args.device).eval()

    if args.model_type == "sdar":
        from speculative_decoding.draft import patch_sdpa_eval_attention
        patch_sdpa_eval_attention(model)

    print(f"Loading dataset ({args.dataset})...")
    if args.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        prompt_key = "question"
    elif args.dataset in ("humaneval", "mbpp"):
        ds = load_dataset(args.dataset, split="test")
        prompt_key = "prompt" if "prompt" in ds.column_names else "text"
    else:
        ds = load_dataset(args.dataset, split="train")
        prompt_key = ds.column_names[0]

    if len(ds) > args.num_samples:
        ds = ds.select(range(args.num_samples))
    print(f"  samples={len(ds)}")

    all_metric_rows = []
    sample_indices = []
    step_maps_list = []
    prompt_lens_list = []
    n_errors = 0

    for idx in tqdm(range(len(ds)), desc="sanity_check"):
        text = ds[idx][prompt_key]
        messages = [{"role": "user", "content": text}]
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = text
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        input_ids = torch.tensor(
            [prompt_ids + [mask_token_id] * args.block_length],
            dtype=torch.long, device=args.device,
        )

        try:
            response_ids, step_map, p1_logits, p1_probs = run_native_inference(
                model, input_ids, prompt_len, args.block_length,
                args.denoising_steps, mask_token_id, args.model_type,
                threshold=args.threshold,
            )
        except Exception as e:
            n_errors += 1
            print(f"  [Phase1 error idx={idx}]: {e}")
            torch.cuda.empty_cache()
            continue

        del input_ids
        torch.cuda.empty_cache()

        try:
            p2_logits, p2_probs, per_pos_loss = run_causal_mask_forward(
                model, prompt_ids, response_ids, step_map, prompt_len,
                args.block_length, args.denoising_steps,
                block_causal_prompt, mask_token_id, args.model_type,
            )
        except Exception as e:
            n_errors += 1
            print(f"  [Phase2 error idx={idx}]: {e}")
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

        metrics = compute_metrics(
            p1_logits, p1_probs, p2_logits, p2_probs,
            response_ids, args.block_length,
        )
        for r in metrics:
            r["ce_loss"] = per_pos_loss[r["position"]]
        all_metric_rows.extend(metrics)
        sample_indices.append(idx)
        step_maps_list.append(step_map.tolist())
        prompt_lens_list.append(prompt_len)

    if not all_metric_rows:
        print(f"No valid results ({n_errors} errors). Aborting.")
        sys.exit(1)

    csv_path = os.path.join(args.output_dir, "per_position.csv")
    write_csv(all_metric_rows, csv_path, sample_indices, step_maps_list, prompt_lens_list)
    print(f"  CSV: {csv_path}")

    stats = aggregate_stats(all_metric_rows)

    config_info = {
        "model_path": args.model_path,
        "model_type": args.model_type,
        "block_length": args.block_length,
        "denoising_steps": args.denoising_steps,
        "mode": args.mode,
        "num_samples": len(ds),
        "errors": n_errors,
        "seed": args.seed,
        "device": args.device,
        "mask_token_id": mask_token_id,
    }

    summary = {"config": config_info, "stats": stats}
    json_path = os.path.join(args.output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"  JSON: {json_path}")

    md_path = os.path.join(args.output_dir, "summary.md")
    write_summary_md(stats, config_info, md_path)
    print(f"  MD:   {md_path}")

    try_plot(stats, args.output_dir)

    o = stats["overall"]
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model_type} bl={args.block_length} ds={args.denoising_steps} mode={args.mode}")
    print(f"  Positions: {o['total_positions']}, Errors: {n_errors}")
    print(f"  Top-1 Agreement: {o['top1_agreement_rate']:.4f}")
    print(f"  Cosine Sim:      {o['cosine_sim']['mean']:.4f} (std={o['cosine_sim']['std']:.4f})")
    print(f"  KL(NM):         {o['kl_native_to_newmask']['mean']:.4f}")
    print(f"  KL(MN):         {o['kl_newmask_to_native']['mean']:.4f}")
    print(f"  JS Divergence:   {o['js_divergence']['mean']:.4f}")
    print(f"  TopK5 Overlap:   {o['topk5_overlap']['mean']:.1f}/5")
    print(f"  TopK10 Overlap:  {o['topk10_overlap']['mean']:.1f}/10")
    print(f"  Gold LogP Diff:  {o['gold_logprob_diff']['mean']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
