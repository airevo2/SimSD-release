"""
Speculative Decoding for SDAR Block Diffusion Models.

Pipeline:
  1. Draft model (SDAR-4B) generates one block via standard block diffusion
  2. Target model (SDAR-8B) verifies using block-wise causal mask (single forward)
  3. MRS (Modified Rejection Sampling) accepts/rejects per position
      speculative_branch / mrs_verify_order

Modes:
  single_block  draft + verify 1 block per prompt (alignment / ablation test)
  multi_block   full sequential generation over multiple blocks

Multi-block  speculative_decoding/bench/`speculative_generate(..., return_timings=True)`

Usage:

  # 1.  OpenCompass  torch / transformers / datasets
  conda activate opencompass

  #  draft/target  cuda:1 0  --draft_device / --target_device

  # 2.

  #  dllm_causal_train :
  python speculative_decoding/speculative_decode.py --config speculative_decoding/configs/single_block_test.yaml

  #  speculative_decoding :
  python speculative_decode.py --config configs/single_block_test.yaml
  python speculative_decode.py --config configs/multi_block_default.yaml
  python speculative_decode.py --mode single_block --block_length 4 --num_samples 20

  #  MRSfinal_ids=draft draft_id  target argmax :
  python speculative_decode.py --config ... --speculative_branch per_position_compare

  # MRS  0L-1  step_map :
  python speculative_decode.py --config ... --mrs_verify_order position

:

  • ImportError: huggingface-hub>=0.30.0,<1.0 ... but found huggingface-hub==1.7.1
    conda activate opencompass
    pip install "huggingface-hub>=0.30.0,<1.0"
    #  evaluation/environment.yml : pip install huggingface-hub==0.35.0
    # : pip install -r speculative_decoding/requirements_speculative.txt

  • CUDA OOM  torch  ~/.local  conda  conda
    conda activate opencompass && pip install torch ...  #  opencompass

    PYTHONNOUSERSITE=1 python speculative_decode.py ...
     SPEC_DECODE_STRIP_USER_SITE=1 python speculative_decode.py ...
     torch  conda  ~/.local ModuleNotFoundError: torch
"""

import json
import os
import sys

#  sys.path  ~/.local conda
if os.environ.get("SPEC_DECODE_STRIP_USER_SITE", "").lower() in ("1", "true", "yes"):
    _strip = [p for p in list(sys.path) if p and "/.local/" in p.replace("\\", "/")]
    for _p in _strip:
        while _p in sys.path:
            sys.path.remove(_p)

# SDAR model uses @torch.compile on fused_flex_attention; flex_attention's
# warnings.warn breaks dynamo tracing. Disable compile to avoid graph breaks.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
import argparse
import time
import torch
import torch._dynamo
import torch.nn.functional as F
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from speculative_decoding.config import SpecConfig, load_config, default_config
from speculative_decoding.report_paths import report_date_str, run_subdir
from speculative_decoding.draft import (
    draft_one_block,
    draft_one_block_batch,
    patch_sdpa_eval_attention,
)
from speculative_decoding.verify import (
    patch_multi_block_mask_fn,
    target_verify_forward,
    target_verify_forward_multi,
    target_verify_forward_multi_batch,
)
from speculative_decoding.mrs import mrs_verify, greedy_match_verify


def _select_verify_fn(cfg):
    """Return the token-verify function based on cfg.speculative_branch.

    - "mrs" (default): stochastic MRS  correct when draft samples multinomial.
    - "greedy_match": argmax-greedy match verification  correct when BOTH
      draft and native use argmax, which is the current codebase default (see
      draft.py:322). For greedy inference this is the lossless choice.
    """
    branch = getattr(cfg, "speculative_branch", "mrs")
    if branch == "greedy_match":
        return greedy_match_verify
    return mrs_verify
# Visualization removed in release; stubs preserve CLI/config compat
# (--save_plots, --ascii_demo flags become no-ops).
def plot_single_block(*_a, **_kw): return None
def plot_multi_block(*_a, **_kw): return None
def print_draft_steps(*_a, **_kw): return ""
def print_verify_layout(*_a, **_kw): return ""
def print_mrs_result(*_a, **_kw): return ""


def _out_prefix(cfg):
    """YAML config name as output file prefix (e.g. multi_block_default_)."""
    return f"{cfg.config_name}_" if getattr(cfg, "config_name", None) else ""


def _run_dir(cfg):
    """Run output directory: results/{YYYY-MM-DD}/{prefix}{tag}/"""
    return run_subdir(cfg.output_dir, _out_prefix(cfg), cfg.tag)


MASK_TOKEN_ID = 151669


# ───────────────────── Single-Block Test ──────────────────────


def _per_position_rows(draft_ids, draft_probs, target_probs, step_map, top1_match, per_pos_kl):
    """ position  batch  i=0..L-1 """
    draft_top1 = draft_probs.argmax(dim=-1).tolist()
    target_top1 = target_probs.argmax(dim=-1).tolist()
    rows = []
    for i in range(len(draft_ids)):
        rows.append({
            "pos": i,
            "step_map": int(step_map[i]),
            "draft_id": int(draft_ids[i]),
            "draft_top1": int(draft_top1[i]),
            "target_top1": int(target_top1[i]),
            "top1_argmax_match": int(top1_match[i]),
            "draft_id_eq_target_top1": int(draft_ids[i] == target_top1[i]),
            "kl_qp": float(per_pos_kl[i]),
        })
    return rows


def run_single_block(draft_model, target_model, prompt_ids, cfg, pad_token_id, padded_len,
                     ascii_demo=False):
    """
    Draft 1 block  target verify  MRS per_position_compare

    Returns dict with draft_ids, final_ids, n_accepted, per-position probs, etc.
    When ascii_demo=True, prints ASCII visualization of the process.
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    branch = getattr(cfg, "speculative_branch", "mrs")
    mrs_order = getattr(cfg, "mrs_verify_order", "position")

    step_records = []
    def _step_cb(records):
        step_records[:] = records

    with torch.cuda.device(draft_device):
        draft_ids, draft_probs, step_map = draft_one_block(
            draft_model, prompt_ids, cfg.block_length, cfg.denoising_steps,
            draft_device, cfg.mask_token_id,
            step_callback=_step_cb if ascii_demo else None,
            use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            sampling=getattr(cfg, "draft_sampling", "argmax"),
        )

    ascii_lines = []
    if ascii_demo and step_records:
        s = print_draft_steps(
            cfg.block_length, cfg.denoising_steps, step_records, cfg.mask_token_id,
        )
        print(s)
        ascii_lines.append(s)

    if ascii_demo:
        s = print_verify_layout(
            len(prompt_ids), draft_ids, step_map,
            cfg.block_length, cfg.block_size, cfg.mask_token_id,
        )
        print(s)
        ascii_lines.append(s)

    with torch.cuda.device(target_device):
        target_probs = target_verify_forward(
            target_model, prompt_ids, [], draft_ids, step_map,
            cfg.block_length, cfg.block_size, pad_token_id, padded_len,
            cfg.mask_token_id,
            use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
            use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            share_mask_position=getattr(cfg, "share_mask_position", True),
        )

    # Dual-GPU placement: bring draft_probs onto target_probs.device so
    # kl_div / mrs_verify don't cross cards.
    if draft_probs.device != target_probs.device:
        draft_probs = draft_probs.to(target_probs.device)

    draft_top1 = draft_probs.argmax(dim=-1).tolist()
    target_top1 = target_probs.argmax(dim=-1).tolist()
    top1_match = [int(d == t) for d, t in zip(draft_top1, target_top1)]

    per_pos_kl = F.kl_div(
        torch.log(target_probs + 1e-10), draft_probs, reduction="none", log_target=False,
    ).sum(dim=-1).tolist()

    if branch == "per_position_compare":
        final_ids = list(draft_ids)
        n_accepted = None
        bonus_token = None
        draft_eq_target_mode_count = sum(
            1 for i in range(len(draft_ids)) if draft_ids[i] == target_top1[i]
        )
    else:
        final_ids, n_accepted, bonus_token = mrs_verify(
            draft_ids, draft_probs, target_probs, step_map,
            verify_order_mode=mrs_order,
        )
        draft_eq_target_mode_count = None

    if ascii_demo:
        if branch == "mrs":
            s = print_mrs_result(
                draft_ids, final_ids, n_accepted, bonus_token,
                cfg.block_length, cfg.mask_token_id,
            )
        else:
            s = (
                f"[per_position_compare]  MRSfinal_ids  draft "
                f"draft_id==target_top1: {draft_eq_target_mode_count}/{cfg.block_length}"
            )
        print(s)
        ascii_lines.append(s)

    want_breakdown = (
        branch == "per_position_compare"
        or getattr(cfg, "per_position_breakdown", False)
    )
    per_position = _per_position_rows(
        draft_ids, draft_probs, target_probs, step_map, top1_match, per_pos_kl,
    ) if want_breakdown else None

    out = {
        "speculative_branch": branch,
        "mrs_verify_order": mrs_order if branch == "mrs" else None,
        "draft_ids": draft_ids,
        "final_ids": final_ids,
        "step_map": step_map,
        "n_accepted": n_accepted,
        "bonus_token": bonus_token,
        "top1_match": top1_match,
        "per_pos_kl": per_pos_kl,
    }
    if draft_eq_target_mode_count is not None:
        out["draft_eq_target_mode_count"] = draft_eq_target_mode_count
    if per_position is not None:
        out["per_position"] = per_position
    if ascii_demo and ascii_lines:
        out["_ascii_demo_text"] = "\n".join(ascii_lines)
    return out


def eval_single_block(draft_model, target_model, tokenizer, all_prompt_ids, cfg):
    """Run single-block test across all prompts, print summary."""
    run_dir = _run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, "sample_details.jsonl")
    print(f"\n{'='*60}")
    print(f"Single-Block Test: bl={cfg.block_length}, ds={cfg.denoising_steps}")
    print(
        f"  speculative_branch={getattr(cfg, 'speculative_branch', 'mrs')}  "
        f"mrs_verify_order={getattr(cfg, 'mrs_verify_order', 'position')}  "
        f"per_position_breakdown={getattr(cfg, 'per_position_breakdown', False)}"
    )
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    max_prompt_len = max(len(p) for p in all_prompt_ids)
    raw_max = max_prompt_len + 2 * cfg.block_length
    padded_len = ((raw_max + 127) // 128) * 128

    all_accepted = []
    all_top1 = []
    all_kl = []
    n_errors = 0

    ascii_demo = getattr(cfg, "ascii_demo", False)
    error_records = []  #  run_errors.jsonl results
    fout = open(output_file, "w")
    fout.write(
        "# : idx | prompt_len | speculative_branch | draft_ids | final_ids | "
        "step_map | n_accepted(None=MRS) | bonus_token | top1_match | per_pos_kl | per_position()\n"
    )
    for idx in tqdm(range(len(all_prompt_ids)), desc=f"[{cfg.tag}]"):
        prompt_ids = all_prompt_ids[idx]
        torch._dynamo.reset()
        show_ascii = ascii_demo and idx == 0
        if show_ascii:
            print(f"\n{'#'*60}\n  ASCII Demo: sample idx=0\n{'#'*60}")
        try:
            result = run_single_block(
                draft_model, target_model, prompt_ids, cfg, pad_token_id, padded_len,
                ascii_demo=show_ascii,
            )
        except Exception as e:
            n_errors += 1
            err_msg = f"{type(e).__name__}: {e}"
            error_records.append({"idx": idx, "prompt_len": len(prompt_ids), "error": err_msg})
            if n_errors <= 5:
                print(f"  [Error idx={idx}]: {e}")
            continue

        if getattr(cfg, "speculative_branch", "mrs") == "per_position_compare":
            all_accepted.append(result.get("draft_eq_target_mode_count", 0))
        else:
            all_accepted.append(result["n_accepted"])
        all_top1.extend(result["top1_match"])
        all_kl.extend(result["per_pos_kl"])

        _ascii = result.pop("_ascii_demo_text", None)
        if _ascii is not None:
            ascii_file = os.path.join(run_dir, "ascii_demo.txt")
            with open(ascii_file, "w") as af:
                af.write("=== ASCII Demo: block DraftTargetMRS  ===\n")
                af.write(f"sample_idx=0 tag={cfg.tag} bl={cfg.block_length} ds={cfg.denoising_steps}\n\n")
                af.write(_ascii)
            print(f"  ASCII demo saved to {ascii_file}")
        record = {"idx": idx, "prompt_len": len(prompt_ids), **result}
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

        if (idx + 1) % cfg.log_interval == 0:
            n = len(all_accepted)
            avg_acc = sum(all_accepted) / n
            avg_t1 = sum(all_top1) / len(all_top1) if all_top1 else 0
            avg_kl = sum(all_kl) / len(all_kl) if all_kl else 0
            if getattr(cfg, "speculative_branch", "mrs") == "per_position_compare":
                acc_label = "avg_draft_eq_target_top1"
            else:
                acc_label = "avg_accepted"
            print(f"\n  [{idx+1}/{len(all_prompt_ids)}] "
                  f"{acc_label}={avg_acc:.2f}/{cfg.block_length}  "
                  f"top1_match={avg_t1:.4f}  avg_kl={avg_kl:.4f}  errors={n_errors}")

    fout.close()
    if error_records:
        err_path = os.path.join(run_dir, "run_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as ef:
            for rec in error_records:
                ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {len(error_records)} failed samples logged to {err_path}")

    _print_single_block_summary(
        all_accepted, all_top1, all_kl, n_errors, cfg, output_file, run_dir,
    )


def _print_single_block_summary(all_accepted, all_top1, all_kl, n_errors, cfg, output_file, run_dir):
    n = len(all_accepted)
    if n == 0:
        print("  No valid results.")
        print("  :  draft/verify/MRS sample_details.jsonl ")
        print("   [Error idx=...] run_errors.jsonl")
        #  results
        summary = {
            "_desc": "block speculative decoding ",
            "valid_samples": 0,
            "errors": n_errors,
            "speculative_branch": getattr(cfg, "speculative_branch", "mrs"),
            "note": " GPU CUDA",
            "config": cfg.to_dict(),
        }
        summary_file = os.path.join(run_dir, "run_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        log_file = os.path.join(run_dir, "run_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== block speculative decoding===\n")
            f.write(f"errors={n_errors} total_prompts={n_errors} (approx)\n")
        print(f"  Summary (degenerate) saved to {summary_file}")
        return

    avg_acc = sum(all_accepted) / n
    avg_t1 = sum(all_top1) / len(all_top1) if all_top1 else 0
    avg_kl = sum(all_kl) / len(all_kl) if all_kl else 0

    acc_dist = defaultdict(int)
    for a in all_accepted:
        acc_dist[a] += 1

    ppc = getattr(cfg, "speculative_branch", "mrs") == "per_position_compare"
    acc_title = (
        "draft_id==target_top1 count / block (MRS )"
        if ppc
        else "MRS accepted tokens / block"
    )
    dist_title = (
        "Distribution of draft_id==target_top1 count"
        if ppc
        else "Acceptance distribution"
    )

    lines = [
        f"\n  --- Single-Block Summary (bl={cfg.block_length}, ds={cfg.denoising_steps}) ---",
        f"  speculative_branch: {getattr(cfg, 'speculative_branch', 'mrs')}",
        f"  Samples: {n}, Errors: {n_errors}",
        f"  {acc_title}: {avg_acc:.2f} / {cfg.block_length}",
        f"  Avg top1 match (argmax vs argmax):  {avg_t1:.4f}",
        f"  Avg KL(q||p):    {avg_kl:.4f}",
        f"\n  --- {dist_title} ---",
        f"  {'Count/bin':>10} {'Samples':>7} {'Fraction':>10}",
    ]
    for k in sorted(acc_dist.keys()):
        frac = acc_dist[k] / n
        lines.append(f"  {k:>10} {acc_dist[k]:>7} {frac:>10.4f}")

    for line in lines:
        print(line)

    summary = {
        "_desc": "block speculative decoding ",
        "valid_samples": n,
        "errors": n_errors,
        "speculative_branch": getattr(cfg, "speculative_branch", "mrs"),
        "avg_accepted": None if ppc else avg_acc,
        "avg_draft_eq_target_top1": avg_acc if ppc else None,
        "avg_top1_match": avg_t1,
        "avg_kl": avg_kl,
        "acceptance_distribution": dict(acc_dist),
        "config": cfg.to_dict(),
    }
    summary_file = os.path.join(run_dir, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {summary_file}")

    log_file = os.path.join(run_dir, "run_log.txt")
    with open(log_file, "w") as f:
        f.write("=== block speculative decoding  ===\n")
        f.write(f"tag={cfg.tag} bl={cfg.block_length} ds={cfg.denoising_steps}\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"  Log saved to {log_file}")

    if getattr(cfg, "save_plots", True):
        try:
            plot_path = plot_single_block(run_dir, output_file, cfg.block_length)
            if plot_path:
                print(f"  Plots saved to {plot_path}")
        except Exception as e:
            print(f"  [Warning] Plot failed: {e}")


# ──────────────────── Multi-Block Generation ──────────────────


def _cuda_sync(*devices):
    """Device-aware cuda sync.

    ``torch.cuda.synchronize()`` with no argument targets the current device
    (typically cuda:0), which does nothing useful when draft/target live on
    other devices and, combined with the Liger SwiGLU Triton kernel on a
    non-current device, triggers "Pointer argument (at 0) cannot be accessed
    from Triton (cpu tensor?)" on the next forward. Pass the actual
    draft/target devices so every device with pending work gets synced.
    """
    if not torch.cuda.is_available():
        return
    if not devices:
        torch.cuda.synchronize()
        return
    seen = set()
    for d in devices:
        if d is None:
            continue
        key = str(d) if not isinstance(d, torch.device) else (d.index if d.index is not None else d.type)
        if key in seen:
            continue
        seen.add(key)
        torch.cuda.synchronize(d)


def _redraft_partial_fill(
    draft_model, target_model, prompt_ids, committed_blocks, final_ids,
    cfg, pad_token_id, padded_len, draft_device, target_device,
):
    """Stage D: re-draft + re-verify the remaining positions of a partial block.

    See plan/08_v3_stale_pad_stage_d.md §4.2. When MRS rejects mid-block, the
    original target_probs[len(final_ids):] were conditioned on draft tokens
    that are no longer in the committed context (the rejected/resampled
    position). target-argmax pad over those probs therefore reads off a stale
    distribution. Stage D fix: re-run draft on the post-reject prefix
    [prompt + committed_blocks + final_ids] for n_remain positions, then
    re-run target verify under a synthetic step_map where final_ids is
    "revealed first" within the current draft block. The new
    target_probs[len(final_ids):] are conditioned on the actually-committed
    prefix; argmax of those is the correct fill.

    Returns the n_remain padding tokens (list[int]); empty if n_remain == 0.
    """
    n_filled = len(final_ids)
    n_remain = cfg.block_length - n_filled
    if n_remain <= 0:
        return []

    # 1) Re-draft on the post-reject prefix. block_length=n_remain so the
    #    diffusion only fills the open positions. cuda graph cache key
    #    includes block_length, so n_remain ∈ {1, 2, 3} triggers up to 3
    #    extra captures the first time they're hit.
    prefix = (
        list(prompt_ids)
        + [t for blk in committed_blocks for t in blk]
        + list(final_ids)
    )
    with torch.cuda.device(draft_device):
        new_draft_ids, _new_draft_probs, _new_step_map = draft_one_block(
            draft_model, prefix, n_remain, cfg.denoising_steps,
            draft_device, cfg.mask_token_id,
            use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            sampling=getattr(cfg, "draft_sampling", "argmax"),
        )

    # 2) Re-verify. target_verify_forward / build_verify_sequence assume a
    #    single block_length for every block, so we cannot pass final_ids as
    #    a "partial accepted block". Instead, treat the current draft block
    #    as the full cfg.block_length-wide block where final_ids occupy the
    #    leading positions and new_draft_ids fill the tail. The step_map
    #    encodes the visibility we want under the diffusion mask:
    #      • final_ids positions: step=0  token_label=1 (revealed first)
    #      • new_draft_ids positions: step=denoising_steps-1
    #        token_label=block_size, i.e. "revealed last" in the block
    #        (still strictly < mask token_label = block_size+1, so the mask
    #         predictions can attend to them; new positions can attend to
    #         final_ids; final_ids attend to nothing in-block  same as the
    #         original first-revealed semantics).
    synth_draft_ids = list(final_ids) + list(new_draft_ids)
    synth_step_map = (
        [0] * n_filled
        + [cfg.denoising_steps - 1] * n_remain
    )
    with torch.cuda.device(target_device):
        new_target_probs = target_verify_forward(
            target_model, prompt_ids, committed_blocks,
            synth_draft_ids, synth_step_map,
            cfg.block_length, cfg.block_size, pad_token_id, padded_len,
            cfg.mask_token_id,
            use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
            use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            share_mask_position=getattr(cfg, "share_mask_position", True),
        )

    return new_target_probs[n_filled:].argmax(dim=-1).tolist()


def speculative_generate(draft_model, target_model, prompt_ids, cfg,
                         pad_token_id, padded_len, eos_ids=None,
                         return_timings: bool = False):
    """
    Full speculative decoding loop: draft  verify  MRS per block.

    Dispatches to ``speculative_generate_pipelined`` when ``cfg.pipeline`` is
    True  the pipelined path overlaps draft_{N+1} with verify_N on dual GPUs.
    The serial path below remains untouched for the default case.

    Args:
        return_timings: If True, also return per-block GPU-timed breakdown
            (draft / target verify / MRS+padding). Third return value is a dict.

    Returns:
        (generated_ids, stats) or (generated_ids, stats, timing_info)
    """
    if getattr(cfg, "use_kv_cache", False):
        # Cache-aware K=1 nopipe path. Maintains DynamicCache for both models,
        # prefills prompt, then each spec iter only forwards the current block
        # attending to cached prefix. See cache_aware.py for design notes.
        # K>1, pipelined, and batched paths are not yet supported via cache.
        if getattr(cfg, "K", 1) > 1:
            raise NotImplementedError(
                "use_kv_cache=True with K>1 is not implemented yet. Set K=1 "
                "or disable use_kv_cache."
            )
        if getattr(cfg, "pipeline", False):
            from speculative_decoding.cache_aware import (
                speculative_generate_with_kv_cache_pipelined,
            )
            return speculative_generate_with_kv_cache_pipelined(
                draft_model, target_model, prompt_ids, cfg,
                pad_token_id, padded_len, eos_ids,
                return_timings=return_timings,
            )
        from speculative_decoding.cache_aware import (
            speculative_generate_with_kv_cache,
        )
        return speculative_generate_with_kv_cache(
            draft_model, target_model, prompt_ids, cfg,
            pad_token_id, padded_len, eos_ids,
            return_timings=return_timings,
        )
    if getattr(cfg, "pipeline", False):
        return speculative_generate_pipelined(
            draft_model, target_model, prompt_ids, cfg,
            pad_token_id, padded_len, eos_ids,
            return_timings=return_timings,
        )
    if getattr(cfg, "K", 1) > 1:
        return _speculative_generate_K(
            draft_model, target_model, prompt_ids, cfg,
            pad_token_id, padded_len, eos_ids,
            return_timings=return_timings,
        )
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    context_ids = list(prompt_ids)
    accepted_blocks = []
    all_generated = []
    stats = {
        "per_block_accepted": [],
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "total_bonus_tokens": 0,
    }
    per_block_timings = [] if return_timings else None

    for blk_idx in range(cfg.num_blocks):
        if return_timings:
            _cuda_sync(draft_device, target_device)
            t0 = time.perf_counter()

        # Align thread-local current device with the model's device  Liger
        # SwiGLU (and other non-dispatcher Triton kernels) reads
        # torch.cuda.current_stream() without a device arg, so it fetches the
        # stream of the current device. If that differs from the tensor
        # device, Triton launch fails with "Pointer argument (at 0) cannot
        # be accessed from Triton (cpu tensor?)". Seen on dual-GPU bench and
        # on any config where the default device (cuda:0) doesn't match the
        # model's device.
        # Early-stop: when draft_steps_per_block ∈ (0, denoising_steps), draft
        # only runs that many denoising iterations and leaves the rest as MASK
        # (step_map=-1). The cuda_graph cache key for draft is
        # (model, seq_len, block_length, device, mask_token_id)  it does NOT
        # include denoising_steps, so early-stop just calls g.replay() fewer
        # times against the same captured graph; safe to keep cuda_graph on.
        _early_k = int(getattr(cfg, "draft_steps_per_block", -1))
        _early_active = 0 < _early_k < cfg.denoising_steps
        _use_graph = getattr(cfg, "use_cuda_graph", False)
        with torch.cuda.device(draft_device):
            draft_ids, draft_probs, step_map = draft_one_block(
                draft_model, context_ids, cfg.block_length, cfg.denoising_steps,
                draft_device, cfg.mask_token_id,
                use_cuda_graph=_use_graph,
                sampling=getattr(cfg, "draft_sampling", "argmax"),
                early_stop_steps=_early_k,
            )
        # Count drafted (i.e. actually proposed by draft) tokens. With
        # early-stop, undrafted positions are NOT charged as draft tokens.
        n_drafted = sum(1 for s in step_map if s >= 0)
        stats["total_draft_tokens"] += n_drafted

        if return_timings:
            _cuda_sync(draft_device)
            t1 = time.perf_counter()

        with torch.cuda.device(target_device):
            target_probs = target_verify_forward(
                target_model, prompt_ids, accepted_blocks, draft_ids, step_map,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
                use_cuda_graph=_use_graph,
                share_mask_position=getattr(cfg, "share_mask_position", True),
            )

        # Dual-GPU placement: normalize device before MRS
        if draft_probs.device != target_probs.device:
            draft_probs = draft_probs.to(target_probs.device)

        # Early-stop: for positions the draft never proposed (step_map[i] == -1),
        # commit target argmax. We do this by overwriting draft_ids[i] with
        # target_probs[i].argmax() and draft_probs[i] with target_probs[i],
        # which makes MRS's q/p ratio = 1  guaranteed accept (MRS still runs in
        # position order; this never triggers a rejection on these positions).
        # Snapshot which positions were truly drafted so we can compute a
        # MRS-acceptance rate that excludes the auto-accepted positions.
        was_drafted = [s >= 0 for s in step_map]
        if _early_active and any(not d for d in was_drafted):
            for i in range(cfg.block_length):
                if not was_drafted[i]:
                    tgt_argmax = int(target_probs[i].argmax().item())
                    draft_ids[i] = tgt_argmax
                    draft_probs[i] = target_probs[i].clone()
                    # Restore step_map to a valid value to keep "step_map" verify
                    # ordering well-defined (we use position ordering by default).
                    step_map[i] = cfg.denoising_steps - 1

        if return_timings:
            _cuda_sync(target_device)
            t2 = time.perf_counter()

        mrs_order = getattr(cfg, "mrs_verify_order", "position")
        _verify_fn = _select_verify_fn(cfg)
        final_ids, n_accepted, bonus_token = _verify_fn(
            draft_ids, draft_probs, target_probs, step_map,
            verify_order_mode=mrs_order,
        )
        stats["per_block_accepted"].append(n_accepted)
        stats["total_accepted_tokens"] += n_accepted
        if bonus_token is not None:
            stats["total_bonus_tokens"] += 1

        # Real-draft acceptance accounting (early-stop aware). Auto-accepted
        # (non-drafted) positions never trigger MRS rejection, so any reject
        # is at a real-drafted position. Count both accept and "seen" only
        # over real-drafted slots to recover a meaningful α.
        accepted_real = sum(
            1 for i in range(min(n_accepted, cfg.block_length)) if was_drafted[i]
        )
        seen_real = accepted_real
        if n_accepted < cfg.block_length and was_drafted[n_accepted]:
            seen_real += 1
        stats.setdefault("total_real_drafted_accepted", 0)
        stats.setdefault("total_real_drafted_seen", 0)
        stats["total_real_drafted_accepted"] += accepted_real
        stats["total_real_drafted_seen"] += seen_real

        # Committed context layout: pad partial blocks with draft's original
        # samples (NOT mask_token_id). Rationale: block-diffusion training has
        # no MASK tokens in past committed blocks; committing MASKs puts target
        # attention in OOD state on subsequent blocks. draft_ids[j:] are legal
        # tokens drawn from draft's conditional distribution -- empirically
        # much closer to native's context than MASK, at the cost of deviating
        # slightly from strict MRS-lossless.
        block_tokens = list(final_ids)
        if len(block_tokens) < cfg.block_length:
            fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")
            if fill_mode == "truncate":
                # Stage E: commit as variable-length partial block (no pad).
                # build_verify_sequence handles via running offset; subsequent
                # block computes target_probs without the rejected/discarded
                # tail in its condition set.
                pass
            elif fill_mode == "truncate_no_bonus":
                # Stage G: like truncate but also drop the MRS bonus token
                # (= last element of final_ids when reject happened). Keeps
                # at least 1 token committed to avoid empty-block stalls.
                if len(block_tokens) >= 2:
                    block_tokens = block_tokens[:-1]
            elif fill_mode == "target_argmax_all":
                # Stage F: replace WHOLE block with target argmax  discards
                # MRS bonus and the accepted draft prefix. Tests whether the
                # accepted-prefix tokens are themselves polluting the context
                # vs. just the partial-pad tail.
                block_tokens = target_probs.argmax(dim=-1).tolist()
            elif fill_mode == "redraft":
                block_tokens += _redraft_partial_fill(
                    draft_model, target_model, prompt_ids,
                    accepted_blocks, block_tokens,
                    cfg, pad_token_id, padded_len,
                    draft_device, target_device,
                )
            elif fill_mode == "target_argmax":
                tgt_pad = target_probs.argmax(dim=-1).tolist()
                block_tokens += tgt_pad[len(block_tokens):cfg.block_length]
            else:
                block_tokens += list(draft_ids[len(block_tokens):cfg.block_length])

        accepted_blocks.append(block_tokens)
        context_ids.extend(block_tokens)
        all_generated.extend(final_ids)

        if return_timings:
            _cuda_sync(target_device)
            t3 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                "draft_s": t1 - t0,
                "target_verify_s": t2 - t1,
                "mrs_and_commit_s": t3 - t2,
                "block_wall_s": t3 - t0,
            })

        if eos_ids and any(t in eos_ids for t in final_ids):
            break

    while all_generated and all_generated[-1] == cfg.mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated, stats

    timing_info = {
        "per_block": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_and_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
    }
    return all_generated, stats, timing_info


def _speculative_generate_K(draft_model, target_model, prompt_ids, cfg,
                            pad_token_id, padded_len, eos_ids=None,
                            return_timings: bool = False):
    """K>1 speculative loop: per iter, draft K blocks then one target verify
    over all K. MRS runs block-by-block; on first reject the remaining draft
    blocks in this iter are discarded (their context is now invalid).

    Rollback: the `committed_blocks` / `context_ids` snapshot is the ground-
    truth anchor. draft blocks are speculatively appended to a working copy
    but only promoted to committed after MRS accepts (or re-samples) them.
    On first-reject within an iter, the working copy is truncated back to
    committed_blocks + re-sampled block, and the iter ends.
    """
    K = int(cfg.K)
    assert cfg.num_blocks % K == 0, f"num_blocks={cfg.num_blocks} % K={K} != 0"
    iters = cfg.num_blocks // K

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    committed_blocks: list = []        # fully-verified blocks, ground truth
    all_generated: list = []
    stats = {
        "per_block_accepted": [],
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "total_bonus_tokens": 0,
    }
    per_block_timings = [] if return_timings else None
    early_stop = False

    for it in range(iters):
        if early_stop:
            break

        # ---- phase 1: draft K blocks optimistically on top of committed context ----
        speculative_context = list(prompt_ids) + [
            tok for blk in committed_blocks for tok in blk
        ]
        draft_blocks: list = []
        draft_probs_list: list = []
        step_maps: list = []

        if return_timings:
            _cuda_sync(draft_device, target_device)
            t_iter_start = time.perf_counter()

        with torch.cuda.device(draft_device):
            for _k in range(K):
                draft_ids, draft_probs, step_map = draft_one_block(
                    draft_model, speculative_context,
                    cfg.block_length, cfg.denoising_steps,
                    draft_device, cfg.mask_token_id,
                    use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
                    sampling=getattr(cfg, "draft_sampling", "argmax"),
                )
                draft_blocks.append(draft_ids)
                draft_probs_list.append(draft_probs)
                step_maps.append(step_map)
                speculative_context = speculative_context + list(draft_ids)
                stats["total_draft_tokens"] += cfg.block_length

        if return_timings:
            _cuda_sync(draft_device)
            t_after_draft = time.perf_counter()

        # ---- phase 2: one target verify over all K draft blocks ----
        with torch.cuda.device(target_device):
            target_probs_list = target_verify_forward_multi(
                target_model, prompt_ids, committed_blocks,
                draft_blocks, step_maps,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
                use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
                share_mask_position=getattr(cfg, "share_mask_position", True),
            )

        if return_timings:
            _cuda_sync(target_device)
            t_after_verify = time.perf_counter()

        # ---- phase 3: MRS block-by-block with first-reject early-exit ----
        mrs_order = getattr(cfg, "mrs_verify_order", "position")
        _verify_fn = _select_verify_fn(cfg)
        for k in range(K):
            draft_probs_k = draft_probs_list[k]
            target_probs_k = target_probs_list[k]
            if draft_probs_k.device != target_probs_k.device:
                draft_probs_k = draft_probs_k.to(target_probs_k.device)

            final_ids, n_accepted, bonus_token = _verify_fn(
                draft_blocks[k], draft_probs_k, target_probs_k, step_maps[k],
                verify_order_mode=mrs_order,
            )
            stats["per_block_accepted"].append(n_accepted)
            stats["total_accepted_tokens"] += n_accepted
            if bonus_token is not None:
                stats["total_bonus_tokens"] += 1

            # See note at line 567 (single-block path): pad with draft's
            # original samples, not MASK, to keep committed context in-dist.
            block_tokens = list(final_ids)
            if len(block_tokens) < cfg.block_length:
                fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")
                if fill_mode == "truncate":
                    # Stage E: variable-length commit (no pad).
                    pass
                elif fill_mode == "truncate_no_bonus":
                    # Stage G: drop MRS bonus token before truncating.
                    if len(block_tokens) >= 2:
                        block_tokens = block_tokens[:-1]
                elif fill_mode == "target_argmax_all":
                    # Stage F: replace WHOLE block with target argmax.
                    block_tokens = target_probs_k.argmax(dim=-1).tolist()
                elif fill_mode == "redraft":
                    block_tokens += _redraft_partial_fill(
                        draft_model, target_model, prompt_ids,
                        committed_blocks, block_tokens,
                        cfg, pad_token_id, padded_len,
                        draft_device, target_device,
                    )
                elif fill_mode == "target_argmax":
                    tgt_pad = target_probs_k.argmax(dim=-1).tolist()
                    block_tokens += tgt_pad[len(block_tokens):cfg.block_length]
                else:
                    block_tokens += list(
                        draft_blocks[k][len(block_tokens):cfg.block_length]
                    )
            committed_blocks.append(block_tokens)
            all_generated.extend(final_ids)

            rejected = n_accepted < cfg.block_length

            if eos_ids and any(t in eos_ids for t in final_ids):
                early_stop = True

            if rejected or early_stop:
                # remaining draft blocks in this iter were speculated against
                # a context that no longer matches committed  discard them.
                break

        if return_timings:
            _cuda_sync(target_device)
            t_iter_end = time.perf_counter()
            per_block_timings.append({
                "iter_idx": it,
                "draft_s": t_after_draft - t_iter_start,
                "target_verify_s": t_after_verify - t_after_draft,
                "mrs_and_commit_s": t_iter_end - t_after_verify,
                "block_wall_s": t_iter_end - t_iter_start,
                "K": K,
            })

    while all_generated and all_generated[-1] == cfg.mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated, stats

    timing_info = {
        "per_block": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_and_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
    }
    return all_generated, stats, timing_info


def speculative_generate_batch(draft_model, target_model, prompts_batch, cfg,
                               pad_token_id, padded_len, eos_ids=None,
                               return_timings: bool = False):
    """Batched speculative generation.

    Per-row independent context, MRS, and (under no_eos_stop) synchronized
    block count. Calls ``draft_one_block_batch`` × K + a single batched
    ``target_verify_forward_multi_batch`` per spec iter, then ``mrs_verify``
    in a per-row Python loop.

    Returns:
      list of (generated_ids, stats) tuples (length B), and optionally a list
      of timing dicts (length B; same shape as the single-prompt path's).

    Notes:
      - Rows must share the same ctx_len at the start of each draft call.
        We enforce this by left-padding all prompts to the longest ctx_len
        with ``pad_token_id``. Under ``--no_eos_stop`` and a fixed
        ``num_blocks``, every row commits exactly ``cfg.block_length`` tokens
        per spec iter, so context lengths stay synchronized for the whole run.
      - K>1: per-row early-exit flag. Once row i hits a reject inside the
        K-block iter, we stop committing more blocks for row i in this iter
        (the remaining draft blocks were speculated against an invalidated
        context). Other rows continue.
      - EOS handling: ``--no_eos_stop`` (eos_ids=None) is the supported mode
        for batch>1. When eos_ids is provided, we keep all rows in lockstep
        and do not early-exit individual rows; the whole batch breaks when
        ANY row emits an EOS token. This matches single-prompt semantics for
        a synchronized-decode bench but isn't the right policy for serving.
    """
    B = len(prompts_batch)
    assert B > 0
    K = int(getattr(cfg, "K", 1))
    assert cfg.num_blocks % K == 0, f"num_blocks={cfg.num_blocks} % K={K} != 0"
    iters = cfg.num_blocks // K

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    # Left-pad all prompts to the longest length to keep ctx_len uniform across
    # rows (required by draft_one_block_batch). Pad token = pad_token_id.
    max_plen = max(len(p) for p in prompts_batch)
    padded_prompts = [
        [pad_token_id] * (max_plen - len(p)) + list(p)
        for p in prompts_batch
    ]
    context_ids_batch = [list(p) for p in padded_prompts]
    accepted_blocks_batch: list = [[] for _ in range(B)]
    all_generated_batch: list = [[] for _ in range(B)]
    stats_batch = [
        {
            "per_block_accepted": [],
            "total_draft_tokens": 0,
            "total_accepted_tokens": 0,
            "total_bonus_tokens": 0,
        }
        for _ in range(B)
    ]
    per_block_timings_batch: list = [[] for _ in range(B)] if return_timings else None
    early_stop = False  # batch-wide; only used when eos_ids is set

    mrs_order = getattr(cfg, "mrs_verify_order", "position")
    _verify_fn = _select_verify_fn(cfg)
    use_cuda_graph = getattr(cfg, "use_cuda_graph", False)

    for it in range(iters):
        if early_stop:
            break

        if return_timings:
            _cuda_sync(draft_device, target_device)
            t_iter_start = time.perf_counter()

        # ---- phase 1: draft K blocks (per-row context, batched forwards) ----
        # speculative_context_batch is what we feed into draft each step;
        # it accumulates across the K draft blocks within this iter.
        speculative_context_batch = [list(c) for c in context_ids_batch]
        draft_blocks_per_k: list = []        # K × (B × bl)
        draft_probs_per_k: list = []         # K × Tensor(B, bl, V)
        step_maps_per_k: list = []           # K × (B × bl)

        with torch.cuda.device(draft_device):
            for _k in range(K):
                draft_ids_b, draft_probs_b, step_map_b = draft_one_block_batch(
                    draft_model, speculative_context_batch,
                    cfg.block_length, cfg.denoising_steps,
                    draft_device, cfg.mask_token_id,
                    use_cuda_graph=use_cuda_graph,
                    sampling=getattr(cfg, "draft_sampling", "argmax"),
                )
                draft_blocks_per_k.append(draft_ids_b)
                draft_probs_per_k.append(draft_probs_b)
                step_maps_per_k.append(step_map_b)
                for i in range(B):
                    speculative_context_batch[i] = (
                        speculative_context_batch[i] + list(draft_ids_b[i])
                    )
                    stats_batch[i]["total_draft_tokens"] += cfg.block_length

        if return_timings:
            _cuda_sync(draft_device)
            t_after_draft = time.perf_counter()

        # ---- phase 2: one batched target verify over K draft blocks ----
        # Reshape per-row K blocks: draft_blocks_batch[i] = [block_0, ..., block_{K-1}]
        draft_blocks_batch = [
            [draft_blocks_per_k[k][i] for k in range(K)] for i in range(B)
        ]
        step_maps_batch = [
            [step_maps_per_k[k][i] for k in range(K)] for i in range(B)
        ]
        with torch.cuda.device(target_device):
            target_probs_batch = target_verify_forward_multi_batch(
                target_model, padded_prompts, accepted_blocks_batch,
                draft_blocks_batch, step_maps_batch,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
                use_cuda_graph=use_cuda_graph,
                share_mask_position=getattr(cfg, "share_mask_position", True),
            )
        # target_probs_batch: list[list[Tensor(bl, V)]] of shape (B, K)

        if return_timings:
            _cuda_sync(target_device)
            t_after_verify = time.perf_counter()

        # ---- phase 3: per-row MRS block-by-block, per-row early-exit ----
        # Rejected rows must still commit draft tokens for the remaining
        # K blocks in this iter so all rows' context_ids stay length-aligned
        # (draft_one_block_batch requires uniform ctx_len across rows). The
        # discarded blocks are not counted as accepted in stats / generated.
        rejected_in_iter = [False] * B  # row-level reject flag for this iter
        for k in range(K):
            for i in range(B):
                draft_ids = draft_blocks_per_k[k][i]
                if rejected_in_iter[i]:
                    # Fallback: commit draft tokens as-is (no MRS), keep batch
                    # rows aligned but don't count toward accepted/generated.
                    block_tokens = list(draft_ids)
                    accepted_blocks_batch[i].append(block_tokens)
                    context_ids_batch[i].extend(block_tokens)
                    continue

                # draft_probs_per_k[k] is a CPU tensor (B, bl, V); index per-row
                draft_probs_i = draft_probs_per_k[k][i]
                target_probs_i = target_probs_batch[i][k]
                if draft_probs_i.device != target_probs_i.device:
                    draft_probs_i = draft_probs_i.to(target_probs_i.device)

                final_ids, n_accepted, bonus_token = _verify_fn(
                    draft_ids, draft_probs_i, target_probs_i,
                    step_maps_per_k[k][i],
                    verify_order_mode=mrs_order,
                )
                stats_batch[i]["per_block_accepted"].append(n_accepted)
                stats_batch[i]["total_accepted_tokens"] += n_accepted
                if bonus_token is not None:
                    stats_batch[i]["total_bonus_tokens"] += 1

                # Pad partial block with draft's own samples, same as single path.
                block_tokens = list(final_ids)
                if len(block_tokens) < cfg.block_length:
                    fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")
                    if fill_mode == "truncate":
                        # Stage E: variable-length commit (no pad). Note: rows
                        # in this batch may now have different total committed
                        # lengths, which is OK because per-row context_ids
                        # advances by len(block_tokens) (line below); the only
                        # batch-alignment requirement was for draft_one_block_batch
                        # uniform ctx_len, which still holds because every row
                        # committed by exactly len(block_tokens) tokens.
                        # (Rows that rejected on different positions will diverge
                        # in ctx_len; if that breaks batched draft, this mode
                        # falls back to per-row sequential. Acceptable for
                        # ablation  production batch should not use truncate.)
                        pass
                    elif fill_mode == "truncate_no_bonus":
                        # Stage G: drop MRS bonus token before truncating.
                        if len(block_tokens) >= 2:
                            block_tokens = block_tokens[:-1]
                    elif fill_mode == "target_argmax_all":
                        # Stage F: replace WHOLE block with target argmax.
                        block_tokens = target_probs_i.argmax(dim=-1).tolist()
                    elif fill_mode == "redraft":
                        # Per-row, single-prompt redraft. padded_prompts[i] is
                        # the left-padded prompt the target verify pipeline
                        # uses for this row; pass it as prompt_ids to keep
                        # build_verify_sequence position arithmetic consistent.
                        block_tokens += _redraft_partial_fill(
                            draft_model, target_model, padded_prompts[i],
                            accepted_blocks_batch[i], block_tokens,
                            cfg, pad_token_id, padded_len,
                            draft_device, target_device,
                        )
                    elif fill_mode == "target_argmax":
                        tgt_pad = target_probs_i.argmax(dim=-1).tolist()
                        block_tokens += tgt_pad[len(block_tokens):cfg.block_length]
                    else:
                        block_tokens += list(
                            draft_ids[len(block_tokens):cfg.block_length]
                        )
                accepted_blocks_batch[i].append(block_tokens)
                context_ids_batch[i].extend(block_tokens)
                all_generated_batch[i].extend(final_ids)

                if n_accepted < cfg.block_length:
                    rejected_in_iter[i] = True

                if eos_ids and any(t in eos_ids for t in final_ids):
                    early_stop = True

            if early_stop:
                break

        if return_timings:
            _cuda_sync(target_device)
            t_iter_end = time.perf_counter()
            for i in range(B):
                per_block_timings_batch[i].append({
                    "iter_idx": it,
                    "draft_s": t_after_draft - t_iter_start,
                    "target_verify_s": t_after_verify - t_after_draft,
                    "mrs_and_commit_s": t_iter_end - t_after_verify,
                    "block_wall_s": t_iter_end - t_iter_start,
                    "K": K,
                })

    # Strip trailing MASKs from each row and build per-row return tuples.
    out_rows = []
    for i in range(B):
        gen = all_generated_batch[i]
        while gen and gen[-1] == cfg.mask_token_id:
            gen.pop()
        if return_timings:
            timing_info = {
                "per_block": per_block_timings_batch[i],
                "total_draft_s": sum(b["draft_s"] for b in per_block_timings_batch[i]),
                "total_target_verify_s": sum(
                    b["target_verify_s"] for b in per_block_timings_batch[i]
                ),
                "total_mrs_and_commit_s": sum(
                    b["mrs_and_commit_s"] for b in per_block_timings_batch[i]
                ),
                "total_block_wall_s": sum(
                    b["block_wall_s"] for b in per_block_timings_batch[i]
                ),
            }
            out_rows.append((gen, stats_batch[i], timing_info))
        else:
            out_rows.append((gen, stats_batch[i]))
    return out_rows


def speculative_generate_pipelined(draft_model, target_model, prompt_ids, cfg,
                                   pad_token_id, padded_len, eos_ids=None,
                                   return_timings: bool = False):
    """
    Pipelined speculative decoding: draft_{N+1} runs concurrently with verify_N.

    ──  overlap ──
     dual-GPU  (draft_device != target_device),
      • draft_one_block    draft_device  forward +  .cpu()/.item() sync,
         sync  draft GPU  stream,  target GPU;
      • target_verify_forward  kernel  target GPU  stream,
         mrs_verify  target_probs  non-blocking .
     "launch verify_N"  "run draft_{N+1}"
    MRS_N  ( Python  verify_N,  draft_{N+1}),
    verify_N  draft_{N+1} .
     ≈ max(T_draft, T_verify) + T_mrs,
    T_draft + T_verify + T_mrs  min(T_draft, T_verify).

    ──  +  ──
    draft_{N+1}  context.  MRS_N ,  block_N .
    :  block_N ** ( block_tokens == draft_ids_N),
     context_ids + draft_ids_N  opt_context  draft_{N+1}.
     MRS_N :
      •  (n_accepted == block_length): opt_context  context,
        draft_{N+1} , .
      •  (~15%  block  rejection,  reject  MRS  residual
        ): opt_context  context , draft_{N+1} ,
         context  draft_{N+1}.
    bonus_token  context_ids,  n_accepted == block_length .

    ──  ──
     draft_one_block / target_verify_forward / mrs_verify ,
    ; stats / timing / eos  speculative_generate .
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    context_ids = list(prompt_ids)
    accepted_blocks = []
    all_generated = []
    stats = {
        "per_block_accepted": [],
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "total_bonus_tokens": 0,
        "pipeline_reused_next_draft": 0,   # : draft_{N+1}
        "pipeline_rollback_next_draft": 0, # : draft_{N+1}
    }
    per_block_timings = [] if return_timings else None

    def _run_draft(ctx):
        #  draft_device  Liger/Triton pin .
        with torch.cuda.device(draft_device):
            return draft_one_block(
                draft_model, ctx, cfg.block_length, cfg.denoising_steps,
                draft_device, cfg.mask_token_id,
                use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
                sampling=getattr(cfg, "draft_sampling", "argmax"),
            )

    def _run_verify(d_ids, s_map):
        #  enqueue verify kernel,  cuda_sync;  tensor  CPU
        # ( mrs_verify  .item())  target_device.
        # : return_on_device=True  target_verify_forward  .cpu(),
        #  D2H  CPU  verify forward ,
        # draft_{N+1}  overlap .
        with torch.cuda.device(target_device):
            return target_verify_forward(
                target_model, prompt_ids, accepted_blocks, d_ids, s_map,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
                return_on_device=True,
                use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
                share_mask_position=getattr(cfg, "share_mask_position", True),
            )

    # :  block 0  draft.
    draft_ids, draft_probs, step_map = _run_draft(context_ids)
    stats["total_draft_tokens"] += cfg.block_length

    mrs_order = getattr(cfg, "mrs_verify_order", "position")
    _verify_fn = _select_verify_fn(cfg)

    for blk_idx in range(cfg.num_blocks):
        if return_timings:
            _cuda_sync(draft_device, target_device)
            t0 = time.perf_counter()

        # (1)  verify_N  target GPU (non-blocking launch).
        #      GPU tensor,  mrs_verify  async.
        target_probs = _run_verify(draft_ids, step_map)

        # (2) ,  draft_{N+1} .
        #      target GPU  verify_N, ,  min(T_d, T_v).
        is_last = (blk_idx == cfg.num_blocks - 1)
        next_draft = None
        if not is_last:
            opt_context = context_ids + list(draft_ids)
            # _run_draft  draft GPU  .cpu() / .item() ,
            # draft stream; target  verify_N kernel .
            next_draft = _run_draft(opt_context)

        # (3)  draft_probs  target_probs  (dual-GPU ).
        #     mrs_verify  CPU  target_probs   target GPU.
        if draft_probs.device != target_probs.device:
            draft_probs = draft_probs.to(target_probs.device)

        if return_timings:
            #  "verify kernel launch  MRS " ;
            #  pipelined  draft_{N+1} overlap .
            _cuda_sync(target_device)
            t1 = time.perf_counter()

        # (4) MRS_N:  sync .  n_accepted, .
        final_ids, n_accepted, bonus_token = _verify_fn(
            draft_ids, draft_probs, target_probs, step_map,
            verify_order_mode=mrs_order,
        )
        stats["per_block_accepted"].append(n_accepted)
        stats["total_accepted_tokens"] += n_accepted
        if bonus_token is not None:
            stats["total_bonus_tokens"] += 1

        # (5)  block_N  context. block_tokens  block_length
        #     (bonus_token  context   serial ).
        #     6  fill /K-wide . truncate / target_argmax_all
        #      reject  block_tokens == draft_ids,  (6)
        #     full_accept .
        block_tokens = list(final_ids)
        if len(block_tokens) < cfg.block_length:
            fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")
            if fill_mode == "truncate":
                pass
            elif fill_mode == "truncate_no_bonus":
                if len(block_tokens) >= 2:
                    block_tokens = block_tokens[:-1]
            elif fill_mode == "target_argmax_all":
                block_tokens = target_probs.argmax(dim=-1).tolist()
            elif fill_mode == "redraft":
                # Stage D: extra draft+verify on post-reject prefix. Runs
                # synchronously inside this iter, eating into pipeline overlap
                # for reject blocks. truncate (Stage E) avoids this cost.
                block_tokens += _redraft_partial_fill(
                    draft_model, target_model, prompt_ids,
                    accepted_blocks, block_tokens,
                    cfg, pad_token_id, padded_len,
                    draft_device, target_device,
                )
            elif fill_mode == "target_argmax":
                tgt_pad = target_probs.argmax(dim=-1).tolist()
                block_tokens += tgt_pad[len(block_tokens):cfg.block_length]
            else:
                block_tokens += list(draft_ids[len(block_tokens):cfg.block_length])

        accepted_blocks.append(block_tokens)
        context_ids.extend(block_tokens)
        all_generated.extend(final_ids)

        # (6)  draft :  or .
        if not is_last:
            # Block_tokens != draft_ids  ( partial-fill  reject )
            # opt_context  context , . truncate
            # len(block_tokens) < block_length  fall through  rollback.
            full_accept = (
                n_accepted == cfg.block_length
                and list(block_tokens) == list(draft_ids)
            )
            if full_accept:
                # opt_context == context_ids, next_draft .
                draft_ids, draft_probs, step_map = next_draft
                stats["pipeline_reused_next_draft"] += 1
            else:
                # :  next_draft,  context .
                #  pipelined  "draft ",  (1 - α^block_length)
                # ;  α  (≳ 0.8) .
                draft_ids, draft_probs, step_map = _run_draft(context_ids)
                stats["pipeline_rollback_next_draft"] += 1
            stats["total_draft_tokens"] += cfg.block_length

        if return_timings:
            _cuda_sync(draft_device, target_device)
            t3 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                #  pipelined  "draft_s" ,
                #  0  target_verify_s / mrs_and_commit_s
                # .  block ,  speedup
                #  serial  block_wall_s.
                "draft_s": 0.0,
                "target_verify_s": t1 - t0,
                "mrs_and_commit_s": t3 - t1,
                "block_wall_s": t3 - t0,
            })

        if eos_ids and any(t in eos_ids for t in final_ids):
            break

    while all_generated and all_generated[-1] == cfg.mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated, stats

    timing_info = {
        "per_block": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_and_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "pipeline": True,
    }
    return all_generated, stats, timing_info


def eval_multi_block(draft_model, target_model, tokenizer, all_prompt_ids, cfg):
    """Run multi-block speculative decoding across all prompts."""
    run_dir = _run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    output_file = os.path.join(run_dir, "sample_details.jsonl")
    print(f"\n{'='*60}")
    print(f"Multi-Block Spec Decode: bl={cfg.block_length}, ds={cfg.denoising_steps}, nb={cfg.num_blocks}")
    print(f"Output: {output_file}")
    print(f"{'='*60}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None

    max_prompt_len = max(len(p) for p in all_prompt_ids)
    # verify  = prompt + [accepted × 2bl] + [current × 2bl];
    # accepted=num_blocks-1 + current=1   num_blocks  2  bl tokens
    raw_max = max_prompt_len + cfg.num_blocks * 2 * cfg.block_length
    padded_len = ((raw_max + 127) // 128) * 128

    all_accept_rates = []
    all_per_block = []
    n_errors = 0
    error_records = []

    fout = open(output_file, "w")
    fout.write("# : idx | prompt_len | gen_len | accept_rate | per_block_accepted | total_draft | total_accepted | total_bonus | gen_text_preview\n")
    for idx in tqdm(range(len(all_prompt_ids)), desc=f"[{cfg.tag}]"):
        prompt_ids = all_prompt_ids[idx]
        # NOTE: torch._dynamo.reset() was here but it invalidates the
        # Inductor-compiled flex_attention Triton kernels used by the patched
        # multi-block causal mask, causing "Triton [CUDA]: invalid resource
        # handle" on the next launch. The compile cache is bounded anyway
        # (one entry per unique shape), so dropping the reset is safe.
        try:
            generated, stats = speculative_generate(
                draft_model, target_model, prompt_ids, cfg,
                pad_token_id, padded_len, eos_ids,
            )
        except Exception as e:
            n_errors += 1
            error_records.append({"idx": idx, "prompt_len": len(prompt_ids), "error": f"{type(e).__name__}: {e}"})
            if n_errors <= 5:
                print(f"  [Error idx={idx}]: {e}")
            continue

        total_draft = stats["total_draft_tokens"]
        total_accepted = stats["total_accepted_tokens"]
        accept_rate = total_accepted / total_draft if total_draft > 0 else 0.0
        all_accept_rates.append(accept_rate)
        all_per_block.extend(stats["per_block_accepted"])

        gen_text = tokenizer.decode(generated, skip_special_tokens=True)
        record = {
            "idx": idx, "prompt_len": len(prompt_ids), "gen_len": len(generated),
            "accept_rate": accept_rate,
            "per_block_accepted": stats["per_block_accepted"],
            "total_draft": total_draft, "total_accepted": total_accepted,
            "total_bonus": stats["total_bonus_tokens"],
            "gen_text_preview": gen_text[:200],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

        if (idx + 1) % cfg.log_interval == 0:
            avg_ar = sum(all_accept_rates) / len(all_accept_rates)
            avg_pb = sum(all_per_block) / len(all_per_block) if all_per_block else 0
            print(f"\n  [{idx+1}/{len(all_prompt_ids)}] avg_accept_rate={avg_ar:.4f}  "
                  f"avg_per_block={avg_pb:.2f}/{cfg.block_length}  errors={n_errors}")

    fout.close()
    if error_records:
        err_path = os.path.join(run_dir, "run_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as ef:
            for rec in error_records:
                ef.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {len(error_records)} failed samples logged to {err_path}")

    _print_multi_block_summary(all_accept_rates, all_per_block, n_errors, cfg, output_file, run_dir)


def _print_multi_block_summary(all_accept_rates, all_per_block, n_errors, cfg, output_file, run_dir):
    n = len(all_accept_rates)
    if n == 0:
        print("  No valid results.")
        print("  : sample_details.jsonl ")
        summary = {
            "_desc": "block speculative decoding ",
            "valid_samples": 0,
            "errors": n_errors,
            "note": " [Error idx=...]  run_errors.jsonl",
            "config": cfg.to_dict(),
        }
        with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(os.path.join(run_dir, "run_log.txt"), "w", encoding="utf-8") as f:
            f.write(f"errors={n_errors}\n")
        print(f"  Summary (degenerate) saved to {os.path.join(run_dir, 'run_summary.json')}")
        return

    avg_accept = sum(all_accept_rates) / n
    avg_per_block = sum(all_per_block) / len(all_per_block) if all_per_block else 0

    acc_dist = defaultdict(int)
    for a in all_per_block:
        acc_dist[a] += 1

    lines = [
        f"\n  --- Multi-Block Summary (bl={cfg.block_length}, ds={cfg.denoising_steps}, nb={cfg.num_blocks}) ---",
        f"  Samples: {n}, Errors: {n_errors}",
        f"  Avg accept rate:        {avg_accept:.4f}",
        f"  Avg accepted per block: {avg_per_block:.2f} / {cfg.block_length}",
        f"\n  --- Acceptance distribution ---",
        f"  {'Accepted':>10} {'Count':>7} {'Fraction':>10}",
    ]
    total_blk = len(all_per_block)
    for k in sorted(acc_dist.keys()):
        frac = acc_dist[k] / total_blk if total_blk > 0 else 0
        lines.append(f"  {k:>10} {acc_dist[k]:>7} {frac:>10.4f}")

    for line in lines:
        print(line)

    summary = {
        "_desc": "block speculative decoding ",
        "valid_samples": n,
        "errors": n_errors,
        "avg_accept_rate": avg_accept,
        "avg_per_block_accepted": avg_per_block,
        "acceptance_distribution": dict(acc_dist),
        "config": cfg.to_dict(),
    }
    summary_file = os.path.join(run_dir, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved to {summary_file}")

    log_file = os.path.join(run_dir, "run_log.txt")
    with open(log_file, "w") as f:
        f.write("=== block speculative decoding  ===\n")
        f.write(f"tag={cfg.tag} bl={cfg.block_length} nb={cfg.num_blocks}\n\n")
        for line in lines:
            f.write(line + "\n")
    print(f"  Log saved to {log_file}")

    if getattr(cfg, "save_plots", True):
        try:
            plot_path = plot_multi_block(run_dir, output_file, cfg.block_length)
            if plot_path:
                print(f"  Plots saved to {plot_path}")
        except Exception as e:
            print(f"  [Warning] Plot failed: {e}")


# ──────────────────────────── Data Loading ────────────────────────────────────


def load_prompts(tokenizer, cfg):
    """Load dataset and tokenize prompts."""
    if cfg.dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split=cfg.dataset_split)
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            messages = [{"role": "user", "content": ds[idx]["question"]}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompt_ids.append(tokenizer.encode(prompt, add_special_tokens=False))
    elif cfg.dataset == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            prompt_text = ds[idx]["prompt"]
            messages = [{"role": "user", "content":
                         f"Read the following function signature and docstring, "
                         f"and fully implement the function described. "
                         f"Your response should only contain the code for this function.\n{prompt_text}"}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            all_prompt_ids.append(tokenizer.encode(prompt, add_special_tokens=False))
    elif cfg.dataset == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            prompt_text = ds[idx]["prompt"]
            test_list = ds[idx].get("test_list") or []
            tests_block = "\n".join(test_list) if test_list else ""
            user_content = (
                "You are a Python coding assistant. Write a Python function "
                "that satisfies the problem description and passes the given "
                "tests. Return only the function code.\n\n"
                f"Problem:\n{prompt_text}\n\nTests:\n{tests_block}"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=384).input_ids
            all_prompt_ids.append(ids)
    elif cfg.dataset == "ifeval":
        ds = load_dataset("google/IFEval", split="train")
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            prompt_text = ds[idx]["prompt"]
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=384).input_ids
            all_prompt_ids.append(ids)
    elif cfg.dataset == "mmlu":
        # MMLU "all" split  4-way multiple choice across 57 subjects.
        # Default test split has ~14k items; we cap to cfg.num_samples.
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            q = ds[idx]["question"]
            ch = ds[idx]["choices"]
            user_content = (
                "The following is a multiple-choice question. Output only the "
                "letter (A, B, C, or D) of the correct answer.\n\n"
                f"Question: {q}\n"
                f"A. {ch[0]}\n"
                f"B. {ch[1]}\n"
                f"C. {ch[2]}\n"
                f"D. {ch[3]}\n\n"
                "Answer:"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=512).input_ids
            all_prompt_ids.append(ids)
    elif cfg.dataset == "triviaqa":
        # closed-book TriviaQA: question only, no Wikipedia context
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
        ds = ds.select(range(min(cfg.num_samples, len(ds))))
        all_prompt_ids = []
        for idx in range(len(ds)):
            question = ds[idx]["question"]
            user_content = (
                "Answer the following trivia question with a short factual "
                "answer (a name, place, date, or short phrase). "
                "Respond with the answer only, no explanation.\n\n"
                f"Question: {question}\nAnswer:"
            )
            messages = [{"role": "user", "content": user_content}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=384).input_ids
            all_prompt_ids.append(ids)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    return ds, all_prompt_ids


# ──────────────────────────── Entry Point ─────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Speculative Decoding for SDAR")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default=None, choices=["single_block", "multi_block"])
    parser.add_argument("--draft_model", type=str, default=None)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--denoising_steps", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--draft_device", type=str, default=None)
    parser.add_argument("--target_device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--save_plots", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument("--ascii_demo", type=lambda x: x.lower() in ("1", "true", "yes"), default=None)
    parser.add_argument(
        "--block_causal_prompt",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help="True=SDAR block-level causal; False=legacy prompt sees mask",
    )
    parser.add_argument(
        "--speculative_branch",
        type=str,
        default=None,
        choices=["mrs", "per_position_compare"],
        help="mrs=MRS; per_position_compare=MRS( single_block)",
    )
    parser.add_argument(
        "--mrs_verify_order",
        type=str,
        default=None,
        choices=["step_map", "position"],
        help="MRS : step_map=unmask; position=0L-1",
    )
    parser.add_argument(
        "--per_position_breakdown",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=None,
        help=" jsonl  per_position mrs ",
    )
    args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(args).items() if k != "config"}

    if args.config:
        cfg = load_config(args.config, cli_overrides)
    else:
        cfg = default_config(cli_overrides)

    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)

    run_dir = _run_dir(cfg)
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nReport date: {report_date_str()}  (override: SPEC_DECODE_REPORT_DATE=YYYY-MM-DD)")
    print(f"Report output directory: {os.path.abspath(run_dir)}")
    print(f"  sample_details.jsonl   ")
    print(f"  run_summary.json       ")
    print(f"  run_log.txt            ")
    print(f"  run_config.yaml        ")
    if getattr(cfg, "save_plots", True):
        print(f"  plots.png              ")
    if getattr(cfg, "ascii_demo", False):
        print(f"  ascii_demo.txt         ASCII()")

    if getattr(cfg, "speculative_branch", "mrs") == "per_position_compare" and cfg.mode != "single_block":
        raise ValueError("speculative_branch=per_position_compare  mode=single_block")

    config_dump = os.path.join(run_dir, "run_config.yaml")
    import yaml
    with open(config_dump, "w") as f:
        f.write("# speculative decoding \n")
        f.write(f"# tag={cfg.tag} mode={cfg.mode}\n\n")
        yaml.dump(cfg.to_dict(), f, default_flow_style=False)
    print(f"Config dumped to {config_dump}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    draft_dev = torch.device(cfg.draft_device)
    target_dev = torch.device(cfg.target_device)
    if draft_dev != target_dev:
        print(f"[dual-GPU] draft on {draft_dev}, target on {target_dev}  "
              f"split placement enabled (device normalization at MRS/kl_div).")
    else:
        print(f"[single-GPU] draft and target both on {draft_dev}")

    print(f"Loading draft model: {cfg.draft_model} -> {draft_dev}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model, trust_remote_code=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        cfg.draft_model, trust_remote_code=True, torch_dtype=torch_dtype,
    ).to(draft_dev).eval()

    print(f"Loading target model: {cfg.target_model} -> {target_dev}")
    target_model = AutoModelForCausalLM.from_pretrained(
        cfg.target_model, trust_remote_code=True, torch_dtype=torch_dtype,
    ).to(target_dev).eval()

    patch_multi_block_mask_fn(target_model, block_causal_prompt=cfg.block_causal_prompt)
    print(f"Patched target model create_multi_block_causal_mask (block_causal_prompt={cfg.block_causal_prompt})")

    # ── pipeline / target_eval_sdpa  ──
    # SDARAttention.forward (eval )  `if torch.all(attention_mask):`,
    # 0-dim bool tensor  Python `if`  GPUCPU sync, 36  × ~1ms ≈ 36ms.
    #  pipeline  verify_N  CPU dispatch  ~50ms,
    #  draft_{N+1}  overlap. patch_sdpa_eval_attention  forward
    #  " SDPA " ( mask  all-ones,
    #  dead code),  sync.  idempotent, draft  use_cuda_graph
    #  patch  attn class.
    # use_cuda_graph implies target_eval_sdpa  the verify graph is only
    # captured on the SDPA-eval path. Flip the flag eagerly so the patch below
    # fires and the verify call sites pick the graphed route.
    if getattr(cfg, "use_cuda_graph", False) and not getattr(cfg, "target_eval_sdpa", False):
        cfg.target_eval_sdpa = True
        print("[cfg] use_cuda_graph=True  forcing target_eval_sdpa=True")

    if getattr(cfg, "target_eval_sdpa", False) or getattr(cfg, "pipeline", False):
        if patch_sdpa_eval_attention(target_model):
            print("Patched target SDARAttention.forward  SDPA-only (skip per-layer torch.all sync)")
        if getattr(cfg, "pipeline", False) and patch_sdpa_eval_attention(draft_model):
            print("Patched draft SDARAttention.forward  SDPA-only (pipeline draft must be async)")

    print(f"Loading dataset: {cfg.dataset} ({cfg.dataset_split}), n={cfg.num_samples}")
    _, all_prompt_ids = load_prompts(tokenizer, cfg)
    print(f"Loaded {len(all_prompt_ids)} prompts")

    if cfg.mode == "single_block":
        eval_single_block(draft_model, target_model, tokenizer, all_prompt_ids, cfg)
    elif cfg.mode == "multi_block":
        eval_multi_block(draft_model, target_model, tokenizer, all_prompt_ids, cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    print("\n" + "=" * 60)
    print(f"Speculative decoding ({cfg.mode}) done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
