"""Draft-probs ablation v3  replace draft's per-block probs with a single
K-wide target_verify forward, same call target uses.

v1 intra-block bidirectional  bit-mismatch from 4D-mask intra_md rule.
v2 per-block K=1 target_verify  still mismatch: block k in K=1 call has fewer
   prior blocks than it would in target's K-wide call (d_0..d_{k-1} get
   training-layout vs. accepted-layout treatment  different token_labels).
v3 single K-wide target_verify call to compute ALL K draft_probs at once.

For self_draft (draft==target==same model), draft's call and target's call are
bit-identical  p == q everywhere  MRS always accepts  committed tokens =
draft_ids = what draft_one_block produced = what native would produce.

Expected result: self_draft pass@1 == native pass@1 exactly on all datasets.
Cost: draft side pays one full K-wide verify forward per spec iter  same as
target. So draft latency doubles (not a production path; purely a correctness
diagnostic).

Implementation: patch _speculative_generate_K so that after draft generates K
blocks, a target_verify_forward_multi(draft_model, ...) call fills in
draft_probs_list, replacing the mixed-step per_pos_probs from draft_one_block.

Usage:
  CUDA_VISIBLE_DEVICES=6 python draft_probs_ablation_v3.py \\
      --dataset gsm8k --num_samples 40 --branch mrs \\
      --target_model dllm_causal_train/inference/model/SDAR-8B-Chat --skip_cross
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding import speculative_decode as _sd_mod  # noqa: E402
from speculative_decoding.draft import draft_one_block  # noqa: E402
from speculative_decoding.verify import target_verify_forward_multi  # noqa: E402


def _speculative_generate_K_v3(draft_model, target_model, prompt_ids, cfg,
                               pad_token_id, padded_len, eos_ids=None,
                               return_timings: bool = False):
    """Drop-in replacement for _speculative_generate_K that overwrites draft's
    per-pos probs with a K-wide draft-side verify forward.

    Mostly copied from speculative_decode.py::_speculative_generate_K with one
    change marked `[v3]`.
    """
    from speculative_decoding.speculative_decode import _cuda_sync, _select_verify_fn

    K = int(cfg.K)
    assert cfg.num_blocks % K == 0, f"num_blocks={cfg.num_blocks} % K={K} != 0"
    iters = cfg.num_blocks // K

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    committed_blocks: list = []
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
                draft_probs_list.append(draft_probs)  # [v3] overwritten below
                step_maps.append(step_map)
                speculative_context = speculative_context + list(draft_ids)
                stats["total_draft_tokens"] += cfg.block_length

        # [v3] Overwrite draft_probs_list with one K-wide target_verify call on
        #      draft_model. For self_draft (draft==target), this is bit-for-bit
        #      identical to target's call below  p==q  MRS always accepts.
        with torch.cuda.device(draft_device):
            draft_probs_list = target_verify_forward_multi(
                draft_model, prompt_ids, committed_blocks,
                draft_blocks, step_maps,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=True,  # matches target's call
                return_on_device=False,
                use_cuda_graph=False,
            )

        if return_timings:
            _cuda_sync(draft_device)
            t_after_draft = time.perf_counter()

        with torch.cuda.device(target_device):
            target_probs_list = target_verify_forward_multi(
                target_model, prompt_ids, committed_blocks,
                draft_blocks, step_maps,
                cfg.block_length, cfg.block_size, pad_token_id, padded_len,
                cfg.mask_token_id,
                use_eval_sdpa=getattr(cfg, "target_eval_sdpa", False),
                use_cuda_graph=getattr(cfg, "use_cuda_graph", False),
            )

        if return_timings:
            _cuda_sync(target_device)
            t_after_verify = time.perf_counter()

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
                    from speculative_decoding.speculative_decode import (
                        _redraft_partial_fill,
                    )
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
        "per_block_timings": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_and_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
    }
    return all_generated, stats, timing_info


_sd_mod._speculative_generate_K = _speculative_generate_K_v3
print("[ablation v3] patched _speculative_generate_K  K-wide draft-side verify "
      "(self_draft α should = 1.0 bit-for-bit)", flush=True)


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_v3_kwide"])
    self_draft_compare.main()
