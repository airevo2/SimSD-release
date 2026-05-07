"""
 SDAR  block  speculative  draft draft_one_block

 4B draft + 8B verify  speculative
 Target  model  checkpoint
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from speculative_decoding.draft import draft_one_block, draft_one_block_batch


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def native_multi_block_generate(
    model,
    prompt_ids: List[int],
    num_blocks: int,
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], Dict[str, Any]]:
    """
     `num_blocks`  block `draft_one_block`

    Returns:
        generated_ids, stats
        stats: {"num_blocks_run": int, "per_block_timings": ...}  return_timings
    """
    device = next(model.parameters()).device
    context_ids = list(prompt_ids)
    all_generated: List[int] = []
    per_block: List[Dict[str, float]] = []

    n_run = 0
    for blk_idx in range(num_blocks):
        if return_timings:
            _cuda_sync()
            t0 = time.perf_counter()

        draft_ids, _, _ = draft_one_block(
            model,
            context_ids,
            block_length,
            denoising_steps,
            device,
            mask_token_id,
            use_cuda_graph=use_cuda_graph,
        )

        if return_timings:
            _cuda_sync()
            t1 = time.perf_counter()
            per_block.append({
                "block_idx": float(blk_idx),
                "block_gen_s": t1 - t0,
                "block_wall_s": t1 - t0,
            })

        context_ids.extend(draft_ids)
        all_generated.extend(draft_ids)
        n_run += 1

        if eos_ids and any(t in eos_ids for t in draft_ids):
            break

    while all_generated and all_generated[-1] == mask_token_id:
        all_generated.pop()

    stats: Dict[str, Any] = {
        "num_blocks_run": n_run,
        "total_draft_tokens": n_run * block_length,
    }
    if return_timings:
        stats["per_block_timings"] = per_block
        stats["total_block_wall_s"] = sum(b["block_wall_s"] for b in per_block)
    return all_generated, stats


def native_multi_block_generate_batch(
    model,
    prompts_batch: List[List[int]],
    num_blocks: int,
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    pad_token_id: int = 0,
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
    use_cuda_graph: bool = False,
) -> List[Tuple[List[int], Dict[str, Any]]]:
    """Batched native multi-block generate via ``draft_one_block_batch``.

    Returns: list of (generated_ids, stats) tuples (length B). Same per-row
    semantics as ``native_multi_block_generate`` (no verify, just block diffusion).

    Rows are kept synchronized: all rows commit ``block_length`` tokens per
    iteration. Under ``--no_eos_stop`` (eos_ids=None) every row runs all
    ``num_blocks``. With eos_ids set, the whole batch breaks when any row
    emits an EOS token.
    """
    device = next(model.parameters()).device
    B = len(prompts_batch)
    assert B > 0
    max_plen = max(len(p) for p in prompts_batch)
    # Left-pad to keep ctx_len uniform (required by draft_one_block_batch).
    context_ids_batch = [
        [pad_token_id] * (max_plen - len(p)) + list(p)
        for p in prompts_batch
    ]
    all_generated_batch: List[List[int]] = [[] for _ in range(B)]
    per_block_lists: List[List[Dict[str, float]]] = [[] for _ in range(B)]
    n_run = 0
    early_stop = False

    for blk_idx in range(num_blocks):
        if early_stop:
            break
        if return_timings:
            _cuda_sync()
            t0 = time.perf_counter()

        with torch.cuda.device(device):
            draft_ids_batch, _, _ = draft_one_block_batch(
                model, context_ids_batch,
                block_length, denoising_steps,
                device, mask_token_id,
                use_cuda_graph=use_cuda_graph,
            )

        if return_timings:
            _cuda_sync()
            t1 = time.perf_counter()

        for i in range(B):
            context_ids_batch[i] = context_ids_batch[i] + list(draft_ids_batch[i])
            all_generated_batch[i].extend(draft_ids_batch[i])
            if return_timings:
                per_block_lists[i].append({
                    "block_idx": float(blk_idx),
                    "block_gen_s": t1 - t0,
                    "block_wall_s": t1 - t0,
                })
            if eos_ids and any(t in eos_ids for t in draft_ids_batch[i]):
                early_stop = True
        n_run += 1

    out_rows: List[Tuple[List[int], Dict[str, Any]]] = []
    for i in range(B):
        gen = all_generated_batch[i]
        while gen and gen[-1] == mask_token_id:
            gen.pop()
        s: Dict[str, Any] = {
            "num_blocks_run": n_run,
            "total_draft_tokens": n_run * block_length,
        }
        if return_timings:
            s["per_block_timings"] = per_block_lists[i]
            s["total_block_wall_s"] = sum(b["block_wall_s"] for b in per_block_lists[i])
        out_rows.append((gen, s))
    return out_rows
