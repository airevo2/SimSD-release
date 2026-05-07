"""Draft-probs ablation v1.5_fused  fuse v1.5's extra forward into the last
denoising step.

v1.5 costs N+1 forwards per drafted block (N denoising + 1 extra intra_md
forward). Observation: the extra forward's input is nearly identical to the
last denoising forward (all positions are fully decoded when it runs); only
the own-block mask rule differs (bidirectional vs intra_md step-causal).

v1.5_fused skips the extra forward entirely: the LAST denoising forward uses
the intra_md step-causal mask over the own-block region, and its softmax is
returned as per_pos_probs for all block positions. Draft cost drops from N+1
to N forwards (−25% at N=4) at the possible cost of a small α drop (the last
denoising forward no longer sees the block bidirectionally).

Implementation: ``speculative_decoding.draft.draft_one_block`` already accepts
``use_intra_md_last_step=True`` (see draft.py). This module monkey-patches
``draft.draft_one_block`` and ``speculative_decode.draft_one_block`` to a thin
wrapper that forces the flag on.

Usage (quality check):
  CUDA_VISIBLE_DEVICES=6 python draft_probs_ablation_v1_5_fused.py \\
      --dataset gsm8k --num_samples 40 --branch mrs \\
      --target_model inference/model/SDAR-8B-Chat --skip_cross
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding import draft as _draft_mod  # noqa: E402
from speculative_decoding import speculative_decode as _sd_mod  # noqa: E402

MASK_TOKEN_ID = 151669

_orig_draft_one_block = _draft_mod.draft_one_block


def _draft_one_block_v1_5_fused(
    model,
    context_ids,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    step_callback=None,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """draft_one_block with use_intra_md_last_step=True forced on."""
    return _orig_draft_one_block(
        model, context_ids, block_length, denoising_steps,
        device, mask_token_id=mask_token_id, step_callback=step_callback,
        use_cuda_graph=use_cuda_graph,
        use_intra_md_last_step=True,
    )


_draft_mod.draft_one_block = _draft_one_block_v1_5_fused
_sd_mod.draft_one_block = _draft_one_block_v1_5_fused
print("[ablation v1.5_fused] patched draft.draft_one_block  last denoising "
      "step uses intra_md step-causal mask (extra forward fused away).",
      flush=True)


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_v1_5_fused"])
    self_draft_compare.main()
