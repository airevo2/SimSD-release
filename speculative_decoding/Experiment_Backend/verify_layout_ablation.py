"""Verify-layout ablation: self_draft with NATIVE-equivalent verify.

Goal:
  Test whether the 4D multi-block mask + [data|mask] interleave layout used by
  ``target_verify_forward_multi`` is the root cause of self_draft's -22.5pp /
  -35.0pp pass@1 regression on humaneval / mbpp.

Method:
  Monkey-patch ``speculative_decoding.speculative_decode.target_verify_forward_multi``
  with a K-separate-forwards implementation that mimics native's block-causal
  layout:

      For each draft block k (k = 0 .. K-1):
        seq = [prompt | committed_0..M-1 | d_0 | ... | d_{k-1} | MASK*bl]
        attention = _build_block_causal_attn(L, block_length)        # (1,L,L)
        forward(input_ids=seq, attention_mask=attention)              # no tokenlbls
        probs_k = softmax(logits[last bl positions])

  Then run the standard self_draft_compare pipeline. MRS / greedy_match get
  the same target_probs_list shape as before; only the verify layout changed.

Expected:
  If 4D-mask layout is the root cause of coding regression, self_draft pass@1
  on humaneval / mbpp should move back toward native under this ablation.

Usage:
  python verify_layout_ablation.py --dataset humaneval --num_samples 40 --branch mrs
  python verify_layout_ablation.py --dataset mbpp     --num_samples 40 --branch mrs
  python verify_layout_ablation.py --dataset gsm8k    --num_samples 20 --branch mrs
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Import target_verify_forward_multi via speculative_decode module so we can
# replace the *bound name* (speculative_decode.py uses it by its local alias).
from speculative_decoding import speculative_decode as _sd_mod  # noqa: E402
from speculative_decoding.draft import _build_block_causal_attn  # noqa: E402

MASK_TOKEN_ID = 151669


def _native_equiv_verify_forward_multi(
    model,
    prompt_ids,
    accepted_blocks,
    draft_blocks,
    step_maps,
    block_length,
    block_size,
    pad_token_id,
    padded_len,
    mask_token_id=MASK_TOKEN_ID,
    use_eval_sdpa: bool = False,
    return_on_device: bool = False,
    use_cuda_graph: bool = False,
):
    """Native-equivalent replacement for target_verify_forward_multi.

    Runs K separate forwards, each with standard block-causal attention over
    ``[prompt | committed | d_0..d_{k-1} | MASK*bl]``. Returns a list of K
    ``(block_length, vocab_size)`` probability tensors, same shape/semantics
    as the original. No 4D multi-block mask, no [data|mask] interleave
    each forward is bit-identical in layout to what native's draft_one_block
    does at block (M+k).

    Arguments ``block_size``, ``pad_token_id``, ``padded_len``, ``use_eval_sdpa``,
    ``use_cuda_graph`` are accepted for signature parity but ignored; the
    block-causal path doesn't need pad_tokens or the 4D mask cache.
    """
    device = next(model.parameters()).device
    K = len(draft_blocks)
    assert K >= 1 and len(step_maps) == K

    committed_flat = [t for blk in accepted_blocks for t in blk]
    base = list(prompt_ids) + committed_flat
    probs_per_block: List[torch.Tensor] = []

    for k in range(K):
        drafted_flat = [t for blk in draft_blocks[:k] for t in blk]
        ctx = base + drafted_flat
        seq = ctx + [mask_token_id] * block_length
        seq_len = len(seq)
        ctx_len = len(ctx)

        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        attn_mask = _build_block_causal_attn(seq_len, block_length, device)

        with torch.cuda.device(device), torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[0, ctx_len:ctx_len + block_length]

        target_probs = F.softmax(logits.float(), dim=-1)
        probs_per_block.append(
            target_probs if return_on_device else target_probs.cpu()
        )

    return probs_per_block


# ── Apply the monkey-patch BEFORE importing self_draft_compare so that module's
#    transitive imports of speculative_decode see the replacement.
_sd_mod.target_verify_forward_multi = _native_equiv_verify_forward_multi
print("[ablation] patched speculative_decode.target_verify_forward_multi "
      " native-equivalent K-forward block-causal verify", flush=True)


# Now drive the standard comparison. Delegate CLI + scoring to self_draft_compare.
from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    # Tag results so they don't collide with the un-ablated self_draft run.
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "verify_layout_ablation_native_equiv"])
    # The ablation path is slower (K× forwards). CUDA graph is already off in
    # self_draft_compare's _build_spec_cfg, so nothing else to disable.
    self_draft_compare.main()
