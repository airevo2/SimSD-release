"""Draft-probs ablation v2  EXACT 4D-mask rules on the draft side.

v1 (draft_probs_ablation.py) used an intra-block *bidirectional* mask for the
extra forward that overwrites draft.per_pos_probs. That differs from the 4D
mask's ``intra_md`` rule (mask(t) only sees data(t' < t), not all data in the
block), so draft's q-probs still didn't match target's p-probs bit-for-bit.

Result in v1 (N=40 MRS seed=42):
    mbpp   +30pp pass@1 (closed the gap)
    humaneval +15pp
    gsm8k  −10pp (made it worse)

Hypothesis: the v1 regression on gsm8k comes from the rule mismatch. If draft
runs its extra forward with the EXACT same build_verify_sequence_multi +
create_multi_block_causal_mask code that target uses, then for self_draft
p == q bit-for-bit  α  1.0 and the ambiguity disappears.

This v2 calls target_verify_forward_multi on the single just-drafted block
after draft_one_block's denoising loop finishes. Extra cost per draft block:
one forward with seq_len ≈ prompt + (M+1)*2*block_length vs v1's
prompt + (M+1)*block_length  a ~10-20% longer forward on top of v1's.

Usage:
  CUDA_VISIBLE_DEVICES=6 python draft_probs_ablation_v2.py \\
      --dataset mbpp --num_samples 40 --branch mrs \\
      --target_model dllm_causal_train/inference/model/SDAR-8B-Chat \\
      --skip_cross
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
from speculative_decoding.verify import target_verify_forward_multi  # noqa: E402

MASK_TOKEN_ID = 151669

_orig_draft_one_block = _draft_mod.draft_one_block


class _DraftCtx:
    """Per-spec-iteration context that the patched draft_one_block uses to
    reconstruct (prompt_ids, committed_blocks) needed by
    target_verify_forward_multi.

    _speculative_generate_K doesn't pass prompt/committed to draft_one_block
    directly  it builds ``speculative_context = prompt + committed + earlier
    draft``. We reverse-engineer it by remembering the initial prompt and
    tracking block lengths. The outer wrapper sets this before each iter.
    """
    prompt_len: int = 0
    block_length: int = 4
    block_size: int = 4
    pad_token_id: int = 0
    padded_len: int = 0
    model = None  # target model (== draft for self_draft)

    def __init__(self): pass


CTX = _DraftCtx()


def _draft_one_block_v2(
    model,
    context_ids,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    step_callback=None,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """draft_one_block + overwrite per_pos_probs using target_verify_forward_multi
    on this single block. Gives bit-for-bit match with target's p-probs under
    self_draft.
    """
    draft_ids, _stale_probs, step_map = _orig_draft_one_block(
        model, context_ids, block_length, denoising_steps,
        device, mask_token_id, step_callback=step_callback,
        use_cuda_graph=use_cuda_graph,
    )

    # Reconstruct (prompt, committed). context_ids = prompt + committed
    # (+ any already-drafted blocks from this iter, which behave as additional
    # "accepted" prefix for 4D-mask purposes since they sit in the prefix of
    # target's verify sequence too).
    if CTX.model is None:
        # Ablation context not set up  fall back to stale probs. Should not
        # happen under the driver below.
        return draft_ids, _stale_probs, step_map

    prompt_len = CTX.prompt_len
    bl = CTX.block_length
    assert block_length == bl

    prompt_ids = list(context_ids[:prompt_len])
    post_prompt = list(context_ids[prompt_len:])
    # post_prompt is a flat concat of completed blocks of length bl each.
    assert len(post_prompt) % bl == 0, (
        f"post_prompt_len={len(post_prompt)} not a multiple of bl={bl}"
    )
    accepted_blocks = [
        post_prompt[i:i + bl]
        for i in range(0, len(post_prompt), bl)
    ]

    # Call target_verify_forward_multi with draft_blocks=[this one block].
    # NOTE: target_verify_forward_multi expects use_eval_sdpa=True for the
    # SDPA-patched path, matching what _speculative_generate_K passes to the
    # actual target verify. Same graph-disable as self_draft_compare.
    probs_list = target_verify_forward_multi(
        CTX.model, prompt_ids, accepted_blocks,
        [draft_ids], [step_map],
        block_length, CTX.block_size, CTX.pad_token_id, CTX.padded_len,
        mask_token_id=mask_token_id,
        use_eval_sdpa=True,
        return_on_device=False,
        use_cuda_graph=False,
    )
    assert len(probs_list) == 1
    per_pos_probs_final = probs_list[0]  # (block_length, vocab_size), on CPU
    return draft_ids, per_pos_probs_final, step_map


_draft_mod.draft_one_block = _draft_one_block_v2
_sd_mod.draft_one_block = _draft_one_block_v2

print("[ablation v2] patched draft.draft_one_block  target_verify_forward_multi "
      "for per-pos probs (EXACT 4D-mask rules, should give self_draft α1.0)",
      flush=True)


# ── Context plumbing: we need prompt_len, block geometry, target model ref.
#    Wrap speculative_generate so we know these at the moment of draft call.
_orig_spec_generate = _sd_mod.speculative_generate


def _spec_generate_with_ctx(draft_model, target_model, prompt_ids, cfg,
                            pad_token_id, padded_len, eos_ids=None,
                            return_timings: bool = False):
    CTX.prompt_len = len(prompt_ids)
    CTX.block_length = int(cfg.block_length)
    CTX.block_size = int(cfg.block_size)
    CTX.pad_token_id = int(pad_token_id)
    CTX.padded_len = int(padded_len)
    # Target is the authoritative "4D-mask rules" model. For self_draft they
    # are the same object; for cross-draft v2 this still pulls target-side
    # probs onto the draft side (which is the point).
    CTX.model = target_model
    return _orig_spec_generate(draft_model, target_model, prompt_ids, cfg,
                               pad_token_id, padded_len, eos_ids,
                               return_timings=return_timings)


_sd_mod.speculative_generate = _spec_generate_with_ctx

# HFSpeculativeBackend did `from speculative_decoding.speculative_decode import
# speculative_generate`, so the bench path also needs the rebind.
from speculative_decoding.bench.backends.hf import speculative as _hf_spec_mod  # noqa: E402
_hf_spec_mod.speculative_generate = _spec_generate_with_ctx


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_final_v2_4dmask"])
    self_draft_compare.main()
