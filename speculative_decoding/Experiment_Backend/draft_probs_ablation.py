"""Draft-probs ablation: overwrite draft's per-pos probs with a fully-decoded
final-step snapshot, so draft's q-distribution matches the context that the
4D-mask target evaluates against.

Background (why the layout ablation gave the opposite of what we hoped):
  The layout-level ablation (verify_layout_ablation.py) replaced the 4D-mask
  verify with K separate block-causal forwards at all-MASK context. Self-draft
  α dropped from 0.83  0.38 on gsm8k. Digging in:

    - draft.per_pos_probs[p] is set when position p is first unmasked, using
      that step's forward. Step 0 sees all-MASK; step s>0 sees a partial-fill
      context. So draft.per_pos_probs is a *mixed-step* snapshot: each pos
      has probs conditioned on a different context.

    - 4D-mask verify has each block's mask positions attend to their own-block
      DATA positions (intra_md rule in create_multi_block_causal_mask). That
      is a *fully-decoded* context  target sees every other position's final
      draft token.

    - Mixed-step draft probs vs fully-decoded target probs  q ≠ p even for
      self-draft  α < 1.

  This ablation aligns the two snapshots by overwriting draft's per-pos probs
  with one final forward at the fully-decoded block. If the 4D-mask layout
  itself is sound, self-draft α should now approach 1.0 and the mbpp /
  humaneval pass@1 gap should close.

Method:
  Monkey-patch speculative_decoding.draft.draft_one_block. Inner logic is
  unchanged, but after the denoising loop commits tokens we do one extra
  forward over [prompt | committed | d_block] (all positions filled, no MASK)
  and re-extract probs at the block positions. This is the same context the
  4D-mask target evaluates under.

Usage:
  CUDA_VISIBLE_DEVICES=6 python draft_probs_ablation.py \\
      --dataset humaneval --num_samples 40 --branch mrs \\
      --target_model dllm_causal_train/inference/model/SDAR-8B-Chat \\
      --skip_cross
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding import draft as _draft_mod  # noqa: E402
from speculative_decoding import speculative_decode as _sd_mod  # noqa: E402
from speculative_decoding.draft import _build_block_causal_attn  # noqa: E402

MASK_TOKEN_ID = 151669

_orig_draft_one_block = _draft_mod.draft_one_block


def _draft_one_block_final_probs(
    model,
    context_ids,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    step_callback=None,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """Same as draft_one_block, but replaces per_pos_probs with a final-step
    snapshot taken from one extra fully-decoded forward.

    We delegate the actual denoising to the original function (keeps every
    detail identical  unmask schedule, CUDA graph cache, step_callback) and
    then do one extra forward at the committed block to overwrite per-pos
    probs. Step_map is preserved as-is (it's used by MRS's position-order
    mode anyway, not for conditioning).
    """
    draft_ids, _per_pos_probs_stale, step_map = _orig_draft_one_block(
        model, context_ids, block_length, denoising_steps,
        device, mask_token_id, step_callback=step_callback,
        use_cuda_graph=use_cuda_graph,
    )

    # One extra forward at the fully-decoded block, matching 4D-mask's
    # intra_md context: own-block MASK positions attend bidirectionally to
    # all own-block DATA positions. Cross-block is standard block-causal.
    ctx_len = len(context_ids)
    seq = list(context_ids) + list(draft_ids)
    seq_len = len(seq)
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)

    attn_mask = _build_block_causal_attn(seq_len, block_length, device).clone()
    block_start = ctx_len
    block_end = ctx_len + block_length
    attn_mask[0, block_start:block_end, block_start:block_end] = 1.0

    with torch.cuda.device(device), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits[0, ctx_len:ctx_len + block_length]

    vocab_size = model.config.vocab_size
    per_pos_probs_final = F.softmax(logits.float(), dim=-1).cpu()
    assert per_pos_probs_final.shape == (block_length, vocab_size)

    return draft_ids, per_pos_probs_final, step_map


_draft_mod.draft_one_block = _draft_one_block_final_probs
# speculative_decode.py does `from ... import draft_one_block`, so rebind there too.
_sd_mod.draft_one_block = _draft_one_block_final_probs
print("[ablation] patched draft.draft_one_block  final-step per-pos probs "
      "(one extra forward at fully-decoded block with intra-block bidirectional mask)",
      flush=True)


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_final_step_ablation"])
    self_draft_compare.main()
