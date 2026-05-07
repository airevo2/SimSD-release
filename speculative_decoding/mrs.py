"""Modified Rejection Sampling (MRS) for speculative decoding verification.

Also hosts ``greedy_match_verify``: the argmax-greedy analogue of MRS, for when
both draft and target decode via ``argmax`` (no stochastic sampling). MRS is
defined for stochastic sampling; when the actual draft/target sampling is
greedy, MRS's acceptance rule produces a biased output distribution that
diverges from native greedy decoding. See ``Experiment_Backend/plan.md §3bis``.
"""

import torch
import torch.nn.functional as F


def mrs_verify(draft_ids, draft_probs, target_probs, step_map, verify_order_mode="position"):
    """
    MRS verification: for each position, accept draft token with
    prob min(1, q(x)/p(x)); on rejection resample from norm(max(0, q-p)).

    Verify order is controlled by ``verify_order_mode``:

    - ``position`` (default, **correct**): block positions 0  L-1. Identical to
      AR speculative decoding's commit-order semantics; line 88's
      ``range(first_reject_pos)`` is a tight prefix of visited positions.
    - ``step_map`` (**known-buggy, do not use**): denoising unmask order. When
      the stepposition mapping is not the identity permutation, the truncation
      on line 88 silently drops already-accepted positions whose block index is
      ≥ ``first_reject_pos``, producing structural token loss in the emitted
      stream (see ``Experiment_Backend/results/quality/SUMMARY.md``, 2026-04-21
      gsm8k / humaneval / mbpp pass@1 regressions of 21–66 pp). Kept only as a
      flag value for reproducing pre-fix behavior; rewriting its semantics to
      handle holes requires ``mrs_verify`` to return a position-indexed output
      plus an "uncommitted positions" set, which the caller
      (``_speculative_generate_K`` etc.) would also need to learn.

    First rejection truncates; output remains in position order.

    Args:
        draft_ids: list[int], length = block_length
        draft_probs: Tensor (block_length, vocab_size)
        target_probs: Tensor (block_length, vocab_size)
        step_map: list[int], 0-indexed step when each position was unmasked
        verify_order_mode: ``"step_map"`` | ``"position"``

    Returns:
        accepted_ids: list[int], final token ids in position order (len <= block_length)
        n_accepted: int, positions accepted before first rejection (in position order)
        bonus_token: int or None, resampled token at rejection point
    """
    block_length = len(draft_ids)
    # Dual-GPU placement: draft_probs may live on draft_device and target_probs
    # on target_device. All downstream ops here (residual = target - draft, etc.)
    # require same-device tensors. Normalize draft_probs onto target's device.
    if draft_probs.device != target_probs.device:
        draft_probs = draft_probs.to(target_probs.device)
    if verify_order_mode == "step_map":
        verify_order = sorted(range(block_length), key=lambda i: step_map[i])
    elif verify_order_mode == "position":
        verify_order = list(range(block_length))
    else:
        raise ValueError(
            f"verify_order_mode must be 'step_map' or 'position', got {verify_order_mode!r}"
        )

    accepted_positions = set()
    first_reject_pos = None
    bonus_token = None

    for pos in verify_order:
        x_i = draft_ids[pos]
        p_x = draft_probs[pos, x_i].item()
        q_x = target_probs[pos, x_i].item()

        if p_x <= 0:
            residual = F.relu(target_probs[pos].clone())
            residual_sum = residual.sum()
            if residual_sum > 0:
                residual = residual / residual_sum
            else:
                residual = torch.ones(residual.shape[-1]) / residual.shape[-1]
            bonus_token = torch.multinomial(residual, 1).item()
            first_reject_pos = pos
            break

        accept_prob = min(1.0, q_x / p_x)
        u = torch.rand(1).item()

        if u < accept_prob:
            accepted_positions.add(pos)
        else:
            residual = F.relu(target_probs[pos] - draft_probs[pos])
            residual_sum = residual.sum()
            if residual_sum > 0:
                residual = residual / residual_sum
            else:
                residual = target_probs[pos].clone()
            bonus_token = torch.multinomial(residual, 1).item()
            first_reject_pos = pos
            break

    # Build output in position order: accepted positions < first_reject_pos, then bonus
    if first_reject_pos is None:
        return list(draft_ids), block_length, None

    accepted_in_order = [i for i in range(first_reject_pos) if i in accepted_positions]
    accepted_ids = [draft_ids[i] for i in accepted_in_order]
    n_accepted = len(accepted_ids)
    accepted_ids.append(bonus_token)
    return accepted_ids, n_accepted, bonus_token


def greedy_match_verify(draft_ids, draft_probs, target_probs, step_map,
                        verify_order_mode="position"):
    """Greedy (argmax) speculative verification.

    For native-argmax-greedy decoding, the lossless spec rule is:
    ``accept draft[pos] iff draft[pos] == argmax(target_probs[pos])``. On the
    first mismatch, commit ``argmax(target_probs[pos])`` as the bonus and
    truncate. Only ``verify_order_mode="position"`` is supported (the
    prefix-truncation semantics match block-position order).

    Signature mirrors ``mrs_verify`` so callers can swap them freely.

    Returns:
        accepted_ids: list[int], final committed token ids (len <= block_length)
        n_accepted: int, number of matching positions before first mismatch
        bonus_token: int or None, target's argmax at the mismatch position
    """
    block_length = len(draft_ids)
    if draft_probs.device != target_probs.device:
        draft_probs = draft_probs.to(target_probs.device)
    if verify_order_mode != "position":
        raise ValueError(
            "greedy_match_verify only supports verify_order_mode='position' "
            f"(got {verify_order_mode!r}); step_map mode does not give a "
            "prefix-of-visited-positions truncation."
        )

    target_argmax = target_probs.argmax(dim=-1).tolist()
    for pos in range(block_length):
        if draft_ids[pos] != target_argmax[pos]:
            accepted_ids = list(draft_ids[:pos])
            bonus_token = target_argmax[pos]
            accepted_ids.append(bonus_token)
            return accepted_ids, pos, bonus_token
    return list(draft_ids), block_length, None
