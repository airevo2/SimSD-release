"""
Multi-Block Causal Mask for dLLM Training.

Combines:
  1. Block-internal causal mask (one-step teacher forcing per block)
  2. Block-inter causal ordering (later blocks see earlier blocks' DATA only)

Training sequence layout:
  [Prompt | B0_data B0_mask | B1_data B1_mask | ... | BN_data BN_mask]

Attention rules:
  - Prompt: visible to everyone; optionally sees all mask tokens
  - Within same block: data/mask step-based causal mask (same as single-block)
  - Across blocks: block_i sees block_j's DATA (j<i), NOT mask
  - Padding: isolated
"""

import torch
from typing import Optional


def create_multi_block_causal_mask(
    token_labels: torch.LongTensor,
    block_ids: torch.LongTensor,
    block_size: int,
    block_causal_prompt: bool = True,
) -> torch.Tensor:
    """
    Generate attention mask for multi-block causal mask training.

    Args:
        token_labels: (B, L)
            - 0: Prompt
            - 1 ~ block_size: Data token (step number within block)
            - block_size + 1: Mask token
            - -1: Padding
        block_ids: (B, L)
            - -1: Prompt or Padding
            - 0, 1, 2, ...: Block index for data/mask tokens
        block_size: int, denoising steps per block
        block_causal_prompt:
            True:  prompt block-level causal (SDAR-style), prompt  data/mask
            False: , prompt  prompt + mask,  prompt

    Returns:
        attn_mask: (B, 1, L, L) bool tensor, True = visible
    """
    B, L = token_labels.shape
    device = token_labels.device

    # ── Token types ──
    is_prompt = (token_labels == 0)
    is_data = (token_labels > 0) & (token_labels <= block_size)
    is_mask = (token_labels == (block_size + 1))
    is_pad = (token_labels == -1)

    # ── Time steps (per-block) ──
    # Data tokens: label value is the step (1..block_size)
    # Mask tokens: map to corresponding data token's step within same block
    time_steps = token_labels.clone().float()
    time_steps[is_pad] = float("inf")
    time_steps[is_prompt] = 0

    # ── Vectorized mask↔data time-step pairing (replaces O(num_blocks) Python loop) ──
    # For each mask token, copy the time_step of the data token with the same
    # within-block rank (position-order pairing). Uses scatter/gather on a dense
    # lookup table so the cost is O(L), independent of num_blocks.
    max_blk = block_ids.max().item() + 1
    if max_blk > 0:
        valid = (block_ids >= 0)                       # exclude prompt / pad
        safe_blk = block_ids.clamp(min=0)              # -1  0 for safe indexing

        # --- within-block DATA rank (0-indexed) ---
        data_long = is_data.long()                     # (B, L)
        data_cum = data_long.cumsum(dim=1)             # running count of data tokens
        blk_data_cnt = torch.zeros(B, max_blk, device=device, dtype=torch.long)
        blk_data_cnt.scatter_add_(1, safe_blk, data_long * valid.long())
        blk_data_pfx = blk_data_cnt.cumsum(dim=1) - blk_data_cnt   # exclusive prefix
        data_rank = (data_cum - blk_data_pfx.gather(1, safe_blk) - 1).clamp(min=0)

        # --- within-block MASK rank (0-indexed) ---
        mask_long = is_mask.long()
        mask_cum = mask_long.cumsum(dim=1)
        blk_mask_cnt = torch.zeros(B, max_blk, device=device, dtype=torch.long)
        blk_mask_cnt.scatter_add_(1, safe_blk, mask_long * valid.long())
        blk_mask_pfx = blk_mask_cnt.cumsum(dim=1) - blk_mask_cnt
        mask_rank = (mask_cum - blk_mask_pfx.gather(1, safe_blk) - 1).clamp(min=0)

        # --- dense lookup table: table[b, block_id, rank] = time_step ---
        # Stride must cover BOTH data ranks (we scatter into) AND mask ranks
        # (we gather from). When the SDAR draft early-stops (some block
        # positions are still MASK in the DATA segment), n_mask > n_data per
        # block, so mask_rank can exceed n_data-1. Without this, the gather
        # at flat_m_idx goes OOB  device-side assert. Unpaired mask reads
        # land on inf (table init) and fall through to the existing
        # ``finite.any()`` guard below, leaving their time_step at the
        # initial (block_size + 1) value (= "never revealed").
        max_rank = max(
            blk_data_cnt.max().item(),
            blk_mask_cnt.max().item(),
            1,
        )
        stride_blk = max_rank
        stride_b = max_blk * stride_blk
        table = torch.full((B * stride_b,), float("inf"), device=device)

        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        flat_idx = b_idx * stride_b + safe_blk * stride_blk + data_rank

        data_sel = is_data & valid
        table.scatter_(0, flat_idx[data_sel], time_steps[data_sel])

        # --- gather for mask tokens & assign ---
        mask_sel = is_mask & valid
        flat_m_idx = b_idx * stride_b + safe_blk * stride_blk + mask_rank
        gathered = table[flat_m_idx[mask_sel]]
        finite = gathered.isfinite()
        if finite.any():
            rows, cols = mask_sel.nonzero(as_tuple=True)
            time_steps[rows[finite], cols[finite]] = gathered[finite]

    # ── Expand to (B, 1, L, L) for pairwise comparison ──
    type_vals = torch.zeros_like(token_labels)  # 0=prompt/pad, 1=data, 2=mask
    type_vals[is_data] = 1
    type_vals[is_mask] = 2

    type_i = type_vals[:, None, :, None]   # (B, 1, L, 1) row (query)
    type_j = type_vals[:, None, None, :]   # (B, 1, 1, L) col (key)
    time_i = time_steps[:, None, :, None]
    time_j = time_steps[:, None, None, :]
    blkid_i = block_ids[:, None, :, None].float()
    blkid_j = block_ids[:, None, None, :].float()

    is_prompt_i = is_prompt.view(B, 1, L, 1)
    is_prompt_j = is_prompt.view(B, 1, 1, L)
    is_pad_i = is_pad.view(B, 1, L, 1)
    is_pad_j = is_pad.view(B, 1, 1, L)

    if block_causal_prompt:
        # : prompt block-level causal (SDAR-style)
        prompt_cumpos = is_prompt.long().cumsum(dim=1) - 1
        prompt_blk = prompt_cumpos // block_size
        prompt_blk_i = prompt_blk[:, None, :, None]
        prompt_blk_j = prompt_blk[:, None, None, :]
        rule_prompt = is_prompt_i & is_prompt_j & (prompt_blk_j <= prompt_blk_i)
        rule_see_prompt = (~is_prompt_i) & (~is_pad_i) & is_prompt_j
    else:
        # : prompt  prompt + mask,  prompt
        rule_prompt = is_prompt_i & is_prompt_j
        is_mask_j = is_mask.view(B, 1, 1, L)
        rule_prompt = rule_prompt | (is_prompt_i & is_mask_j)
        rule_see_prompt = is_prompt_j.expand(B, 1, L, L)

    # ── Same block, intra-block causal rules ──
    same_block = (blkid_i == blkid_j) & (blkid_i >= 0)

    # Data(t) sees Data(t' <= t)
    intra_dd = same_block & (type_i == 1) & (type_j == 1) & (time_j <= time_i)
    # Data(t) sees Mask(t' > t)
    intra_dm = same_block & (type_i == 1) & (type_j == 2) & (time_j > time_i)
    # Mask(t) sees Data(t' < t)
    intra_md = same_block & (type_i == 2) & (type_j == 1) & (time_j < time_i)
    # Mask(t) sees Mask(t' >= t)
    intra_mm = same_block & (type_i == 2) & (type_j == 2) & (time_j >= time_i)

    # ── Cross-block, later block sees earlier block's DATA only ──
    cross_block_data = (blkid_i > blkid_j) & (blkid_j >= 0) & (type_j == 1)

    # ── Combine ──
    final_mask = (
        rule_prompt
        | rule_see_prompt
        | intra_dd
        | intra_dm
        | intra_md
        | intra_mm
        | cross_block_data
    )

    # Padding isolation
    final_mask = final_mask & (~is_pad_i) & (~is_pad_j)

    return final_mask.to(dtype=torch.bool)


# ──────────────────── Helper: Build training sequence from rollout ────────────────────


def build_training_sequence(
    prompt_ids: list,
    response_ids: list,
    block_step_maps: list,
    block_length: int,
    block_size: int,
    mask_token_id: int,
):
    """
    Convert rollout data to training sequence with token_labels and block_ids.

    Args:
        prompt_ids: list of prompt token ids
        response_ids: list of response token ids (len = num_blocks * block_length)
        block_step_maps: list of per-block step_maps, each len block_length
                         values 0..block_size-1 (0-indexed from rollout)
        block_length: tokens per block
        block_size: denoising steps (= block_length for our setup)
        mask_token_id: token id for MASK

    Returns:
        input_ids: list  [prompt | B0_data | B0_mask | B1_data | B1_mask | ...]
        token_labels: list  [0...0 | step_map+1 | bs+1...bs+1 | step_map+1 | bs+1...bs+1 | ...]
        block_ids: list  [-1...-1 | 0...0 | 0...0 | 1...1 | 1...1 | ...]
        labels: list  [-100... | -100... | GT_token | -100... | GT_token | ...]
        position_ids: list  [0..P-1 | P..P+bl-1 | P..P+bl-1 | P+bl..P+2bl-1 | P+bl..P+2bl-1 | ...]
    """
    prompt_len = len(prompt_ids)
    num_blocks = len(block_step_maps)

    input_ids = list(prompt_ids)
    token_labels = [0] * prompt_len
    block_id_list = [-1] * prompt_len
    labels = [-100] * prompt_len
    position_ids = list(range(prompt_len))

    for blk_idx in range(num_blocks):
        blk_start = blk_idx * block_length
        blk_end = blk_start + block_length
        blk_tokens = response_ids[blk_start:blk_end]
        step_map = block_step_maps[blk_idx]

        # Skip blocks beyond actual response
        if blk_start >= len(response_ids):
            break

        # Pad if last block is shorter
        actual_len = len(blk_tokens)
        if actual_len < block_length:
            blk_tokens = blk_tokens + [mask_token_id] * (block_length - actual_len)
            step_map = step_map[:actual_len] + [-1] * (block_length - actual_len)

        # Block positions in the original response
        blk_pos_start = prompt_len + blk_idx * block_length
        blk_positions = list(range(blk_pos_start, blk_pos_start + block_length))

        # Data region: GT tokens with step labels (1-indexed)
        input_ids.extend(blk_tokens)
        token_labels.extend([s + 1 if s >= 0 else -1 for s in step_map])
        block_id_list.extend([blk_idx] * block_length)
        labels.extend([-100] * block_length)  # no loss on data
        position_ids.extend(blk_positions)

        # Mask region: MASK tokens
        input_ids.extend([mask_token_id] * block_length)
        token_labels.extend([block_size + 1 if s >= 0 else -1 for s in step_map])
        block_id_list.extend([blk_idx] * block_length)
        # loss on mask positions  predict GT; skip padded positions
        labels.extend([blk_tokens[j] if step_map[j] >= 0 else -100 for j in range(block_length)])
        position_ids.extend(blk_positions)  # same positions as data (RoPE sharing)

    return input_ids, token_labels, block_id_list, labels, position_ids


# ──────────────────── Test & Visualization ────────────────────


def test_with_rollout_data():
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load a sample from rollout data
    data_path = "./data/sdar8b_rollout_gsm8k_bl4.jsonl"
    with open(data_path) as f:
        rec = json.loads(f.readline())

    prompt_ids = rec["prompt_ids"]
    response_ids = rec["response_ids"]
    block_step_maps = rec["block_step_maps"]
    cfg = rec["gen_config"]
    block_length = cfg["block_length"]
    block_size = cfg["denoising_steps"]
    mask_token_id = 151669  # SDAR mask token

    # Only use first 3 blocks for visualization clarity
    num_vis_blocks = 3
    vis_response = response_ids[: num_vis_blocks * block_length]
    vis_step_maps = block_step_maps[:num_vis_blocks]

    print(f"Prompt len: {len(prompt_ids)}, Block length: {block_length}, Block size: {block_size}")
    print(f"Using {num_vis_blocks} blocks for visualization")
    print(f"Step maps: {vis_step_maps}")

    # Build training sequence
    input_ids, token_labels, block_id_list, labels, position_ids = build_training_sequence(
        prompt_ids, vis_response, vis_step_maps,
        block_length, block_size, mask_token_id,
    )

    print(f"\nTraining sequence length: {len(input_ids)}")
    print(f"  Prompt: {len(prompt_ids)} tokens")
    print(f"  Response region: {num_vis_blocks} blocks × {block_length} × 2 (data+mask) = {num_vis_blocks * block_length * 2} tokens")
    print(f"  Total: {len(input_ids)}")

    # Print token_labels and block_ids
    p = len(prompt_ids)
    print(f"\ntoken_labels (after prompt):")
    for i in range(num_vis_blocks):
        d_start = p + i * 2 * block_length
        m_start = d_start + block_length
        print(f"  Block {i} data: {token_labels[d_start:d_start+block_length]}")
        print(f"  Block {i} mask: {token_labels[m_start:m_start+block_length]}")

    print(f"\nblock_ids (after prompt):")
    for i in range(num_vis_blocks):
        d_start = p + i * 2 * block_length
        m_start = d_start + block_length
        print(f"  Block {i} data: {block_id_list[d_start:d_start+block_length]}")
        print(f"  Block {i} mask: {block_id_list[m_start:m_start+block_length]}")

    print(f"\nposition_ids (after prompt):")
    for i in range(num_vis_blocks):
        d_start = p + i * 2 * block_length
        m_start = d_start + block_length
        print(f"  Block {i} data: {position_ids[d_start:d_start+block_length]}")
        print(f"  Block {i} mask: {position_ids[m_start:m_start+block_length]}")

    # Convert to tensors
    tl = torch.tensor([token_labels], dtype=torch.long)
    bi = torch.tensor([block_id_list], dtype=torch.long)

    # Generate masks for both modes
    mask_new = create_multi_block_causal_mask(tl, bi, block_size, block_causal_prompt=True)
    mask_new_np = mask_new[0, 0].cpu().numpy().astype(float)

    mask_legacy = create_multi_block_causal_mask(tl, bi, block_size, block_causal_prompt=False)
    mask_legacy_np = mask_legacy[0, 0].cpu().numpy().astype(float)

    # Shared layout info
    boundaries = [p]
    for i in range(num_vis_blocks):
        boundaries.append(p + (2 * i + 1) * block_length)
        boundaries.append(p + (2 * i + 2) * block_length)

    region_labels = ["Prompt"]
    for i in range(num_vis_blocks):
        region_labels.append(f"B{i}_data")
        region_labels.append(f"B{i}_mask")

    region_centers = [p / 2]
    for i in range(num_vis_blocks):
        d_center = p + (2 * i) * block_length + block_length / 2
        m_center = p + (2 * i + 1) * block_length + block_length / 2
        region_centers.append(d_center)
        region_centers.append(m_center)

    # Visualize side by side
    fig, axes = plt.subplots(1, 2, figsize=(28, 12))

    for ax, mask_np, title_suffix in [
        (axes[0], mask_legacy_np, "Legacy (prompt sees all prompt + mask)"),
        (axes[1], mask_new_np, "New (prompt block-level causal)"),
    ]:
        ax.imshow(mask_np, cmap="Blues", interpolation="nearest", aspect="equal")
        for b in boundaries:
            ax.axhline(y=b - 0.5, color="red", linewidth=0.8, linestyle="--")
            ax.axvline(x=b - 0.5, color="red", linewidth=0.8, linestyle="--")
        ax.set_xticks(region_centers)
        ax.set_xticklabels(region_labels, rotation=45, fontsize=8)
        ax.set_yticks(region_centers)
        ax.set_yticklabels(region_labels, fontsize=8)
        ax.set_title(
            f"Multi-Block Causal Mask  {title_suffix}\n"
            f"(bl={block_length}, bs={block_size}, {num_vis_blocks} blocks)",
            fontsize=11,
        )
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    fig.suptitle("Legacy vs New Multi-Block Causal Mask", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = "./test/multi_block_causal_mask.png"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {out_path}")
    plt.close()

    # Verify some attention rules
    print("\n── Verification ──")
    L = len(input_ids)

    # Check: B1_data should see B0_data but NOT B0_mask
    b1_data_start = p + 2 * block_length  # block 1 data starts here
    b0_data_start = p
    b0_mask_start = p + block_length

    b1d_sees_b0d = mask_np[b1_data_start, b0_data_start]
    b1d_sees_b0m = mask_np[b1_data_start, b0_mask_start]
    print(f"B1_data[0] sees B0_data[0]: {bool(b1d_sees_b0d)} (expected: True)")
    print(f"B1_data[0] sees B0_mask[0]: {bool(b1d_sees_b0m)} (expected: False)")

    # Check: B0_data should NOT see B1_data
    b0d_sees_b1d = mask_np[b0_data_start, b1_data_start]
    print(f"B0_data[0] sees B1_data[0]: {bool(b0d_sees_b1d)} (expected: False)")

    # Check: B1_mask should see B0_data
    b1_mask_start = p + 3 * block_length
    b1m_sees_b0d = mask_np[b1_mask_start, b0_data_start]
    print(f"B1_mask[0] sees B0_data[0]: {bool(b1m_sees_b0d)} (expected: True)")

    # Check: data/mask sees all prompt
    prompt_visible = mask_np[b1_data_start, 0]
    print(f"B1_data[0] sees Prompt[0]: {bool(prompt_visible)} (expected: True)")

    # Check: Prompt does NOT see data/mask (prompt comes before response)
    prompt_sees_b0m = mask_np[0, b0_mask_start]
    print(f"Prompt[0] sees B0_mask[0]: {bool(prompt_sees_b0m)} (expected: False)")
    prompt_sees_b0d = mask_np[0, b0_data_start]
    print(f"Prompt[0] sees B0_data[0]: {bool(prompt_sees_b0d)} (expected: False)")

    # Check: Prompt block-level causal
    # Prompt has 47 tokens, block_size=4  prompt blocks: 0..3(blk0), 4..7(blk1), ..., 44..46(blk11)
    # Last prompt token (idx 46, blk11) should see first prompt token (idx 0, blk0)
    last_p = p - 1
    print(f"Prompt[last] sees Prompt[0]: {bool(mask_np[last_p, 0])} (expected: True)")
    # First prompt token (idx 0, blk0) should NOT see token in blk1 (idx 4)
    if p > block_size:
        print(f"Prompt[0] sees Prompt[{block_size}]: {bool(mask_np[0, block_size])} (expected: False)")
        # Token in blk1 (idx 4) should see token in blk0 (idx 0)
        print(f"Prompt[{block_size}] sees Prompt[0]: {bool(mask_np[block_size, 0])} (expected: True)")

    print("\nDone!")


if __name__ == "__main__":
    test_with_rollout_data()
