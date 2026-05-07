"""
ASCII visualization of single-block speculative decoding process.

Demonstrates:
  1. Draft: step-by-step filling of MASK positions via block diffusion
  2. Target: causal mask layout [Prompt|B_data|B_mask] and verification
  3. MRS: accept/reject result per position
"""


def _char(pos_val, mask_id):
    """One-char representation: # = MASK, . = filled token."""
    return "#" if pos_val == mask_id else "."


def print_draft_steps(block_length, denoising_steps, step_records, mask_token_id=151669):
    """
    Print ASCII diagram of draft block diffusion steps.

    step_records: list of (step_idx, block_ids, unmask_positions, step_map)
      - step_idx: 0-based
      - block_ids: list[int] length block_length
      - unmask_positions: list of indices unmasked this step
      - step_map: list[int] after this step
    """
    w = max(60, block_length * 2 + 30)
    lines = [
        "",
        "+" + "-" * (w - 2) + "+",
        "|  DRAFT: Block Diffusion (fill MASK -> tokens step by step)" + " " * max(0, w - 62) + "|",
        "+" + "-" * (w - 2) + "+",
    ]
    for step_idx, block_ids, unmask_pos, step_map in step_records:
        row = [_char(block_ids[i], mask_token_id) for i in range(block_length)]
        row_str = "".join(row)
        unmask_str = ",".join(str(p) for p in sorted(unmask_pos)) if unmask_pos else "-"
        step_map_str = "".join(str(s) for s in step_map)
        line = f"|  Step {step_idx}:  [{row_str}]  unmask pos [{unmask_str}]  step_map={step_map_str}"
        lines.append(line + " " * max(0, w - len(line) - 2) + "|")
    lines.append("+" + "-" * (w - 2) + "+")
    lines.append("  Legend: # = MASK, . = predicted token")
    lines.append("")
    return "\n".join(lines)


def print_verify_layout(prompt_len, draft_ids, step_map, block_length, block_size, mask_token_id=151669):
    """
    Print ASCII diagram of target verification input layout.

    Layout: [Prompt | B_data | B_mask]
    """
    p = "P" * min(prompt_len, 16)
    if prompt_len > 16:
        p = p + ".."
    data_row = "".join(_char(t, mask_token_id) for t in draft_ids)
    mask_row = "#" * block_length
    step_str = "".join(str(s) for s in step_map)
    w = max(60, block_length * 2 + 30)
    lines = [
        "",
        "+" + "-" * (w - 2) + "+",
        "|  TARGET: Causal Mask Verification (single forward)" + " " * max(0, w - 50) + "|",
        "+" + "-" * (w - 2) + "+",
        f"|  Layout: [Prompt({prompt_len}) | B_data | B_mask]" + " " * max(0, w - 45) + "|",
        "|" + " " * (w - 2) + "|",
        f"|  Prompt:  {p}" + " " * max(0, w - 14 - len(p)) + "|",
        f"|  B_data:  [{data_row}]  (draft as DATA, step_map={step_str})" + " " * max(0, w - 45 - block_length * 2) + "|",
        f"|  B_mask:  [{mask_row}]  (target predicts here)" + " " * max(0, w - 40 - block_length * 2) + "|",
        "|" + " " * (w - 2) + "|",
        "|  Target attends: prompt + B_data; predicts logits at B_mask" + " " * max(0, w - 58) + "|",
        "+" + "-" * (w - 2) + "+",
        "",
    ]
    return "\n".join(lines)


def print_mrs_result(draft_ids, final_ids, n_accepted, bonus_token, block_length, mask_token_id=151669):
    """
    Print ASCII diagram of MRS accept/reject per position.

    n_accepted = number of draft tokens accepted before first rejection.
    When bonus_token is not None, we rejected at position n_accepted and resampled.
    """
    results = []
    for i in range(block_length):
        if i < n_accepted:
            results.append("OK ")   # draft accepted
        elif i == n_accepted and bonus_token is not None:
            results.append("SUB")   # rejected at this pos, resampled
        else:
            results.append("REJ")  # truncated (never reached)
    result_str = " ".join(results)
    w = max(60, block_length * 4 + 20)
    lines = [
        "",
        "+" + "-" * (w - 2) + "+",
        "|  MRS: Modified Rejection Sampling (accept/reject per position)" + " " * max(0, w - 57) + "|",
        "+" + "-" * (w - 2) + "+",
        f"|  Pos:    " + "  ".join(f"{i}" for i in range(block_length)) + " " * max(0, w - 12 - block_length * 3) + "|",
        f"|  Result: {result_str}" + " " * max(0, w - 10 - len(result_str)) + "|",
        f"|  Accepted: {n_accepted}/{block_length}, bonus_token={bonus_token}" + " " * max(0, w - 38) + "|",
        "+" + "-" * (w - 2) + "+",
        "  Legend: OK=accepted draft, SUB=resampled, REJ=rejected",
        "",
    ]
    return "\n".join(lines)


