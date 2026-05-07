import torch
from typing import Optional

def create_causal_mask_from_labels(
    token_labels: torch.LongTensor,
    block_size: int,
    device: torch.device = None,
    block_causal_prompt: bool = True,
    mask_test: bool = False,
) -> torch.Tensor:
    """
     token_labels  dLLM Causal Mask  Attention Mask

    Args:
        token_labels: (Batch, Seq_Len)
            - 0: Prompt
            - 1 ~ block_size: Data Token ( Step)
            - block_size + 1: Mask Token ( Token)
            - -1: Padding
        block_size: int,
        device: torch.device
        block_causal_prompt: bool,
            True (): Prompt  block_size , block-level causal,  data/mask
            False (): Prompt  Prompt +  Mask,  Prompt
        mask_test: bool,  True Mask  Prompt +  Mask Data

    Returns:
        attn_mask: (Batch, 1, Seq_Len, Seq_Len)
                   1 (True) 0 (False)
    """
    if device is None:
        device = token_labels.device

    B, L = token_labels.shape

    # =========================================================
    # 1.  Token
    # =========================================================
    # Type 0: Prompt
    # Type 1: Data (GT)
    # Type 2: Mask (Query)
    # Type 3: Padding

    is_prompt = (token_labels == 0)
    is_data = (token_labels > 0) & (token_labels <= block_size)
    is_mask = (token_labels == block_size + 1)
    is_pad = (token_labels == -1)

    # =========================================================
    # 2.  Time Step
    # =========================================================
    # Data Token  label  (1..T)
    # Mask Token  label  (block_size+1) Data
    # Prompt  0
    # Padding  inf

    time_steps = token_labels.clone().float()

    for b in range(B):
        data_vals = time_steps[b, is_data[b]]
        num_data = data_vals.shape[0]
        mask_indices = torch.nonzero(is_mask[b], as_tuple=True)[0]

        if mask_indices.shape[0] == num_data:
             time_steps[b, mask_indices] = data_vals
        elif mask_indices.shape[0] > 0:
            min_len = min(num_data, mask_indices.shape[0])
            time_steps[b, mask_indices[:min_len]] = data_vals[:min_len]

    time_steps[is_pad] = float('inf')

    type_vals = torch.zeros_like(token_labels) # 1=Data, 2=Mask
    type_vals[is_data] = 1
    type_vals[is_mask] = 2
    type_i = type_vals[:, None, :, None]       # (B, 1, L, 1) -> Row
    type_j = type_vals[:, None, None, :]       # (B, 1, 1, L) -> Col

    time_i = time_steps[:, None, :, None]      # (B, 1, L, 1)
    time_j = time_steps[:, None, None, :]      # (B, 1, 1, L)

    is_prompt_j = is_prompt.view(B, 1, 1, L)
    is_pad_i = is_pad.view(B, 1, L, 1)
    is_pad_j = is_pad.view(B, 1, 1, L)

    is_prompt_i = is_prompt.view(B, 1, L, 1)

    if block_causal_prompt:
        # ── : Prompt block-level causal (SDAR style) ──
        # Prompt  block_size  block  block
        # Prompt  data/mask
        prompt_cumpos = is_prompt.long().cumsum(dim=1) - 1
        prompt_blk = prompt_cumpos // block_size
        prompt_blk_i = prompt_blk[:, None, :, None]
        prompt_blk_j = prompt_blk[:, None, None, :]
        rule_prompt = is_prompt_i & is_prompt_j & (prompt_blk_j <= prompt_blk_i)
        rule_see_prompt = (~is_prompt_i) & (~is_pad_i) & is_prompt_j
    else:
        # ── : Prompt  Prompt +  Mask Prompt ──
        rule_prompt = is_prompt_i & is_prompt_j
        is_mask_j = is_mask.view(B, 1, 1, L)
        rule_prompt |= is_prompt_i & is_mask_j  # prompt sees mask
        rule_see_prompt = is_prompt_j.expand(B, 1, L, L)  # everyone sees prompt

    # ── Data  ──
    mask_data_data = (type_i == 1) & (type_j == 1) & (time_j <= time_i)
    mask_data_mask = (type_i == 1) & (type_j == 2) & (time_j > time_i)

    if mask_test:
        mask_mask_data = False
        is_mask_i = is_mask.view(B, 1, L, 1)
        is_mask_j_broad = is_mask.view(B, 1, 1, L)
        mask_mask_mask = is_mask_i & is_mask_j_broad
    else:
        mask_mask_data = (type_i == 2) & (type_j == 1) & (time_j < time_i)
        mask_mask_mask = (type_i == 2) & (type_j == 2) & (time_j >= time_i)

    final_mask = (
        rule_prompt |
        rule_see_prompt |
        mask_data_data |
        mask_data_mask |
        mask_mask_data |
        mask_mask_mask
    )

    #  Padding (Padding  Padding)
    final_mask = final_mask & (~is_pad_i) & (~is_pad_j)

    return final_mask.to(dtype=torch.bool) #