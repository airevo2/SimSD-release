"""Target model: block-wise causal mask verification forward."""

import os
import sys
import time
import torch
import torch.nn.functional as F

MASK_TOKEN_ID = 151669


# Verify-side CUDA graph cache. Only the use_eval_sdpa=True branch is graph-
# eligible: the training branch goes through fused_flex_attention which has
# dynamic-shape logic we don't want to audit inside a capture. padded_len is
# fixed per run (computed once from max_plen + (num_blocks+1)*2*block_length,
# aligned to 128), so a single graph replays for every block of every prompt.
# Key: (id(model), batch, padded_len, block_size, str(device)).
# Value: (CUDAGraph, static_input, static_tl, static_bi, static_pos,
#         static_mask, static_logits).
# Legacy key without batch is used for the single-prompt API to preserve
# cache hits from before the batched refactor.
_VERIFY_GRAPH_CACHE: dict = {}


def _capture_verify_forward_graph(model, padded_len, block_size, device,
                                  mask_token_id=MASK_TOKEN_ID):
    """Capture CUDA graph for a single SDPA-eval verify forward.

    Preconditions:
      - Caller is using use_eval_sdpa=True (training branch is not graphed).
      - ``padded_len`` and ``block_size`` are fixed across all verify calls.
      - ``patch_sdpa_eval_attention(model)`` has been (or will be) called;
        otherwise the upstream ``torch.all(attention_mask)`` per-layer sync
        would make capture fail.

    The 4D attention mask is built OUTSIDE the capture region per call
    ``create_multi_block_causal_mask`` contains ``.item()`` syncs  and its
    contents get copied into a persistent bool buffer before each ``g.replay()``.
    Same pattern for input_ids / token_labels / block_ids / position_ids.

    Returns the graph + persistent buffers. ``static_logits`` is the lm_head
    output tensor (captured via forward hook during capture); replays write
    new logits into the same storage.
    """
    from speculative_decoding.draft import patch_sdpa_eval_attention
    patch_sdpa_eval_attention(model)
    # modeling_sdar gates a torch.nonzero-based RoPE-copy path on
    # `token_labels is not None and not hasattr(self, '_skip_rope_copy')`.
    # nonzero() is data-dependent so it's forbidden during CUDA graph capture.
    # The hasattr check lives on SDARModel (model.model), NOT the outer
    # SDARForCausalLM. For multi-block verify, position_ids between data and
    # mask are already shared, so skipping is a no-op behaviorally.
    getattr(model, "model", model)._skip_rope_copy = True

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)

        static_input = torch.full((1, padded_len), mask_token_id,
                                  dtype=torch.long, device=device)
        static_tl = torch.zeros((1, padded_len), dtype=torch.long, device=device)
        static_bi = torch.full((1, padded_len), -1, dtype=torch.long,
                               device=device)
        static_pos = torch.zeros((1, padded_len), dtype=torch.long,
                                 device=device)
        static_mask = torch.zeros((1, 1, padded_len, padded_len),
                                  dtype=torch.bool, device=device)

        captured = {}

        def hook_fn(_m, _i, output):
            captured["logits"] = output.detach()

        hook = model.lm_head.register_forward_hook(hook_fn)
        orig_bs = model.config.block_size
        model.config.block_size = block_size
        try:
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(5):
                    model(
                        input_ids=static_input,
                        attention_mask=static_mask,
                        token_labels=static_tl,
                        position_ids=static_pos,
                    )
            torch.cuda.current_stream(device).wait_stream(s)
            torch.cuda.synchronize(device)

            g = torch.cuda.CUDAGraph()
            # Same device/stream pinning as draft.py: torch.cuda.graph() with
            # stream=None falls back to a class-level default capture stream
            # that gets pinned to the FIRST device used, causing cross-device
            # failures if draft captured earlier on another GPU.
            with torch.cuda.graph(g, stream=s,
                                  capture_error_mode="thread_local"), \
                    torch.no_grad():
                model(
                    input_ids=static_input,
                    attention_mask=static_mask,
                    token_labels=static_tl,
                    position_ids=static_pos,
                )
            static_logits = captured["logits"]
        finally:
            hook.remove()
            model.config.block_size = orig_bs

    return (g, static_input, static_tl, static_bi, static_pos, static_mask,
            static_logits)


def _get_verify_forward_graph(model, padded_len, block_size, device,
                              mask_token_id=MASK_TOKEN_ID):
    key = (id(model), int(padded_len), int(block_size), str(device))
    entry = _VERIFY_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_verify_forward_graph(
            model, padded_len, block_size, device, mask_token_id)
        _VERIFY_GRAPH_CACHE[key] = entry
    return entry


def _capture_verify_forward_graph_batch(model, batch, padded_len, block_size,
                                        device, mask_token_id=MASK_TOKEN_ID):
    """Batched variant of _capture_verify_forward_graph.

    Same assumptions (use_eval_sdpa path, patched SDPA kernel). All persistent
    buffers gain a batch dim:
      static_input / static_tl / static_bi / static_pos: (B, padded_len)
      static_mask:  (B, 1, padded_len, padded_len)    per-row 4D mask
      static_logits:(B, padded_len, vocab_size)        lm_head output
    """
    from speculative_decoding.draft import patch_sdpa_eval_attention
    patch_sdpa_eval_attention(model)
    getattr(model, "model", model)._skip_rope_copy = True

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)

        static_input = torch.full((batch, padded_len), mask_token_id,
                                  dtype=torch.long, device=device)
        static_tl = torch.zeros((batch, padded_len), dtype=torch.long,
                                device=device)
        static_bi = torch.full((batch, padded_len), -1, dtype=torch.long,
                               device=device)
        static_pos = torch.zeros((batch, padded_len), dtype=torch.long,
                                 device=device)
        static_mask = torch.zeros((batch, 1, padded_len, padded_len),
                                  dtype=torch.bool, device=device)

        captured = {}

        def hook_fn(_m, _i, output):
            captured["logits"] = output.detach()

        hook = model.lm_head.register_forward_hook(hook_fn)
        orig_bs = model.config.block_size
        model.config.block_size = block_size
        try:
            s = torch.cuda.Stream(device=device)
            s.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(s), torch.no_grad():
                for _ in range(5):
                    model(
                        input_ids=static_input,
                        attention_mask=static_mask,
                        token_labels=static_tl,
                        position_ids=static_pos,
                    )
            torch.cuda.current_stream(device).wait_stream(s)
            torch.cuda.synchronize(device)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s,
                                  capture_error_mode="thread_local"), \
                    torch.no_grad():
                model(
                    input_ids=static_input,
                    attention_mask=static_mask,
                    token_labels=static_tl,
                    position_ids=static_pos,
                )
            static_logits = captured["logits"]
        finally:
            hook.remove()
            model.config.block_size = orig_bs

    return (g, static_input, static_tl, static_bi, static_pos, static_mask,
            static_logits)


def _get_verify_forward_graph_batch(model, batch, padded_len, block_size,
                                    device, mask_token_id=MASK_TOKEN_ID):
    key = (id(model), int(batch), int(padded_len), int(block_size), str(device))
    entry = _VERIFY_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_verify_forward_graph_batch(
            model, batch, padded_len, block_size, device, mask_token_id)
        _VERIFY_GRAPH_CACHE[key] = entry
    return entry


def clear_verify_graph_cache():
    """Release all captured verify graphs. Call when model is reloaded / moved."""
    _VERIFY_GRAPH_CACHE.clear()

# Set SPEC_VERIFY_DEBUG=1 to print per-stage CPU dispatch time inside
# target_verify_forward. Useful for diagnosing pipeline overlap failures
# any stage > 5ms is a CPU sync point that blocks the dispatching thread
# and prevents draft_{N+1} from launching concurrently on the other GPU.
_DEBUG = os.environ.get("SPEC_VERIFY_DEBUG", "").lower() in ("1", "true", "yes")
_DEBUG_STATS = {"calls": 0, "stages": {}}


def _dbg_tic():
    return time.perf_counter() if _DEBUG else 0.0


def _dbg_toc(stage, t0):
    if not _DEBUG:
        return
    dt_ms = (time.perf_counter() - t0) * 1000.0
    s = _DEBUG_STATS["stages"].setdefault(stage, [0.0, 0])
    s[0] += dt_ms
    s[1] += 1


def reset_verify_debug_stats():
    _DEBUG_STATS["calls"] = 0
    _DEBUG_STATS["stages"].clear()


def report_verify_debug_stats(prefix: str = ""):
    """Print per-stage mean CPU dispatch time across all calls. Any stage with
    mean ≫ 1ms in pipeline mode is a sync point worth fixing."""
    if not _DEBUG or not _DEBUG_STATS["stages"]:
        return
    print(f"{prefix}[verify-debug] calls={_DEBUG_STATS['calls']}")
    for k, (total, n) in sorted(_DEBUG_STATS["stages"].items()):
        print(f"{prefix}  {k:36s}  mean={total/max(n,1):7.3f} ms  n={n}")


def patch_multi_block_mask_fn(model, block_causal_prompt: bool = True):
    """
    Monkey-patch the model module's create_multi_block_causal_mask to use
    the fixed version from new_attn_multi_block.py.

    Args:
        block_causal_prompt: True = SDAR block-level causal (prompt doesn't see data/mask)
                             False = legacy (prompt sees all prompt + mask tokens)
    """
    from new_attn_multi_block import create_multi_block_causal_mask
    model_module = sys.modules[type(model).__module__]
    bcp = block_causal_prompt  # capture in closure

    def patched(token_labels, block_ids, block_size, **kwargs):
        mask_4d = create_multi_block_causal_mask(
            token_labels, block_ids, block_size, bcp,
        )
        return mask_4d.squeeze(1)

    model_module.create_multi_block_causal_mask = patched


def build_verify_sequence(prompt_ids, accepted_blocks, draft_ids, step_map,
                          block_length, block_size, mask_token_id=MASK_TOKEN_ID,
                          share_mask_position: bool = True):
    """
    Build verification sequence for the target model.

    Layout (inference verify; mask copies of accepted blocks dropped):
      [Prompt | AcceptedB0_data | ... | AcceptedB_{N-1}_data | Draft_data Draft_mask]

    Accepted blocks contribute their committed data only (step_label=1, no
    mask copy)  their logits are never read by MRS, and the legacy training-
    style mask duplication doubled verify seq_len for no benefit.
    Draft block keeps its data + mask pair so the within-block step-causal
    attention mask can pair them. Draft block uses ``step_map`` for token_labels.

    Variable-length accepted blocks are supported (Stage E "truncate" mode):
    each accepted block contributes ``len(blk_tokens)`` data positions only.
    Running ``cur_pos`` advances by ``actual_len`` per accepted block (no
    second advance for a mask copy).
    """
    prompt_len = len(prompt_ids)

    input_ids = list(prompt_ids)
    token_labels = [0] * prompt_len
    block_id_list = [-1] * prompt_len
    labels = [-100] * prompt_len
    position_ids = list(range(prompt_len))

    cur_pos = prompt_len
    # Inference verify only needs logits at the *current* draft block's mask
    # positions (consumed by MRS). The legacy training-style layout that
    # duplicated each accepted block as data + mask doubled the verify
    # seq_len for no logit benefit (accepted-block mask logits are never
    # read). Drop the mask copies for accepted blocks; keep data only.
    # The draft block below still keeps its data + mask pair for the
    # within-block step-causal attention mask.
    for blk_idx, blk_tokens in enumerate(accepted_blocks):
        actual_len = len(blk_tokens)
        blk_positions = list(range(cur_pos, cur_pos + actual_len))
        cur_pos += actual_len

        input_ids.extend(blk_tokens)
        token_labels.extend([1] * actual_len)
        block_id_list.extend([blk_idx] * actual_len)
        labels.extend([-100] * actual_len)
        position_ids.extend(blk_positions)

    cur_blk_idx = len(accepted_blocks)
    data_positions = list(range(cur_pos, cur_pos + block_length))
    cur_pos += block_length

    input_ids.extend(draft_ids)
    # Early-stop: positions where draft never unmasked have step_map[i] = -1.
    # In the DATA segment they should look like still-MASK to the target
    # token_label = block_size + 1, so the multi-block causal mask treats them
    # exactly like the trailing MASK segment.
    token_labels.extend([
        s + 1 if s >= 0 else (block_size + 1)
        for s in step_map
    ])
    block_id_list.extend([cur_blk_idx] * block_length)
    labels.extend([-100] * block_length)
    position_ids.extend(data_positions)

    if share_mask_position:
        # Default SDAR layout: mask token at within-block rank k reuses the
        # SAME RoPE position id as the data token at rank k.
        mask_positions = data_positions
    else:
        # Ablation: mask gets distinct sequential positions right after data.
        # Breaks training-time RoPE alignment on purpose; visibility rules
        # (token_labels / block_ids / attn-mask) are unchanged.
        mask_positions = list(range(cur_pos, cur_pos + block_length))
        cur_pos += block_length

    input_ids.extend([mask_token_id] * block_length)
    token_labels.extend([block_size + 1] * block_length)
    block_id_list.extend([cur_blk_idx] * block_length)
    labels.extend(draft_ids)
    position_ids.extend(mask_positions)

    return input_ids, token_labels, block_id_list, labels, position_ids


def build_verify_sequence_multi(prompt_ids, accepted_blocks, draft_blocks,
                                step_maps, block_length, block_size,
                                mask_token_id=MASK_TOKEN_ID,
                                share_mask_position: bool = True):
    """Multi-draft-block variant of build_verify_sequence.

    Layout (inference verify; mask copies of accepted blocks dropped):
      [Prompt | Acc_0_data | ... | Acc_{M-1}_data
             | Draft_0_data Draft_0_mask | ... | Draft_{K-1}_data Draft_{K-1}_mask]

    Each draft block k uses its own step_map; block_id = M+k. The resulting
    probs can be sliced per mask window (K windows total, one per draft block).
    """
    assert len(draft_blocks) == len(step_maps) and len(draft_blocks) >= 1
    prompt_len = len(prompt_ids)
    input_ids = list(prompt_ids)
    token_labels = [0] * prompt_len
    block_id_list = [-1] * prompt_len
    labels = [-100] * prompt_len
    position_ids = list(range(prompt_len))

    # Running offset (Stage E "truncate" support). Identical to legacy
    # ``blk_idx * block_length`` layout when all accepted blocks are full-length.
    # Accepted-block mask copies are dropped for inference verify (see
    # build_verify_sequence comment): only their data is kept; the K draft
    # blocks below still carry data + mask pairs.
    cur_pos = prompt_len
    for blk_idx, blk_tokens in enumerate(accepted_blocks):
        actual_len = len(blk_tokens)
        blk_positions = list(range(cur_pos, cur_pos + actual_len))
        cur_pos += actual_len

        input_ids.extend(blk_tokens)
        token_labels.extend([1] * actual_len)
        block_id_list.extend([blk_idx] * actual_len)
        labels.extend([-100] * actual_len)
        position_ids.extend(blk_positions)

    M = len(accepted_blocks)
    for k, (draft_ids, step_map) in enumerate(zip(draft_blocks, step_maps)):
        cur_blk_idx = M + k
        data_positions = list(range(cur_pos, cur_pos + block_length))
        cur_pos += block_length

        input_ids.extend(draft_ids)
        token_labels.extend([
            s + 1 if s >= 0 else (block_size + 1)
            for s in step_map
        ])
        block_id_list.extend([cur_blk_idx] * block_length)
        labels.extend([-100] * block_length)
        position_ids.extend(data_positions)

        if share_mask_position:
            # Default: mask reuses data's RoPE positions (training-time layout).
            mask_positions = data_positions
        else:
            # Ablation: mask gets distinct positions; advances cur_pos so the
            # next draft block sits after both data + mask of this one.
            mask_positions = list(range(cur_pos, cur_pos + block_length))
            cur_pos += block_length

        input_ids.extend([mask_token_id] * block_length)
        token_labels.extend([block_size + 1] * block_length)
        block_id_list.extend([cur_blk_idx] * block_length)
        labels.extend(draft_ids)
        position_ids.extend(mask_positions)

    return input_ids, token_labels, block_id_list, labels, position_ids


def target_verify_forward_multi(
    model, prompt_ids, accepted_blocks, draft_blocks, step_maps,
    block_length, block_size, pad_token_id, padded_len,
    mask_token_id=MASK_TOKEN_ID, use_eval_sdpa: bool = False,
    return_on_device: bool = False,
    use_cuda_graph: bool = False,
    share_mask_position: bool = True,
):
    """K-block verify: one target forward returns K probs tensors (one per draft block).

    Call this instead of ``target_verify_forward`` when ``K = len(draft_blocks) > 1``.
    The caller MRS-verifies block 0..K-1 sequentially; on first reject the
    remaining draft blocks are discarded (they were speculated against a
    context that may now disagree).
    """
    device = next(model.parameters()).device
    graph_ok = (use_cuda_graph and use_eval_sdpa and device.type == "cuda")
    K = len(draft_blocks)
    assert K >= 1 and len(step_maps) == K

    _DEBUG_STATS["calls"] += 1

    t0 = _dbg_tic()
    input_ids, token_labels, block_id_list, labels_list, position_ids = \
        build_verify_sequence_multi(
            prompt_ids, accepted_blocks, draft_blocks, step_maps,
            block_length, block_size, mask_token_id,
            share_mask_position=share_mask_position,
        )
    real_len = len(input_ids)
    pad_len = padded_len - real_len
    if pad_len < 0:
        raise ValueError(
            f"Sequence length {real_len} exceeds padded_len {padded_len} "
            f"(K={K}, accepted={len(accepted_blocks)}, prompt={len(prompt_ids)})"
        )

    input_ids_padded = input_ids + [pad_token_id] * pad_len
    token_labels_padded = token_labels + [-1] * pad_len
    block_ids_padded = block_id_list + [-1] * pad_len
    labels_padded = labels_list + [-100] * pad_len
    position_ids_padded = position_ids + [0] * pad_len
    _dbg_toc("01_build_seq_python", t0)

    t0 = _dbg_tic()
    input_t = torch.tensor([input_ids_padded], dtype=torch.long, device=device)
    tl_t = torch.tensor([token_labels_padded], dtype=torch.long, device=device)
    bi_t = torch.tensor([block_ids_padded], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels_padded], dtype=torch.long, device=device)
    pos_t = torch.tensor([position_ids_padded], dtype=torch.long, device=device)
    _dbg_toc("02_h2d_tensors", t0)

    captured = {}

    def hook_fn(module, inp, output):
        captured["logits"] = output.detach()

    t0 = _dbg_tic()
    hook = None
    orig_bs = model.config.block_size
    model.config.block_size = block_size
    if use_eval_sdpa:
        model.eval()
        from new_attn_multi_block import create_multi_block_causal_mask as _mbc_mask_fn
        attn_mask_4d = _mbc_mask_fn(tl_t, bi_t, block_size, block_causal_prompt=True)
    else:
        model.train()
    if not graph_ok:
        hook = model.lm_head.register_forward_hook(hook_fn)
    _dbg_toc("03_mask_build", t0)
    try:
        _dev = input_t.device
        t0 = _dbg_tic()
        with torch.cuda.device(_dev), torch.no_grad():
            if graph_ok:
                entry = _get_verify_forward_graph(
                    model, padded_len, block_size, _dev, mask_token_id)
                (g, s_input, s_tl, s_bi, s_pos, s_mask, s_logits) = entry
                s_input.copy_(input_t)
                s_tl.copy_(tl_t)
                s_bi.copy_(bi_t)
                s_pos.copy_(pos_t)
                s_mask.copy_(attn_mask_4d)
                g.replay()
                captured["logits"] = s_logits
            elif use_eval_sdpa:
                model(
                    input_ids=input_t,
                    attention_mask=attn_mask_4d,
                    token_labels=tl_t,
                    position_ids=pos_t,
                )
            else:
                model(
                    input_ids=input_t,
                    token_labels=tl_t,
                    block_ids=bi_t,
                    labels=labels_t,
                    position_ids=pos_t,
                )
        _dbg_toc("04_model_forward_dispatch", t0)

        t0 = _dbg_tic()
        logits = captured["logits"]
        # Variable-length accepted blocks (Stage E "truncate"): each contributes
        # len(blk) positions in the new layout (mask copies dropped).
        # Draft blocks are always full-length data + mask, so contribute
        # k * 2 * block_length per preceding draft block.
        acc_total = sum(len(blk) for blk in accepted_blocks)
        probs_per_block = []
        for k in range(K):
            mask_start = (
                len(prompt_ids) + acc_total + k * 2 * block_length + block_length
            )
            mask_end = mask_start + block_length
            mask_logits = logits[0, mask_start:mask_end]
            target_probs = F.softmax(mask_logits.float(), dim=-1)
            probs_per_block.append(
                target_probs if return_on_device else target_probs.cpu()
            )
        _dbg_toc("05_softmax_and_return", t0)
        return probs_per_block
    finally:
        if hook is not None:
            hook.remove()
        model.eval()
        model.config.block_size = orig_bs


def target_verify_forward(model, prompt_ids, accepted_blocks, draft_ids, step_map,
                          block_length, block_size, pad_token_id, padded_len,
                          mask_token_id=MASK_TOKEN_ID, use_eval_sdpa: bool = False,
                          return_on_device: bool = False,
                          use_cuda_graph: bool = False,
                          share_mask_position: bool = True):
    """
    Single forward pass through target model with block-wise causal mask.

    ``use_eval_sdpa``: if True, keep model in eval(), build the 4D multi-block
    bool mask externally and pass via ``attention_mask``  SDPA path (same kernel
    draft uses). Skips the training branch / patched create_multi_block_causal_mask
    entirely. See sweep_forward.hf_verify_run docstring for rationale.

    ``return_on_device``: False  return .cpu() tensor (default; keeps original
    semantics and triggers a D2H sync that waits for the verify forward to finish).
    True  return on target GPU, no sync. **Only the pipelined speculative path
    should pass True**: it needs verify_N to stay async so draft_{N+1} can
    overlap on the other GPU; MRS later triggers the sync via its own .item().

    ``use_cuda_graph``: if True AND ``use_eval_sdpa=True`` AND device is CUDA,
    replay a captured graph instead of running eager. padded_len / block_size
    are the cache key so a single capture amortizes across all blocks and
    samples. First call hits the capture (~6 extra forwards); subsequent calls
    skip kernel launch overhead. The training branch is never graphed
    regardless of the flag (flex_attention has dynamic-shape logic).

    Returns:
        target_probs: Tensor (block_length, vocab_size) on CPU (default) or target GPU
    """
    device = next(model.parameters()).device
    graph_ok = (use_cuda_graph and use_eval_sdpa and device.type == "cuda")

    _DEBUG_STATS["calls"] += 1

    t0 = _dbg_tic()
    input_ids, token_labels, block_id_list, labels_list, position_ids = \
        build_verify_sequence(
            prompt_ids, accepted_blocks, draft_ids, step_map,
            block_length, block_size, mask_token_id,
            share_mask_position=share_mask_position,
        )

    real_len = len(input_ids)
    pad_len = padded_len - real_len
    if pad_len < 0:
        raise ValueError(f"Sequence length {real_len} exceeds padded_len {padded_len}")

    input_ids_padded = input_ids + [pad_token_id] * pad_len
    token_labels_padded = token_labels + [-1] * pad_len
    block_ids_padded = block_id_list + [-1] * pad_len
    labels_padded = labels_list + [-100] * pad_len
    position_ids_padded = position_ids + [0] * pad_len
    _dbg_toc("01_build_seq_python", t0)

    t0 = _dbg_tic()
    input_t = torch.tensor([input_ids_padded], dtype=torch.long, device=device)
    tl_t = torch.tensor([token_labels_padded], dtype=torch.long, device=device)
    bi_t = torch.tensor([block_ids_padded], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels_padded], dtype=torch.long, device=device)
    pos_t = torch.tensor([position_ids_padded], dtype=torch.long, device=device)
    _dbg_toc("02_h2d_tensors", t0)

    captured = {}

    def hook_fn(module, inp, output):
        captured["logits"] = output.detach()

    t0 = _dbg_tic()
    hook = None
    orig_bs = model.config.block_size
    model.config.block_size = block_size
    if use_eval_sdpa:
        model.eval()
        from new_attn_multi_block import create_multi_block_causal_mask as _mbc_mask_fn
        # NOTE: _mbc_mask_fn  .item() (block_ids.max(), blk_data_cnt.max()),
        #  GPUCPU sync,  mask ,  ~1-2ms,
        #  SDARAttention.forward  torch.all sync (~36ms).
        #  graph-ok  mask  capture .
        attn_mask_4d = _mbc_mask_fn(tl_t, bi_t, block_size, block_causal_prompt=True)
    else:
        model.train()
    if not graph_ok:
        hook = model.lm_head.register_forward_hook(hook_fn)
    _dbg_toc("03_mask_build", t0)
    try:
        # Pin current CUDA device for the duration of the forward. Without this,
        # in dual-GPU runs (draft cuda:0, target cuda:1) the current device can
        # still be the draft's device when control reaches here, and Liger's
        # Triton SwiGLU kernel launches on the wrong stream  "invalid resource
        # handle". torch.cuda.device accepts torch.device or string.
        _dev = input_t.device
        t0 = _dbg_tic()
        with torch.cuda.device(_dev), torch.no_grad():
            if graph_ok:
                entry = _get_verify_forward_graph(
                    model, padded_len, block_size, _dev, mask_token_id)
                (g, s_input, s_tl, s_bi, s_pos, s_mask, s_logits) = entry
                # Copy per-call inputs into the persistent buffers the graph
                # was captured against. The bool mask has to match exactly
                # static_mask is (1,1,padded,padded) matching attn_mask_4d.
                s_input.copy_(input_t)
                s_tl.copy_(tl_t)
                s_bi.copy_(bi_t)
                s_pos.copy_(pos_t)
                s_mask.copy_(attn_mask_4d)
                g.replay()
                captured["logits"] = s_logits
            elif use_eval_sdpa:
                model(
                    input_ids=input_t,
                    attention_mask=attn_mask_4d,
                    token_labels=tl_t,
                    position_ids=pos_t,
                )
            else:
                model(
                    input_ids=input_t,
                    token_labels=tl_t,
                    block_ids=bi_t,
                    labels=labels_t,
                    position_ids=pos_t,
                )
        #   pipeline :  dispatch  < 5ms.
        #  ≥ 30ms,  model  per-layer sync (: SDARAttention
        # eval  torch.all(attention_mask)  sync,  patch_sdpa_eval_attention
        # ).  RoPE-copy  torch.nonzero (data-dependent shape).
        _dbg_toc("04_model_forward_dispatch", t0)

        t0 = _dbg_tic()
        logits = captured["logits"]
        # Variable-length accepted blocks (Stage E "truncate"): each contributes
        # len(blk) positions in the new layout (mask copies dropped).
        # Reduces to (M * block_length) when full.
        acc_total = sum(len(blk) for blk in accepted_blocks)
        mask_start = len(prompt_ids) + acc_total + block_length
        mask_end = mask_start + block_length
        mask_logits = logits[0, mask_start:mask_end]
        target_probs = F.softmax(mask_logits.float(), dim=-1)
        result = target_probs if return_on_device else target_probs.cpu()
        # :  return_on_device=False , .cpu()  D2H + sync,
        #  model forward  GPU ,  dispatch.
        #  return_on_device=True (pipeline ) ,  enqueue softmax
        # kernel,  sync  mrs_verify  .item() / .tolist().
        _dbg_toc("05_softmax_and_return", t0)
        return result
    finally:
        if hook is not None:
            hook.remove()
        model.eval()
        model.config.block_size = orig_bs


def target_verify_forward_multi_batch(
    model, prompts_batch, accepted_blocks_batch, draft_blocks_batch,
    step_maps_batch, block_length, block_size, pad_token_id, padded_len,
    mask_token_id=MASK_TOKEN_ID, use_eval_sdpa: bool = False,
    return_on_device: bool = False, use_cuda_graph: bool = False,
    share_mask_position: bool = True,
):
    """Batched K-block verify.

    Args (all length B, all rows must have same K):
        prompts_batch:         list[list[int]]
        accepted_blocks_batch: list[list[list[int]]]    B rows of M blocks × bl
        draft_blocks_batch:    list[list[list[int]]]    B rows of K blocks × bl
        step_maps_batch:       list[list[list[int]]]    B rows of K step_maps

    All rows share the same M (accepted count) and K (draft lookahead). Prompts
    may differ in length; shorter sequences are right-padded with pad_token_id
    inside the common padded_len. The 4D attention mask is built per-row and
    stacked into (B, 1, padded_len, padded_len).

    Returns:
        probs_per_block_batch: list[list[Tensor(bl, V)]] of shape (B, K)
    """
    device = next(model.parameters()).device
    graph_ok = (use_cuda_graph and use_eval_sdpa and device.type == "cuda")
    B = len(prompts_batch)
    assert B == len(accepted_blocks_batch) == len(draft_blocks_batch) \
        == len(step_maps_batch)
    assert B > 0
    K = len(draft_blocks_batch[0])
    M = len(accepted_blocks_batch[0])
    for i in range(B):
        assert len(draft_blocks_batch[i]) == K
        assert len(step_maps_batch[i]) == K
        assert len(accepted_blocks_batch[i]) == M, \
            "All rows must share the same accepted-block count"

    _DEBUG_STATS["calls"] += 1

    # Per-row build + right-pad into uniform padded_len.
    t0 = _dbg_tic()
    input_rows, tl_rows, bi_rows, labels_rows, pos_rows, real_lens = \
        [], [], [], [], [], []
    for i in range(B):
        iid, tl, bi, lb, pos = build_verify_sequence_multi(
            prompts_batch[i], accepted_blocks_batch[i], draft_blocks_batch[i],
            step_maps_batch[i], block_length, block_size, mask_token_id,
            share_mask_position=share_mask_position,
        )
        real_lens.append(len(iid))
        pad_len = padded_len - len(iid)
        if pad_len < 0:
            raise ValueError(
                f"Row {i}: sequence length {len(iid)} exceeds padded_len "
                f"{padded_len} (K={K}, M={M}, prompt={len(prompts_batch[i])})"
            )
        input_rows.append(iid + [pad_token_id] * pad_len)
        tl_rows.append(tl + [-1] * pad_len)
        bi_rows.append(bi + [-1] * pad_len)
        labels_rows.append(lb + [-100] * pad_len)
        pos_rows.append(pos + [0] * pad_len)
    _dbg_toc("01_build_seq_python", t0)

    t0 = _dbg_tic()
    input_t = torch.tensor(input_rows, dtype=torch.long, device=device)
    tl_t = torch.tensor(tl_rows, dtype=torch.long, device=device)
    bi_t = torch.tensor(bi_rows, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels_rows, dtype=torch.long, device=device)
    pos_t = torch.tensor(pos_rows, dtype=torch.long, device=device)
    _dbg_toc("02_h2d_tensors", t0)

    captured = {}

    def hook_fn(_m, _i, output):
        captured["logits"] = output.detach()

    t0 = _dbg_tic()
    hook = None
    orig_bs = model.config.block_size
    model.config.block_size = block_size
    if use_eval_sdpa:
        model.eval()
        from new_attn_multi_block import create_multi_block_causal_mask as _mbc_mask_fn
        # _mbc_mask_fn expects (B, L) token_labels / block_ids; it builds the
        # 4D mask per row, so the batched call works directly.
        attn_mask_4d = _mbc_mask_fn(tl_t, bi_t, block_size,
                                    block_causal_prompt=True)
    else:
        model.train()
    if not graph_ok:
        hook = model.lm_head.register_forward_hook(hook_fn)
    _dbg_toc("03_mask_build", t0)
    try:
        _dev = input_t.device
        t0 = _dbg_tic()
        with torch.cuda.device(_dev), torch.no_grad():
            if graph_ok:
                entry = _get_verify_forward_graph_batch(
                    model, B, padded_len, block_size, _dev, mask_token_id)
                (g, s_input, s_tl, s_bi, s_pos, s_mask, s_logits) = entry
                s_input.copy_(input_t)
                s_tl.copy_(tl_t)
                s_bi.copy_(bi_t)
                s_pos.copy_(pos_t)
                s_mask.copy_(attn_mask_4d)
                g.replay()
                captured["logits"] = s_logits
            elif use_eval_sdpa:
                model(
                    input_ids=input_t,
                    attention_mask=attn_mask_4d,
                    token_labels=tl_t,
                    position_ids=pos_t,
                )
            else:
                model(
                    input_ids=input_t,
                    token_labels=tl_t,
                    block_ids=bi_t,
                    labels=labels_t,
                    position_ids=pos_t,
                )
        _dbg_toc("04_model_forward_dispatch", t0)

        t0 = _dbg_tic()
        logits = captured["logits"]  # (B, padded_len, V)
        probs_per_block_batch = []
        for i in range(B):
            prompt_len_i = len(prompts_batch[i])
            # Variable-length accepted blocks (Stage E "truncate"): per-row
            # accepted-region length = sum of len(blk) in the new layout
            # (mask copies dropped; only data positions remain).
            acc_total_i = sum(len(blk) for blk in accepted_blocks_batch[i])
            row_probs = []
            for k in range(K):
                mask_start = (
                    prompt_len_i + acc_total_i + k * 2 * block_length + block_length
                )
                mask_end = mask_start + block_length
                mask_logits = logits[i, mask_start:mask_end]
                target_probs = F.softmax(mask_logits.float(), dim=-1)
                row_probs.append(
                    target_probs if return_on_device else target_probs.cpu()
                )
            probs_per_block_batch.append(row_probs)
        _dbg_toc("05_softmax_and_return", t0)
        return probs_per_block_batch
    finally:
        if hook is not None:
            hook.remove()
        model.eval()
        model.config.block_size = orig_bs
