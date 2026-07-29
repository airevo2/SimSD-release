"""Draft model: standard multi-step block diffusion generation."""

import torch
import torch.nn.functional as F

from speculative_decoding import adapters
from speculative_decoding.adapters.sdar import MASK_TOKEN_ID as SDAR_MASK_TOKEN_ID

#: Legacy default for the no-KV-cache path, which predates multi-family support
#: and is only ever run on SDAR. Every cache-aware entry point takes the id from
#: ``cfg.mask_token_id`` instead — see ``adapters.get(family).mask_token_id``.
MASK_TOKEN_ID = SDAR_MASK_TOKEN_ID


_SDPA_PATCHED_CLASSES: set = set()


def patch_sdpa_eval_attention(model) -> bool:
    """Permanently replace the family's eval-mode attention forward with an
    SDPA-only variant. Idempotent: patching the same class twice is a no-op.

    Why: SDAR's upstream eval path (``modeling_sdar.SDARAttention.forward``)
    dispatches between flash_attn and SDPA via ``if torch.all(attention_mask)``.
    That call triggers a GPUCPU sync, which (a) forbids ``torch.cuda.graph``
    capture, and (b) is dead logic for our use  draft's block-causal mask is
    never all-ones, so the SDPA branch is the only one that ever fires.

    Locking both draft and target (under ``--target_eval_sdpa``) to the same
    SDPA kernel path also keeps their latency profiles directly comparable.

    Returns True if the class was patched this call, False if already patched
    or if no adapter recognises the model. Training-mode path
    (fused_flex_attention) is untouched where the family has one.
    """
    adapter = adapters.detect(model)
    if adapter is None:
        return False
    attn_cls = adapter.find_attn_class(model)

    from speculative_decoding.cache_aware import _install_attn_forward
    first = _install_attn_forward(
        attn_cls, "eval", lambda: adapters.make_eval_forward(adapter, attn_cls))
    _SDPA_PATCHED_CLASSES.add(attn_cls)
    adapters.rebind_hooked_forwards(model, attn_cls)
    return first


def _build_block_causal_attn(seq_len, block_length, device):
    """Build block-level causal attention mask (same as generate.py).

    Returns shape (1, seq_len, seq_len); SDPA broadcasts across batch dim,
    so this single mask works for any batch size.
    """
    num_blocks = (seq_len + block_length - 1) // block_length
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device))
    full_mask = block_mask.repeat_interleave(block_length, dim=0) \
                          .repeat_interleave(block_length, dim=1)
    return full_mask[:seq_len, :seq_len].unsqueeze(0)  # (1, seq_len, seq_len)


def _compute_unmask_schedule(block_length, denoising_steps):
    """Pre-compute the per-step unmask count for the block diffusion schedule.

    Returns a list of length `denoising_steps` giving how many positions get
    unmasked at each step. The formula matches the original data-dependent
    logic: at step s (1-indexed), remaining = denoising_steps - s + 1 and
    n_unmask = ceil(num_masked / remaining). Since all batch rows start with
    the same num_masked = block_length and the schedule is identical, we can
    compute it once in Python and avoid data-dependent .item() / nonzero ops
    (which would forbid CUDA-graph capture at batch > 1).
    """
    schedule = []
    remaining_masked = block_length
    for step in range(1, denoising_steps + 1):
        remaining_steps = denoising_steps - step + 1
        # ceil(remaining_masked / remaining_steps)
        n_unmask = min(-(-remaining_masked // remaining_steps), remaining_masked)
        schedule.append(n_unmask)
        remaining_masked -= n_unmask
    # any leftover from rounding gets appended to the last step
    if remaining_masked > 0:
        schedule[-1] += remaining_masked
    return schedule


# Key: (id(model), batch, seq_len, block_length, str(device)).
# Value: (CUDAGraph, static_ids, static_mask, static_logits).
# Shapes are stable across the N denoising forwards within one draft_one_block
# call (same ctx_len + block_length), so a single capture is replayed N times.
# Across calls ctx_len grows by block_length, so we cache one graph per unique
# (batch, seq_len) pair. Single-prompt API omits batch=1 from the legacy key
# format to preserve cache hits from before the batched refactor.
_DRAFT_GRAPH_CACHE: dict = {}


def _capture_draft_forward_graph(model, seq_len, block_length, device,
                                 mask_token_id=MASK_TOKEN_ID):
    """Capture a CUDA graph for a single fixed-shape forward pass.

    Preconditions / assumptions:
      - SDAR diffusion decode uses no KV cache, so ``use_cache=False`` keeps
        the output structure static across replays.
      - ``input_ids`` shape = (1, seq_len), ``attention_mask`` shape =
        (1, seq_len, seq_len); both fixed for this cache entry.
      - Caller is responsible for writing inputs into ``static_ids`` before
        each ``g.replay()`` (either by aliasing ``static_ids`` as the working
        buffer or via ``.copy_``). Reading ``static_logits`` after replay.
      - Forces ``patch_sdpa_eval_attention(model)`` (idempotent) so the
        eval-mode attention path is graph-safe and matches the kernel verify
        already uses under ``--target_eval_sdpa``.
    """
    patch_sdpa_eval_attention(model)
    with torch.cuda.device(device):
        # Flush any in-flight work left over from prior captures / backends on
        # this device (e.g. the native role captured a graph on another GPU
        # first, which lazily initialized ``torch.cuda.graph.default_capture_stream``
        # on *that* device; we now need a device-local capture stream).
        torch.cuda.synchronize(device)

        static_ids = torch.full((1, seq_len), mask_token_id,
                                dtype=torch.long, device=device)
        static_mask = _build_block_causal_attn(seq_len, block_length, device)

        # Warmup on a side stream so any lazy allocations / autotune happen
        # OUTSIDE the capture region. Without this, the first replay can hit
        # uninitialized workspace and crash.
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                _ = model(input_ids=static_ids, attention_mask=static_mask,
                          use_cache=False, return_dict=True)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        # Capture on the SAME stream we warmed up on, AND explicitly on this
        # device. torch.cuda.graph(...) with stream=None falls back to a
        # class-level ``default_capture_stream`` that is created on the device
        # active at FIRST use  so a prior capture on another GPU leaves the
        # default pinned to that GPU, producing cross-device "operation not
        # permitted when stream is capturing" errors here. capture_error_mode=
        # thread_local narrows failure attribution away from other threads
        # (e.g. gpu_grabber polling on the main stream).
        with torch.cuda.graph(g, stream=s,
                              capture_error_mode="thread_local"), \
                torch.no_grad():
            out = model(input_ids=static_ids, attention_mask=static_mask,
                        use_cache=False, return_dict=True)
        static_logits = out.logits  # persistent output buffer owned by the graph

    return g, static_ids, static_mask, static_logits


def _get_draft_forward_graph(model, seq_len, block_length, device,
                             mask_token_id=MASK_TOKEN_ID):
    """Return (possibly capture) the cached graph for this shape."""
    key = (id(model), int(seq_len), int(block_length), str(device))
    entry = _DRAFT_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_draft_forward_graph(
            model, seq_len, block_length, device, mask_token_id)
        _DRAFT_GRAPH_CACHE[key] = entry
    return entry


def _capture_draft_forward_graph_batch(model, batch, seq_len, block_length,
                                       device, mask_token_id=MASK_TOKEN_ID):
    """Batched variant: captures a graph with static_ids shape (batch, seq_len).

    The block-causal attention mask stays (1, seq_len, seq_len)  SDPA
    broadcasts it across the batch dim when building attention weights of
    shape (B, H, L, L), so we don't materialize a per-row mask.
    """
    patch_sdpa_eval_attention(model)
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)

        static_ids = torch.full((batch, seq_len), mask_token_id,
                                dtype=torch.long, device=device)
        static_mask = _build_block_causal_attn(seq_len, block_length, device)

        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                _ = model(input_ids=static_ids, attention_mask=static_mask,
                          use_cache=False, return_dict=True)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s,
                              capture_error_mode="thread_local"), \
                torch.no_grad():
            out = model(input_ids=static_ids, attention_mask=static_mask,
                        use_cache=False, return_dict=True)
        static_logits = out.logits

    return g, static_ids, static_mask, static_logits


def _get_draft_forward_graph_batch(model, batch, seq_len, block_length, device,
                                   mask_token_id=MASK_TOKEN_ID):
    """Return (possibly capture) the cached graph for this (batch, seq_len) shape."""
    key = (id(model), int(batch), int(seq_len), int(block_length), str(device))
    entry = _DRAFT_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_draft_forward_graph_batch(
            model, batch, seq_len, block_length, device, mask_token_id)
        _DRAFT_GRAPH_CACHE[key] = entry
    return entry


def clear_draft_graph_cache():
    """Release all captured draft graphs. Call when model is reloaded / moved."""
    _DRAFT_GRAPH_CACHE.clear()


def draft_one_block(model, context_ids, block_length, denoising_steps, device,
                    mask_token_id=MASK_TOKEN_ID, step_callback=None,
                    use_cuda_graph=False, use_intra_md_last_step=False,
                    sampling: str = "argmax",
                    early_stop_steps: int = -1):
    """
    Generate one block via standard multi-step block diffusion.

    Args:
        model: draft SDAR model (already on device, eval mode)
        context_ids: list[int], prompt + previously accepted tokens
        block_length: int, tokens per block
        denoising_steps: int, number of denoising iterations
        device: torch.device
        mask_token_id: int
        step_callback: optional callable(step_records) called at end with
            step_records = [(step_idx, block_ids, unmask_pos_list, step_map), ...]
        use_intra_md_last_step: if True, the LAST denoising step runs with an
            intra_md step-causal mask over the own-block region (position i
            attends to j iff step_j < step_i; still-masked positions are
            pre-assigned step=denoising_steps-1). Probs from this last forward
            are returned as ``draft_probs`` for ALL block positions, replacing
            the per-step-unmask saves. This fuses v1.5's "extra forward" into
            the last denoising step, cutting draft cost from N+1 to N forwards
            per block. Incompatible with ``use_cuda_graph`` on the last step:
            falls back to eager for that step.

    Returns:
        draft_ids: list[int], length = block_length
        draft_probs: Tensor (block_length, vocab_size)
        step_map: list[int], 0-indexed step when each position was unmasked
    """
    vocab_size = model.config.vocab_size
    input_list = list(context_ids) + [mask_token_id] * block_length
    seq_len = len(input_list)

    # Early-stop: cap how many denoising-step iterations the draft runs.
    # ``early_stop_steps <= 0`` or ``>= denoising_steps``  behave identically
    # to the legacy schedule (run the full ``denoising_steps`` iterations and
    # fill any leftover MASK in the trailing block). When ``0 < early_stop_steps
    # < denoising_steps``, we run only that many iterations, skip the
    # leftover-fill block, and emit ``step_map[i] = -1`` for any position that
    # remained MASK so the caller (verify + MRS) can route those positions
    # through ``target_argmax`` instead of through draft.
    _early_stop_active = (
        isinstance(early_stop_steps, int)
        and 0 < early_stop_steps < denoising_steps
    )
    _effective_steps = early_stop_steps if _early_stop_active else denoising_steps

    graph_entry = None
    if use_cuda_graph and device.type == "cuda":
        # Warning: graph capture requires a real forward, which costs ~4 extra
        # forwards on the first hit for this seq_len. Subsequent replays are
        # cheap. Cache is keyed by (model, seq_len, block_length, device).
        graph_entry = _get_draft_forward_graph(
            model, seq_len, block_length, device, mask_token_id)
        g, static_ids, static_mask, static_logits = graph_entry
        # Write current input into the persistent buffer the graph was
        # captured against. All subsequent in-place writes target this same
        # buffer, so the next g.replay() sees the updated tokens.
        seed = torch.tensor([input_list], dtype=torch.long, device=device)
        static_ids.copy_(seed)
        input_ids = static_ids
        attention_mask = static_mask
    else:
        input_ids = torch.tensor([input_list], dtype=torch.long, device=device)
        attention_mask = _build_block_causal_attn(seq_len, block_length, device)

    ctx_len = len(context_ids)
    step_map = [0] * block_length
    per_pos_probs = torch.zeros(block_length, vocab_size)
    step_records = []

    for step in range(1, _effective_steps + 1):
        resp = input_ids[0, ctx_len:]
        is_masked = resp == mask_token_id
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        use_fused_mask_this_step = (
            use_intra_md_last_step
            and step == denoising_steps
            and not _early_stop_active
        )

        # Pin current CUDA device so Liger's Triton SwiGLU kernel launches on
        # the right stream when draft and target live on different GPUs.
        with torch.cuda.device(device), torch.no_grad():
            if use_fused_mask_this_step:
                # intra_md step-causal mask over the own-block region.
                # Still-masked positions get step = denoising_steps - 1
                # (they will be unmasked by THIS forward).
                sm_cur = list(step_map)
                for p_idx in torch.nonzero(is_masked, as_tuple=True)[0].tolist():
                    sm_cur[p_idx] = step - 1
                attn_m = _build_block_causal_attn(
                    seq_len, block_length, device).clone()
                sm_t = torch.as_tensor(sm_cur, dtype=torch.long, device=device)
                step_causal = (sm_t[None, :] < sm_t[:, None]).to(attn_m.dtype)
                attn_m[0, ctx_len:ctx_len + block_length,
                       ctx_len:ctx_len + block_length] = step_causal
                outputs = model(input_ids=input_ids, attention_mask=attn_m)
                logits = outputs.logits[0, ctx_len:]
            elif graph_entry is not None:
                g.replay()
                logits = static_logits[0, ctx_len:]
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, ctx_len:]

        probs = F.softmax(logits.float(), dim=-1)
        if sampling == "multinomial":
            # Sample once per position from the per-position softmax. MRS's
            # output==target invariant requires draft_id ~ p, not argmax(p).
            # Confidence still uses the *sampled* token's probability, which
            # acts as an unbiased estimator of token-level "draft conviction"
            # for the unmask-ordering heuristic.
            pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            pred_ids = torch.argmax(probs, dim=-1)
        confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

        if use_fused_mask_this_step:
            # Replace per_pos_probs for ALL block positions with this step's
            # softmax. Matches the v1.5 extra-forward contract.
            per_pos_probs = probs.detach().cpu()

        remaining = denoising_steps - step + 1
        n_unmask = min(-(-num_masked // remaining), num_masked)

        masked_idx = torch.nonzero(is_masked, as_tuple=True)[0]
        _, topk_idx = confidence[masked_idx].topk(n_unmask)
        unmask_pos = masked_idx[topk_idx]

        if step_callback is not None:
            block_before = input_ids[0, ctx_len:].tolist()
            step_records.append((step - 1, block_before, [p.item() for p in unmask_pos], list(step_map)))

        for pos in unmask_pos:
            input_ids[0, ctx_len + pos] = pred_ids[pos]
            step_map[pos.item()] = step - 1
            per_pos_probs[pos.item()] = probs[pos].cpu()

    # Ensure exactly block_length positions are filled (no MASK left).
    # Skipped under early-stop: leftover MASK positions are intentional and
    # will be filled by target argmax during verify; mark them with
    # step_map[i] = -1 so the caller can route them.
    resp = input_ids[0, ctx_len:]
    still_masked = resp == mask_token_id
    if still_masked.any() and not _early_stop_active:
        if step_callback is not None:
            block_before = input_ids[0, ctx_len:].tolist()
            pos_list = torch.nonzero(still_masked, as_tuple=True)[0]
            step_records.append((denoising_steps - 1, block_before, [p.item() for p in pos_list], list(step_map)))
        # Pin current CUDA device so Liger's Triton SwiGLU kernel launches on
        # the right stream when draft and target live on different GPUs.
        with torch.cuda.device(device), torch.no_grad():
            if graph_entry is not None:
                g.replay()
                logits = static_logits[0, ctx_len:]
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, ctx_len:]
        probs = F.softmax(logits.float(), dim=-1)
        if sampling == "multinomial":
            pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            pred_ids = torch.argmax(probs, dim=-1)
        for pos in torch.nonzero(still_masked, as_tuple=True)[0]:
            p = pos.item()
            input_ids[0, ctx_len + p] = pred_ids[p]
            step_map[p] = denoising_steps - 1
            per_pos_probs[p] = probs[p].cpu()
    elif still_masked.any() and _early_stop_active:
        # Mark unrevealed positions: they remain mask_token_id in draft_ids
        # and get sentinel step_map = -1 (caller routes them via target argmax).
        for pos in torch.nonzero(still_masked, as_tuple=True)[0]:
            step_map[pos.item()] = -1

    if step_callback is not None and step_records:
        step_callback(step_records)

    draft_ids = input_ids[0, ctx_len:].tolist()
    return draft_ids, per_pos_probs, step_map


def draft_one_block_batch(model, context_ids_batch, block_length, denoising_steps,
                          device, mask_token_id=MASK_TOKEN_ID,
                          use_cuda_graph=False,
                          skip_per_pos_probs=False,
                          sampling: str = "argmax"):
    """Batched variant of draft_one_block. Rows in context_ids_batch must share
    the same length (caller pads / chunks). No step_callback support.

    Args:
        skip_per_pos_probs: if True, do not accumulate or transfer the
            ``(B, block_length, vocab_size)`` probability tensor  returns a
            zero-filled placeholder. Use when the caller overwrites
            ``per_pos_probs`` itself (v1.5 ablations: an extra intra_md
            forward replaces these probs anyway). Avoids the per-step
            ``.cpu()`` of a 38 MB / bs=16 tensor that otherwise dominates
            graph-mode wall-clock.

    Returns:
        draft_ids_batch: list[list[int]], shape (B, block_length)
        per_pos_probs_batch: Tensor (B, block_length, vocab_size) on CPU
        step_map_batch: list[list[int]], shape (B, block_length)

    Implementation note (graph-mode optimization, 2026-04-25):
        Earlier versions did one ``probs.cpu()`` and one ``topk_pos.cpu()``
        per denoising step plus a Python (B × n_unmask) loop. On bs=16,
        ``probs`` is ``(16, 4, 152000)`` float32 = 38 MB  4 H2D syncs per
        step × 4 steps + Python loop overhead = ~150 ms / call, dominating
        spec_ms. We now keep both ``step_map_batch`` and
        ``per_pos_probs_batch`` on GPU via ``scatter_`` + advanced indexing,
        and do a single ``.cpu()`` at the very end.
    """
    assert len(context_ids_batch) > 0
    B = len(context_ids_batch)
    ctx_len = len(context_ids_batch[0])
    for row in context_ids_batch:
        assert len(row) == ctx_len, "context_ids_batch rows must share length"
    seq_len = ctx_len + block_length
    vocab_size = model.config.vocab_size

    # Deterministic per-step unmask counts  same for all rows, so we can avoid
    # data-dependent .item() / nonzero ops that forbid CUDA-graph replay.
    schedule = _compute_unmask_schedule(block_length, denoising_steps)

    graph_entry = None
    if use_cuda_graph and device.type == "cuda":
        graph_entry = _get_draft_forward_graph_batch(
            model, B, seq_len, block_length, device, mask_token_id)
        g, static_ids, static_mask, static_logits = graph_entry
        ctx_tensor = torch.tensor(context_ids_batch, dtype=torch.long, device=device)
        mask_cols = torch.full((B, block_length), mask_token_id,
                               dtype=torch.long, device=device)
        seed = torch.cat([ctx_tensor, mask_cols], dim=1)  # (B, seq_len)
        static_ids.copy_(seed)
        input_ids = static_ids
        attention_mask = static_mask
    else:
        init = [list(row) + [mask_token_id] * block_length for row in context_ids_batch]
        input_ids = torch.tensor(init, dtype=torch.long, device=device)
        attention_mask = _build_block_causal_attn(seq_len, block_length, device)

    # GPU-side accumulators  do one .cpu() at end instead of per-step.
    step_map_t = torch.zeros(B, block_length, dtype=torch.long, device=device)
    if not skip_per_pos_probs:
        per_pos_probs_gpu = torch.zeros(
            B, block_length, vocab_size, dtype=torch.float32, device=device,
        )

    neg_inf = float("-inf")
    batch_idx_full = torch.arange(B, device=device)  # (B,)

    for step_idx, n_unmask in enumerate(schedule):
        if n_unmask == 0:
            break

        with torch.cuda.device(device), torch.no_grad():
            if graph_entry is not None:
                g.replay()
                logits = static_logits[:, ctx_len:, :]  # (B, block_length, V)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, ctx_len:, :]

        probs = F.softmax(logits.float(), dim=-1)              # (B, bl, V)
        if sampling == "multinomial":
            # multinomial wants a 2-D tensor; flatten (B, bl)  (B*bl, V).
            B_, bl_, V_ = probs.shape
            pred_ids = torch.multinomial(
                probs.reshape(B_ * bl_, V_), num_samples=1
            ).view(B_, bl_)
        else:
            pred_ids = torch.argmax(probs, dim=-1)             # (B, bl)
        confidence = probs.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)  # (B, bl)

        resp = input_ids[:, ctx_len:]                          # (B, bl)
        is_masked = resp == mask_token_id                      # (B, bl)
        masked_conf = torch.where(is_masked, confidence,
                                  torch.full_like(confidence, neg_inf))
        _, topk_pos = masked_conf.topk(n_unmask, dim=-1)       # (B, n_unmask)

        batch_idx = batch_idx_full.unsqueeze(-1).expand(-1, n_unmask)
        abs_pos = ctx_len + topk_pos
        new_tokens = pred_ids.gather(1, topk_pos)              # (B, n_unmask)
        input_ids[batch_idx, abs_pos] = new_tokens

        # GPU-side step_map update (scatter, no .cpu()).
        step_map_t.scatter_(
            1, topk_pos,
            torch.full_like(topk_pos, step_idx, dtype=step_map_t.dtype),
        )

        # GPU-side per_pos_probs update via advanced indexing (slice along V).
        if not skip_per_pos_probs:
            per_pos_probs_gpu[batch_idx, topk_pos] = probs[batch_idx, topk_pos]

    # Fallback pass for any residual MASKs (should not happen with a correct
    # schedule, but mirrors the single-prompt path). The .any() here forces a
    # single sync  acceptable since this only runs at end-of-loop.
    resp = input_ids[:, ctx_len:]
    still_masked = resp == mask_token_id
    if bool(still_masked.any()):
        with torch.cuda.device(device), torch.no_grad():
            if graph_entry is not None:
                g.replay()
                logits = static_logits[:, ctx_len:, :]
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, ctx_len:, :]
        probs = F.softmax(logits.float(), dim=-1)
        if sampling == "multinomial":
            B_, bl_, V_ = probs.shape
            pred_ids = torch.multinomial(
                probs.reshape(B_ * bl_, V_), num_samples=1
            ).view(B_, bl_)
        else:
            pred_ids = torch.argmax(probs, dim=-1)
        # Vectorized fill: where still_masked, write pred_ids; else keep existing.
        new_resp = torch.where(still_masked, pred_ids, input_ids[:, ctx_len:])
        input_ids[:, ctx_len:] = new_resp
        step_map_t = torch.where(
            still_masked,
            torch.full_like(step_map_t, denoising_steps - 1),
            step_map_t,
        )
        if not skip_per_pos_probs:
            mask_3d = still_masked.unsqueeze(-1)               # (B, bl, 1)
            per_pos_probs_gpu = torch.where(mask_3d, probs, per_pos_probs_gpu)

    # Single .cpu() at the very end  replaces N×B-row .cpu() per step.
    draft_ids_batch = input_ids[:, ctx_len:].cpu().tolist()
    step_map_batch = step_map_t.cpu().tolist()
    if skip_per_pos_probs:
        per_pos_probs_batch = torch.zeros(B, block_length, vocab_size)
    else:
        per_pos_probs_batch = per_pos_probs_gpu.cpu()
    return draft_ids_batch, per_pos_probs_batch, step_map_batch
