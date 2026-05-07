"""v1.5 batched draft patch  same intra_md extra-forward semantics as
``draft_probs_ablation_v1_5`` but for the batched ``draft_one_block_batch``
entry. Used by ``bench_v1_5_batch.py`` and any caller that does batched spec
decoding.

The extra forward runs once with shape (B, seq_len). Attention mask is
``(B, 1, seq_len, seq_len)`` because the intra_md sub-region differs per row
(driven by per-row step_map). The block-causal prefix region is the same for
all rows, so we broadcast that and only fill in the per-row intra_md block at
``[ctx:ctx+bl, ctx:ctx+bl]``.

Per-call CUDA-event timings (denoising / extra_forward / softmax_cpu) are
appended to ``_V15_BATCH_TIMINGS`` and flushed by the bench wrapper.

CUDA-graph optimizations (2026-04-25):
  1. Pass ``skip_per_pos_probs=True`` to the underlying ``draft_one_block_batch``
      its accumulated per_pos_probs is THROWN AWAY by v1.5 (we overwrite it
     with the extra-forward softmax). Saves a full ``.cpu()`` of a
     ``(B, bl, V)`` float32 tensor per draft call.
  2. When ``use_cuda_graph=True``, capture the **whole extra-forward path**
     into a CUDA graph: the step_causal mask construction, the slice-write
     into the static attention buffer, AND the model forward. Step_map is fed
     through a static GPU buffer (``copy_`` per call). The block-causal base
     part of the static mask is set once at capture time and never touched
     during replay.

Usage (via bench_v1_5_batch.py):
  python speculative_decoding/Experiment_Backend/bench_v1_5_batch.py \\
    --compare both --runtime hf --batch 8 --K 1 \\
    --use_cuda_graph --target_eval_sdpa --no_eos_stop ...
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
from speculative_decoding.draft import (  # noqa: E402
    _build_block_causal_attn,
    patch_sdpa_eval_attention,
)

MASK_TOKEN_ID = 151669

_orig_draft_one_block_batch = _draft_mod.draft_one_block_batch

# Per-call timing list (batched  one entry per draft_one_block_batch call,
# i.e. one entry per spec-iter draft step regardless of B).
_V15_BATCH_TIMINGS: List[dict] = []

# Cache: key -> (g, static_ids, static_step_map, static_attn_mask, static_logits, block_start, block_length)
# Key format: (id(model), B, seq_len, block_length, str(device), "v15_extra_batch")
_V15_BATCH_GRAPH_CACHE: dict = {}


def _capture_v15_extra_graph_batch(model, B, seq_len, block_length, device,
                                   mask_token_id=MASK_TOKEN_ID):
    """Capture a CUDA graph for the v1.5 batched extra forward.

    Captures: step_causal mask computation from static_step_map  slice-write
    into static_attn_mask intra-block region  model forward  logits read.

    During replay the caller must:
      static_ids.copy_(seq_tensor)             # (B, seq_len) int64
      static_step_map.copy_(step_map_tensor)   # (B, block_length) int64
      g.replay()
      logits = static_logits[:, ctx_len:ctx_len+block_length, :]

    The block-causal prefix region of static_attn_mask is set ONCE here and
    never touched during replay (it's stable for given seq_len, block_length).
    Only the (B, 1, ctx:ctx+bl, ctx:ctx+bl) intra-block sub-region gets
    rewritten each replay from the current static_step_map.
    """
    patch_sdpa_eval_attention(model)
    ctx_len = seq_len - block_length  # rows have uniform ctx_len here
    block_start = ctx_len

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)

        # Static buffers
        static_ids = torch.full((B, seq_len), mask_token_id,
                                dtype=torch.long, device=device)
        static_step_map = torch.zeros((B, block_length),
                                      dtype=torch.long, device=device)
        # Pre-built block-causal mask, expanded to (B, 1, L, L). The "outside"
        # of the intra-block sub-region stays at this value forever.
        base = _build_block_causal_attn(seq_len, block_length, device)  # (1, L, L)
        static_attn_mask = (
            base.unsqueeze(0).expand(B, 1, seq_len, seq_len).contiguous()
        )

        # Warmup on a side stream  exercise mask-write + forward with current
        # buffer values. This also primes any lazy autotune.
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                sm = static_step_map
                step_causal = (sm.unsqueeze(1) < sm.unsqueeze(2)).to(static_attn_mask.dtype)
                static_attn_mask[
                    :, 0,
                    block_start:block_start + block_length,
                    block_start:block_start + block_length,
                ] = step_causal
                _ = model(input_ids=static_ids,
                          attention_mask=static_attn_mask,
                          use_cache=False, return_dict=True)
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s,
                              capture_error_mode="thread_local"), \
                torch.no_grad():
            sm = static_step_map
            step_causal = (sm.unsqueeze(1) < sm.unsqueeze(2)).to(static_attn_mask.dtype)
            static_attn_mask[
                :, 0,
                block_start:block_start + block_length,
                block_start:block_start + block_length,
            ] = step_causal
            out = model(input_ids=static_ids,
                        attention_mask=static_attn_mask,
                        use_cache=False, return_dict=True)
        static_logits = out.logits

    return (g, static_ids, static_step_map, static_attn_mask, static_logits,
            block_start, block_length)


def _get_v15_extra_graph_batch(model, B, seq_len, block_length, device,
                               mask_token_id=MASK_TOKEN_ID):
    key = (id(model), int(B), int(seq_len), int(block_length),
           str(device), "v15_extra_batch")
    entry = _V15_BATCH_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_v15_extra_graph_batch(
            model, B, seq_len, block_length, device, mask_token_id,
        )
        _V15_BATCH_GRAPH_CACHE[key] = entry
    return entry


def _draft_one_block_batch_v1_5(
    model,
    context_ids_batch,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    use_cuda_graph: bool = False,
    sampling: str = "argmax",
) -> Tuple[List[List[int]], torch.Tensor, List[List[int]]]:
    """draft_one_block_batch + intra_md extra forward (batched).

    Returns same shapes as the original ``draft_one_block_batch``:
      draft_ids_batch:        list[list[int]],  shape (B, block_length)
      per_pos_probs_batch:    Tensor (B, block_length, vocab_size) on CPU
      step_map_batch:         list[list[int]],  shape (B, block_length)

    The returned ``per_pos_probs_batch`` is the softmax of the v1.5 extra
    forward (intra_md mask), not the per-step accumulated probs from the
    denoising loop  same semantics as the single-prompt v1.5 patch.
    """
    is_cuda = (device.type == "cuda")
    if is_cuda:
        evt_t0 = torch.cuda.Event(enable_timing=True)
        evt_t1 = torch.cuda.Event(enable_timing=True)
        evt_t2 = torch.cuda.Event(enable_timing=True)
        evt_t3 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            evt_t0.record()

    # ---- denoising: skip per_pos_probs accumulation since v1.5 overwrites it ----
    draft_ids_batch, _stale_probs, step_map_batch = _orig_draft_one_block_batch(
        model, context_ids_batch, block_length, denoising_steps,
        device, mask_token_id, use_cuda_graph=use_cuda_graph,
        skip_per_pos_probs=True,
        sampling=sampling,
    )

    if is_cuda:
        with torch.cuda.device(device):
            evt_t1.record()

    B = len(context_ids_batch)
    ctx_len = len(context_ids_batch[0])
    seq_len = ctx_len + block_length
    block_start = ctx_len

    # Build per-row inputs ONCE (small).
    seqs = [list(context_ids_batch[i]) + list(draft_ids_batch[i]) for i in range(B)]
    input_ids_t = torch.tensor(seqs, dtype=torch.long, device=device)
    sm_tensor = torch.tensor(step_map_batch, dtype=torch.long, device=device)  # (B, bl)

    if use_cuda_graph and is_cuda:
        # ---- graph path ----
        (g, static_ids, static_step_map, static_attn_mask, static_logits,
         _block_start, _block_length) = _get_v15_extra_graph_batch(
            model, B, seq_len, block_length, device, mask_token_id,
        )
        static_ids.copy_(input_ids_t)
        static_step_map.copy_(sm_tensor)
        with torch.cuda.device(device), torch.no_grad():
            g.replay()
            logits = static_logits[:, ctx_len:ctx_len + block_length, :]  # (B, bl, V)
    else:
        # ---- eager path ----
        # SDARAttention's SDPA-eval forward expects attn_mask broadcastable
        # to (B, H, L, L)  use shape (B, 1, L, L).
        base = _build_block_causal_attn(seq_len, block_length, device)  # (1, L, L)
        attn_mask = base.unsqueeze(0).expand(B, 1, seq_len, seq_len).clone()  # (B,1,L,L)
        step_causal = (sm_tensor.unsqueeze(1) < sm_tensor.unsqueeze(2)).to(attn_mask.dtype)
        attn_mask[:, 0,
                  block_start:block_start + block_length,
                  block_start:block_start + block_length] = step_causal

        with torch.cuda.device(device), torch.no_grad():
            outputs = model(input_ids=input_ids_t, attention_mask=attn_mask)
            logits = outputs.logits[:, ctx_len:ctx_len + block_length, :]

    if is_cuda:
        with torch.cuda.device(device):
            evt_t2.record()

    per_pos_probs_batch = F.softmax(logits.float(), dim=-1).cpu()
    assert per_pos_probs_batch.shape[0] == B
    assert per_pos_probs_batch.shape[1] == block_length

    if is_cuda:
        with torch.cuda.device(device):
            evt_t3.record()
            evt_t3.synchronize()
        _V15_BATCH_TIMINGS.append({
            "denoising_ms": evt_t0.elapsed_time(evt_t1),
            "extra_forward_ms": evt_t1.elapsed_time(evt_t2),
            "softmax_cpu_ms": evt_t2.elapsed_time(evt_t3),
            "B": B,
            "ctx_len": ctx_len,
            "block_length": block_length,
            "seq_len": seq_len,
            "graph": bool(use_cuda_graph and is_cuda),
        })

    return draft_ids_batch, per_pos_probs_batch, step_map_batch


def get_timings() -> List[dict]:
    out = list(_V15_BATCH_TIMINGS)
    _V15_BATCH_TIMINGS.clear()
    return out


_draft_mod.draft_one_block_batch = _draft_one_block_batch_v1_5
_sd_mod.draft_one_block_batch = _draft_one_block_batch_v1_5
print("[ablation v1.5_batch] patched draft.draft_one_block_batch  batched "
      "intra_md extra forward (graph-captured under use_cuda_graph=True; "
      "skip_per_pos_probs=True passed to underlying denoising); CUDA-event "
      "timings recorded into _V15_BATCH_TIMINGS.",
      flush=True)
