"""Draft-probs ablation v1.5_graph  v1.5 with the extra forward captured
into a CUDA graph.

Motivation (from T0 / T1 findings):
- v1.5's extra forward costs ~44 ms on 1.7B (K8). Draft is ~96% of spec time
  on single-GPU, so this forward is a real bottleneck.
- T2 fused (one-forward-less) crashed α (0.73  0.34) and pass@1 (−12.5 pp on
  gsm8k self_draft)  not salvageable.
- extra forward's *shape* is stable per (seq_len, block_length) key  only the
  token values and the intra-block step-causal sub-mask change per call. CUDA
  graph replay with static buffers + per-call ``copy_`` of inputs is ideal.

Cache key:  (id(model), seq_len, block_length, str(device), "v15_extra")
Buffers:    static_ids (1, seq_len), static_mask (1, seq_len, seq_len)
Per call:   static_ids.copy_(seq); static_mask.copy_(mask); g.replay();
            read softmax from static_logits[0, ctx_len:ctx_len+block_length]

Usage:
  python speculative_decoding/Experiment_Backend/bench_v1_5_graph.py \\
      --compare both --runtime hf \\
      --draft_model inference/model/SDAR-1_7B-Chat \\
      --target_model inference/model/SDAR-8B-Chat \\
      --K 8 --num_blocks 32 ...
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
    _build_block_causal_attn, patch_sdpa_eval_attention,
)

MASK_TOKEN_ID = 151669

_orig_draft_one_block = _draft_mod.draft_one_block

# key -> (g, static_ids, static_mask, static_logits)
_V15_EXTRA_GRAPH_CACHE: dict = {}

_V15_TIMINGS: List[dict] = []


def _capture_v15_extra_graph(model, seq_len, block_length, device,
                             mask_token_id=MASK_TOKEN_ID):
    """Capture a CUDA graph for the v1.5 extra forward.

    Static buffers are mutated by the caller before each replay:
      static_ids.copy_(seq_tensor)       # shape (1, seq_len)
      static_mask.copy_(attn_mask)       # shape (1, seq_len, seq_len)
      g.replay()
      logits = static_logits             # shape (1, seq_len, vocab_size)

    Uses the SDPA-eval path (idempotent patch), so the forward is graph-safe.
    """
    patch_sdpa_eval_attention(model)
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)

        static_ids = torch.full((1, seq_len), mask_token_id,
                                dtype=torch.long, device=device)
        static_mask = _build_block_causal_attn(seq_len, block_length, device).clone()

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


def _get_v15_extra_graph(model, seq_len, block_length, device,
                         mask_token_id=MASK_TOKEN_ID):
    key = (id(model), int(seq_len), int(block_length), str(device), "v15_extra")
    entry = _V15_EXTRA_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_v15_extra_graph(
            model, seq_len, block_length, device, mask_token_id)
        _V15_EXTRA_GRAPH_CACHE[key] = entry
    return entry


def _draft_one_block_v1_5_graph(
    model,
    context_ids,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    step_callback=None,
    use_cuda_graph: bool = False,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """v1.5 with graph-accelerated extra forward."""
    is_cuda = (device.type == "cuda")
    if is_cuda:
        evt_t0 = torch.cuda.Event(enable_timing=True)
        evt_t1 = torch.cuda.Event(enable_timing=True)
        evt_t2 = torch.cuda.Event(enable_timing=True)
        evt_t3 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            evt_t0.record()

    draft_ids, _stale_probs, step_map = _orig_draft_one_block(
        model, context_ids, block_length, denoising_steps,
        device, mask_token_id, step_callback=step_callback,
        use_cuda_graph=use_cuda_graph,
    )

    if is_cuda:
        with torch.cuda.device(device):
            evt_t1.record()

    ctx_len = len(context_ids)
    seq = list(context_ids) + list(draft_ids)
    seq_len = len(seq)
    block_start = ctx_len

    if is_cuda:
        g, static_ids, static_mask, static_logits = _get_v15_extra_graph(
            model, seq_len, block_length, device, mask_token_id)
        seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
        static_ids.copy_(seq_tensor)

        # Build mask in-place on device: start from block-causal base, then
        # overwrite own-block region with intra_md step-causal.
        attn_base = _build_block_causal_attn(seq_len, block_length, device)
        sm = torch.as_tensor(step_map, dtype=torch.long, device=device)
        step_causal = (sm[None, :] < sm[:, None]).to(attn_base.dtype)
        # Write straight into the static buffer to avoid an extra allocation.
        static_mask.copy_(attn_base)
        static_mask[0, block_start:block_start + block_length,
                    block_start:block_start + block_length] = step_causal

        with torch.cuda.device(device), torch.no_grad():
            g.replay()
            logits = static_logits[0, ctx_len:ctx_len + block_length]
    else:
        # CPU / non-CUDA fallback: eager path.
        input_ids = torch.tensor([seq], dtype=torch.long, device=device)
        attn_mask = _build_block_causal_attn(seq_len, block_length, device).clone()
        sm = torch.as_tensor(step_map, dtype=torch.long, device=device)
        step_causal = (sm[None, :] < sm[:, None]).to(attn_mask.dtype)
        attn_mask[0, block_start:block_start + block_length,
                  block_start:block_start + block_length] = step_causal
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits[0, ctx_len:ctx_len + block_length]

    if is_cuda:
        with torch.cuda.device(device):
            evt_t2.record()

    per_pos_probs_final = F.softmax(logits.float(), dim=-1).cpu()

    if is_cuda:
        with torch.cuda.device(device):
            evt_t3.record()
            evt_t3.synchronize()
        _V15_TIMINGS.append({
            "denoising_ms": evt_t0.elapsed_time(evt_t1),
            "extra_forward_ms": evt_t1.elapsed_time(evt_t2),
            "softmax_cpu_ms": evt_t2.elapsed_time(evt_t3),
            "ctx_len": ctx_len,
            "block_length": block_length,
            "seq_len": seq_len,
        })

    return draft_ids, per_pos_probs_final, step_map


def get_timings() -> List[dict]:
    out = list(_V15_TIMINGS)
    _V15_TIMINGS.clear()
    return out


_draft_mod.draft_one_block = _draft_one_block_v1_5_graph
_sd_mod.draft_one_block = _draft_one_block_v1_5_graph
print("[ablation v1.5_graph] patched draft.draft_one_block  extra intra_md "
      "forward captured into CUDA graph (shared across calls of same seq_len).",
      flush=True)


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_v1_5_graph"])
    self_draft_compare.main()
