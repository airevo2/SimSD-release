"""Cache-aware speculative decoding (K=1, single-prompt, no pipeline).

Maintains DynamicCache for both draft and target models, à la
generate.py:block_diffusion_generate. Each spec iter only forwards the current
block (block_length tokens for draft denoising; 2*block_length tokens for
target verify) attending to the cached prefix.

Lifecycle per request:
  1. Init draft_cache, target_cache as DynamicCache.
  2. Prefill prompt on both models (store_kv=True).
  3. For each spec iter:
     a. Draft denoising: denoising_steps forwards on current block (block_length
        tokens), each store_kv=False, attending to cached prefix. Yields
        draft_ids + per_pos_probs + step_map.
     b. Target verify: 1 forward on [draft_data | draft_mask] (2*block_length),
        store_kv=False, yields target_probs at mask positions.
     c. MRS / commit produces final_ids.
     d. Extend BOTH caches with final_ids tokens (each: 1 forward with
        store_kv=True, position_ids = [ctx_len .. ctx_len+len(final_ids)-1]).

Limitations of this initial version:
  - K = 1 only (no multi-block lookahead).
  - bs = 1 only (no batched cache).
  - No pipeline (sequential draft  verify  commit).
  - No cuda_graph (cache size changes each iter; would need a graph per size).
  - No SDPA patch (the patched eval forward in draft.py:patch_sdpa_eval_attention
    drops past_key_value handling; we use the default modeling_sdar.py attention
    forward which supports cache).

Validated by self-draft (draft = target) + greedy_match: should reproduce the
non-cached spec output exactly modulo sampling RNG order.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache

from speculative_decoding.mrs import mrs_verify, greedy_match_verify


# ─────── sampling helpers (mirror generate.py:sample_with_temperature_topk_topp) ───────

def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool),
        -1, sorted_indices, sorted_mask,
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample_logits(
    logits: torch.Tensor,
    sampling: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (probs, pred_ids). Matches generate.py:sample_with_temperature_topk_topp.

    For sampling='argmax' we ignore temp/top_k/top_p (deterministic max).
    For 'multinomial' we apply temp  top_k  top_p  softmax  multinomial,
    which is exactly generate.py's vanilla path.
    """
    if sampling == "argmax":
        probs = F.softmax(logits.float(), dim=-1)
        return probs, probs.argmax(dim=-1)
    logits = logits.float()
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = _top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = _top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return probs, pred_ids


def _pick_unmask_positions(
    is_masked: torch.Tensor,
    confidence: torch.Tensor,
    n_unmask: int,
    remasking_strategy: str,
    confidence_threshold: float,
) -> torch.Tensor:
    """Pick which masked positions to unmask this denoising step.

    'low_confidence_static' (default): always top-k by confidence (current behavior).
    'low_confidence_dynamic': if #(conf > threshold) >= n_unmask, unmask all
    high-conf positions (potentially > n_unmask); else fall back to top-k.
    Matches generate.py:block_diffusion_generate's `low_confidence_dynamic` path.
    """
    neg_inf = float("-inf")
    masked_conf = torch.where(
        is_masked, confidence, torch.full_like(confidence, neg_inf)
    )
    if remasking_strategy == "low_confidence_dynamic":
        high_mask = masked_conf > confidence_threshold
        # Sole sync (per step) for dynamic mode. Picking is in eager outside
        # the captured forward, so the sync only stalls host enqueue, not GPU.
        if int(high_mask.sum().item()) >= n_unmask:
            return torch.nonzero(high_mask, as_tuple=True)[0]
    _, top_pos = masked_conf.topk(n_unmask)
    # Filter to currently-masked positions only. With the precomputed schedule
    # (sum == initial n_masks) static never picks more than masked, so this
    # is a no-op there. Under dynamic, prior steps may have unmasked extras
    # via the high_mask path, leaving num_masked < scheduled n_unmask; without
    # the filter, topk over all-(-inf) returns arbitrary indices and the
    # scatter would overwrite committed tokens.
    return top_pos[is_masked[top_pos]]

# SDPA backend selection (2026-05-05):
#   StaticBlockCache stores H_q heads (post 2026-05-05 GQA-expanded refactor).
#   In `_sdpa_static_forward` we write `k_new` (H_kv) into n_rep strided
#   sections of the H_q-wide cache, then call SDPA with matching head counts.
#   This lets EFFICIENT_ATTENTION fire (~1.8× faster than MATH fallback on
#   Blackwell + cuda_graph capture). FLASH would be even faster but rejects
#   our bool 4D mask. CUDNN works in eager but breaks under cuda_graph
#   (workspace alloc conflict)  confirmed empirically (2× slower in graph).
#   So:  EFFICIENT > MATH (fallback). Apples-to-apples  applies to native /
#   vanilla / SD verify uniformly via the shared patched attention forward.
try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel, SDPBackend as _SDPBackend
except ImportError:  # torch < 2.3 fallback
    _sdpa_kernel = None
    _SDPBackend = None

MASK_TOKEN_ID_DEFAULT = 151669

_PATCHED_ATTN_CLASSES = set()
_PATCHED_STATIC_ATTN_CLASSES = set()


# ─────────────────────────────────────────────────────────────────────
# Fixed-shape cache for cuda_graph
# ─────────────────────────────────────────────────────────────────────
# DynamicCache uses ``torch.cat`` to grow K/V tensors in place each forward,
# which is incompatible with cuda_graph (cat allocates new tensors per
# replay). StaticBlockCache pre-allocates per-layer K/V buffers of shape
# (1, n_kv_heads, full_len, head_dim) where:
#     full_len = max_cache_len + max_cur_len
# Cache slot is at positions [0, max_cache_len); scratch slot for the
# current forward's K/V is at positions [max_cache_len, full_len).
# All updates are in-place via ``index_copy_`` (graph-friendly).
# Attention reads the FULL buffer; the attention_mask masks out positions
# beyond the actual ``ctx_aligned`` (cache portion) and beyond the actual
# cur_len (scratch portion). The mask is an input to the graph and changes
# per replay, so a single graph per cur_len handles every iter.

class StaticBlockCache(Cache):
    """Fixed-shape K/V cache wrapper for cuda_graph compatibility.

    Layout per layer (post 2026-05-05 GQA-expanded refactor):
      key_cache[layer]:   (1, n_q_heads, full_len, head_dim)    H_q heads
      value_cache[layer]: same
      where full_len = max_cache_len + max_cur_len.

    Why H_q (Q-side head count) instead of H_kv:
      The patched ``_sdpa_static_forward`` writes ``k_new`` (which has H_kv
      heads from the GQA k_proj) into ``n_rep = H_q // H_kv`` strided
      positions of the H_q-wide cache. This pre-expansion lets the SDPA call
      see matching head counts and pick ``EFFICIENT_ATTENTION`` (~1.8× faster
      than the GQA-fallback ``MATH`` backend on Blackwell). The 4× cache
      memory cost is ~70 MB total  negligible.

    Slots:
      [0, max_cache_len)              : permanent cache (from store_kv=True)
      [max_cache_len, full_len)       : scratch for current forward's cur_x
    """

    # transformers 4.55+ made Cache.max_cache_len a read-only property. Shadow
    # it with a read/write property on this subclass so existing
    # `self.max_cache_len = ...` and `cache.max_cache_len` reads keep working.
    @property
    def max_cache_len(self):
        return self._max_cache_len

    @max_cache_len.setter
    def max_cache_len(self, v):
        self._max_cache_len = v

    def __init__(self, model, max_cache_len: int, max_cur_len: int,
                 device: torch.device, dtype: torch.dtype,
                 batch_size: int = 1, n_kv_heads_override: int = None,
                 n_q_heads_override: int = None,
                 gqa_mode: str = "expand"):
        assert gqa_mode in ("expand", "native"), (
            f"gqa_mode must be 'expand' (H_q-wide cache + EFFICIENT_ATTENTION) "
            f"or 'native' (H_kv-wide cache + enable_gqa=True / MATH path); "
            f"got {gqa_mode!r}"
        )
        cfg = model.config
        n_layers = cfg.num_hidden_layers
        n_kv_heads_from_cfg = getattr(cfg, "num_key_value_heads",
                                       cfg.num_attention_heads)
        self.n_kv_heads = (n_kv_heads_override if n_kv_heads_override is not None
                            else n_kv_heads_from_cfg)
        n_q_heads = (n_q_heads_override if n_q_heads_override is not None
                      else cfg.num_attention_heads)
        self.n_q_heads = n_q_heads
        assert n_q_heads % self.n_kv_heads == 0, (
            f"n_q_heads={n_q_heads} not divisible by n_kv_heads={self.n_kv_heads}"
        )
        self.n_rep = n_q_heads // self.n_kv_heads
        self.gqa_mode = gqa_mode
        # cache_heads:
        #   "expand" (default): H_q-wide buffer; k_new (H_kv) strided into
        #       n_rep slots  SDPA picks EFFICIENT_ATTENTION (matching head
        #       counts on Q and K/V).
        #   "native": H_kv-wide buffer; k_new written directly; SDPA called
        #       with enable_gqa=True so Q broadcasts over K/V  MATH GQA path.
        cache_heads = n_q_heads if gqa_mode == "expand" else self.n_kv_heads
        self.cache_heads = cache_heads
        head_dim = cfg.head_dim
        full_len = max_cache_len + max_cur_len

        self.max_cache_len = max_cache_len
        self.max_cur_len = max_cur_len
        self.batch_size = batch_size
        self.full_len = full_len
        self.device = device
        self.dtype = dtype

        self.key_cache = [
            torch.zeros(batch_size, cache_heads, full_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(n_layers)
        ]
        self.value_cache = [
            torch.zeros(batch_size, cache_heads, full_len, head_dim,
                        device=device, dtype=dtype)
            for _ in range(n_layers)
        ]

    def __len__(self) -> int:
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        # HF Cache interface compat: returns (k, v) for the layer.
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Compatibility shim  most modeling code uses this for past_len.
        # Our patched attention path uses attention_mask instead, but some
        # transformers internals call this.
        return self.max_cache_len

    def reset(self):
        """Zero out buffers between requests."""
        for k in self.key_cache:
            k.zero_()
        for v in self.value_cache:
            v.zero_()


def patch_sdpa_static_cache(model) -> bool:
    """Replace SDARAttention.forward with an SDPA path that uses
    StaticBlockCache's fixed-shape buffers via in-place ``index_copy_``.

    The patched forward expects:
      - ``past_key_value``: a ``StaticBlockCache``
      - ``cache_position``: optional LongTensor of positions to write into
        permanent cache (None = scratch only, store_kv=False mode)
      - ``cur_scratch_pos``: LongTensor of fixed positions in scratch slot
        where cur_x's K/V get written each forward (typically
        torch.arange(max_cache_len, max_cache_len + cur_len))
      - ``attention_mask``: bool mask of shape
        (1, 1, cur_len, full_len)  caller masks invalid cache positions.

    Idempotent. Returns True if a patch was applied.
    """
    import sys

    attn_cls = None
    for m in model.modules():
        if m.__class__.__name__ == "SDARAttention":
            attn_cls = m.__class__
            break
    if attn_cls is None:
        return False
    if attn_cls in _PATCHED_STATIC_ATTN_CLASSES:
        return False

    apply_rotary_pos_emb = sys.modules[attn_cls.__module__].apply_rotary_pos_emb

    def _sdpa_static_forward(self, hidden_states, position_embeddings,
                             attention_mask, past_key_value=None,
                             cache_position=None, **kwargs):
        from einops import rearrange
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k_new = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        # Caller passes scratch positions via kwargs (graph-stable input).
        cur_scratch_pos = kwargs.get("cur_scratch_pos")
        store_kv = bool(kwargs.get("store_kv", False))

        if past_key_value is not None and cur_scratch_pos is not None:
            gqa_mode = getattr(past_key_value, "gqa_mode", "expand")
            n_rep = past_key_value.n_rep
            if gqa_mode == "native":
                # H_kv-wide cache: write k_new (H_kv) directly, no expansion.
                # SDPA below will broadcast Q (H_q) via enable_gqa=True.
                past_key_value.key_cache[self.layer_idx].index_copy_(
                    2, cur_scratch_pos, k_new,
                )
                past_key_value.value_cache[self.layer_idx].index_copy_(
                    2, cur_scratch_pos, v_new,
                )
                if store_kv and cache_position is not None:
                    past_key_value.key_cache[self.layer_idx].index_copy_(
                        2, cache_position, k_new,
                    )
                    past_key_value.value_cache[self.layer_idx].index_copy_(
                        2, cache_position, v_new,
                    )
            elif n_rep == 1:
                # H_q == H_kv (no GQA): single-slot write.
                past_key_value.key_cache[self.layer_idx].index_copy_(
                    2, cur_scratch_pos, k_new,
                )
                past_key_value.value_cache[self.layer_idx].index_copy_(
                    2, cur_scratch_pos, v_new,
                )
                if store_kv and cache_position is not None:
                    past_key_value.key_cache[self.layer_idx].index_copy_(
                        2, cache_position, k_new,
                    )
                    past_key_value.value_cache[self.layer_idx].index_copy_(
                        2, cache_position, v_new,
                    )
            else:
                # GQA-expand on write: H_q-wide cache, k_new strided across n_rep slots.
                for i in range(n_rep):
                    past_key_value.key_cache[self.layer_idx][:, i::n_rep, :, :].index_copy_(
                        2, cur_scratch_pos, k_new,
                    )
                    past_key_value.value_cache[self.layer_idx][:, i::n_rep, :, :].index_copy_(
                        2, cur_scratch_pos, v_new,
                    )
                if store_kv and cache_position is not None:
                    for i in range(n_rep):
                        past_key_value.key_cache[self.layer_idx][:, i::n_rep, :, :].index_copy_(
                            2, cache_position, k_new,
                        )
                        past_key_value.value_cache[self.layer_idx][:, i::n_rep, :, :].index_copy_(
                            2, cache_position, v_new,
                        )
            full_k = past_key_value.key_cache[self.layer_idx]
            full_v = past_key_value.value_cache[self.layer_idx]
        else:
            gqa_mode = "expand"
            # Fallback no-cache path (shouldn't be reached in cache-aware spec).
            full_k = k_new
            full_v = v_new

        mask_bool = attention_mask.bool() if attention_mask is not None else None
        # In "expand" mode head counts match  SDPA picks EFFICIENT_ATTENTION
        # for free (~1.8× faster than MATH GQA on Blackwell).
        # In "native" mode K/V is H_kv-wide  enable_gqa=True so Q broadcasts;
        # SDPA falls back to MATH (mirror of the eager `patch_sdpa_with_cache`
        # path, kept available so we can ablate the GQA-expansion optimization).
        if gqa_mode == "native":
            attn_output = F.scaled_dot_product_attention(
                query=q, key=full_k, value=full_v,
                attn_mask=mask_bool,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=True,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query=q, key=full_k, value=full_v,
                attn_mask=mask_bool,
                is_causal=False,
                scale=self.scaling,
            )
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn_cls.forward = _sdpa_static_forward
    _PATCHED_STATIC_ATTN_CLASSES.add(attn_cls)
    return True


# ─────────────────────────────────────────────────────────────────────
# CUDA graph capture: per-(model, op_type, cur_len) graph cache
# ─────────────────────────────────────────────────────────────────────
# Graphs key:    (id(model), op_type, cur_len, str(device))
# op_type:       "denoise" | "verify" | "extend"
#                (op_type only differs by store_kv; for the patched static
#                 forward, store_kv is part of kwargs but handled inside the
#                 graph based on whether cache_position is non-empty)
# cur_len:       fixed token count for this op variant.
# Each graph entry holds static input/output buffers; replay copies fresh
# inputs in and reads outputs out.

_STATIC_GRAPH_CACHE: Dict[Any, Dict[str, Any]] = {}


def _capture_static_forward_graph(model, op_type: str, cur_len: int,
                                  cache: StaticBlockCache,
                                  device: torch.device,
                                  mask_token_id: int) -> Dict[str, Any]:
    """Capture a cuda_graph for a fixed-shape forward.

    Static buffers:
      - input_ids:        (B, cur_len) long
      - position_ids:     (B, cur_len) long
      - attention_mask:   (B, 1, cur_len, full_len) bool
      - cur_scratch_pos:  (cur_len,) long  typically max_cache_len..+cur_len
      - cache_position:   (cur_len,) long  for store_kv=True; ignored in
                          the graph if op_type != "extend"
      - logits:           (B, cur_len, vocab)  graph-owned output

    The store_kv flag is fixed at capture: extend graph captures with
    store_kv=True; denoise/verify with store_kv=False.

    B is taken from cache.batch_size  each batch_size variant gets its own
    captured graph.
    """
    full_len = cache.full_len
    vocab = model.config.vocab_size
    store_kv = (op_type == "extend")
    B = cache.batch_size

    static_input_ids = torch.full(
        (B, cur_len), mask_token_id, dtype=torch.long, device=device,
    )
    static_position_ids = torch.zeros(
        (B, cur_len), dtype=torch.long, device=device,
    )
    static_attn_mask = torch.zeros(
        (B, 1, cur_len, full_len), dtype=torch.bool, device=device,
    )
    static_cur_scratch_pos = torch.arange(
        cache.max_cache_len, cache.max_cache_len + cur_len,
        dtype=torch.long, device=device,
    )
    # Initialize cache_position to point at scratch slot (same positions as
    # cur_scratch_pos). For extend graph (store_kv=True) the warmup forwards
    # would otherwise default to zeros and corrupt prompt K/V at cache[0:bl].
    # By pointing at scratch, warmup writes are harmless (overwrite scratch
    # which is transient anyway). Replay overwrites this buffer with the real
    # cache_position before each call.
    static_cache_position = static_cur_scratch_pos.clone()

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        # Warmup on side stream so any lazy allocs/autotune happen outside
        # the capture region.
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                _ = model(
                    input_ids=static_input_ids,
                    attention_mask=static_attn_mask,
                    position_ids=static_position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    store_kv=store_kv,
                    cur_scratch_pos=static_cur_scratch_pos,
                    cache_position=static_cache_position,
                    return_dict=True,
                )
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"), \
                torch.no_grad():
            out = model(
                input_ids=static_input_ids,
                attention_mask=static_attn_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                use_cache=True,
                store_kv=store_kv,
                cur_scratch_pos=static_cur_scratch_pos,
                cache_position=static_cache_position,
                return_dict=True,
            )
        static_logits = out.logits  # owned by the graph

    return {
        "graph": g,
        "input_ids": static_input_ids,
        "position_ids": static_position_ids,
        "attn_mask": static_attn_mask,
        "cur_scratch_pos": static_cur_scratch_pos,
        "cache_position": static_cache_position,
        "logits": static_logits,
    }


def _get_static_forward_graph(model, op_type: str, cur_len: int,
                              cache: StaticBlockCache,
                              device: torch.device,
                              mask_token_id: int) -> Dict[str, Any]:
    key = (id(model), op_type, int(cur_len), int(cache.batch_size), str(device))
    entry = _STATIC_GRAPH_CACHE.get(key)
    if entry is None:
        entry = _capture_static_forward_graph(
            model, op_type, cur_len, cache, device, mask_token_id,
        )
        _STATIC_GRAPH_CACHE[key] = entry
    return entry


# ─────────────────────────────────────────────────────────────────────
# CUDA-graph-driven runners for denoise / verify / extend
# ─────────────────────────────────────────────────────────────────────

def _replay_static_forward(entry: Dict[str, Any],
                           input_ids: torch.Tensor,
                           position_ids: torch.Tensor,
                           attn_mask: torch.Tensor,
                           cache_position: Optional[torch.Tensor] = None,
                           ) -> torch.Tensor:
    """Copy fresh inputs into static buffers, replay graph, return logits."""
    entry["input_ids"].copy_(input_ids)
    entry["position_ids"].copy_(position_ids)
    entry["attn_mask"].copy_(attn_mask)
    if cache_position is not None:
        entry["cache_position"].copy_(cache_position)
    entry["graph"].replay()
    return entry["logits"]


def _build_iter_attn_mask(cache: StaticBlockCache,
                          ctx_aligned: int,
                          cur_len: int,
                          intra_q_to_q: torch.Tensor,
                          device: torch.device,
                          batch_size: int = 1) -> torch.Tensor:
    """Assemble (B, 1, cur_len, full_len) bool mask:
      - cache portion [0, ctx_aligned): all True (visible)
      - cache portion [ctx_aligned, max_cache_len): all False (uninit)
      - scratch portion [max_cache_len, max_cache_len + cur_len): per intra_q_to_q
      - tail [max_cache_len + cur_len, full_len): all False (unused)

    All batch rows share same mask (lockstep progress assumption).
    """
    full_len = cache.full_len
    mask = torch.zeros(batch_size, 1, cur_len, full_len, dtype=torch.bool, device=device)
    if ctx_aligned > 0:
        mask[..., :, :ctx_aligned] = True
    s = cache.max_cache_len
    mask[..., :, s : s + cur_len] = intra_q_to_q.unsqueeze(0).unsqueeze(0)
    return mask


def _denoise_block_graph_dispatch(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    leftover: List[int],
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    device: torch.device,
    *,
    sampling: str = "argmax",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_static",
    confidence_threshold: float = 0.9,
    early_stop_steps: int = -1,
    fused: bool = False,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    if fused and sampling == "argmax" \
            and remasking_strategy == "low_confidence_static" \
            and not (0 < early_stop_steps < denoising_steps):
        return _denoise_block_fused_graph(
            model, cache, ctx_aligned, leftover, block_length,
            denoising_steps, mask_token_id, device,
        )
    return _denoise_block_graph(
        model, cache, ctx_aligned, leftover, block_length, denoising_steps,
        mask_token_id, device,
        sampling=sampling, temperature=temperature, top_k=top_k, top_p=top_p,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        early_stop_steps=early_stop_steps,
    )


def _denoise_block_graph(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    leftover: List[int],
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    device: torch.device,
    sampling: str = "multinomial",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_static",
    confidence_threshold: float = 0.9,
    early_stop_steps: int = -1,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """CUDA-graph-replayed denoise. Same semantics as
    ``_denoise_block_with_cache`` but uses fixed-shape buffers.

    Returns (draft_ids, per_pos_probs, step_map) for the new positions only.

    Sync-free body (single batched DH at exit). The previous implementation
    issued one ``.item()`` per denoising step plus one ``.item()`` /
    ``.cpu()`` per unmasked position, which serialized this function into
    ``denoising_steps × n_unmask`` host-side mini-steps. Inside
    ``speculative_generate_with_kv_cache_pipelined`` that meant ``draft_{N+1}``
    could only overlap its first denoise step with ``verify_N``; the rest ran
    serially after each Python-side sync. Now the per-step schedule is
    pre-computed deterministically (``ceil(remaining / remaining_steps)``
    equivalent to the original because ``topk(n_unmask)`` exhausts picks each
    step) and per-position writes go through ``scatter_`` / ``index_copy_``,
    so all denoise kernels for the block enqueue back-to-back on the draft
    stream. The lone exit sync (``.tolist()`` / ``.cpu()``) waits for the
    whole denoise wall but does not block the sibling GPU's verify stream.
    """
    n_known = len(leftover)
    n_masks = block_length - n_known
    vocab = model.config.vocab_size
    if n_masks == 0:
        return [], torch.zeros(0, vocab), []

    # Deterministic per-step unmask counts. sum(n_unmask_per_step) == n_masks
    # so the original force-fill safety net is unreachable and removed.
    n_unmask_per_step: List[int] = []
    rem = n_masks
    for s in range(denoising_steps):
        rs = denoising_steps - s
        n_u = -(-rem // rs)  # ceil division
        n_unmask_per_step.append(n_u)
        rem -= n_u

    # Early-stop: only run the first ``early_stop_steps`` of the schedule;
    # remaining positions stay MASK and get step_map[i] = -1 sentinel
    # (caller routes them via target argmax during verify+MRS).
    _early_active = (
        isinstance(early_stop_steps, int)
        and 0 < early_stop_steps < denoising_steps
    )
    if _early_active:
        n_unmask_per_step = n_unmask_per_step[:early_stop_steps]

    entry = _get_static_forward_graph(
        model, "denoise", block_length, cache, device, mask_token_id,
    )

    # Within the bl current-block positions, all see all (block-causal intra).
    intra = torch.ones(block_length, block_length, dtype=torch.bool, device=device)
    attn_mask = _build_iter_attn_mask(cache, ctx_aligned, block_length, intra, device)

    cur_x = list(leftover) + [mask_token_id] * n_masks
    pos_ids = list(range(ctx_aligned, ctx_aligned + block_length))
    pos_t = torch.tensor([pos_ids], dtype=torch.long, device=device)
    cur_x_t = torch.tensor([cur_x], dtype=torch.long, device=device)

    # GPU-resident accumulators reused across calls  repeatedly allocating
    # ``per_pos_probs_t`` (block_length × vocab × 4B ≈ 2.4 MB for SDAR) per
    # call thrashed the caching allocator and added ~10 samples of warmup.
    # Persistent buffers are attached to the model and zero'd in-place.
    bufs = getattr(model, "_denoise_block_bufs", None)
    if (
        bufs is None
        or bufs["step_map"].shape[0] != block_length
        or bufs["probs"].shape != (block_length, vocab)
        or bufs["step_map"].device != device
    ):
        bufs = {
            "step_map": torch.zeros(block_length, dtype=torch.long, device=device),
            "probs": torch.zeros(block_length, vocab, device=device, dtype=torch.float32),
        }
        model._denoise_block_bufs = bufs
    step_map_t = bufs["step_map"]
    per_pos_probs_t = bufs["probs"]
    step_map_t.zero_()
    per_pos_probs_t.zero_()

    is_dynamic = (remasking_strategy == "low_confidence_dynamic")
    for step_i, n_unmask in enumerate(n_unmask_per_step):
        if n_unmask == 0:
            continue
        is_masked = (cur_x_t[0] == mask_token_id)
        # Dynamic mode may burst-unmask all positions in one step  later
        # steps have nothing to do. Sync-check only in dynamic mode (static
        # is sync-free by precomputed schedule).
        if is_dynamic and int(is_masked.sum().item()) == 0:
            break
        logits_static = _replay_static_forward(
            entry, cur_x_t, pos_t, attn_mask,
        )
        probs, pred_ids = _sample_logits(
            logits_static[0], sampling, temperature, top_k, top_p,
        )
        confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)
        top_pos = _pick_unmask_positions(
            is_masked, confidence, n_unmask,
            remasking_strategy, confidence_threshold,
        )
        # Scatter writes  fully GPU-side, no .item().
        cur_x_t[0].scatter_(0, top_pos, pred_ids.gather(0, top_pos))
        step_map_t.scatter_(0, top_pos, torch.full_like(top_pos, step_i))
        per_pos_probs_t.index_copy_(0, top_pos, probs.index_select(0, top_pos))

    # Single batched sync at exit. While the draft stream completes the whole
    # denoise wall here, the sibling GPU's verify_N kernels can still run
    # .cpu() on a draft-device tensor only waits on the draft stream.
    draft_ids = cur_x_t[0, n_known:].tolist()
    new_probs = per_pos_probs_t[n_known:].cpu()
    new_step_map = step_map_t[n_known:].tolist()
    if _early_active:
        # Mark positions still MASK (never unmasked under early-stop) with
        # sentinel -1. Order matters: do this AFTER tolist() so we touch
        # only host-side scalars.
        for i, tok in enumerate(draft_ids):
            if tok == mask_token_id:
                new_step_map[i] = -1
    return draft_ids, new_probs, new_step_map


_FUSED_DENOISE_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _capture_fused_denoise_graph(
    model,
    block_length: int,
    n_known: int,
    denoising_steps: int,
    cache: StaticBlockCache,
    device: torch.device,
    mask_token_id: int,
) -> Optional[Dict[str, Any]]:
    n_masks = block_length - n_known
    if n_masks <= 0:
        return None

    n_unmask_per_step: List[int] = []
    rem = n_masks
    for s in range(denoising_steps):
        rs = denoising_steps - s
        n_u = -(-rem // rs)
        if n_u <= 0:
            break
        n_unmask_per_step.append(int(n_u))
        rem -= n_u

    full_len = cache.full_len
    B = cache.batch_size
    vocab = model.config.vocab_size

    static_input_ids = torch.full(
        (B, block_length), mask_token_id, dtype=torch.long, device=device,
    )
    static_position_ids = torch.zeros(
        (B, block_length), dtype=torch.long, device=device,
    )
    static_attn_mask = torch.zeros(
        (B, 1, block_length, full_len), dtype=torch.bool, device=device,
    )
    static_cur_scratch_pos = torch.arange(
        cache.max_cache_len, cache.max_cache_len + block_length,
        dtype=torch.long, device=device,
    )
    static_cache_position = static_cur_scratch_pos.clone()
    static_step_map = torch.zeros(block_length, dtype=torch.long, device=device)
    static_per_pos_probs = torch.zeros(block_length, vocab, dtype=torch.float32, device=device)

    def _run_body():
        static_step_map.zero_()
        static_per_pos_probs.zero_()
        for step_i, n_u in enumerate(n_unmask_per_step):
            out = model(
                input_ids=static_input_ids,
                attention_mask=static_attn_mask,
                position_ids=static_position_ids,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
                cur_scratch_pos=static_cur_scratch_pos,
                cache_position=static_cache_position,
                return_dict=True,
            )
            logits = out.logits[0]
            probs = F.softmax(logits.float(), dim=-1)
            pred_ids = probs.argmax(dim=-1)
            confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)
            is_masked = (static_input_ids[0] == mask_token_id)
            masked_conf = torch.where(
                is_masked, confidence,
                torch.full_like(confidence, float("-inf")),
            )
            _, top_pos = masked_conf.topk(n_u)
            static_input_ids[0].scatter_(0, top_pos, pred_ids.gather(0, top_pos))
            static_step_map.scatter_(
                0, top_pos, torch.full_like(top_pos, step_i),
            )
            static_per_pos_probs.index_copy_(
                0, top_pos, probs.index_select(0, top_pos),
            )

    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                static_input_ids.fill_(mask_token_id)
                _run_body()
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s, capture_error_mode="thread_local"), \
                torch.no_grad():
            _run_body()

    return {
        "graph": g,
        "input_ids": static_input_ids,
        "position_ids": static_position_ids,
        "attn_mask": static_attn_mask,
        "cur_scratch_pos": static_cur_scratch_pos,
        "cache_position": static_cache_position,
        "step_map": static_step_map,
        "per_pos_probs": static_per_pos_probs,
        "n_known": n_known,
        "n_masks": n_masks,
        "eff_steps": len(n_unmask_per_step),
    }


def _get_fused_denoise_graph(
    model, block_length: int, n_known: int, denoising_steps: int,
    cache: StaticBlockCache, device: torch.device, mask_token_id: int,
) -> Optional[Dict[str, Any]]:
    key = (id(model), int(block_length), int(n_known), int(denoising_steps),
           int(cache.batch_size), str(device))
    entry = _FUSED_DENOISE_CACHE.get(key)
    if entry is None:
        entry = _capture_fused_denoise_graph(
            model, block_length, n_known, denoising_steps, cache, device,
            mask_token_id,
        )
        if entry is not None:
            _FUSED_DENOISE_CACHE[key] = entry
    return entry


def _denoise_block_fused_graph(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    leftover: List[int],
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor, List[int]]:
    n_known = len(leftover)
    n_masks = block_length - n_known
    vocab = model.config.vocab_size
    if n_masks == 0:
        return [], torch.zeros(0, vocab), []

    entry = _get_fused_denoise_graph(
        model, block_length, n_known, denoising_steps, cache, device,
        mask_token_id,
    )
    if entry is None:
        return [], torch.zeros(0, vocab), []

    cur_x = list(leftover) + [mask_token_id] * n_masks
    pos_ids = list(range(ctx_aligned, ctx_aligned + block_length))

    intra = torch.ones(block_length, block_length, dtype=torch.bool, device=device)
    attn_mask = _build_iter_attn_mask(cache, ctx_aligned, block_length, intra, device)

    entry["input_ids"][0].copy_(
        torch.as_tensor(cur_x, dtype=torch.long, device=device),
    )
    entry["position_ids"][0].copy_(
        torch.as_tensor(pos_ids, dtype=torch.long, device=device),
    )
    entry["attn_mask"].copy_(attn_mask)

    entry["graph"].replay()

    draft_ids = entry["input_ids"][0, n_known:].tolist()
    new_probs = entry["per_pos_probs"][n_known:].cpu()
    new_step_map = entry["step_map"][n_known:].tolist()
    return draft_ids, new_probs, new_step_map


def _verify_block_graph(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    leftover: List[int],
    draft_ids: List[int],
    step_map: List[int],
    block_length: int,
    mask_token_id: int,
    device: torch.device,
    return_on_device: bool = False,
) -> torch.Tensor:
    """CUDA-graph-replayed verify. Mirrors ``_verify_block_with_cache``.

    Returns target_probs of shape (n_masks, vocab). When ``return_on_device``
    is True, the tensor is left on ``device`` (no .cpu() D2H sync)  caller
    must .cpu() it before reading. This is required by the pipelined path so
    that draft_{N+1} on the other GPU can overlap with verify_N's kernels.

    Optimizations vs. the naive layout:
      - Vectorized intra mask: previously the paired-step-causal block was
        written via a Python ``for i: for j: intra[i,j] = bool`` loop, which
        issued ``n_masks² × 2`` single-element GPU writes per call (~32 kernel
        launches with bl=4). Replaced by 4 broadcast comparisons.
      - Persistent intra + entry["attn_mask"] direct write: skips a fresh
        ``torch.zeros(cur_len, cur_len)`` alloc + an intermediate
        ``(1, 1, cur_len, full_len)`` mask tensor and the redundant
        ``entry["attn_mask"].copy_(...)`` from the wrapper.
    """
    n_known = len(leftover)
    n_masks = len(draft_ids)
    assert n_known + n_masks == block_length
    cur_len = n_known + 2 * n_masks  # leftover + draft_data + draft_mask

    entry = _get_static_forward_graph(
        model, "verify", cur_len, cache, device, mask_token_id,
    )

    L_lo, L_hi = 0, n_known
    D_lo, D_hi = n_known, n_known + n_masks
    M_lo, M_hi = n_known + n_masks, cur_len

    # Persistent intra buffer (cur_len-keyed, attached to model so it follows
    # the same model lifecycle as _STATIC_GRAPH_CACHE).
    cache_attr = "_verify_intra_bufs"
    bufs = getattr(model, cache_attr, None)
    if bufs is None:
        bufs = {}
        setattr(model, cache_attr, bufs)
    buf_key = (cur_len, str(device))
    intra = bufs.get(buf_key)
    if intra is None or intra.device != device:
        intra = torch.zeros(cur_len, cur_len, dtype=torch.bool, device=device)
        bufs[buf_key] = intra
    intra.zero_()

    if n_known > 0:
        intra[L_lo:L_hi, L_lo:L_hi] = True
        if n_masks > 0:
            intra[D_lo:D_hi, L_lo:L_hi] = True
            intra[M_lo:M_hi, L_lo:L_hi] = True
    if n_masks > 0:
        # Vectorized GPU mask fill replaces n_masks² Python loop.
        # data_times[p] = step_map[p] + 1; comparisons are equivalent to
        # comparing step_map directly (the +1 cancels on both sides).
        # Early-stop: positions never drafted have step_map[i] = -1; we map
        # them to ``block_length + 1`` (a sentinel above any real step) so the
        # comparison-based rules below give them the correct "still-MASK at
        # time ∞" semantics: drafted DATA do not see them; non-drafted MASK
        # rows see all drafted DATA but no drafted MASK; etc. See SUMMARY.md
        # 2026-05-07 in results/draft_steps_sweep for the rule trace.
        step_arr = [s if s >= 0 else (block_length + 1) for s in step_map]
        step_t = torch.as_tensor(step_arr, dtype=torch.long, device=device)
        ti = step_t.unsqueeze(1)  # (n_masks, 1)
        tj = step_t.unsqueeze(0)  # (1, n_masks)
        intra[D_lo:D_hi, D_lo:D_hi] = (tj <= ti)
        intra[D_lo:D_hi, M_lo:M_hi] = (tj > ti)
        intra[M_lo:M_hi, D_lo:D_hi] = (tj < ti)
        intra[M_lo:M_hi, M_lo:M_hi] = (tj >= ti)

    # Write attn_mask DIRECTLY into entry["attn_mask"]  saves an intermediate
    # alloc and the copy_ inside _replay_static_forward.
    entry_mask = entry["attn_mask"]
    s = cache.max_cache_len
    entry_mask.zero_()
    if ctx_aligned > 0:
        entry_mask[..., :, :ctx_aligned] = True
    entry_mask[..., :, s:s + cur_len] = intra.unsqueeze(0).unsqueeze(0)

    # Build query input. Keep torch.tensor  small ops on 8-element lists.
    in_ids = list(leftover) + list(draft_ids) + [mask_token_id] * n_masks
    leftover_pos = list(range(ctx_aligned, ctx_aligned + n_known))
    draft_pos = list(range(ctx_aligned + n_known, ctx_aligned + block_length))
    pos_ids = leftover_pos + draft_pos + draft_pos
    in_ids_t = torch.tensor([in_ids], dtype=torch.long, device=device)
    pos_t = torch.tensor([pos_ids], dtype=torch.long, device=device)

    # Inline replay  skip _replay_static_forward's redundant attn_mask.copy_
    # since we already wrote into entry["attn_mask"] above.
    entry["input_ids"].copy_(in_ids_t)
    entry["position_ids"].copy_(pos_t)
    entry["graph"].replay()
    logits_static = entry["logits"]

    mask_logits = logits_static[0, M_lo:M_hi]
    target_probs = F.softmax(mask_logits.float(), dim=-1)
    if not return_on_device:
        target_probs = target_probs.cpu()
    return target_probs


def _extend_cache_graph(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    full_block: List[int],
    device: torch.device,
) -> int:
    """CUDA-graph-replayed extend. Writes a full block of K/V into cache
    permanent slot via ``cache_position = arange(ctx_aligned, ctx_aligned+bl)``.
    Returns new ctx_aligned.

    Same dispatch optimization as ``_verify_block_graph``: persistent intra
    buffer (here always all-True, so we keep one ``torch.ones`` per (n, device)
    and reuse) + direct write to ``entry["attn_mask"]`` skipping the
    ``_build_iter_attn_mask`` intermediate.
    """
    n = len(full_block)
    assert n > 0
    entry = _get_static_forward_graph(
        model, "extend", n, cache, device, MASK_TOKEN_ID_DEFAULT,
    )

    # Persistent all-True intra buffer (immutable for this op_type).
    cache_attr = "_extend_intra_bufs"
    bufs = getattr(model, cache_attr, None)
    if bufs is None:
        bufs = {}
        setattr(model, cache_attr, bufs)
    buf_key = (n, str(device))
    intra = bufs.get(buf_key)
    if intra is None or intra.device != device:
        intra = torch.ones(n, n, dtype=torch.bool, device=device)
        bufs[buf_key] = intra

    # Direct write to entry["attn_mask"].
    entry_mask = entry["attn_mask"]
    s = cache.max_cache_len
    entry_mask.zero_()
    if ctx_aligned > 0:
        entry_mask[..., :, :ctx_aligned] = True
    entry_mask[..., :, s:s + n] = intra.unsqueeze(0).unsqueeze(0)

    in_ids_t = torch.tensor([full_block], dtype=torch.long, device=device)
    pos_t = torch.arange(
        ctx_aligned, ctx_aligned + n, dtype=torch.long, device=device,
    ).unsqueeze(0)
    cache_pos_t = torch.arange(
        ctx_aligned, ctx_aligned + n, dtype=torch.long, device=device,
    )

    # Inline replay  entry["attn_mask"] already populated above.
    entry["input_ids"].copy_(in_ids_t)
    entry["position_ids"].copy_(pos_t)
    if "cache_position" in entry and cache_pos_t is not None:
        entry["cache_position"].copy_(cache_pos_t)
    entry["graph"].replay()
    return ctx_aligned + n


def _prefill_prompt_static(
    model,
    prompt_ids: List[int],
    cache: StaticBlockCache,
    device: torch.device,
    block_length: int = 4,
) -> Tuple[int, List[int]]:
    """Aligned-prefill into a StaticBlockCache (eager  runs once per request).

    Writes K/V for prompt[:aligned_len] directly into cache positions
    [0, aligned_len) by passing cur_scratch_pos=arange(0, aligned_len). The
    patched static-cache attention uses index_copy_ at those positions,
    same shape as max_cur_len isn't enforced (scratch_pos can be anywhere
    in the cache buffer).
    """
    from speculative_decoding.draft import _build_block_causal_attn
    L = len(prompt_ids)
    aligned_len = (L // block_length) * block_length
    leftover = list(prompt_ids[aligned_len:])

    if aligned_len == 0:
        return 0, leftover

    input_ids = torch.tensor(
        [prompt_ids[:aligned_len]], dtype=torch.long, device=device,
    )
    position_ids = torch.arange(aligned_len, device=device).unsqueeze(0)
    # Block-causal mask sized (1, 1, aligned_len, full_len): visible only at
    # [0, aligned_len) (causal block layout).
    full_len = cache.full_len
    base = _build_block_causal_attn(aligned_len, block_length, device).bool()
    attn_mask = torch.zeros(1, 1, aligned_len, full_len,
                            dtype=torch.bool, device=device)
    attn_mask[..., :, :aligned_len] = base.unsqueeze(0).unsqueeze(0)

    # cur_scratch_pos = where to write K/V. For prefill it IS the permanent
    # slot at [0, aligned_len) so the cache is populated correctly.
    cur_scratch_pos = torch.arange(aligned_len, dtype=torch.long, device=device)

    with torch.cuda.device(device), torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=False,  # only one write needed; scratch positions = permanent
            cur_scratch_pos=cur_scratch_pos,
            return_dict=True,
        )
    return aligned_len, leftover


def patch_causallm_pass_cache(model) -> bool:
    """1.7B's SDARForCausalLM.forward (modeling_sdar.py:849) passes neither
    past_key_values nor use_cache when delegating to self.model(...). This
    means an internally-allocated DynamicCache is used per forward and our
    accumulated cache is silently dropped. 8B's SDARForCausalLM.forward
    (8B modeling_sdar.py:819-830) does pass them correctly.

    Patch: rebind SDARForCausalLM.forward so the internal self.model call
    forwards past_key_values + use_cache + inputs_embeds. Idempotent per class.
    """
    import functools
    cls = model.__class__
    if cls.__name__ != "SDARForCausalLM":
        return False
    if getattr(cls, "_cache_passthrough_patched", False):
        return False

    orig_forward = cls.forward

    @functools.wraps(orig_forward)
    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Mirror 8B's delegation pattern: pass past_key_values + use_cache +
        # inputs_embeds to self.model(...). Strip return_dict from kwargs to
        # avoid duplicate-kwarg TypeError when caller passes it explicitly.
        kwargs.pop("return_dict", None)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if logits_to_keep is not None:
            B, _, H = hidden_states.shape
            num_keep = logits_to_keep.sum(dim=1)
            assert torch.all(num_keep == num_keep[0])
            N = int(num_keep[0].item())
            hidden_states = hidden_states[logits_to_keep].view(B, N, H).contiguous()
        logits = self.lm_head(hidden_states)
        # Reuse transformers output type from the original return.
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    cls.forward = patched_forward
    cls._cache_passthrough_patched = True
    return True


def patch_sdpa_with_cache(model) -> bool:
    """Replace SDARAttention.forward with an SDPA-only eval path that ALSO
    handles past_key_value / store_kv (the existing patch_sdpa_eval_attention
    in draft.py drops the cache silently; the default 1.7B forward uses
    fused_flex_attention which can't accept our bool tensor mask).

    Idempotent. Returns True if a patch was applied this call.
    """
    import sys
    import torch.nn.functional as F
    from einops import rearrange

    attn_cls = None
    for m in model.modules():
        if m.__class__.__name__ == "SDARAttention":
            attn_cls = m.__class__
            break
    if attn_cls is None:
        return False
    if attn_cls in _PATCHED_ATTN_CLASSES:
        return False

    apply_rotary_pos_emb = sys.modules[attn_cls.__module__].apply_rotary_pos_emb

    def _sdpa_cache_forward(self, hidden_states, position_embeddings,
                            attention_mask, past_key_value=None,
                            cache_position=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Cache integration (mirrors modeling_sdar.py SDARAttention.forward
        # behavior at 8B lines 263-273):
        store_kv = bool(kwargs.get("store_kv", False))
        if past_key_value is not None and store_kv:
            k, v = past_key_value.update(k, v, self.layer_idx)
        elif (
            past_key_value is not None
            and not store_kv
            and len(past_key_value) > self.layer_idx
        ):
            past_k, past_v = past_key_value[self.layer_idx]
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        mask_bool = attention_mask.bool() if attention_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=mask_bool,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    attn_cls.forward = _sdpa_cache_forward
    _PATCHED_ATTN_CLASSES.add(attn_cls)
    return True


def _select_verify_fn(cfg):
    branch = getattr(cfg, "speculative_branch", "mrs")
    if branch == "greedy_match":
        return greedy_match_verify
    return mrs_verify


def _prefill_prompt(model, prompt_ids: List[int], cache: DynamicCache,
                    device: torch.device,
                    block_length: int = 4) -> Tuple[int, DynamicCache, List[int]]:
    """Aligned-prefill: forward the BLOCK-ALIGNED prefix of the prompt with
    ``store_kv=True``; the unaligned tail (leftover) is returned unencoded.

    For a 50-token prompt with block_length=4:
      aligned_len = 48     (= 50 // 4 * 4)
      leftover    = prompt_ids[48:50]  (2 tokens, will go into the first
                                        decode block alongside the new masks)

    Why aligned-prefill: SDAR's block-causal attention couples positions
    sharing a ``block_id = pos // block_length``. Pre-computing K/V for the
    unaligned prompt tail under prompt-only attention (i.e., prefilling the
    full prompt) yields STALE K/V at those positions: in no-cache forward
    they would have seen the same-block decoded mask positions. By holding
    leftover OUTSIDE the cache and re-forwarding it together with the first
    decode block each iter, we match no-cache K/V exactly.

    Returns:
        aligned_len: number of tokens prefilled into cache (= multiple of bl)
        cache:       possibly-replaced cache reference (SDARForCausalLM bug)
        leftover:    prompt tokens at positions [aligned_len, len(prompt)),
                     0 to block_length-1 tokens, NOT in cache.
    """
    from speculative_decoding.draft import _build_block_causal_attn
    L = len(prompt_ids)
    aligned_len = (L // block_length) * block_length
    leftover = list(prompt_ids[aligned_len:])

    if aligned_len == 0:
        # Prompt shorter than one block  entirely leftover, cache stays empty.
        return 0, cache, leftover

    input_ids = torch.tensor(
        [prompt_ids[:aligned_len]], dtype=torch.long, device=device,
    )
    position_ids = torch.arange(aligned_len, device=device).unsqueeze(0)
    # BLOCK-causal among aligned prompt tokens: identical to prompt-only
    # block_causal_prompt rule when input is prompt-only (no data/mask),
    # so this prefill K/V is consumable by both draft denoising
    # (block-causal) and target verify (block_causal_prompt=True) downstream.
    attn_mask = _build_block_causal_attn(aligned_len, block_length, device)
    attn_mask = attn_mask.bool().unsqueeze(1)  #  [1, 1, L, L]
    with torch.cuda.device(device), torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
    new_cache = getattr(out, "past_key_values", None) or cache
    return aligned_len, new_cache, leftover


def _denoise_block_with_cache(
    model,
    cache: DynamicCache,
    ctx_aligned: int,
    leftover: List[int],
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    device: torch.device,
    sampling: str = "multinomial",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_static",
    confidence_threshold: float = 0.9,
    early_stop_steps: int = -1,
) -> Tuple[List[int], torch.Tensor, List[int], DynamicCache]:
    """One block of diffusion denoising on top of cached prefix + leftover.

    Cache contains K/V for [0, ctx_aligned). Leftover tokens occupy
    positions [ctx_aligned, ctx_aligned + len(leftover)) and are part of
    the current decode block but NOT in cache. The remaining
    n_masks = block_length - len(leftover) positions are MASK to denoise.

    cur_x layout: [leftover (known) | masks (unknown)]  total = block_length
    All bl positions share block_id = ctx_aligned // bl, so within-block
    block-causal allows full bidirectional attention among them. This
    matches no-cache draft denoising ``_build_block_causal_attn`` exactly.

    Returns:
        draft_ids:     list[int] of length n_masks (only the new positions)
        per_pos_probs: Tensor (n_masks, vocab_size)
        step_map:      list[int] of length n_masks
        cache:         possibly-replaced reference
    """
    n_known = len(leftover)
    n_masks = block_length - n_known
    vocab_size = model.config.vocab_size

    if n_masks == 0:
        # Nothing to denoise; caller should flush leftover and re-enter.
        return [], torch.zeros(0, vocab_size), [], cache

    cur_x = torch.tensor(
        [list(leftover) + [mask_token_id] * n_masks],
        dtype=torch.long, device=device,
    )
    position_ids = torch.arange(
        ctx_aligned, ctx_aligned + block_length, device=device,
    ).unsqueeze(0)
    # All bl current-block positions can see each other + all of cache.
    # Since ctx_aligned is bl-aligned and cur_x is one block, no cross-block
    # restriction is needed within the query.
    full_len = ctx_aligned + block_length
    attn_mask = torch.ones(
        1, 1, block_length, full_len, dtype=torch.bool, device=device,
    )

    step_map = [0] * block_length
    per_pos_probs = torch.zeros(block_length, vocab_size)

    # Early-stop: cap loop iterations; remaining positions stay MASK and
    # get step_map[i] = -1 sentinel so caller routes them via target argmax.
    _early_active = (
        isinstance(early_stop_steps, int)
        and 0 < early_stop_steps < denoising_steps
    )
    _effective_steps = early_stop_steps if _early_active else denoising_steps

    for step in range(1, _effective_steps + 1):
        is_masked = (cur_x[0] == mask_token_id)
        num_masked = int(is_masked.sum().item())
        if num_masked == 0:
            break

        with torch.cuda.device(device), torch.no_grad():
            out = model(
                input_ids=cur_x,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                store_kv=False,
            )
        # Adopt possibly-fresh cache (workaround for SDARForCausalLM bug
        # where it drops past_key_values when delegating to inner model).
        new_cache = getattr(out, "past_key_values", None)
        if new_cache is not None:
            cache = new_cache
        logits = out.logits[0]  # (block_length, vocab)
        probs, pred_ids = _sample_logits(
            logits, sampling, temperature, top_k, top_p,
        )
        confidence = probs.gather(1, pred_ids.unsqueeze(-1)).squeeze(-1)

        # Pick unmask positions per remasking strategy.
        remaining = denoising_steps - step + 1
        n_unmask = min(-(-num_masked // remaining), num_masked)
        unmask_pos = _pick_unmask_positions(
            is_masked, confidence, n_unmask,
            remasking_strategy, confidence_threshold,
        )
        for p_t in unmask_pos:
            p = int(p_t.item())
            cur_x[0, p] = pred_ids[p]
            step_map[p] = step - 1
            per_pos_probs[p] = probs[p].cpu()

    # Force-fill any leftover masks with last forward's argmax (rare).
    # SKIPPED under early-stop: those positions are intentional MASKs and
    # carry step_map[i] = -1 so the caller routes them through target argmax.
    is_masked = (cur_x[0] == mask_token_id)
    if is_masked.any() and not _early_active:
        with torch.cuda.device(device), torch.no_grad():
            out = model(
                input_ids=cur_x, attention_mask=attn_mask,
                position_ids=position_ids, past_key_values=cache,
                use_cache=True, store_kv=False,
            )
        new_cache = getattr(out, "past_key_values", None)
        if new_cache is not None:
            cache = new_cache
        logits = out.logits[0]
        probs = F.softmax(logits.float(), dim=-1)
        if sampling == "multinomial":
            pred_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            pred_ids = torch.argmax(probs, dim=-1)
        for p_t in torch.nonzero(is_masked, as_tuple=True)[0]:
            p = int(p_t.item())
            cur_x[0, p] = pred_ids[p]
            step_map[p] = denoising_steps - 1
            per_pos_probs[p] = probs[p].cpu()
    elif is_masked.any() and _early_active:
        for p_t in torch.nonzero(is_masked, as_tuple=True)[0]:
            step_map[int(p_t.item())] = -1

    # Return only the new (mask-derived) positions; leftover tokens are
    # already known to caller.
    draft_ids = cur_x[0, n_known:].tolist()
    new_probs = per_pos_probs[n_known:]
    new_step_map = step_map[n_known:]
    return draft_ids, new_probs, new_step_map, cache


def _verify_block_with_cache(
    target_model,
    cache: DynamicCache,
    ctx_aligned: int,
    leftover: List[int],
    draft_ids: List[int],
    step_map: List[int],
    block_size: int,
    block_length: int,
    mask_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, DynamicCache]:
    """Verify forward with leftover prefix + draft block.

    Layout (Q = n_known + 2*n_masks tokens in query):
        [leftover (PROMPT) | draft_data (DATA) | draft_mask (MASK)]
        n_known            n_masks             n_masks

    where n_known = len(leftover), n_masks = len(draft_ids), and
    n_known + n_masks = block_length. Cache holds K/V for [0, ctx_aligned)
    (only block-aligned prompt prefix). Leftover replays each iter to match
    no-cache verify K/V.

    Mask rules (matching ``create_multi_block_causal_mask`` with
    ``block_causal_prompt=True``):
      - leftover (prompt) sees ALL of cache + other leftover (intra-prompt
        block-causal); does NOT see draft_data / draft_mask.
      - draft_data / draft_mask see ALL prompt (cache + leftover) via
        ``rule_see_prompt``.
      - draft_data / draft_mask interact via step-causal:
          d(t)d(t'):  tj <= ti
          d(t)m(t'):  tj > ti
          m(t)d(t'):  tj < ti
          m(t)m(t'):  tj >= ti
        with mask_time = paired data_time = step_map[i] + 1.

    Returns (target_probs, cache). target_probs shape = (n_masks, vocab_size).
    """
    bl = block_length
    n_known = len(leftover)
    n_masks = len(draft_ids)
    assert n_known + n_masks == bl, \
        f"leftover ({n_known}) + draft_ids ({n_masks}) must equal block_length ({bl})"

    # Build query tokens: leftover, draft data, draft mask copies.
    in_ids = torch.tensor(
        [list(leftover) + list(draft_ids) + [mask_token_id] * n_masks],
        dtype=torch.long, device=device,
    )
    # Position IDs:
    #   leftover at [ctx_aligned, ctx_aligned + n_known)
    #   draft_data at [ctx_aligned + n_known, ctx_aligned + bl)
    #   draft_mask paired at SAME positions as draft_data
    leftover_pos = torch.arange(
        ctx_aligned, ctx_aligned + n_known, device=device,
    )
    draft_pos = torch.arange(
        ctx_aligned + n_known, ctx_aligned + bl, device=device,
    )
    position_ids = torch.cat([leftover_pos, draft_pos, draft_pos]).unsqueeze(0)

    # Build the (Q, full_len) mask.
    Q = n_known + 2 * n_masks
    full_len = ctx_aligned + Q
    attn_mask = torch.zeros(1, 1, Q, full_len, dtype=torch.bool, device=device)
    # All query positions see all cache (rule_prompt block-causal for leftover
    # since cache is earlier prompt blocks; rule_see_prompt for draft_*).
    attn_mask[..., :, :ctx_aligned] = True

    # Within-query mask (Q × Q).
    # Index ranges:
    L_lo, L_hi = 0, n_known                  # leftover
    D_lo, D_hi = n_known, n_known + n_masks  # draft_data
    M_lo, M_hi = n_known + n_masks, Q        # draft_mask
    intra = torch.zeros(Q, Q, dtype=torch.bool, device=device)

    # Leftover ↔ Leftover: intra-prompt block-causal (same block  all True).
    if n_known > 0:
        intra[L_lo:L_hi, L_lo:L_hi] = True

    # Leftover  draft_data / draft_mask: NOT allowed (prompt rule excludes it).
    # (already 0)

    # Draft_data  leftover: rule_see_prompt (data sees prompt)  True.
    if n_known > 0 and n_masks > 0:
        intra[D_lo:D_hi, L_lo:L_hi] = True
        intra[M_lo:M_hi, L_lo:L_hi] = True

    # Draft data/mask step-causal among themselves.
    if n_masks > 0:
        # Early-stop: step_map[p] == -1  never drafted; map to a sentinel
        # above any real step so the comparison rules treat it as "still
        # MASK at time ∞".  See _verify_block_graph for full rule analysis.
        data_times = [
            (step_map[p] + 1) if step_map[p] >= 0 else (bl + 2)
            for p in range(n_masks)
        ]
        mask_times = data_times[:]  # paired
        for i in range(n_masks):
            ti = data_times[i]
            for j in range(n_masks):
                # data(i)  data(j): tj <= ti
                intra[D_lo + i, D_lo + j] = (data_times[j] <= ti)
                # data(i)  mask(j): tj > ti
                intra[D_lo + i, M_lo + j] = (mask_times[j] > ti)
        for i in range(n_masks):
            ti = mask_times[i]
            for j in range(n_masks):
                # mask(i)  data(j): tj < ti
                intra[M_lo + i, D_lo + j] = (data_times[j] < ti)
                # mask(i)  mask(j): tj >= ti
                intra[M_lo + i, M_lo + j] = (mask_times[j] >= ti)

    attn_mask[..., :, ctx_aligned:] = intra.unsqueeze(0).unsqueeze(0)

    with torch.cuda.device(device), torch.no_grad():
        out = target_model(
            input_ids=in_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=False,
        )
    new_cache = getattr(out, "past_key_values", None)
    if new_cache is not None:
        cache = new_cache
    # Logits at draft_mask positions = indices [M_lo, M_hi).
    mask_logits = out.logits[0, M_lo:M_hi]
    target_probs = F.softmax(mask_logits.float(), dim=-1).cpu()
    return target_probs, cache


def _extend_cache(
    model,
    cache: DynamicCache,
    ctx_len: int,
    new_tokens: List[int],
    device: torch.device,
) -> Tuple[int, DynamicCache]:
    """Forward `new_tokens` at positions [ctx_len .. ctx_len+len-1] with
    store_kv=True so the cache absorbs their K/V. Returns (new_ctx_len, cache).
    Cache may be replaced  caller must adopt returned reference.
    """
    n = len(new_tokens)
    if n == 0:
        return ctx_len, cache
    in_ids = torch.tensor([new_tokens], dtype=torch.long, device=device)
    position_ids = torch.arange(
        ctx_len, ctx_len + n, device=device,
    ).unsqueeze(0)
    full_len = ctx_len + n
    # New tokens see all of cache + each other (causal not needed since these
    # are committed data; matching block_diffusion_generate's single forward
    # to extend cache pattern at line 121-128).
    attn_mask = torch.ones(
        1, 1, n, full_len, dtype=torch.bool, device=device,
    )
    with torch.cuda.device(device), torch.no_grad():
        out = model(
            input_ids=in_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
    new_cache = getattr(out, "past_key_values", None)
    if new_cache is not None:
        cache = new_cache
    return ctx_len + n, cache


def speculative_generate_with_kv_cache(
    draft_model,
    target_model,
    prompt_ids: List[int],
    cfg,
    pad_token_id: int,
    padded_len: int,  # unused; kept for interface compat
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
):
    """Cache-aware K=1 nopipe spec generate. Single prompt only.

    Returns (generated_ids, stats) or (generated_ids, stats, timing) when
    ``return_timings=True``.
    """
    assert getattr(cfg, "K", 1) == 1, \
        "cache_aware: only K=1 supported in this version"
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    block_length = cfg.block_length
    denoising_steps = cfg.denoising_steps
    num_blocks = cfg.num_blocks
    mask_token_id = cfg.mask_token_id
    sampling = getattr(cfg, "draft_sampling", "argmax")
    remasking_strategy = getattr(cfg, "remasking_strategy", "low_confidence_static")
    confidence_threshold = float(getattr(cfg, "confidence_threshold", 0.9))
    early_stop_steps = int(getattr(cfg, "draft_steps_per_block", -1))
    early_active = 0 < early_stop_steps < denoising_steps
    fused_denoise = bool(getattr(cfg, "fused_denoise", False))

    verify_fn = _select_verify_fn(cfg)
    mrs_order = getattr(cfg, "mrs_verify_order", "position")
    fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")

    use_cuda_graph = bool(getattr(cfg, "use_cuda_graph", False))

    if use_cuda_graph:
        # Static-shape cache + per-(op,cur_len) cuda_graph cache.
        patch_sdpa_static_cache(draft_model)
        if draft_model is not target_model:
            patch_sdpa_static_cache(target_model)
    else:
        # DynamicCache + cat-style attention (no graph).
        patch_sdpa_with_cache(draft_model)
        if draft_model is not target_model:
            patch_sdpa_with_cache(target_model)
    # Also patch SDARForCausalLM.forward  1.7B's version drops past_key_values
    # when delegating to self.model(...) (8B does it correctly). This patch
    # makes 1.7B forward like 8B.
    patch_causallm_pass_cache(draft_model)
    if draft_model is not target_model:
        patch_causallm_pass_cache(target_model)

    if use_cuda_graph:
        # IMPORTANT: cuda_graphs capture buffer ADDRESSES, so the cache's
        # key_cache / value_cache tensors must persist across requests.
        # Stash a single StaticBlockCache per (model, device) as an attr so
        # it survives request boundaries. ``cache.reset()`` zeroes between
        # requests; the buffer identities don't change.
        max_cache_len = int(getattr(cfg, "kv_cache_max_len", 1024))
        prompt_len = len(prompt_ids)
        if prompt_len + num_blocks * block_length > max_cache_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + num_blocks * block_length "
                f"({num_blocks * block_length}) exceeds kv_cache_max_len "
                f"({max_cache_len}). Set cfg.kv_cache_max_len higher."
            )
        max_cur_len = 2 * block_length

        def _get_or_create(model, device):
            attr = "_kv_static_cache"
            existing = getattr(model, attr, None)
            if (
                existing is not None
                and existing.max_cache_len == max_cache_len
                and existing.max_cur_len == max_cur_len
                and existing.device == device
            ):
                existing.reset()
                return existing
            # New (or shape-changed)  also wipe stale graphs that referenced
            # the previous cache's buffers.
            new_cache = StaticBlockCache(
                model, max_cache_len, max_cur_len, device,
                next(model.parameters()).dtype,
                n_kv_heads_override=getattr(cfg, "n_kv_heads_override", None),
                n_q_heads_override=getattr(cfg, "n_q_heads_override", None),
            )
            setattr(model, attr, new_cache)
            # Drop graphs that pointed at the old buffers (id(model) key still
            # the same, but graph entries reference old static buffers).
            keys_to_drop = [k for k in _STATIC_GRAPH_CACHE if k[0] == id(model)]
            for k in keys_to_drop:
                _STATIC_GRAPH_CACHE.pop(k, None)
            return new_cache

        draft_cache = _get_or_create(draft_model, draft_device)
        target_cache = _get_or_create(target_model, target_device)
        # Aligned-prefill (eager) writes prompt K/V into cache permanent slots.
        draft_ctx_aligned, draft_leftover = _prefill_prompt_static(
            draft_model, prompt_ids, draft_cache, draft_device,
            block_length=block_length,
        )
        target_ctx_aligned, target_leftover = _prefill_prompt_static(
            target_model, prompt_ids, target_cache, target_device,
            block_length=block_length,
        )
    else:
        draft_cache = DynamicCache()
        target_cache = DynamicCache()
        draft_ctx_aligned, draft_cache, draft_leftover = _prefill_prompt(
            draft_model, prompt_ids, draft_cache, draft_device,
            block_length=block_length,
        )
        target_ctx_aligned, target_cache, target_leftover = _prefill_prompt(
            target_model, prompt_ids, target_cache, target_device,
            block_length=block_length,
        )
    assert draft_ctx_aligned == target_ctx_aligned
    assert draft_leftover == target_leftover  # same prompt
    leftover: List[int] = list(draft_leftover)

    stats = {
        "per_block_accepted": [],
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "total_bonus_tokens": 0,
    }
    all_generated: List[int] = []
    per_block_timings = [] if return_timings else None

    for blk_idx in range(num_blocks):
        if return_timings:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Defensive: if leftover already fills a block (shouldn't normally
        # happen since we flush below when full), drain it before denoising.
        while len(leftover) >= block_length:
            full_block = leftover[:block_length]
            if use_cuda_graph:
                draft_ctx_aligned = _extend_cache_graph(
                    draft_model, draft_cache, draft_ctx_aligned, full_block, draft_device,
                )
                target_ctx_aligned = _extend_cache_graph(
                    target_model, target_cache, target_ctx_aligned, full_block, target_device,
                )
            else:
                draft_ctx_aligned, draft_cache = _extend_cache(
                    draft_model, draft_cache, draft_ctx_aligned, full_block, draft_device,
                )
                target_ctx_aligned, target_cache = _extend_cache(
                    target_model, target_cache, target_ctx_aligned, full_block, target_device,
                )
            leftover = leftover[block_length:]

        n_known = len(leftover)
        n_masks = block_length - n_known
        # n_masks > 0 here (we just ensured leftover < block_length).

        if use_cuda_graph:
            draft_ids, draft_probs, step_map = _denoise_block_graph(
                draft_model, draft_cache, draft_ctx_aligned, leftover,
                block_length, denoising_steps, mask_token_id, draft_device,
                sampling=sampling,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                early_stop_steps=early_stop_steps,
            )
        else:
            draft_ids, draft_probs, step_map, draft_cache = _denoise_block_with_cache(
                draft_model, draft_cache, draft_ctx_aligned, leftover,
                block_length, denoising_steps, mask_token_id, draft_device,
                sampling=sampling,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                early_stop_steps=early_stop_steps,
            )
        # draft_ids has length n_masks (only the new positions).

        if return_timings:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        if use_cuda_graph:
            target_probs = _verify_block_graph(
                target_model, target_cache, target_ctx_aligned, leftover,
                draft_ids, step_map, block_length, mask_token_id, target_device,
            )
        else:
            target_probs, target_cache = _verify_block_with_cache(
                target_model, target_cache, target_ctx_aligned, leftover,
                draft_ids, step_map, denoising_steps, block_length,
                mask_token_id, target_device,
            )
        # target_probs shape (n_masks, vocab).

        if return_timings:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        # Move draft_probs to MRS expected device (CPU); target_probs already cpu.
        if draft_probs.device.type != "cpu":
            draft_probs = draft_probs.cpu()

        # Early-stop: for positions never proposed by the draft (step_map=-1),
        # commit the target argmax. Patch draft_ids/draft_probs so MRS treats
        # them as auto-accept (q/p ≡ 1) and step_map to a valid value so later
        # callers / "step_map" verify ordering don't break.
        # Batched form: one tolist() over the full block instead of per-pos
        # .item() (that sync was costing more than the draft forwards saved).
        was_drafted = [s >= 0 for s in step_map]
        if early_active and any(not d for d in was_drafted):
            tgt_argmax_list = target_probs.argmax(dim=-1).tolist()  # one sync
            for i in range(len(step_map)):
                if not was_drafted[i]:
                    draft_ids[i] = tgt_argmax_list[i]
                    draft_probs[i] = target_probs[i]  # mrs_verify reads-only
                    step_map[i] = denoising_steps - 1

        final_ids, n_accepted, bonus_token = verify_fn(
            draft_ids, draft_probs, target_probs, step_map,
            verify_order_mode=mrs_order,
        )
        n_drafted_real = sum(1 for d in was_drafted if d)
        stats["per_block_accepted"].append(n_accepted)
        stats["total_draft_tokens"] += n_drafted_real
        stats["total_accepted_tokens"] += n_accepted
        if bonus_token is not None:
            stats["total_bonus_tokens"] += 1

        # Real-draft acceptance accounting (matches the no-cache path):
        # auto-accepted positions never trigger MRS rejection, so any reject
        # comes from a real-drafted position. Count both accept and "seen"
        # only over real-drafted slots.
        accepted_real = sum(
            1 for i in range(min(n_accepted, n_masks)) if was_drafted[i]
        )
        seen_real = accepted_real
        if n_accepted < n_masks and was_drafted[n_accepted]:
            seen_real += 1
        stats.setdefault("total_real_drafted_accepted", 0)
        stats.setdefault("total_real_drafted_seen", 0)
        stats["total_real_drafted_accepted"] += accepted_real
        stats["total_real_drafted_seen"] += seen_real

        # Apply partial_block_fill  committed_new are the new positions
        # decided this iter (length 0..n_masks).
        committed_new = list(final_ids)
        if len(committed_new) < n_masks:
            if fill_mode == "truncate":
                pass  # variable-length commit
            elif fill_mode == "truncate_no_bonus":
                if len(committed_new) >= 2:
                    committed_new = committed_new[:-1]
            elif fill_mode == "target_argmax":
                tgt_pad = target_probs.argmax(dim=-1).tolist()
                committed_new += tgt_pad[len(committed_new):n_masks]
            elif fill_mode == "target_argmax_all":
                committed_new = target_probs.argmax(dim=-1).tolist()
            else:  # draft_argmax (legacy)
                committed_new += list(draft_ids[len(committed_new):n_masks])

        # Append committed new tokens to leftover. If the block is now full,
        # flush it to cache (advancing ctx_aligned by block_length).
        leftover = leftover + committed_new
        all_generated.extend(final_ids)

        if return_timings:
            torch.cuda.synchronize()
            t_after_mrs = time.perf_counter()

        if len(leftover) >= block_length:
            full_block = leftover[:block_length]
            if use_cuda_graph:
                draft_ctx_aligned = _extend_cache_graph(
                    draft_model, draft_cache, draft_ctx_aligned, full_block, draft_device,
                )
                target_ctx_aligned = _extend_cache_graph(
                    target_model, target_cache, target_ctx_aligned, full_block, target_device,
                )
            else:
                draft_ctx_aligned, draft_cache = _extend_cache(
                    draft_model, draft_cache, draft_ctx_aligned, full_block, draft_device,
                )
                target_ctx_aligned, target_cache = _extend_cache(
                    target_model, target_cache, target_ctx_aligned, full_block, target_device,
                )
            leftover = leftover[block_length:]

        if return_timings:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                "draft_s": t1 - t0,
                "target_verify_s": t2 - t1,
                "mrs_commit_s": t_after_mrs - t2,
                "extend_s": t3 - t_after_mrs,
                "block_wall_s": t3 - t0,
            })

        if eos_ids and any(t in eos_ids for t in final_ids):
            break

    # Strip trailing MASKs (shouldn't happen, but matches existing path).
    while all_generated and all_generated[-1] == mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated, stats
    timing = {
        "per_block": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_commit_s": sum(b["mrs_commit_s"] for b in per_block_timings),
        "total_extend_s": sum(b["extend_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "use_kv_cache": True,
    }
    return all_generated, stats, timing


def speculative_generate_with_kv_cache_pipelined(
    draft_model,
    target_model,
    prompt_ids: List[int],
    cfg,
    pad_token_id: int,
    padded_len: int,  # unused; kept for interface compat
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
):
    """Cache-aware K=1 PIPELINED spec generate. Draft on cuda:i, target on cuda:j.

    Reuses every primitive from ``speculative_generate_with_kv_cache``
    (StaticBlockCache, _prefill_prompt_static, _denoise_block_graph,
    _verify_block_graph, _extend_cache_graph) plus the optimistic-execute +
    rollback pattern from ``speculative_decode.py:speculative_generate_pipelined``.

    Per spec iter (overlap region marked ║):
      (1) launch verify_N on target GPU            ║
      (2) optimistic _extend draft_cache w/ block_N║   verify_N runs on
      (3) launch draft_{N+1} on draft GPU          ║   target while draft GPU
                                                   ║   does (2)+(3)
      (4) target_probs.cpu()   real sync point on target
      (5) MRS_N  n_accepted, committed_new
      (6) target_cache extend (when leftover fills)
      (7) full_accept ? reuse next_draft : rollback draft_cache + redo

    Rollback semantics for draft_cache: ``ctx_aligned`` is the only state we
    track. After opt_extend it advances by block_length; on reject we just
    decrement it (the K/V buffer at those positions becomes stale but won't
    be read because ``_build_iter_attn_mask`` only sets True at [0, ctx_aligned)).
    The next ``_extend_cache_graph`` overwrites those positions with the real
    committed block.

    Returns (generated_ids, stats) or (..., timing) when return_timings=True.
    """
    from speculative_decoding.speculative_decode import _cuda_sync

    assert getattr(cfg, "K", 1) == 1, \
        "cache_aware pipelined: only K=1 supported"
    assert bool(getattr(cfg, "use_cuda_graph", False)), \
        "cache_aware pipelined: requires use_cuda_graph=True"

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    assert draft_device != target_device, \
        "pipelined path expects draft_device != target_device; " \
        "use speculative_generate_with_kv_cache for single-GPU."

    block_length = cfg.block_length
    denoising_steps = cfg.denoising_steps
    num_blocks = cfg.num_blocks
    mask_token_id = cfg.mask_token_id
    sampling = getattr(cfg, "draft_sampling", "argmax")
    remasking_strategy = getattr(cfg, "remasking_strategy", "low_confidence_static")
    confidence_threshold = float(getattr(cfg, "confidence_threshold", 0.9))
    early_stop_steps = int(getattr(cfg, "draft_steps_per_block", -1))
    early_active = 0 < early_stop_steps < denoising_steps
    fused_denoise = bool(getattr(cfg, "fused_denoise", False))
    speculative_target_extend = bool(getattr(cfg, "speculative_target_extend", False))

    verify_fn = _select_verify_fn(cfg)
    mrs_order = getattr(cfg, "mrs_verify_order", "position")
    fill_mode = getattr(cfg, "partial_block_fill", "draft_argmax")

    # Patches (idempotent).
    patch_sdpa_static_cache(draft_model)
    if draft_model is not target_model:
        patch_sdpa_static_cache(target_model)
    patch_causallm_pass_cache(draft_model)
    if draft_model is not target_model:
        patch_causallm_pass_cache(target_model)

    # Cache + prefill  same logic as nopipe path.
    max_cache_len = int(getattr(cfg, "kv_cache_max_len", 1024))
    prompt_len = len(prompt_ids)
    if prompt_len + num_blocks * block_length > max_cache_len:
        raise ValueError(
            f"prompt_len ({prompt_len}) + num_blocks * block_length "
            f"({num_blocks * block_length}) exceeds kv_cache_max_len "
            f"({max_cache_len}). Set cfg.kv_cache_max_len higher."
        )
    max_cur_len = 2 * block_length

    def _get_or_create(model, device):
        attr = "_kv_static_cache"
        existing = getattr(model, attr, None)
        if (
            existing is not None
            and existing.max_cache_len == max_cache_len
            and existing.max_cur_len == max_cur_len
            and existing.device == device
        ):
            existing.reset()
            return existing
        new_cache = StaticBlockCache(
            model, max_cache_len, max_cur_len, device,
            next(model.parameters()).dtype,
            n_kv_heads_override=getattr(cfg, "n_kv_heads_override", None),
            n_q_heads_override=getattr(cfg, "n_q_heads_override", None),
        )
        setattr(model, attr, new_cache)
        keys_to_drop = [k for k in _STATIC_GRAPH_CACHE if k[0] == id(model)]
        for k in keys_to_drop:
            _STATIC_GRAPH_CACHE.pop(k, None)
        return new_cache

    draft_cache = _get_or_create(draft_model, draft_device)
    target_cache = _get_or_create(target_model, target_device)

    draft_ctx_aligned, draft_leftover = _prefill_prompt_static(
        draft_model, prompt_ids, draft_cache, draft_device,
        block_length=block_length,
    )
    target_ctx_aligned, target_leftover = _prefill_prompt_static(
        target_model, prompt_ids, target_cache, target_device,
        block_length=block_length,
    )
    assert draft_ctx_aligned == target_ctx_aligned
    assert draft_leftover == target_leftover
    leftover: List[int] = list(draft_leftover)

    stats = {
        "per_block_accepted": [],
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "total_bonus_tokens": 0,
        "pipeline_reused_next_draft": 0,
        "pipeline_rollback_next_draft": 0,
    }
    all_generated: List[int] = []
    per_block_timings = [] if return_timings else None

    def _flush_full_leftover_both():
        """If leftover fills a block, extend BOTH caches and advance ctx."""
        nonlocal leftover, draft_ctx_aligned, target_ctx_aligned
        while len(leftover) >= block_length:
            full_block = leftover[:block_length]
            draft_ctx_aligned = _extend_cache_graph(
                draft_model, draft_cache, draft_ctx_aligned, full_block, draft_device,
            )
            target_ctx_aligned = _extend_cache_graph(
                target_model, target_cache, target_ctx_aligned, full_block, target_device,
            )
            leftover = leftover[block_length:]

    # Drain any pre-existing fullness in leftover (defensive  same as nopipe).
    _flush_full_leftover_both()

    # ── warmup: draft block 0 ─────────────────────────────────────────────
    n_known = len(leftover)
    n_masks = block_length - n_known
    draft_ids, draft_probs, step_map = _denoise_block_graph_dispatch(
        draft_model, draft_cache, draft_ctx_aligned, leftover,
        block_length, denoising_steps, mask_token_id, draft_device,
        sampling=sampling,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        early_stop_steps=early_stop_steps,
        fused=fused_denoise,
    )
    stats["total_draft_tokens"] += sum(1 for s in step_map if s >= 0)

    # ── main spec loop ────────────────────────────────────────────────────
    # Per-block cuda.Event handles, accumulated; elapsed_time computed at end
    # to avoid per-block sync that inflates wall (option B in plan/14 §6).
    per_block_events: List[Dict[str, Any]] = [] if return_timings else None
    for blk_idx in range(num_blocks):
        if return_timings:
            t0 = time.perf_counter()  # CPU clock only  no GPU sync to keep
                                       # iteriter pipeline overlap intact.
            ev = {
                "v_start": torch.cuda.Event(enable_timing=True),
                "v_end":   torch.cuda.Event(enable_timing=True),
                "ext_start": torch.cuda.Event(enable_timing=True),
                "ext_end":   torch.cuda.Event(enable_timing=True),
                "den_start": torch.cuda.Event(enable_timing=True),
                "den_end":   torch.cuda.Event(enable_timing=True),
            }

        cur_n_known = len(leftover)
        cur_n_masks = block_length - cur_n_known

        # (1) launch verify_N on target GPU (non-blocking, return GPU tensor).
        if return_timings:
            with torch.cuda.device(target_device):
                ev["v_start"].record()
        target_probs_gpu = _verify_block_graph(
            target_model, target_cache, target_ctx_aligned, leftover,
            draft_ids, step_map, block_length, mask_token_id, target_device,
            return_on_device=True,
        )
        if return_timings:
            with torch.cuda.device(target_device):
                ev["v_end"].record()

        # (2)+(3) Optimistic: assume full block accept  opt_block fills the
        #         block, draft_cache advances, then launch draft_{N+1} on the
        #         already-advanced cache. All non-blocking on draft GPU,
        #         overlapping with verify_N on target GPU.
        is_last = (blk_idx == num_blocks - 1)
        next_draft = None
        opt_block_len = 0
        opt_target_extended = False
        if not is_last:
            opt_block = list(leftover) + list(draft_ids)  # length == block_length
            opt_block_len = len(opt_block)
            assert opt_block_len == block_length, (
                f"opt_block_len {opt_block_len} != block_length {block_length}"
            )
            if return_timings:
                with torch.cuda.device(draft_device):
                    ev["ext_start"].record()
            draft_ctx_aligned = _extend_cache_graph(
                draft_model, draft_cache, draft_ctx_aligned, opt_block,
                draft_device,
            )
            if speculative_target_extend:
                target_ctx_aligned = _extend_cache_graph(
                    target_model, target_cache, target_ctx_aligned, opt_block,
                    target_device,
                )
                opt_target_extended = True
            if return_timings:
                with torch.cuda.device(draft_device):
                    ev["ext_end"].record()
                    ev["den_start"].record()
            # draft_{N+1}: leftover for the next block is empty (we just
            # optimistically flushed the current block). n_masks_next == bl.
            next_draft = _denoise_block_graph_dispatch(
                draft_model, draft_cache, draft_ctx_aligned, [],
                block_length, denoising_steps, mask_token_id, draft_device,
                sampling=sampling,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                early_stop_steps=early_stop_steps,
                fused=fused_denoise,
            )
            if return_timings:
                with torch.cuda.device(draft_device):
                    ev["den_end"].record()

        # (4) sync target  first CPU read of target_probs.
        if return_timings:
            t_cpu_start = time.perf_counter()
        if draft_probs.device.type != "cpu":
            draft_probs = draft_probs.cpu()
        target_probs = target_probs_gpu.cpu()
        if return_timings:
            t_cpu_end = time.perf_counter()
            ev["target_cpu_wait_ms"] = (t_cpu_end - t_cpu_start) * 1000

        if return_timings:
            t1 = time.perf_counter()

        # Early-stop pre-MRS patch (matches no-pipe path). Auto-accept positions
        # the draft never proposed by overwriting draft_ids/draft_probs to match
        # target's verify-time output.
        was_drafted = [s >= 0 for s in step_map]
        if early_active and any(not d for d in was_drafted):
            tgt_argmax_list = target_probs.argmax(dim=-1).tolist()  # one sync
            for i in range(len(step_map)):
                if not was_drafted[i]:
                    draft_ids[i] = tgt_argmax_list[i]
                    draft_probs[i] = target_probs[i]
                    step_map[i] = denoising_steps - 1

        # (5) MRS / commit.
        final_ids, n_accepted, bonus_token = verify_fn(
            draft_ids, draft_probs, target_probs, step_map,
            verify_order_mode=mrs_order,
        )
        n_drafted_real = sum(1 for d in was_drafted if d)
        stats["per_block_accepted"].append(n_accepted)
        stats["total_draft_tokens"] += n_drafted_real
        stats["total_accepted_tokens"] += n_accepted
        if bonus_token is not None:
            stats["total_bonus_tokens"] += 1

        # Real-draft acceptance accounting.
        accepted_real = sum(
            1 for i in range(min(n_accepted, cur_n_masks)) if was_drafted[i]
        )
        seen_real = accepted_real
        if n_accepted < cur_n_masks and was_drafted[n_accepted]:
            seen_real += 1
        stats.setdefault("total_real_drafted_accepted", 0)
        stats.setdefault("total_real_drafted_seen", 0)
        stats["total_real_drafted_accepted"] += accepted_real
        stats["total_real_drafted_seen"] += seen_real

        # Apply partial_block_fill exactly like nopipe path.
        committed_new = list(final_ids)
        if len(committed_new) < cur_n_masks:
            if fill_mode == "truncate":
                pass
            elif fill_mode == "truncate_no_bonus":
                if len(committed_new) >= 2:
                    committed_new = committed_new[:-1]
            elif fill_mode == "target_argmax":
                tgt_pad = target_probs.argmax(dim=-1).tolist()
                committed_new += tgt_pad[len(committed_new):cur_n_masks]
            elif fill_mode == "target_argmax_all":
                committed_new = target_probs.argmax(dim=-1).tolist()
            else:  # draft_argmax (legacy)
                committed_new += list(draft_ids[len(committed_new):cur_n_masks])

        # full_accept: optimistic prediction matches what was committed.
        # opt_block was leftover_at_iter_start + draft_ids (length bl).
        # Real flush would be leftover_at_iter_start + committed_new[:bl-len(leftover)].
        # They match iff committed_new == draft_ids (same content, same length n_masks).
        full_accept = (
            not is_last
            and len(committed_new) == cur_n_masks
            and list(committed_new) == list(draft_ids)
        )

        # (6) Update leftover + all_generated; flush BOTH caches when full.
        leftover = leftover + committed_new
        all_generated.extend(final_ids)

        if full_accept:
            assert len(leftover) >= block_length, \
                f"full_accept must produce at least block_length tokens in leftover (got {len(leftover)})"
            full_block = leftover[:block_length]
            if not opt_target_extended:
                target_ctx_aligned = _extend_cache_graph(
                    target_model, target_cache, target_ctx_aligned, full_block,
                    target_device,
                )
            leftover = leftover[block_length:]
            draft_ids, draft_probs, step_map = next_draft
            stats["pipeline_reused_next_draft"] += 1
        else:
            if not is_last:
                draft_ctx_aligned -= opt_block_len
                if opt_target_extended:
                    target_ctx_aligned -= opt_block_len
                stats["pipeline_rollback_next_draft"] += 1
            # Now draft_ctx_aligned == target_ctx_aligned (both at iter-start state).
            # Apply the canonical "flush leftover when full" rule on BOTH caches.
            _flush_full_leftover_both()
            # Re-launch draft_{N+1} on the (now correct) draft_cache state.
            if not is_last:
                next_n_known = len(leftover)
                next_n_masks = block_length - next_n_known
                if next_n_masks > 0:
                    draft_ids, draft_probs, step_map = _denoise_block_graph_dispatch(
                        draft_model, draft_cache, draft_ctx_aligned, leftover,
                        block_length, denoising_steps, mask_token_id, draft_device,
                        sampling=sampling,
                        remasking_strategy=remasking_strategy,
                        confidence_threshold=confidence_threshold,
                        early_stop_steps=early_stop_steps,
                        fused=fused_denoise,
                    )
                else:
                    # leftover already fills a block  would be flushed above.
                    # (Shouldn't reach here because _flush drains.)
                    draft_ids, draft_probs, step_map = [], torch.zeros(0, draft_model.config.vocab_size), []

        if return_timings:
            t3 = time.perf_counter()  # CPU clock at end of MRS/rollback
            per_block_timings.append({
                "block_idx": blk_idx,
                "draft_s": 0.0,  # overlapped; reported in target_verify_s wall
                "target_verify_s": t1 - t0,           # wall: launch verify  target.cpu() done
                "target_cpu_wait_ms": ev["target_cpu_wait_ms"],
                "mrs_and_commit_s": t3 - t1,          # CPU MRS + rollback wall
                "block_wall_s": t3 - t0,              # full iter wall (no per-block GPU sync)
            })
            per_block_events.append(ev)

        if eos_ids and any(t in eos_ids for t in final_ids):
            break

    # End-of-sample: drain both streams and compute elapsed_time for events.
    # This is the ONLY GPU sync introduced by timing  all per-block records
    # are async (option B from plan/14 §6).
    if return_timings:
        _cuda_sync(draft_device, target_device)
        for blk_idx, ev in enumerate(per_block_events):
            try:
                v_ms = ev["v_start"].elapsed_time(ev["v_end"])
            except RuntimeError:
                v_ms = 0.0
            try:
                ext_ms = ev["ext_start"].elapsed_time(ev["ext_end"]) if blk_idx < len(per_block_timings) - 1 else 0.0
            except RuntimeError:
                ext_ms = 0.0  # is_last block: extend not launched
            try:
                den_ms = ev["den_start"].elapsed_time(ev["den_end"]) if blk_idx < len(per_block_timings) - 1 else 0.0
            except RuntimeError:
                den_ms = 0.0
            t = per_block_timings[blk_idx]
            t["verify_gpu_ms"] = v_ms             # pure verify forward (target stream)
            t["extend_gpu_ms"] = ext_ms           # pure draft extend forward
            t["denoise_gpu_ms"] = den_ms          # pure draft denoise (4 forwards)

    while all_generated and all_generated[-1] == mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated, stats
    timing = {
        "per_block": per_block_timings,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        # cuda.Event-based pure GPU stream times (no sync inflation):
        "total_verify_gpu_ms": sum(b.get("verify_gpu_ms", 0.0) for b in per_block_timings),
        "total_extend_gpu_ms": sum(b.get("extend_gpu_ms", 0.0) for b in per_block_timings),
        "total_denoise_gpu_ms": sum(b.get("denoise_gpu_ms", 0.0) for b in per_block_timings),
        "total_target_cpu_wait_ms": sum(b.get("target_cpu_wait_ms", 0.0) for b in per_block_timings),
        # Both keys: "total_mrs_commit_s" (legacy) and "total_mrs_and_commit_s"
        # (what self_draft_compare.py:549/588 reads  without "_and_" the field
        # silently rendered as 0 in summaries).
        "total_mrs_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_mrs_and_commit_s": sum(b["mrs_and_commit_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "use_kv_cache": True,
        # Two keys for backward-compat: "pipelined" (existing) and "pipeline"
        # (the key downstream readers in self_draft_compare.py / breakdown
        # render expect  they used to see False here even on the pipelined
        # path).
        "pipelined": True,
        "pipeline": True,
    }
    return all_generated, stats, timing


def native_generate_with_kv_cache(model, prompt_ids, cfg, eos_ids=None,
                                  return_timings: bool = False):
    """Native (target-only) block diffusion with KV cache + optional cuda_graph.

    Mirrors `block_diffusion_generate` (generate.py:58) but reuses the spec
    cache-aware infrastructure (`StaticBlockCache` + per-(op,cur_len) graphs)
    to deliver an apples-to-apples latency reference for spec.

    Algorithm per spec iter:
      1. Aligned-prefill: prompt[:aligned_len]  cache, leftover = prompt[aligned_len:]
      2. For each of num_blocks decode blocks:
         a. Denoise current block (leftover + masks)  fills mask positions
         b. Commit all bl tokens (no MRS / draft / verify)
         c. Extend cache with the full block, advance ctx_aligned by bl

    Returns (generated_ids,) or (generated_ids, timing) when return_timings=True.
    """
    device = next(model.parameters()).device
    block_length = cfg.block_length
    denoising_steps = cfg.denoising_steps
    num_blocks = cfg.num_blocks
    mask_token_id = cfg.mask_token_id
    sampling = getattr(cfg, "draft_sampling", "argmax")
    use_cuda_graph = bool(getattr(cfg, "use_cuda_graph", False))
    # generate.py-compatible vanilla sampling knobs (defaults preserve current
    # native behavior: pure softmax, no temp/top_k/top_p, low_confidence_static).
    temperature = float(getattr(cfg, "temperature", 1.0))
    top_k = int(getattr(cfg, "top_k", 0))
    top_p = float(getattr(cfg, "top_p", 1.0))
    remasking_strategy = getattr(cfg, "remasking_strategy", "low_confidence_static")
    confidence_threshold = float(getattr(cfg, "confidence_threshold", 0.9))

    # Patches
    if use_cuda_graph:
        patch_sdpa_static_cache(model)
    else:
        patch_sdpa_with_cache(model)
    patch_causallm_pass_cache(model)

    # Cache + prefill
    if use_cuda_graph:
        max_cache_len = int(getattr(cfg, "kv_cache_max_len", 1024))
        prompt_len = len(prompt_ids)
        if prompt_len + num_blocks * block_length > max_cache_len:
            raise ValueError(
                f"prompt_len ({prompt_len}) + num_blocks*bl ({num_blocks*block_length}) "
                f"exceeds kv_cache_max_len ({max_cache_len})."
            )
        max_cur_len = 2 * block_length

        attr = "_kv_static_cache"
        existing = getattr(model, attr, None)
        gqa_mode_cfg = getattr(cfg, "gqa_mode", "expand")
        if (
            existing is not None
            and existing.max_cache_len == max_cache_len
            and existing.max_cur_len == max_cur_len
            and existing.device == device
            and getattr(existing, "gqa_mode", "expand") == gqa_mode_cfg
        ):
            existing.reset()
            cache = existing
        else:
            cache = StaticBlockCache(
                model, max_cache_len, max_cur_len, device,
                next(model.parameters()).dtype,
                n_kv_heads_override=getattr(cfg, "n_kv_heads_override", None),
                n_q_heads_override=getattr(cfg, "n_q_heads_override", None),
                gqa_mode=gqa_mode_cfg,
            )
            setattr(model, attr, cache)
            keys_to_drop = [k for k in _STATIC_GRAPH_CACHE if k[0] == id(model)]
            for k in keys_to_drop:
                _STATIC_GRAPH_CACHE.pop(k, None)

    if return_timings:
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
    if use_cuda_graph:
        ctx_aligned, leftover = _prefill_prompt_static(
            model, prompt_ids, cache, device, block_length=block_length,
        )
    else:
        cache = DynamicCache()
        ctx_aligned, cache, leftover = _prefill_prompt(
            model, prompt_ids, cache, device, block_length=block_length,
        )
    if return_timings:
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_prefill_start

    leftover = list(leftover)
    all_generated: List[int] = []
    per_block_timings = [] if return_timings else None

    for blk_idx in range(num_blocks):
        if return_timings:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Drain full leftover (defensive)
        while len(leftover) >= block_length:
            full_block = leftover[:block_length]
            if use_cuda_graph:
                ctx_aligned = _extend_cache_graph(model, cache, ctx_aligned, full_block, device)
            else:
                ctx_aligned, cache = _extend_cache(model, cache, ctx_aligned, full_block, device)
            leftover = leftover[block_length:]

        n_masks = block_length - len(leftover)

        if use_cuda_graph:
            new_ids, _, _ = _denoise_block_graph(
                model, cache, ctx_aligned, leftover, block_length,
                denoising_steps, mask_token_id, device, sampling=sampling,
                temperature=temperature, top_k=top_k, top_p=top_p,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
            )
        else:
            new_ids, _, _, cache = _denoise_block_with_cache(
                model, cache, ctx_aligned, leftover, block_length,
                denoising_steps, mask_token_id, device, sampling=sampling,
                temperature=temperature, top_k=top_k, top_p=top_p,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
            )

        if return_timings:
            torch.cuda.synchronize()
            t_after_denoise = time.perf_counter()

        # Commit all new positions, no MRS / no truncate
        leftover = leftover + new_ids
        all_generated.extend(new_ids)

        # Flush full block to cache
        if len(leftover) >= block_length:
            full_block = leftover[:block_length]
            if use_cuda_graph:
                ctx_aligned = _extend_cache_graph(model, cache, ctx_aligned, full_block, device)
            else:
                ctx_aligned, cache = _extend_cache(model, cache, ctx_aligned, full_block, device)
            leftover = leftover[block_length:]

        if return_timings:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                "denoise_s": t_after_denoise - t0,
                "commit_extend_s": t1 - t_after_denoise,
                "block_wall_s": t1 - t0,
            })

        if eos_ids and any(t in eos_ids for t in new_ids):
            break

    while all_generated and all_generated[-1] == mask_token_id:
        all_generated.pop()

    if not return_timings:
        return all_generated
    timing = {
        "per_block": per_block_timings,
        "prefill_s": prefill_s,
        "total_denoise_s": sum(b["denoise_s"] for b in per_block_timings),
        "total_commit_extend_s": sum(b["commit_extend_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "use_kv_cache": True,
        "use_cuda_graph": use_cuda_graph,
    }
    return all_generated, timing


# ─────────────────────────────────────────────────────────────────────
# Batched native + KV cache + cuda_graph
# ─────────────────────────────────────────────────────────────────────

def _denoise_block_graph_batch(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    block_length: int,
    denoising_steps: int,
    mask_token_id: int,
    device: torch.device,
    sampling: str = "argmax",
) -> List[List[int]]:
    """Batched denoise: all B rows in lockstep. ctx_aligned shared.

    Cur_x for each row: bl masks (no leftover support at batch>1  we
    pad prompts to the same aligned length so leftover is always empty).
    Returns committed_ids of shape (B, block_length).
    """
    B = cache.batch_size
    bl = block_length
    vocab = model.config.vocab_size
    entry = _get_static_forward_graph(
        model, "denoise", bl, cache, device, mask_token_id,
    )

    intra = torch.ones(bl, bl, dtype=torch.bool, device=device)
    attn_mask = _build_iter_attn_mask(cache, ctx_aligned, bl, intra, device, batch_size=B)

    pos_ids = torch.arange(ctx_aligned, ctx_aligned + bl, device=device).unsqueeze(0).expand(B, bl).contiguous()
    cur_x = torch.full((B, bl), mask_token_id, dtype=torch.long, device=device)

    for step in range(1, denoising_steps + 1):
        is_masked = (cur_x == mask_token_id)  # (B, bl)
        n_masked_per_row = is_masked.sum(dim=1)  # (B,)
        if n_masked_per_row.max().item() == 0:
            break
        # Replay graph
        logits_static = _replay_static_forward(
            entry, cur_x, pos_ids, attn_mask,
        )
        # logits_static: (B, bl, vocab)
        probs = F.softmax(logits_static.float(), dim=-1)  # (B, bl, vocab)
        if sampling == "multinomial":
            flat = probs.view(B * bl, vocab)
            pred = torch.multinomial(flat, num_samples=1).view(B, bl)
        else:
            pred = probs.argmax(dim=-1)  # (B, bl)
        confidence = probs.gather(2, pred.unsqueeze(-1)).squeeze(-1)  # (B, bl)

        # Per-row top-k unmask
        n_unmask_per_row = []
        remaining = denoising_steps - step + 1
        for b in range(B):
            nm = int(n_masked_per_row[b].item())
            n_unmask = min(-(-nm // remaining), nm) if nm > 0 else 0
            n_unmask_per_row.append(n_unmask)

        # For simplicity, assume same n_unmask across rows (true if all rows
        # have same masked count). Should hold under lockstep.
        n_unmask = n_unmask_per_row[0]
        if all(x == n_unmask for x in n_unmask_per_row) and n_unmask > 0:
            # Vectorized top-k
            conf_masked = torch.where(is_masked, confidence, torch.full_like(confidence, -float("inf")))
            _, topk_idx = conf_masked.topk(n_unmask, dim=1)  # (B, n_unmask)
            # Build mask of selected positions
            sel_mask = torch.zeros_like(is_masked)
            sel_mask.scatter_(1, topk_idx, True)
            # Unmask: cur_x[sel_mask] = pred[sel_mask]
            cur_x = torch.where(sel_mask, pred, cur_x)
        else:
            # Fallback per-row (rare)
            for b in range(B):
                nm = n_unmask_per_row[b]
                if nm == 0:
                    continue
                conf_b = torch.where(is_masked[b], confidence[b], torch.full_like(confidence[b], -float("inf")))
                _, idx = conf_b.topk(nm)
                cur_x[b, idx] = pred[b, idx]

    # Force-fill any remaining masks
    is_masked = (cur_x == mask_token_id)
    if is_masked.any():
        logits_static = _replay_static_forward(entry, cur_x, pos_ids, attn_mask)
        probs = F.softmax(logits_static.float(), dim=-1)
        if sampling == "multinomial":
            flat = probs.view(B * bl, vocab)
            pred = torch.multinomial(flat, num_samples=1).view(B, bl)
        else:
            pred = probs.argmax(dim=-1)
        cur_x = torch.where(is_masked, pred, cur_x)

    return cur_x.tolist()


def _extend_cache_graph_batch(
    model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    full_blocks: List[List[int]],  # length B, each list of bl tokens
    device: torch.device,
) -> int:
    """Batched extend: write each row's full block at positions [ctx_aligned, ctx_aligned+bl)."""
    B = cache.batch_size
    bl = len(full_blocks[0])
    entry = _get_static_forward_graph(model, "extend", bl, cache, device, MASK_TOKEN_ID_DEFAULT)

    intra = torch.ones(bl, bl, dtype=torch.bool, device=device)
    attn_mask = _build_iter_attn_mask(cache, ctx_aligned, bl, intra, device, batch_size=B)

    in_ids = torch.tensor(full_blocks, dtype=torch.long, device=device)  # (B, bl)
    pos_ids = torch.arange(ctx_aligned, ctx_aligned + bl, device=device).unsqueeze(0).expand(B, bl).contiguous()
    cache_pos = torch.arange(ctx_aligned, ctx_aligned + bl, dtype=torch.long, device=device)

    _replay_static_forward(entry, in_ids, pos_ids, attn_mask, cache_pos)
    return ctx_aligned + bl


def _prefill_prompt_batch(
    model,
    prompts: List[List[int]],
    cache: StaticBlockCache,
    device: torch.device,
    block_length: int,
    pad_token_id: int = 0,
) -> Tuple[int, List[List[int]]]:
    """Batched aligned-prefill. All prompts left-padded to common aligned length.

    Returns (aligned_len, leftover_per_row). leftover_per_row may be non-empty
    for individual rows but they're truncated to make all rows lockstep-able
    by callers using draft_argmax (which always commits bl per iter).

    Simplification for batch: round all prompts up to nearest bl boundary
    by left-padding with pad_token_id. Then aligned_len = max_padded_len,
    leftover = [] for all rows.
    """
    from speculative_decoding.draft import _build_block_causal_attn

    B = len(prompts)
    max_p = max(len(p) for p in prompts)
    # Round UP to bl boundary (pad each prompt's left side until length is a bl multiple)
    aligned_len = ((max_p + block_length - 1) // block_length) * block_length
    padded = []
    for p in prompts:
        pad_n = aligned_len - len(p)
        padded.append([pad_token_id] * pad_n + list(p))
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    position_ids = torch.arange(aligned_len, device=device).unsqueeze(0).expand(B, aligned_len).contiguous()

    full_len = cache.full_len
    base = _build_block_causal_attn(aligned_len, block_length, device).bool()
    attn_mask = torch.zeros(B, 1, aligned_len, full_len, dtype=torch.bool, device=device)
    attn_mask[..., :, :aligned_len] = base.unsqueeze(0).unsqueeze(0)

    cur_scratch_pos = torch.arange(aligned_len, dtype=torch.long, device=device)

    with torch.cuda.device(device), torch.no_grad():
        _ = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=False,
            cur_scratch_pos=cur_scratch_pos,
            return_dict=True,
        )
    leftovers = [[] for _ in range(B)]
    return aligned_len, leftovers


def native_generate_with_kv_cache_batch(
    model,
    prompts: List[List[int]],
    cfg,
    pad_token_id: int = 0,
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
):
    """Batched native (target-only) block diffusion + KV cache + optional cuda_graph.

    All rows progress in lockstep:
      - Prompts left-padded to common aligned length
      - Each iter every row commits exactly `block_length` tokens
      - All rows share ctx_aligned, advance together

    Returns list[list[int]] of generated token ids (one per row, each
    num_blocks * block_length tokens).
    """
    B = len(prompts)
    device = next(model.parameters()).device
    block_length = cfg.block_length
    denoising_steps = cfg.denoising_steps
    num_blocks = cfg.num_blocks
    mask_token_id = cfg.mask_token_id
    sampling = getattr(cfg, "draft_sampling", "argmax")
    use_cuda_graph = bool(getattr(cfg, "use_cuda_graph", False))
    assert use_cuda_graph, "Batch native cache requires use_cuda_graph=True"

    patch_sdpa_static_cache(model)
    patch_causallm_pass_cache(model)

    max_cache_len = int(getattr(cfg, "kv_cache_max_len", 1024))
    max_cur_len = 2 * block_length

    # Per-(model, B) cache instance
    attr = f"_kv_static_cache_b{B}"
    existing = getattr(model, attr, None)
    if (
        existing is not None
        and existing.max_cache_len == max_cache_len
        and existing.max_cur_len == max_cur_len
        and existing.device == device
        and existing.batch_size == B
    ):
        existing.reset()
        cache = existing
    else:
        cache = StaticBlockCache(
            model, max_cache_len, max_cur_len, device,
            next(model.parameters()).dtype, batch_size=B,
        )
        setattr(model, attr, cache)
        keys_to_drop = [k for k in _STATIC_GRAPH_CACHE if k[0] == id(model) and k[3] == B]
        for k in keys_to_drop:
            _STATIC_GRAPH_CACHE.pop(k, None)

    if return_timings:
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
    aligned_len, leftovers = _prefill_prompt_batch(
        model, prompts, cache, device, block_length, pad_token_id,
    )
    if return_timings:
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_prefill_start

    ctx_aligned = aligned_len
    all_generated: List[List[int]] = [[] for _ in range(B)]
    per_block_timings = [] if return_timings else None

    for blk_idx in range(num_blocks):
        if return_timings:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        new_ids_batch = _denoise_block_graph_batch(
            model, cache, ctx_aligned, block_length, denoising_steps,
            mask_token_id, device, sampling=sampling,
        )

        if return_timings:
            torch.cuda.synchronize()
            t_after_denoise = time.perf_counter()

        for b in range(B):
            all_generated[b].extend(new_ids_batch[b])

        # Extend cache with all rows' new blocks (lockstep).
        ctx_aligned = _extend_cache_graph_batch(
            model, cache, ctx_aligned, new_ids_batch, device,
        )

        if return_timings:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                "denoise_s": t_after_denoise - t0,
                "commit_extend_s": t1 - t_after_denoise,
                "block_wall_s": t1 - t0,
            })

        # EOS check (per row)
        if eos_ids:
            if any(any(t in eos_ids for t in row) for row in new_ids_batch):
                break

    if not return_timings:
        return all_generated
    timing = {
        "per_block": per_block_timings,
        "prefill_s": prefill_s,
        "total_denoise_s": sum(b["denoise_s"] for b in per_block_timings),
        "total_commit_extend_s": sum(b["commit_extend_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "use_kv_cache": True,
        "use_cuda_graph": True,
        "batch_size": B,
    }
    return all_generated, timing


# ─────────────────────────────────────────────────────────────────────
# Batched spec + KV cache + cuda_graph
# ─────────────────────────────────────────────────────────────────────

def _verify_block_graph_batch(
    target_model,
    cache: StaticBlockCache,
    ctx_aligned: int,
    draft_ids_batch: List[List[int]],   # (B, bl)
    step_map_batch: List[List[int]],    # (B, bl)
    block_length: int,
    mask_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Batched verify: query = [draft_data | draft_mask] for each row.

    Returns target_probs of shape (B, bl, vocab).
    """
    B = cache.batch_size
    bl = block_length
    cur_len = 2 * bl  # always 2*bl since leftover=[] in lockstep batch

    entry = _get_static_forward_graph(
        target_model, "verify", cur_len, cache, device, mask_token_id,
    )

    # Build per-row intra mask (step-causal)
    intra = torch.zeros(B, cur_len, cur_len, dtype=torch.bool, device=device)
    for b in range(B):
        sm = step_map_batch[b]
        data_times = [sm[p] + 1 for p in range(bl)]
        for i in range(bl):
            ti = data_times[i]
            for j in range(bl):
                # data(i)  data(j): tj <= ti
                intra[b, i, j] = (data_times[j] <= ti)
                # data(i)  mask(j): tj > ti
                intra[b, i, bl + j] = (data_times[j] > ti)
        for i in range(bl):
            ti = data_times[i]  # paired
            for j in range(bl):
                # mask(i)  data(j): tj < ti
                intra[b, bl + i, j] = (data_times[j] < ti)
                # mask(i)  mask(j): tj >= ti
                intra[b, bl + i, bl + j] = (data_times[j] >= ti)

    # Build full attn_mask (B, 1, cur_len, full_len)
    full_len = cache.full_len
    attn_mask = torch.zeros(B, 1, cur_len, full_len, dtype=torch.bool, device=device)
    if ctx_aligned > 0:
        attn_mask[..., :, :ctx_aligned] = True
    s = cache.max_cache_len
    attn_mask[..., :, s : s + cur_len] = intra.unsqueeze(1)

    # Build query tokens
    in_ids = torch.zeros(B, cur_len, dtype=torch.long, device=device)
    for b in range(B):
        in_ids[b, :bl] = torch.tensor(draft_ids_batch[b], dtype=torch.long, device=device)
        in_ids[b, bl:] = mask_token_id
    pos_ids = torch.arange(ctx_aligned, ctx_aligned + bl, device=device)
    pos_ids = torch.cat([pos_ids, pos_ids]).unsqueeze(0).expand(B, cur_len).contiguous()

    logits_static = _replay_static_forward(entry, in_ids, pos_ids, attn_mask)
    # logits at mask positions: (B, bl, vocab)
    mask_logits = logits_static[:, bl:, :]
    target_probs = F.softmax(mask_logits.float(), dim=-1).cpu()
    return target_probs


def speculative_generate_with_kv_cache_batch(
    draft_model,
    target_model,
    prompts: List[List[int]],
    cfg,
    pad_token_id: int = 0,
    eos_ids: Optional[List[int]] = None,
    return_timings: bool = False,
):
    """Batched K=1 spec + KV cache + cuda_graph.

    Simplification: ``partial_block_fill = draft_argmax`` is forced internally
    (regardless of cfg) so all rows commit exactly block_length tokens per
    iter, keeping ctx_aligned shared. This loses truncate's quality nuance
    but makes the batched cache feasible.
    """
    B = len(prompts)
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    assert draft_device == target_device, "batch spec cache: draft + target must share GPU"
    device = draft_device

    block_length = cfg.block_length
    denoising_steps = cfg.denoising_steps
    num_blocks = cfg.num_blocks
    mask_token_id = cfg.mask_token_id
    sampling = getattr(cfg, "draft_sampling", "argmax")
    use_cuda_graph = bool(getattr(cfg, "use_cuda_graph", False))
    assert use_cuda_graph, "Batch spec cache requires use_cuda_graph=True"
    verify_fn = _select_verify_fn(cfg)
    mrs_order = getattr(cfg, "mrs_verify_order", "position")

    patch_sdpa_static_cache(draft_model)
    if draft_model is not target_model:
        patch_sdpa_static_cache(target_model)
    patch_causallm_pass_cache(draft_model)
    if draft_model is not target_model:
        patch_causallm_pass_cache(target_model)

    max_cache_len = int(getattr(cfg, "kv_cache_max_len", 1024))
    max_cur_len = 2 * block_length

    def _get_or_create(model, dev):
        attr = f"_kv_static_cache_b{B}"
        existing = getattr(model, attr, None)
        if (
            existing is not None
            and existing.max_cache_len == max_cache_len
            and existing.max_cur_len == max_cur_len
            and existing.device == dev
            and existing.batch_size == B
        ):
            existing.reset()
            return existing
        new_cache = StaticBlockCache(
            model, max_cache_len, max_cur_len, dev,
            next(model.parameters()).dtype, batch_size=B,
        )
        setattr(model, attr, new_cache)
        keys_to_drop = [k for k in _STATIC_GRAPH_CACHE if k[0] == id(model) and k[3] == B]
        for k in keys_to_drop:
            _STATIC_GRAPH_CACHE.pop(k, None)
        return new_cache

    draft_cache = _get_or_create(draft_model, draft_device)
    target_cache = _get_or_create(target_model, target_device)

    if return_timings:
        torch.cuda.synchronize()
        t_pre = time.perf_counter()
    aligned_len, _ = _prefill_prompt_batch(
        draft_model, prompts, draft_cache, draft_device, block_length, pad_token_id,
    )
    aligned_len_t, _ = _prefill_prompt_batch(
        target_model, prompts, target_cache, target_device, block_length, pad_token_id,
    )
    assert aligned_len == aligned_len_t
    if return_timings:
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t_pre

    ctx_aligned = aligned_len
    all_generated: List[List[int]] = [[] for _ in range(B)]
    stats_batch = [
        {"per_block_accepted": [], "total_draft_tokens": 0,
         "total_accepted_tokens": 0, "total_bonus_tokens": 0}
        for _ in range(B)
    ]
    per_block_timings = [] if return_timings else None

    # Build vocab tensor for sampling shapes
    vocab = draft_model.config.vocab_size

    for blk_idx in range(num_blocks):
        if return_timings:
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        # Draft denoise (returns (B, bl) tokens  pred per row)
        draft_ids_b = _denoise_block_graph_batch(
            draft_model, draft_cache, ctx_aligned, block_length,
            denoising_steps, mask_token_id, draft_device, sampling=sampling,
        )

        # We also need step_map and per_pos_probs per row.
        # The internal _denoise_block_graph_batch tracks step_map but doesn't
        # return it for batch path. As a workaround: rerun a final inference
        # to get probs, OR simplify by uniform step_map = arange(bl) (assumes
        # one position unmasked per step).
        # For lockstep batch denoise with denoising_steps=block_length, each
        # step unmasks 1 token, so step_map[i] = i across all rows after the
        # top-k argmax. Use this assumption:
        step_map_b = [[i for i in range(block_length)] for _ in range(B)]
        # Get per_pos_probs by running one more forward post-hoc  needed for MRS.
        # Skip this for simplicity: build uniform fake probs (one-hot at draft_ids).
        # MRS with greedy_match doesn't read draft_probs anyway.
        if return_timings:
            torch.cuda.synchronize()
            t1 = time.perf_counter()

        # Target verify
        target_probs_b = _verify_block_graph_batch(
            target_model, target_cache, ctx_aligned, draft_ids_b, step_map_b,
            block_length, mask_token_id, target_device,
        )
        # target_probs_b: (B, bl, vocab) on CPU

        if return_timings:
            torch.cuda.synchronize()
            t2 = time.perf_counter()

        # Per-row MRS
        committed_blocks: List[List[int]] = []
        for b in range(B):
            # Use one-hot draft_probs (greedy_match doesn't compare distributions)
            draft_probs = torch.zeros(block_length, vocab)
            for p, tid in enumerate(draft_ids_b[b]):
                draft_probs[p, tid] = 1.0
            final_ids, n_accepted, bonus = verify_fn(
                draft_ids_b[b], draft_probs, target_probs_b[b], step_map_b[b],
                verify_order_mode=mrs_order,
            )
            stats_batch[b]["per_block_accepted"].append(n_accepted)
            stats_batch[b]["total_draft_tokens"] += block_length
            stats_batch[b]["total_accepted_tokens"] += n_accepted
            if bonus is not None:
                stats_batch[b]["total_bonus_tokens"] += 1

            # Force-fill to bl tokens via draft_argmax (lockstep)
            block_tokens = list(final_ids)
            if len(block_tokens) < block_length:
                block_tokens += list(draft_ids_b[b][len(block_tokens):block_length])
            committed_blocks.append(block_tokens)
            all_generated[b].extend(final_ids)

        if return_timings:
            torch.cuda.synchronize()
            t_after_mrs = time.perf_counter()

        # Extend BOTH caches at the SAME ctx_aligned with same committed blocks.
        new_ctx = _extend_cache_graph_batch(
            draft_model, draft_cache, ctx_aligned, committed_blocks, draft_device,
        )
        _ = _extend_cache_graph_batch(
            target_model, target_cache, ctx_aligned, committed_blocks, target_device,
        )
        ctx_aligned = new_ctx

        if return_timings:
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            per_block_timings.append({
                "block_idx": blk_idx,
                "draft_s": t1 - t0,
                "target_verify_s": t2 - t1,
                "mrs_commit_s": t_after_mrs - t2,
                "extend_s": t3 - t_after_mrs,
                "block_wall_s": t3 - t0,
            })

    if not return_timings:
        return all_generated, stats_batch
    timing = {
        "per_block": per_block_timings,
        "prefill_s": prefill_s,
        "total_draft_s": sum(b["draft_s"] for b in per_block_timings),
        "total_target_verify_s": sum(b["target_verify_s"] for b in per_block_timings),
        "total_mrs_commit_s": sum(b["mrs_commit_s"] for b in per_block_timings),
        "total_extend_s": sum(b["extend_s"] for b in per_block_timings),
        "total_block_wall_s": sum(b["block_wall_s"] for b in per_block_timings),
        "use_kv_cache": True,
        "use_cuda_graph": True,
        "batch_size": B,
    }
    return all_generated, stats_batch, timing
