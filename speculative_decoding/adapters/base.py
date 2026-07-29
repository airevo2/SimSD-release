"""Adapter base class + the family-blind attention forwards built on top of it.

SimSD's algorithm layer is architecture agnostic: every attention mask is built
by us, every ``position_ids`` is passed in explicitly, and the K/V buffers are
sized from ``model.config``. The only thing that genuinely differs between
model families is

  * how one attention module names and shapes its q/k/v projections,
  * which submodule is the output projection, and
  * whether the surrounding plumbing (``*ForCausalLM.forward`` /
    ``*Model.forward`` / ``*DecoderLayer.forward``) drops the kwargs and the
    4D bool mask we need to reach the attention.

An adapter answers exactly those. The three ``make_*_forward`` factories below
then assemble the real forwards once, so the cache-write / SDPA / GQA logic
lives in a single place regardless of family.
"""

from __future__ import annotations

import sys
from typing import Tuple

import torch
import torch.nn.functional as F
from einops import rearrange


class ModelAdapter:
    """One model family's naming conventions. Stateless; instantiate freely."""

    #: short name used in YAML (``family:``) and in log lines
    family: str = "?"

    #: attention module class names to look for when detecting / patching
    attn_class_names: Tuple[str, ...] = ()

    #: the family's MASK placeholder token id (block-diffusion sentinel)
    mask_token_id: int = -1

    #: True when the family's forward can be captured into a cuda_graph.
    #: False disables the graph paths with an explanation instead of failing
    #: deep inside capture.
    supports_cuda_graph: bool = True

    #: Why not, when ``supports_cuda_graph`` is False.
    cuda_graph_blocker: str = ""

    # ── required hooks ────────────────────────────────────────────────
    def qkv(self, mod, hidden_states: torch.Tensor):
        """Project ``hidden_states`` to (q, k, v), each ``(B, H, L, head_dim)``.

        Post q/k-norm, pre-RoPE — i.e. exactly the point where RoPE is applied
        upstream. ``q`` carries H_q heads, ``k``/``v`` carry H_kv heads.
        """
        raise NotImplementedError

    def out_proj(self, mod) -> torch.nn.Module:
        """The attention output projection submodule."""
        raise NotImplementedError

    # ── optional hooks ────────────────────────────────────────────────
    def patch_plumbing(self, model) -> int:
        """Patch whatever sits between ``model(...)`` and the attention forward
        so that our 4D bool mask and our ``store_kv`` / ``cur_scratch_pos``
        kwargs actually arrive. Returns the number of classes patched.
        """
        return 0

    def rotary(self, attn_cls):
        """The family's ``apply_rotary_pos_emb(q, k, cos, sin)``.

        Both supported families expose it as a module-level function next to
        the attention class, so resolving it from the defining module is the
        general case.
        """
        return sys.modules[attn_cls.__module__].apply_rotary_pos_emb

    # ── shared helpers ────────────────────────────────────────────────
    def find_attn_class(self, model):
        """First attention class in ``model`` matching ``attn_class_names``."""
        wanted = set(self.attn_class_names)
        for m in model.modules():
            if m.__class__.__name__ in wanted:
                return m.__class__
        return None

    def matches(self, model) -> bool:
        return self.find_attn_class(model) is not None

    def graph_blocker(self, model):
        """Why this *model instance* cannot be cuda_graph captured, or None.

        Instance-level rather than family-level because for LLaDA2 the answer
        depends on whether the fused MoE dispatch has been installed: the stock
        ``moe_infer`` syncs on the host, the grouped-GEMM one does not. A class
        attribute could not express that.
        """
        if not self.supports_cuda_graph:
            return self.cuda_graph_blocker
        return None


def rebind_hooked_forwards(model, cls) -> int:
    """Re-point accelerate's saved forward at our patched one.

    Needed whenever a model is loaded with ``device_map=`` (i.e. a target
    sharded across GPUs). ``accelerate.hooks.add_hook_to_module`` installs a
    per-*instance* forward and stashes the one it replaced::

        module._old_forward = module.forward     # bound at load time
        module.forward      = new_forward        # calls module._old_forward

    Both happen inside ``from_pretrained``, before we patch anything. The
    instance attribute then shadows the class attribute, so a later
    ``cls.forward = ...`` is never reached — the hook keeps calling the original
    function it captured. Symptom on LLaDA2: the stock attention calls
    ``past_key_value.update()`` unconditionally, so the K/V cache grows on every
    denoise step (64 → 96 → 128) until it stops matching our mask.

    Re-binding ``_old_forward`` to the current ``cls.forward`` puts our version
    back in the chain while keeping the hook itself — and the hook is what
    moves each forward's inputs onto that shard's device for us.

    Call this on **every** patch call, not just the one that patches the class.
    The class patches are idempotent, so an unsharded draft patched first would
    otherwise leave a sharded target's instances still pointing at the original.
    """
    fn = cls.forward
    n = 0
    for m in model.modules():
        if not isinstance(m, cls):
            continue
        old = getattr(m, "_old_forward", None)
        if old is None or getattr(old, "__func__", None) is fn:
            continue
        m._old_forward = fn.__get__(m, cls)
        n += 1
    return n


def is_sharded(model) -> bool:
    """True when the model is spread over more than one device by accelerate."""
    dm = getattr(model, "hf_device_map", None)
    if not dm:
        return False
    return len({str(v) for v in dm.values()}) > 1


# ─────────────────────────────────────────────────────────────────────
# Forward factories
# ─────────────────────────────────────────────────────────────────────
# All three share the same contract with the caller:
#   - ``attention_mask`` is a bool tensor we built; True = visible. It is
#     handed straight to SDPA, never re-derived.
#   - ``position_embeddings`` is the (cos, sin) pair the outer model computed
#     from the ``position_ids`` we passed in.
#   - ``store_kv`` (kwarg) decides whether this forward's K/V is persisted.
# The bodies are transcriptions of the pre-adapter SDAR versions; the only
# change is that q/k/v and the output projection come from the adapter.


def make_static_cache_forward(adapter: ModelAdapter, attn_cls):
    """Forward against ``StaticBlockCache``'s fixed-shape buffers (cuda_graph).

    Expects, via kwargs:
      - ``past_key_value``: a ``StaticBlockCache``
      - ``cur_scratch_pos``: LongTensor of scratch slots for this forward's K/V
      - ``cache_position``: permanent slots to also write when ``store_kv``
      - ``attention_mask``: bool ``(1, 1, cur_len, full_len)``
    """
    apply_rotary_pos_emb = adapter.rotary(attn_cls)

    def _forward(self, hidden_states, position_embeddings, attention_mask,
                 past_key_value=None, cache_position=None, **kwargs):
        q, k_new, v_new = adapter.qkv(self, hidden_states)

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
                # GQA-expand on write: H_q-wide cache, k_new strided across
                # n_rep slots.
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
        # In "expand" mode head counts match → SDPA picks EFFICIENT_ATTENTION
        # for free (~1.8× faster than MATH GQA on Blackwell).
        # In "native" mode K/V is H_kv-wide → enable_gqa=True so Q broadcasts;
        # SDPA falls back to MATH (mirror of the eager cache forward, kept
        # available so we can ablate the GQA-expansion optimization).
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
        attn_output = adapter.out_proj(self)(attn_output)
        return attn_output, None

    return _forward


def make_eager_cache_forward(adapter: ModelAdapter, attn_cls):
    """Forward against a ``DynamicCache`` (no cuda_graph, growing K/V)."""
    apply_rotary_pos_emb = adapter.rotary(attn_cls)

    def _forward(self, hidden_states, position_embeddings, attention_mask,
                 past_key_value=None, cache_position=None, **kwargs):
        q, k, v = adapter.qkv(self, hidden_states)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # store_kv=True → persist this forward's K/V; otherwise read-only
        # concat so the current block can attend to the cached prefix without
        # polluting it.
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
        attn_output = adapter.out_proj(self)(attn_output)
        return attn_output, None

    return _forward


def make_eval_forward(adapter: ModelAdapter, attn_cls):
    """Cache-free SDPA forward for the legacy no-KV-cache draft/verify path.

    Why it exists: the upstream eval paths dispatch between flash-attn and SDPA
    on ``torch.all(attention_mask)``, which forces a GPU→CPU sync (forbidding
    graph capture) and is dead logic here — our block-causal mask is never
    all-ones. The training path is preserved bit-for-bit where the family has
    one.
    """
    apply_rotary_pos_emb = adapter.rotary(attn_cls)
    fused_flex_attention = getattr(
        sys.modules[attn_cls.__module__], "fused_flex_attention", None)

    def _forward(self, hidden_states, position_embeddings, attention_mask,
                 past_key_value=None, cache_position=None, **kwargs):
        q, k, v = adapter.qkv(self, hidden_states)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.training and fused_flex_attention is not None:
            # Preserve training path bit-for-bit.
            attn_output, attn_weights = fused_flex_attention(
                query=q, key=k, value=v,
                attention_mask=attention_mask,
                enable_gqa=True, scale=self.scaling, return_lse=True,
            )
            attn_weights = attn_weights.to(v.dtype) if attn_weights is not None else None
            attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
            return adapter.out_proj(self)(attn_output), attn_weights

        mask_bool = attention_mask.bool() if attention_mask is not None else None
        attn_output = F.scaled_dot_product_attention(
            query=q, key=k, value=v,
            attn_mask=mask_bool,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=True,
        )
        attn_output = rearrange(attn_output, "b h l d -> b l (h d)")
        return adapter.out_proj(self)(attn_output), None

    return _forward
