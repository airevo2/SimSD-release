"""LLaDA2 (inclusionAI, MoE block-diffusion) adapter.

Verified against ``inclusionAI/LLaDA2.0-mini`` remote code
(``modeling_llada2_moe.py``, transformers 4.57 era, imports resolve on our
pinned 4.52.4 given the existing ``TransformersKwargs`` compat shim).

The decode semantics match SDAR closely enough that nothing above the patch
layer changes: block-level tril mask (bidirectional inside a block, causal
across blocks), per-block iterative denoising, confidence-threshold unmasking,
``use_cache=False`` full recompute. What differs is naming and plumbing:

  =====================  ==============================  ==========================
  concept                SDAR                            LLaDA2
  =====================  ==============================  ==========================
  attention module       ``layer.self_attn``             ``layer.attention``
  q/k/v projection       ``q_proj``/``k_proj``/``v_proj`` fused ``query_key_value``
  q/k norm               ``q_norm`` / ``k_norm``         ``query_layernorm`` /
                                                          ``key_layernorm``
  output projection      ``o_proj``                      ``dense``
  RoPE                   full head_dim                   ``partial_rotary_factor``
                                                          = 0.5 (handled inside
                                                          their apply_rotary)
  MASK token             151669                          156895
  =====================  ==============================  ==========================

Two plumbing problems have to be patched out, both in the path between
``model(...)`` and the attention forward:

  1. ``LLaDA2MoeModel.forward`` accepts *only* a ``(B, 1, S, S)`` attention mask
     and pushes it through ``_prepare_4d_causal_attention_mask_for_sdpa``,
     raising otherwise. Our masks are ``(B, 1, cur_len, full_len)`` bool with
     cur_len ≠ full_len (query = current block, key = cache + block), so the
     upstream check rejects them outright.
  2. Neither ``LLaDA2MoeModel.forward`` nor ``LLaDA2MoeDecoderLayer.forward``
     forwards ``**kwargs`` down to the attention call, so our ``store_kv`` /
     ``cur_scratch_pos`` never arrive and the K/V cache silently does nothing.

Both are fixed by replacing those two forwards (see ``patch_plumbing``).
"""

from __future__ import annotations

import torch

from .base import ModelAdapter, rebind_hooked_forwards

#: LLaDA2's block-diffusion MASK placeholder — the default of ``mask_id`` in
#: ``LLaDA2MoeModelLM.generate``. Shared by mini and flash (same tokenizer).
MASK_TOKEN_ID = 156895

#: ``eos_id`` default in the same generate(); also the config's pad_token_id.
EOS_TOKEN_ID = 156892


class Llada2Adapter(ModelAdapter):
    family = "llada2"
    attn_class_names = ("LLaDA2MoeAttention",)
    mask_token_id = MASK_TOKEN_ID
    # Capturability is a property of the *instance*, not the family: the stock
    # moe_infer syncs on the host, the fused grouped-GEMM dispatch does not.
    # See graph_blocker() below.
    supports_cuda_graph = True
    cuda_graph_blocker = ""

    def graph_blocker(self, model):
        from kernels import fused_toggle
        if fused_toggle.is_moe_applied(model):
            return None
        return (
            "LLaDA2MoeSparseMoeBlock.moe_infer does tokens_per_expert.cpu()"
            ".numpy() and loops over the active experts in Python — that host "
            "sync forbids torch.cuda.graph capture. Install the grouped-GEMM "
            "dispatch first: kernels.fused_toggle.apply_moe_to_model(model), "
            "or run eager (use_cuda_graph: false)."
        )

    def qkv(self, mod, hidden_states):
        bsz, q_len, _ = hidden_states.shape
        n_q, n_kv = mod.num_heads, mod.num_key_value_heads
        # One fused projection → split into q / k / v along the head axis.
        qkv = mod.query_key_value(hidden_states).view(
            bsz, q_len, n_q + 2 * n_kv, mod.head_dim,
        )
        q, k, v = qkv.split([n_q, n_kv, n_kv], dim=-2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Upstream norms after the transpose; RMSNorm is over head_dim so the
        # order is irrelevant numerically, but keep it identical anyway.
        if mod.config.use_qk_norm:
            q = mod.query_layernorm(q)
            k = mod.key_layernorm(k)
        return q, k, v

    def out_proj(self, mod):
        return mod.dense

    def patch_plumbing(self, model) -> int:
        inner = getattr(model, "model", None)
        if inner is None:
            return 0
        n = _patch_model_forward(inner.__class__)
        rebind_hooked_forwards(model, inner.__class__)
        if getattr(inner, "layers", None):
            layer_cls = inner.layers[0].__class__
            n += _patch_decoder_layer_forward(layer_cls)
            # A sharded target has an accelerate hook on every decoder layer,
            # and those shadow the class patch above. Without the rebind the
            # stock layer forward runs, drops our store_kv kwarg, and the stock
            # attention then grows the K/V cache on every denoise step.
            rebind_hooked_forwards(model, layer_cls)
        return n


def _patch_model_forward(cls) -> int:
    """Replace ``LLaDA2MoeModel.forward``.

    Two deliberate deviations from upstream:

      * ``attention_mask`` is passed to the layers **verbatim**. We build a bool
        ``(B, 1, cur_len, full_len)`` mask ourselves and hand it to SDPA; the
        upstream ``(B, 1, S, S)`` assertion plus
        ``_prepare_4d_causal_attention_mask_for_sdpa`` would reject it and then
        re-derive causality we do not want.
      * ``**kwargs`` reaches the decoder layers, so ``store_kv`` /
        ``cur_scratch_pos`` / ``cache_position`` survive the trip down.

    Dropped along the way: gradient checkpointing, ``output_attentions``,
    ``output_hidden_states``, ``output_router_logits``. None are used on any
    inference path here, and keeping them would mean keeping the branches that
    make the mask handling ambiguous. The returned cache is the object the
    caller passed in — our attention forwards mutate it in place
    (``index_copy_`` for StaticBlockCache, ``update`` for DynamicCache).
    """
    if getattr(cls, "_simsd_model_forward_patched", False):
        return 0
    from transformers.modeling_outputs import MoeModelOutputWithPast

    def _forward(self, input_ids=None, attention_mask=None, position_ids=None,
                 past_key_values=None, inputs_embeds=None, use_cache=None,
                 output_attentions=None, output_hidden_states=None,
                 output_router_logits=None, return_dict=None,
                 cache_position=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "cannot specify both input_ids and inputs_embeds"
            )
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = inputs_embeds

        if position_ids is None:
            past_seen = (past_key_values.get_seq_length()
                         if past_key_values is not None else 0)
            position_ids = torch.arange(
                past_seen, past_seen + hidden_states.shape[1],
                device=hidden_states.device,
            ).unsqueeze(0)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=bool(use_cache),
                position_embeddings=position_embeddings,
                cache_position=cache_position,
                **kwargs,
            )[0]

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    cls.forward = _forward
    cls._simsd_model_forward_patched = True
    return 1


def _patch_decoder_layer_forward(cls) -> int:
    """Replace ``LLaDA2MoeDecoderLayer.forward`` so ``**kwargs`` reaches
    ``self.attention``, and so the 2-tuple our patched attention returns is
    accepted (upstream unpacks three values).
    """
    if getattr(cls, "_simsd_layer_forward_patched", False):
        return 0

    def _forward(self, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, output_attentions=False,
                 output_router_logits=False, use_cache=False,
                 position_embeddings=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.attention(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + attn_out[0]

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # MoE blocks may return (hidden, router_logits); dense MLPs return a
        # bare tensor. The first ``first_k_dense_replace`` layers are dense.
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states.to(residual.device)
        return (hidden_states,)

    cls.forward = _forward
    cls._simsd_layer_forward_patched = True
    return 1
