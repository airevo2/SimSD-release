"""SDAR adapter — the family the pipeline was originally written against.

Every hook here is a transcription of what ``cache_aware.py`` did inline before
the adapter split, so SDAR runs must stay bit-identical.
"""

from __future__ import annotations

import functools

from .base import ModelAdapter, rebind_hooked_forwards

#: SDAR's block-diffusion MASK placeholder (``<|mask|>`` in the checkpoints'
#: added_tokens.json). Both SDAR-1.7B-Chat and SDAR-8B-Chat agree on it.
MASK_TOKEN_ID = 151669


class SdarAdapter(ModelAdapter):
    family = "sdar"
    attn_class_names = ("SDARAttention",)
    mask_token_id = MASK_TOKEN_ID
    supports_cuda_graph = True

    def qkv(self, mod, hidden_states):
        hidden_shape = (*hidden_states.shape[:-1], -1, mod.head_dim)
        q = mod.q_norm(mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = mod.k_norm(mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return q, k, v

    def out_proj(self, mod):
        return mod.o_proj

    def patch_plumbing(self, model) -> int:
        n = _patch_causallm_pass_cache(model)
        # The root module carries an accelerate hook when the model is sharded,
        # which would shadow the class patch above.
        rebind_hooked_forwards(model, model.__class__)
        return n


def _patch_causallm_pass_cache(model) -> int:
    """1.7B's ``SDARForCausalLM.forward`` (modeling_sdar.py:849) passes neither
    past_key_values nor use_cache when delegating to ``self.model(...)``. That
    means an internally-allocated DynamicCache is used per forward and our
    accumulated cache is silently dropped. 8B's version (8B modeling_sdar.py:
    819-830) does pass them correctly.

    Patch: rebind ``SDARForCausalLM.forward`` so the internal ``self.model``
    call forwards past_key_values + use_cache + inputs_embeds. Idempotent per
    class.
    """
    cls = model.__class__
    if cls.__name__ != "SDARForCausalLM":
        return 0
    if getattr(cls, "_cache_passthrough_patched", False):
        return 0

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
            import torch
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
    return 1
