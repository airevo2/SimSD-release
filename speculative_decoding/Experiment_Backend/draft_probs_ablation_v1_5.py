"""Draft-probs ablation v1.5  one extra forward with intra_md (step-causal)
intra-block rule, matching 4D-mask's rules exactly on the part v1 got wrong.

v1 used ``_build_block_causal_attn`` (block-level lower-triangular, intra-block
bidirectional) for the extra forward  the intra-block region is fully visible,
which aggregates information from ALL peers regardless of step_map ordering.

Target's 4D-mask rule for a mask position i in its own block is ``intra_md``:
the mask at position i sees own-block DATA at positions j where ``step_j <
step_i``. (Plus ``intra_mm`` which lets it see peer mask positions at steps
``>= step_i``  we cannot exactly simulate this in a data-only single-pass
forward, but intra_md captures the dominant causal dependency.)

v1.5 keeps v1's cheap single-extra-forward shape but overrides the intra-block
region to exactly this step-causal rule, using the draft's step_map.

Cost: one extra forward per drafted block, same as v1 (~10% of draft time on
the bench prompts). Much cheaper than v3 (which doubles draft-side verify).

T0 instrumentation: per-call CUDA-event breakdown is appended to
``_V15_TIMINGS`` (``{denoising_ms, extra_forward_ms, softmax_cpu_ms}``). The
bench wrapper (``bench_v1_5.py``) flushes it to ``{out_dir}/v1_5_timings.json``
after the run.

Usage:
  CUDA_VISIBLE_DEVICES=6 python draft_probs_ablation_v1_5.py \\
      --dataset gsm8k --num_samples 40 --branch mrs \\
      --target_model inference/model/SDAR-8B-Chat --skip_cross
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding import draft as _draft_mod  # noqa: E402
from speculative_decoding import speculative_decode as _sd_mod  # noqa: E402
from speculative_decoding.draft import _build_block_causal_attn  # noqa: E402

MASK_TOKEN_ID = 151669

_orig_draft_one_block = _draft_mod.draft_one_block

# Module-level per-call timing list. Each entry = dict with keys:
#   denoising_ms, extra_forward_ms, softmax_cpu_ms, ctx_len, block_length
# The bench wrapper reads / flushes this list; other callers ignore it.
_V15_TIMINGS: List[dict] = []


def _draft_one_block_v1_5(
    model,
    context_ids,
    block_length,
    denoising_steps,
    device,
    mask_token_id=MASK_TOKEN_ID,
    step_callback=None,
    use_cuda_graph: bool = False,
    sampling: str = "argmax",
) -> Tuple[List[int], torch.Tensor, List[int]]:
    """draft_one_block + overwrite per_pos_probs with a single fully-decoded
    forward, using the step-causal intra-block rule (matches 4D-mask intra_md)
    instead of v1's intra-block bidirectional.
    """
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
        sampling=sampling,
    )

    if is_cuda:
        with torch.cuda.device(device):
            evt_t1.record()

    ctx_len = len(context_ids)
    seq = list(context_ids) + list(draft_ids)
    seq_len = len(seq)
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)

    attn_mask = _build_block_causal_attn(seq_len, block_length, device).clone()
    block_start = ctx_len
    sm = torch.as_tensor(step_map, dtype=torch.long, device=device)
    # intra_md: position i attends to position j in same block iff step_j < step_i.
    step_causal = (sm[None, :] < sm[:, None]).to(attn_mask.dtype)
    attn_mask[0, block_start:block_start + block_length,
              block_start:block_start + block_length] = step_causal

    with torch.cuda.device(device), torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attn_mask)
        logits = outputs.logits[0, ctx_len:ctx_len + block_length]

    if is_cuda:
        with torch.cuda.device(device):
            evt_t2.record()

    vocab_size = model.config.vocab_size
    per_pos_probs_final = F.softmax(logits.float(), dim=-1).cpu()
    assert per_pos_probs_final.shape == (block_length, vocab_size)

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
    """Snapshot + clear per-call timing records. Call between / after bench runs."""
    out = list(_V15_TIMINGS)
    _V15_TIMINGS.clear()
    return out


_draft_mod.draft_one_block = _draft_one_block_v1_5
_sd_mod.draft_one_block = _draft_one_block_v1_5
print("[ablation v1.5] patched draft.draft_one_block  one extra forward with "
      "intra_md step-causal rule (matches 4D-mask intra_md exactly); CUDA-event "
      "timings recorded into _V15_TIMINGS.",
      flush=True)


from speculative_decoding.Experiment_Backend import self_draft_compare  # noqa: E402

if __name__ == "__main__":
    if "--tag" not in sys.argv:
        sys.argv.extend(["--tag", "draft_probs_v1_5_intra_md"])
    self_draft_compare.main()
