#!/usr/bin/env python3
"""Pure-forward latency sweep for ONE SDAR model across (block_length, denoising_steps).

For each grid point, times the model in BOTH roles it plays in speculative decoding:
  * draft  role  standard block-causal mask, one forward per denoising step,
                  run num_blocks times (matches draft.draft_one_block shape)
  * target role  multi-block causal mask verify forward, once per committed block
                  (matches verify.build_verify_sequence + model.forward)

model_forward_ms is timed via CUDA events (top-level forward hook). Everything
else (mask construction, input materialization, sampling) is wall-clocked via
StageAcc with sync_after=True at stage boundaries.

NOTE: the `breakdown` block in the output is a **diagnostic / bottleneck
attribution** view  it answers "where did the time go?", not "what is the real
serving latency?". Because sync_after=True serializes async CPU↔GPU overlap,
the summed stages are conservative (slower than a well-pipelined run). Do not
use these numbers as a headline throughput figure.

Self-contained: only depends on speculative_decoding.draft._build_block_causal_attn
and speculative_decoding.verify.{build_verify_sequence, patch_multi_block_mask_fn}.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    os.path.join(os.path.expanduser("~"), ".flashinfer_local"),
)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from speculative_decoding.draft import _build_block_causal_attn  # noqa: E402
from speculative_decoding.verify import (  # noqa: E402
    build_verify_sequence,
    patch_multi_block_mask_fn,
)

MASK_TOKEN_ID = 151669


def _percentile(vs: List[float], p: float) -> float:
    if not vs:
        return float("nan")
    s = sorted(vs)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"n": 0}
    return {
        "n": len(times_ms),
        "mean_ms": round(statistics.mean(times_ms), 4),
        "p50_ms": round(_percentile(times_ms, 50), 4),
        "p95_ms": round(_percentile(times_ms, 95), 4),
        "min_ms": round(min(times_ms), 4),
        "max_ms": round(max(times_ms), 4),
    }


class StageAcc:
    """Per-iter wall-clock stage accumulator.

    Use as a companion to HFForwardTimer: model_forward_ms stays on CUDA events
    for GPU-accurate timing, while everything else (mask build, Python input
    construction, H2D tensor materialization, sampling) is wall-clocked here so
    we can tell whether non-model time dominates an iteration.

    ``sync_after=True`` issues a cuda sync at the stage's trailing edge so any
    GPU work launched inside the stage (e.g. small tensor allocs, H2D copies)
    is attributed to this stage rather than leaking into the next one.
    """

    def __init__(self):
        self.sums: Dict[str, float] = {}

    def add(self, name: str, ms: float):
        self.sums[name] = self.sums.get(name, 0.0) + ms

    @contextmanager
    def stage(self, name: str, sync_after: bool = False):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_after:
                torch.cuda.synchronize()
            self.sums[name] = self.sums.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0


def summarize_breakdown(iters: List[Dict[str, float]],
                        keys: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        out[k] = summarize([d.get(k, 0.0) for d in iters])
    return out


_BREAKDOWN_NOTE = (
    "Diagnostic / bottleneck attribution breakdown  answers 'where did the "
    "time go?'. Stages are wall-clocked with sync_after=True, which serializes "
    "CPU↔GPU overlap and is conservative. model_forward_ms is from CUDA events. "
    "Do NOT use these numbers as serving latency."
)


def build_breakdown_output(iters: List[Dict[str, float]], keys: List[str],
                           n_forwards_per_run: int) -> Dict[str, Any]:
    """Produce diagnostic breakdown with both per-run and per-forward views.

    per_run_breakdown_ms: summary stats across iters for each stage.
    per_forward_breakdown_ms: per-run mean divided by n_forwards_per_run,
      most useful for model_forward_ms (= one forward's mean latency).
    """
    per_run = summarize_breakdown(iters, keys)
    denom = max(1, n_forwards_per_run)
    per_forward = {k: round(per_run[k].get("mean_ms", 0.0) / denom, 4) for k in keys}
    return {
        "per_run_breakdown_ms": per_run,
        "per_forward_breakdown_ms": per_forward,
        "n_forwards_per_run": n_forwards_per_run,
        "notes": _BREAKDOWN_NOTE,
    }


class HFForwardTimer:
    """Top-level forward hook  CUDA-event pairs per model.forward() call."""

    def __init__(self, model):
        self.starts: List[torch.cuda.Event] = []
        self.ends: List[torch.cuda.Event] = []
        self._h_pre = model.register_forward_pre_hook(self._pre)
        self._h_post = model.register_forward_hook(self._post)

    def _pre(self, _mod, _inp):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.starts.append(e)

    def _post(self, _mod, _inp, _out):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ends.append(e)

    def collect(self) -> List[float]:
        torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in zip(self.starts, self.ends)]

    def reset(self):
        self.starts.clear()
        self.ends.clear()

    def close(self):
        self._h_pre.remove()
        self._h_post.remove()


def hf_block_causal_run(model, batch, prompt_len, block_length, denoising_steps,
                        num_blocks, mask_token_id=MASK_TOKEN_ID,
                        acc: Optional[StageAcc] = None):
    """Simulate draft_one_block's forward shape: per block, run denoising_steps
    forwards with a block-causal mask; advance context by random "accepted" tokens.

    If ``acc`` is provided, wall-clock time is split into
    ``mask_build_ms`` / ``sampling_ms`` stages. The model forward itself is NOT
    wrapped  it is timed via CUDA events on HFForwardTimer.
    """
    if acc is None:
        acc = StageAcc()
    device = next(model.parameters()).device
    vocab = model.config.vocab_size
    context = torch.randint(0, vocab, (batch, prompt_len), device=device, dtype=torch.long)

    for _blk in range(num_blocks):
        with acc.stage("mask_build_ms", sync_after=True):
            mask_block = torch.full((batch, block_length), mask_token_id,
                                    device=device, dtype=torch.long)
            input_ids = torch.cat([context, mask_block], dim=1)
            seq_len = input_ids.shape[1]
            attn = _build_block_causal_attn(seq_len, block_length, device)

        for _step in range(denoising_steps):
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)

        with acc.stage("sampling_ms", sync_after=True):
            new_tokens = torch.randint(0, vocab, (batch, block_length),
                                       device=device, dtype=torch.long)
            context = torch.cat([context, new_tokens], dim=1)


def hf_verify_run(target_model, batch, prompt_len, block_length, num_blocks,
                  block_size, padded_len, mask_token_id=MASK_TOKEN_ID,
                  acc: Optional[StageAcc] = None,
                  use_eval_sdpa: bool = False):
    """Simulate target verify forward shape: per block, build a
    [prompt | accepted_data+mask | draft_data+mask] sequence and forward once
    with multi-block causal mask. Reuses build_verify_sequence.

    If ``acc`` is provided, wall-clock time is split into
    ``verify_seq_build_ms`` (pure Python: build_verify_sequence + list padding)
    and ``tensor_materialize_and_h2d_ms`` (torch.tensor creation + H2D copy +
    contiguous). The model forward is NOT wrapped here  it is timed via
    CUDA events.

    ``use_eval_sdpa`` (opt-in) routes the target forward through
    `SDARAttention`'s eval branch  `F.scaled_dot_product_attention`, the same
    kernel path that draft already uses. Rationale:
    - Default path forces `model.train()` so that `SDARForCausalLM.forward`'s
      training branch builds a `BlockMask` via `create_block_mask(lambda...)`
      and dispatches into `fused_flex_attention`  but without `torch.compile`
      that lands on the eager `_flex_attention_hop` fallback (~3.3 ms/layer
      at S=256 on A800).
    - In eval mode, `SDARForCausalLM.forward` forwards whatever `attention_mask`
      the caller supplies straight through `SDARModel.forward` (line 1033) to
      `SDARAttention.forward`, which hits `F.scaled_dot_product_attention`
      (same path as draft) as long as the mask isn't all-True.
    - So we build the 4D multi-block bool mask ourselves (via
      `new_attn_multi_block.create_multi_block_causal_mask`) and pass it via
      `attention_mask=`. We deliberately do NOT pass `block_ids` in this mode
      because that's the signal the training branch uses to build its own mask;
      we still pass `token_labels` because `SDARModel.forward` needs it to copy
      RoPE from `data` positions to paired `mask` positions (line 1007–1021).
    - `labels=None` because we're only timing forward, not CE loss.
    """
    if acc is None:
        acc = StageAcc()
    device = next(target_model.parameters()).device
    prompt_ids = list(range(1, prompt_len + 1))
    accepted_blocks: List[List[int]] = []

    orig_bs = target_model.config.block_size
    target_model.config.block_size = block_size
    if use_eval_sdpa:
        # Eval path: skip the training branch entirely. No `.train()` toggle,
        # no `patch_multi_block_mask_fn` dependency, no `create_block_mask`.
        target_model.eval()
    else:
        target_model.train()  # required by patched multi-block mask code path

    # Lazy import to keep module-import time minimal and avoid pulling
    # new_attn_multi_block into the import graph when use_eval_sdpa is off.
    if use_eval_sdpa:
        from new_attn_multi_block import create_multi_block_causal_mask as _mbc_mask_fn
    try:
        for _blk in range(num_blocks):
            with acc.stage("verify_seq_build_ms"):
                draft_ids = [100] * block_length
                # step_map entries must be in [0, block_size-1] so that token_labels
                # (= step+1) stay within [1, block_size]; otherwise multi-block mask
                # indexing asserts when block_length > block_size (e.g. bl=6,ds=4).
                step_map = [min(i, block_size - 1) for i in range(block_length)]

                input_ids, token_labels, block_ids, labels, position_ids = \
                    build_verify_sequence(
                        prompt_ids, accepted_blocks, draft_ids, step_map,
                        block_length, block_size, mask_token_id,
                    )

                real_len = len(input_ids)
                pad_len = padded_len - real_len
                if pad_len < 0:
                    raise ValueError(
                        f"verify seq len {real_len} > padded_len {padded_len}; raise --padded_len"
                    )
                input_ids = input_ids + [0] * pad_len
                token_labels = token_labels + [-1] * pad_len
                block_ids = block_ids + [-1] * pad_len
                labels = labels + [-100] * pad_len
                position_ids = position_ids + [0] * pad_len

            with acc.stage("tensor_materialize_and_h2d_ms", sync_after=True):
                def to_b(lst):
                    t = torch.tensor(lst, dtype=torch.long, device=device)
                    return t.unsqueeze(0).expand(batch, -1).contiguous()
                ids_b = to_b(input_ids)
                tl_b = to_b(token_labels)
                bi_b = to_b(block_ids)
                lab_b = to_b(labels)
                pos_b = to_b(position_ids)

                # In eval-SDPA mode we build the 4D bool attention mask here
                # so its materialization cost is charged to the same stage
                # that covers tensor creation + H2D. Shape: [B, 1, S, S].
                # SDPA accepts [B,1,S,S] (broadcast over heads) directly.
                attn_mask_4d = (
                    _mbc_mask_fn(tl_b, bi_b, block_size, block_causal_prompt=True)
                    if use_eval_sdpa else None
                )

            with torch.no_grad():
                if use_eval_sdpa:
                    # Eval branch: externally-built mask + token_labels for RoPE
                    # copy. NOT passing block_ids/labels keeps SDARForCausalLM.
                    # forward out of its training path even if somebody toggles
                    # training flags upstream.
                    target_model(
                        input_ids=ids_b,
                        attention_mask=attn_mask_4d,
                        token_labels=tl_b,
                        position_ids=pos_b,
                    )
                else:
                    target_model(
                        input_ids=ids_b,
                        token_labels=tl_b,
                        block_ids=bi_b,
                        labels=lab_b,
                        position_ids=pos_b,
                    )

            accepted_blocks.append(draft_ids)
    finally:
        target_model.eval()
        target_model.config.block_size = orig_bs


def run(args) -> Dict[str, Any]:
    print(f"[sweep-forward] loading {args.model} on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    # IMPORTANT: defer patch_multi_block_mask_fn until after the first draft
    # forward  applying it before any forward call interacts with the Liger
    # SwiGLU Triton kernel via some trust-remote-code path and causes
    # "Pointer argument ... cpu tensor?" errors.

    timer = HFForwardTimer(model)
    _verify_patched = False
    # List-of-one so the nested scope can rebind without `nonlocal`.
    # Guards `torch.compile` so it's applied at most once even though the
    # patch block lives inside the per-tag grid loop.
    _model_compiled = [False]

    grid: List[tuple] = []
    for bl in args.block_lengths:
        for S in {args.fixed_steps, bl}:
            if (bl, S) not in grid:
                grid.append((bl, S))

    results: Dict[str, Any] = {}
    for bl, S in grid:
        tag = f"bl={bl}_ds={S}"
        padded_len = max(
            args.padded_len,
            ((args.prompt_len + (args.num_blocks + 1) * 2 * bl + 127) // 128) * 128,
        )

        print(f"  [{tag}] draft role", flush=True)
        for _ in range(args.warmup):
            hf_block_causal_run(model, args.batch, args.prompt_len, bl, S,
                                args.num_blocks, MASK_TOKEN_ID)
        d_times: List[float] = []
        draft_iter_breakdown: List[Dict[str, float]] = []
        for _ in range(args.iters):
            acc = StageAcc()
            timer.reset()
            torch.cuda.synchronize()
            t_iter = time.perf_counter()
            hf_block_causal_run(model, args.batch, args.prompt_len, bl, S,
                                args.num_blocks, MASK_TOKEN_ID, acc=acc)
            torch.cuda.synchronize()
            iter_wall_ms = (time.perf_counter() - t_iter) * 1000.0
            per_fwd = timer.collect()
            d_times.extend(per_fwd)
            fwd_sum = sum(per_fwd)
            mask_ms = acc.sums.get("mask_build_ms", 0.0)
            samp_ms = acc.sums.get("sampling_ms", 0.0)
            known = fwd_sum + mask_ms + samp_ms
            draft_iter_breakdown.append({
                "end_to_end_stage_ms": iter_wall_ms,
                "model_forward_ms": fwd_sum,
                "mask_build_ms": mask_ms,
                "sampling_ms": samp_ms,
                "other_ms": iter_wall_ms - known,
            })

        if not _verify_patched:
            patch_multi_block_mask_fn(model, block_causal_prompt=True)
            _verify_patched = True
            # Target role runs in training mode (see hf_verify_run), which routes
            # SDARAttention.forward into the `fused_flex_attention` branch. That
            # branch calls `torch.nn.attention.flex_attention` without any
            # `torch.compile` wrapper, so it falls back to the eager
            # `_flex_attention_hop` path -- measured ~3.3 ms/layer at S=256
            # versus ~0.3 ms/layer for the compiled Triton kernel. Opt-in
            # compile here to let Inductor lower flex_attention to Triton; gated
            # behind `--compile` so the default path stays byte-identical.
            if args.compile and not _model_compiled[0]:
                print(f"  [compile] torch.compile(mode={args.compile_mode!r}, "
                      f"dynamic=False)  first target forward absorbs trace cost",
                      flush=True)
                model = torch.compile(
                    model,
                    backend="inductor",
                    mode=args.compile_mode,
                    dynamic=False,   # padded_len fixed per run  avoid recompile churn
                    fullgraph=False,  # SDAR has data-dependent python branches
                )
                _model_compiled[0] = True

        print(f"  [{tag}] target role", flush=True)
        for _ in range(args.warmup):
            hf_verify_run(model, args.batch, args.prompt_len, bl, args.num_blocks,
                          S, padded_len, MASK_TOKEN_ID,
                          use_eval_sdpa=args.target_eval_sdpa)
        v_times: List[float] = []
        verify_iter_breakdown: List[Dict[str, float]] = []
        for _ in range(args.iters):
            acc = StageAcc()
            timer.reset()
            torch.cuda.synchronize()
            t_iter = time.perf_counter()
            hf_verify_run(model, args.batch, args.prompt_len, bl, args.num_blocks,
                          S, padded_len, MASK_TOKEN_ID, acc=acc,
                          use_eval_sdpa=args.target_eval_sdpa)
            torch.cuda.synchronize()
            iter_wall_ms = (time.perf_counter() - t_iter) * 1000.0
            per_fwd = timer.collect()
            v_times.extend(per_fwd)
            fwd_sum = sum(per_fwd)
            seq_ms = acc.sums.get("verify_seq_build_ms", 0.0)
            tmat_ms = acc.sums.get("tensor_materialize_and_h2d_ms", 0.0)
            known = fwd_sum + seq_ms + tmat_ms
            verify_iter_breakdown.append({
                "end_to_end_stage_ms": iter_wall_ms,
                "model_forward_ms": fwd_sum,
                "verify_seq_build_ms": seq_ms,
                "tensor_materialize_and_h2d_ms": tmat_ms,
                "other_ms": iter_wall_ms - known,
            })

        ds = summarize(d_times)
        vs = summarize(v_times)
        draft_nf = args.num_blocks * S
        target_nf = args.num_blocks
        draft_per_step_ms = ds.get("mean_ms", 0) / max(1, draft_nf)
        target_per_step_ms = vs.get("mean_ms", 0) / max(1, target_nf)
        useful_tokens_per_run = args.batch * bl * args.num_blocks
        draft_tput = (useful_tokens_per_run * 1000.0 / ds["mean_ms"]) if ds.get("mean_ms") else 0.0
        target_tput = (useful_tokens_per_run * 1000.0 / vs["mean_ms"]) if vs.get("mean_ms") else 0.0
        results[tag] = {
            "block_length": bl,
            "denoising_steps": S,
            "padded_len": padded_len,
            "batch": args.batch,
            "prompt_len": args.prompt_len,
            "num_blocks": args.num_blocks,
            "draft_role": {
                **ds,
                "n_forwards_per_run": draft_nf,
                "per_step_latency_ms": round(draft_per_step_ms, 4),
                "throughput_tok_per_s": round(draft_tput, 2),
                "breakdown": build_breakdown_output(
                    draft_iter_breakdown,
                    ["end_to_end_stage_ms", "model_forward_ms",
                     "mask_build_ms", "sampling_ms", "other_ms"],
                    draft_nf,
                ),
            },
            "target_role": {
                **vs,
                "n_forwards_per_run": target_nf,
                "per_step_latency_ms": round(target_per_step_ms, 4),
                "throughput_tok_per_s": round(target_tput, 2),
                "breakdown": build_breakdown_output(
                    verify_iter_breakdown,
                    ["end_to_end_stage_ms", "model_forward_ms",
                     "verify_seq_build_ms", "tensor_materialize_and_h2d_ms",
                     "other_ms"],
                    target_nf,
                ),
            },
        }
        print(f"    draft  mean={ds.get('mean_ms', 0):.3f}ms  n={ds.get('n', 0)}")
        print(f"    target mean={vs.get('mean_ms', 0):.3f}ms  n={vs.get('n', 0)}")
        db = results[tag]["draft_role"]["breakdown"]["per_run_breakdown_ms"]
        vb = results[tag]["target_role"]["breakdown"]["per_run_breakdown_ms"]
        dbp = results[tag]["draft_role"]["breakdown"]["per_forward_breakdown_ms"]
        vbp = results[tag]["target_role"]["breakdown"]["per_forward_breakdown_ms"]
        print(
            "    draft  /run   "
            f"e2e={db['end_to_end_stage_ms']['mean_ms']:.2f}  "
            f"fwd={db['model_forward_ms']['mean_ms']:.2f}  "
            f"mask={db['mask_build_ms']['mean_ms']:.2f}  "
            f"sample={db['sampling_ms']['mean_ms']:.2f}  "
            f"other={db['other_ms']['mean_ms']:.2f}"
        )
        print(
            f"    draft  /fwd   fwd={dbp['model_forward_ms']:.3f}  "
            f"(n_fwd={draft_nf})"
        )
        print(
            "    target /run   "
            f"e2e={vb['end_to_end_stage_ms']['mean_ms']:.2f}  "
            f"fwd={vb['model_forward_ms']['mean_ms']:.2f}  "
            f"seqbuild={vb['verify_seq_build_ms']['mean_ms']:.2f}  "
            f"tmat+h2d={vb['tensor_materialize_and_h2d_ms']['mean_ms']:.2f}  "
            f"other={vb['other_ms']['mean_ms']:.2f}"
        )
        print(
            f"    target /fwd   fwd={vbp['model_forward_ms']:.3f}  "
            f"(n_fwd={target_nf})"
        )

    timer.close()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model path (relative to repo root allowed)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--block_lengths", type=int, nargs="+",
                   default=[1, 2, 4, 6, 8, 12, 16])
    p.add_argument("--fixed_steps", type=int, default=4,
                   help="S value used for 'fixed' denoising_steps variant")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt_len", type=int, default=69)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--padded_len", type=int, default=0,
                   help="Floor for padded_len (0 = auto, use the smallest 128-aligned value that fits)")
    # `--compile` wraps the model in `torch.compile` right after the multi-block
    # mask fn is patched. Needed to make the target (training-mode)
    # `fused_flex_attention` path hit the Inductor-lowered Triton kernel
    # instead of the eager `_flex_attention_hop` fallback. Draft role gets
    # compiled as a side effect on its next forward call (harmless, may even
    # help a little). Default off so the baseline path stays byte-identical.
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(model) before target role (fixes uncompiled flex_attention).")
    p.add_argument("--compile_mode", type=str, default="default",
                   choices=["default", "reduce-overhead", "max-autotune"],
                   help="torch.compile mode. 'default' is safest; "
                        "'reduce-overhead' enables CUDA graphs (requires fully static shapes).")
    # Opt-in: route the target forward through the same SDPA eval path draft
    # already uses, by building the multi-block bool mask externally and
    # keeping model in eval mode. See hf_verify_run docstring for why this
    # skips the uncompiled fused_flex_attention path entirely.
    p.add_argument("--target_eval_sdpa", action="store_true",
                   help="Run target in eval mode with externally-built 4D bool "
                        "mask  F.scaled_dot_product_attention (same as draft).")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    if not os.path.isdir(args.model):
        cand = os.path.join(_ROOT, args.model)
        if os.path.isdir(cand):
            args.model = cand

    results = run(args)
    report = {
        "model": args.model,
        "device": args.device,
        "config": {k: v for k, v in vars(args).items() if k != "output"},
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
