#!/usr/bin/env python3
"""Per-submodule latency attribution for SDAR.

Breaks ``model_forward_ms`` (the opaque box in sweep_forward/sweep_regime)
into attention projections (q/k/v/o), attention core (SDPA / flex / flash),
MLP projections (gate/up/down), MLP element-wise (SwiGLU silu*mul), and
residual "norms + other" inside each decoder layer. Also reports a per-model
HBM-bandwidth roofline lower bound (``params * bytes_per_param / HBM_BW``) so
the reader can tell whether observed forward time is at the bandwidth wall or
in launch-overhead / compute-bound territory.

Method: ``nn.Module.register_forward_pre_hook`` + ``_hook`` with
``torch.cuda.Event`` per invocation. Hooks fire once per decoder layer
(SDAR has 28/36/36 layers for 1.7B/4B/8B), so bucket sums are already
per-forward totals  no layer-count multiplier needed.

Buckets aggregated from the SDAR naming (see inference/model/SDAR-*/modeling_sdar.py):
  q_proj, k_proj, v_proj, o_proj     (Linear)
  gate_proj, up_proj, down_proj      (Linear, SwiGLU)
  self_attn   SDARAttention total   (to derive attn_core)
  mlp         SDARMLP total         (to derive mlp_elementwise)
  decoder_layer  SDARDecoderLayer total (to derive norms_other_in_layer)

Caveats:
  - Hook dispatch adds small per-call overhead (~5-15%). Use sweep_forward
    for the *canonical* model_forward_ms; this script is for attribution.
  - The roofline assumes bf16 weights (2 bytes/param) and a single pass over
    weights. It ignores KV-cache reads and activation bytes, so it's a
    *lower* bound, not a target.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault(
    "FLASHINFER_WORKSPACE_BASE",
    os.path.join(os.path.expanduser("~"), ".flashinfer_local"),
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from speculative_decoding.sweep_forward import (  # noqa: E402
    MASK_TOKEN_ID,
    StageAcc,
    hf_block_causal_run,
    hf_verify_run,
)
from speculative_decoding.sweep_regime import padded_len_for  # noqa: E402
from speculative_decoding.draft import (  # noqa: E402
    _build_block_causal_attn,
    _capture_draft_forward_graph,
    clear_draft_graph_cache,
    patch_sdpa_eval_attention,
)
from speculative_decoding.verify import (  # noqa: E402
    _capture_verify_forward_graph,
    clear_verify_graph_cache,
    patch_multi_block_mask_fn,
)


LINEAR_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj")
CLASS_BUCKETS = {
    "SDARAttention": "self_attn",
    "SDARMLP": "mlp",
    "SDARDecoderLayer": "decoder_layer",
}
ALL_BUCKETS = list(LINEAR_SUFFIXES) + list(CLASS_BUCKETS.values())


class SubmoduleTimer:
    """Forward-hook timer that buckets CUDA-event intervals by submodule kind."""

    def __init__(self, model: nn.Module):
        self.buckets: Dict[str, List[Tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            b: [] for b in ALL_BUCKETS
        }
        self._starts: Dict[str, List[torch.cuda.Event]] = {b: [] for b in ALL_BUCKETS}
        self._handles: List[Any] = []

        for name, mod in model.named_modules():
            suffix = name.rsplit(".", 1)[-1]
            if suffix in LINEAR_SUFFIXES and isinstance(mod, nn.Linear):
                self._attach(mod, suffix)
            cls = type(mod).__name__
            if cls in CLASS_BUCKETS:
                self._attach(mod, CLASS_BUCKETS[cls])

    def _attach(self, mod: nn.Module, key: str) -> None:
        starts = self._starts[key]
        bucket = self.buckets[key]

        def pre(_m, _i, key=key):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            starts.append(ev)

        def post(_m, _i, _o, key=key):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            s = starts.pop()
            bucket.append((s, ev))

        self._handles.append(mod.register_forward_pre_hook(pre))
        self._handles.append(mod.register_forward_hook(post))

    def reset(self) -> None:
        for k in self.buckets:
            self.buckets[k].clear()
            self._starts[k].clear()

    def collect_ms(self) -> Dict[str, float]:
        torch.cuda.synchronize()
        return {k: sum(s.elapsed_time(e) for s, e in pairs)
                for k, pairs in self.buckets.items()}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def roofline_ms(params: int, hbm_bw_gb_s: float, bytes_per_param: int) -> float:
    """Single weight-pass time: (params * bytes) / bandwidth."""
    return (params * bytes_per_param) / (hbm_bw_gb_s * 1e9) * 1000.0


def measure_cell(model: nn.Module, timer: SubmoduleTimer, role: str, *,
                 batch: int, prompt_len: int, block_length: int,
                 denoising_steps: int, num_blocks: int, padded_len: int,
                 warmup: int, iters: int,
                 target_eval_sdpa: bool = False
                 ) -> Tuple[Dict[str, float], int]:
    if role == "draft":
        def run_once():
            hf_block_causal_run(
                model, batch, prompt_len, block_length, denoising_steps,
                num_blocks, MASK_TOKEN_ID, acc=StageAcc(),
            )
        n_forwards = num_blocks * denoising_steps
    elif role == "target":
        # Forward the flag through to hf_verify_run so target role shares the
        # same eval/SDPA path introduced in sweep_forward.py. With the flag
        # off the behavior is byte-identical to the original training-branch
        # path; with it on we skip fused_flex_attention entirely and hit
        # F.scaled_dot_product_attention  same kernel draft already uses.
        def run_once():
            hf_verify_run(
                model, batch, prompt_len, block_length, num_blocks,
                denoising_steps, padded_len, MASK_TOKEN_ID, acc=StageAcc(),
                use_eval_sdpa=target_eval_sdpa,
            )
        n_forwards = num_blocks
    else:
        raise ValueError(role)

    for _ in range(warmup):
        run_once()

    agg = {k: 0.0 for k in ALL_BUCKETS}
    for _ in range(iters):
        timer.reset()
        torch.cuda.synchronize()
        run_once()
        ms = timer.collect_ms()
        for k, v in ms.items():
            agg[k] += v
    per_forward = {k: agg[k] / iters / n_forwards for k in agg}
    return per_forward, n_forwards


def _time_forward_cuda_events(closure, warmup: int, iters: int) -> float:
    """Mean per-call forward time in ms via CUDA events. Pure-GPU, no hook
    overhead (unlike SubmoduleTimer). Use for eager vs graph comparison where
    submodule attribution would distort the picture."""
    for _ in range(warmup):
        closure()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        closure()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / iters


def _capture_local_graph(closure, device: torch.device):
    """Capture-and-return a CUDA graph for a closure with fixed-shape inputs.

    Single-file helper (NOT cached): we want a fresh graph per (model, role,
    batch, prompt_len) measurement, so we don't reuse the draft/verify caches
    that key on batch=1 shapes from the speculative pipeline.
    """
    with torch.cuda.device(device):
        torch.cuda.synchronize(device)
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(5):
                closure()
        torch.cuda.current_stream(device).wait_stream(s)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s,
                              capture_error_mode="thread_local"), \
                torch.no_grad():
            closure()
    return g


def measure_single_forward_compare(
    model: nn.Module, role: str, *, batch: int, prompt_len: int,
    block_length: int, block_size: int, num_blocks: int, padded_len: int,
    device: torch.device, mask_token_id: int,
    warmup: int, iters: int,
) -> Dict[str, float]:
    """Compare a single fixed-shape forward pass in eager vs CUDA-graph mode,
    at arbitrary batch size.

    This intentionally does NOT attach SubmoduleTimer hooks: Python-level hooks
    fire once during graph capture and never during replay, so they drop
    per-forward bucket samples and distort attribution. Submodule breakdown
    (eager only) and graph-compare (eager vs graph) are answering different
    questions  keep them separate.

    Shapes:
      - role=draft:  [B, seq_len] input with block-causal 2D mask
                     (seq_len = prompt_len + (num_blocks//2)*block_length
                     + block_length  "mid-run" draft forward shape).
      - role=target: [B, padded_len] input with a 4D multi-block causal bool
                     mask from new_attn_multi_block.create_multi_block_causal_mask
                     (same kernel path verify uses in eval-SDPA mode).

    Returns {'eager_ms', 'graph_ms', 'speedup'}. For launch-bound models
    (1.7B, 4B at small batch) graph removes most per-call overhead  big
    speedup. Pushing batch up shifts the regime toward compute/BW-bound
    speedup shrinks. That's exactly the curve we want to plot.
    """
    patch_sdpa_eval_attention(model)
    # modeling_sdar's RoPE-copy path uses torch.nonzero when token_labels is
    # not None, which crashes during CUDA graph capture (data-dependent shape).
    # The hasattr check lives on SDARModel (model.model), NOT the outer
    # SDARForCausalLM. Multi-block inputs already share position_ids between
    # data and mask, so skipping is semantically correct.
    _inner = getattr(model, "model", model)
    _had_skip = hasattr(_inner, "_skip_rope_copy")
    _inner._skip_rope_copy = True

    if role == "draft":
        seq_len = prompt_len + (num_blocks // 2) * block_length + block_length
        input_ids = torch.full((batch, seq_len), mask_token_id,
                               dtype=torch.long, device=device)
        attn = _build_block_causal_attn(seq_len, block_length, device)
        model.eval()

        def eager_call():
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn,
                      use_cache=False, return_dict=True)

        eager_ms = _time_forward_cuda_events(eager_call, warmup, iters)
        g = _capture_local_graph(eager_call, device)

        def graph_call():
            g.replay()

        graph_ms = _time_forward_cuda_events(graph_call, warmup, iters)
        del g

    elif role == "target":
        from new_attn_multi_block import create_multi_block_causal_mask as _mbc
        from speculative_decoding.verify import build_verify_sequence

        prompt_ids = list(range(1, prompt_len + 1))
        accepted_blocks = [[200] * block_length for _ in range(num_blocks // 2)]
        draft_ids = [100] * block_length
        step_map = [min(i, block_size - 1) for i in range(block_length)]

        ids, tl, bi, _labels, pos = build_verify_sequence(
            prompt_ids, accepted_blocks, draft_ids, step_map,
            block_length, block_size, mask_token_id,
        )
        real_len = len(ids)
        pad = padded_len - real_len
        if pad < 0:
            raise ValueError(
                f"verify seq_len {real_len} exceeds padded_len {padded_len}")
        ids = ids + [0] * pad
        tl = tl + [-1] * pad
        bi = bi + [-1] * pad
        pos = pos + [0] * pad

        def to_b(lst):
            t = torch.tensor(lst, dtype=torch.long, device=device)
            # [1, S]  [B, S]; contiguous() so the graph's input buffer has a
            # normal stride pattern (otherwise capture can bind a view).
            return t.unsqueeze(0).expand(batch, -1).contiguous()

        input_t = to_b(ids)
        tl_t = to_b(tl)
        bi_t = to_b(bi)
        pos_t = to_b(pos)
        # _mbc returns [B,1,S,S] when given [B,S] token_labels/block_ids.
        attn_mask = _mbc(tl_t, bi_t, block_size, block_causal_prompt=True)

        orig_bs = model.config.block_size
        model.config.block_size = block_size
        model.eval()
        try:
            def eager_call():
                with torch.no_grad():
                    model(input_ids=input_t, attention_mask=attn_mask,
                          token_labels=tl_t, position_ids=pos_t)

            eager_ms = _time_forward_cuda_events(eager_call, warmup, iters)
            g = _capture_local_graph(eager_call, device)

            def graph_call():
                g.replay()

            graph_ms = _time_forward_cuda_events(graph_call, warmup, iters)
            del g
        finally:
            model.config.block_size = orig_bs
    else:
        raise ValueError(role)

    if not _had_skip:
        try:
            del _inner._skip_rope_copy
        except AttributeError:
            pass

    return {
        "eager_ms": round(eager_ms, 4),
        "graph_ms": round(graph_ms, 4),
        "speedup": round(eager_ms / graph_ms, 4) if graph_ms > 0 else float("nan"),
    }


def run(args) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    graph_records: List[Dict[str, Any]] = []
    for mpath in args.models:
        full = mpath if os.path.isabs(mpath) else os.path.join(_ROOT, mpath)
        mname = os.path.basename(full.rstrip("/"))
        print(f"[breakdown] loading {full} on {args.device}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            full, trust_remote_code=True, torch_dtype=torch.bfloat16,
        ).to(args.device).eval()
        params = count_params(model)
        params_b = params / 1e9
        rl_ms = roofline_ms(params, args.hbm_bw_gb_s, args.bytes_per_param)
        print(f"  params={params_b:.3f}B  roofline_bw_lower_bound_ms={rl_ms:.3f}",
              flush=True)

        timer = SubmoduleTimer(model)
        patched = False
        try:
            for batch in args.batches:
                for pl in args.prompt_lens:
                    padded = padded_len_for(pl, args.num_blocks, args.block_length)
                    for role in ("draft", "target"):
                        if role == "target" and not patched:
                            patch_multi_block_mask_fn(model, block_causal_prompt=True)
                            patched = True
                        if args.compare_cuda_graph:
                            # Eager-vs-graph single-forward comparison runs
                            # WITHOUT SubmoduleTimer hooks (Python-level hooks
                            # don't re-fire during graph replay  broken
                            # attribution). Numbers here are pure CUDA-event
                            # time around the forward call itself.
                            # target role needs the SDPA-eval kernel path for
                            # graph capture; if the user hasn't enabled eval
                            # mode patch yet (via target_eval_sdpa), the
                            # function does it itself (patch is idempotent).
                            gr = measure_single_forward_compare(
                                model, role,
                                batch=batch, prompt_len=pl,
                                block_length=args.block_length,
                                block_size=args.denoising_steps,
                                num_blocks=args.num_blocks,
                                padded_len=padded,
                                device=torch.device(args.device),
                                mask_token_id=MASK_TOKEN_ID,
                                warmup=args.warmup, iters=args.iters,
                            )
                            grec = {
                                "model": mname, "params_b": round(params_b, 3),
                                "role": role,
                                "batch": batch, "prompt_len": pl,
                                "block_length": args.block_length,
                                "denoising_steps": args.denoising_steps,
                                "num_blocks": args.num_blocks,
                                "padded_len": padded,
                                **gr,
                                "roofline_bw_lower_bound_ms": round(rl_ms, 4),
                            }
                            graph_records.append(grec)
                            print(f"  [graph] {mname:<18} {role:<6} bs={batch} "
                                  f"pl={pl}  eager={gr['eager_ms']:.2f}  "
                                  f"graph={gr['graph_ms']:.2f}  "
                                  f"speedup={gr['speedup']:.2f}x  "
                                  f"roofline≥{rl_ms:.2f}",
                                  flush=True)
                        per_fwd, n_fwds = measure_cell(
                            model, timer, role,
                            batch=batch, prompt_len=pl,
                            block_length=args.block_length,
                            denoising_steps=args.denoising_steps,
                            num_blocks=args.num_blocks, padded_len=padded,
                            warmup=args.warmup, iters=args.iters,
                            target_eval_sdpa=args.target_eval_sdpa,
                        )
                        attn_core = per_fwd["self_attn"] - sum(
                            per_fwd[k] for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
                        mlp_elem = per_fwd["mlp"] - sum(
                            per_fwd[k] for k in ("gate_proj", "up_proj", "down_proj"))
                        norms_other = (per_fwd["decoder_layer"]
                                       - per_fwd["self_attn"] - per_fwd["mlp"])
                        rec = {
                            "model": mname, "params_b": round(params_b, 3),
                            "role": role,
                            # Tag which attention path was exercised so
                            # mixed JSONs (baseline + eval_sdpa) can be told
                            # apart without re-deriving from filename.
                            "target_eval_sdpa": bool(args.target_eval_sdpa),
                            "batch": batch, "prompt_len": pl,
                            "block_length": args.block_length,
                            "denoising_steps": args.denoising_steps,
                            "num_blocks": args.num_blocks, "padded_len": padded,
                            "n_forwards_per_run": n_fwds,
                            **{k: round(v, 4) for k, v in per_fwd.items()},
                            "attn_core": round(attn_core, 4),
                            "mlp_elementwise": round(mlp_elem, 4),
                            "norms_other_in_layer": round(norms_other, 4),
                            "roofline_bw_lower_bound_ms": round(rl_ms, 4),
                            "hbm_bw_gb_s": args.hbm_bw_gb_s,
                            "bytes_per_param": args.bytes_per_param,
                        }
                        records.append(rec)
                        print(f"  {mname:<18} {role:<6} bs={batch} pl={pl}  "
                              f"attn={per_fwd['self_attn']:.2f}  "
                              f"mlp={per_fwd['mlp']:.2f}  "
                              f"layer={per_fwd['decoder_layer']:.2f}  "
                              f"roofline≥{rl_ms:.2f} ms",
                              flush=True)
        finally:
            timer.close()
            del model
            torch.cuda.empty_cache()
    return records, graph_records


def write_json_csv(records: List[Dict[str, Any]], out_dir: str) -> None:
    with open(os.path.join(out_dir, "module_breakdown.json"), "w") as f:
        json.dump(records, f, indent=2)
    fields = sorted({k for r in records for k in r.keys()})
    with open(os.path.join(out_dir, "module_breakdown.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def write_graph_compare(records: List[Dict[str, Any]], out_dir: str) -> None:
    if not records:
        return
    with open(os.path.join(out_dir, "single_forward_graph_compare.json"), "w") as f:
        json.dump(records, f, indent=2)
    fields = sorted({k for r in records for k in r.keys()})
    with open(os.path.join(out_dir, "single_forward_graph_compare.csv"),
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def plot_graph_compare(records: List[Dict[str, Any]], out_dir: str) -> None:
    """Single-forward latency: eager vs cuda_graph, across params and batch.

    Two figures per (role, prompt_len):

    1. ``single_forward_graph_compare_{role}_pl{pl}.png``  eager/graph ms vs
       params (one line per batch size). Visualizes the thesis: graph replay
       drops the launch-overhead floor so latency scales with params (hugs the
       HBM-BW roofline dashed line); eager is ~flat for small models because
       launch overhead swamps compute.

    2. ``single_forward_graph_speedup_{role}_pl{pl}.png``  eager/graph speedup
       vs batch (one line per model). As batch grows, each forward does more
       work per launch so graph's payoff shrinks  speedup should fall with
       batch. If speedup stays high at batch=32, that model is still
       launch-bound even at that scale.
    """
    if not records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[breakdown] matplotlib unavailable: {e}")
        return

    roles = sorted({r["role"] for r in records})
    prompt_lens = sorted({r["prompt_len"] for r in records})

    # ── Figure 1: eager/graph ms vs params, one line per batch ──
    for role in roles:
        for pl in prompt_lens:
            batches = sorted({r["batch"] for r in records
                              if r["role"] == role and r["prompt_len"] == pl})
            if not batches:
                continue
            fig, ax = plt.subplots(figsize=(7, 5))
            plotted_roof = set()
            for b in batches:
                rs = sorted([r for r in records
                             if r["role"] == role and r["prompt_len"] == pl
                             and r["batch"] == b],
                            key=lambda r: r["params_b"])
                xs = [r["params_b"] for r in rs]
                eager = [r["eager_ms"] for r in rs]
                graph = [r["graph_ms"] for r in rs]
                line_e, = ax.plot(xs, eager, marker="o", linestyle="-",
                                  label=f"bs={b} eager")
                ax.plot(xs, graph, marker="s", linestyle="--",
                        color=line_e.get_color(), alpha=0.8,
                        label=f"bs={b} graph")
                for x, e, g in zip(xs, eager, graph):
                    if g > 0:
                        ax.annotate(f"{e/g:.1f}x", xy=(x, g),
                                    xytext=(2, -10),
                                    textcoords="offset points",
                                    fontsize=6, alpha=0.7,
                                    color=line_e.get_color())
                # Roofline (bs-invariant in our formula  weights dominate)
                for r in rs:
                    key = r["model"]
                    if key not in plotted_roof:
                        ax.axhline(r["roofline_bw_lower_bound_ms"],
                                   linestyle=":", color="gray", alpha=0.4)
                        plotted_roof.add(key)
            ax.set_yscale("log")
            ax.set_xlabel("params (B)")
            ax.set_ylabel("ms per forward (log)")
            ax.set_title(
                f"Single-forward latency  role={role}  prompt_len={pl}\n"
                f"eager vs cuda_graph across batches "
                f"(annotations = eager/graph speedup; dotted = HBM roofline)"
            )
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(os.path.join(
                out_dir, f"single_forward_graph_compare_{role}_pl{pl}.png"),
                dpi=150)
            plt.close(fig)

    # ── Figure 2: speedup vs batch, one line per model ──
    for role in roles:
        for pl in prompt_lens:
            models = sorted({r["model"] for r in records
                             if r["role"] == role and r["prompt_len"] == pl},
                            key=lambda m: next(r["params_b"]
                                               for r in records
                                               if r["model"] == m))
            if not models:
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for m in models:
                rs = sorted([r for r in records
                             if r["role"] == role and r["prompt_len"] == pl
                             and r["model"] == m],
                            key=lambda r: r["batch"])
                xs = [r["batch"] for r in rs]
                ys = [r["speedup"] for r in rs]
                params_b = rs[0]["params_b"] if rs else 0.0
                ax.plot(xs, ys, marker="o",
                        label=f"{m} ({params_b:.2f}B)")
            ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5,
                       label="no speedup")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("batch size")
            ax.set_ylabel("eager / graph speedup")
            ax.set_title(
                f"CUDA-graph speedup vs batch  role={role}  prompt_len={pl}\n"
                f"(small model + small batch = launch-bound  big speedup; "
                f"growing batch shrinks the gap)"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(
                out_dir, f"single_forward_graph_speedup_{role}_pl{pl}.png"),
                dpi=150)
            plt.close(fig)


def plot_stacked(records: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[breakdown] matplotlib unavailable: {e}")
        return

    parts = ["q_proj", "k_proj", "v_proj", "o_proj", "attn_core",
             "gate_proj", "up_proj", "down_proj", "mlp_elementwise",
             "norms_other_in_layer"]
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["batch"], r["prompt_len"]), []).append(r)

    for (b, pl), group in groups.items():
        group = sorted(group, key=lambda r: (r["params_b"], r["role"]))
        labels = [f"{r['model']}\n{r['role']}" for r in group]
        data = np.array([[max(0.0, r[p]) for r in group] for p in parts])
        fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(group)), 5.5))
        bottom = np.zeros(len(group))
        for i, name in enumerate(parts):
            ax.bar(labels, data[i], bottom=bottom, label=name)
            bottom += data[i]
        seen = set()
        for i, r in enumerate(group):
            key = r["model"]
            if key in seen:
                continue
            seen.add(key)
            ax.hlines(r["roofline_bw_lower_bound_ms"], i - 0.4, i + 0.4,
                      linestyles="dashed", linewidth=1.2, alpha=0.8)
        ax.set_ylabel("ms per forward (per-layer sums across all layers)")
        ax.set_title(f"Submodule breakdown  batch={b}  prompt_len={pl}  "
                     f"(dashed = HBM roofline lower bound)")
        ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"module_breakdown_bs{b}_pl{pl}.png"),
                    dpi=150)
        plt.close(fig)


def plot_roofline(records: List[Dict[str, Any]], out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[breakdown] matplotlib unavailable: {e}")
        return

    models = sorted({r["model"] for r in records},
                    key=lambda m: next(r["params_b"]
                                       for r in records if r["model"] == m))
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        rs = sorted([r for r in records if r["model"] == m],
                    key=lambda r: (r["batch"], r["prompt_len"], r["role"]))
        xs = list(range(len(rs)))
        observed = [r["decoder_layer"] for r in rs]
        ax.plot(xs, observed, marker="o",
                label=f"{m} observed (decoder_layer sum)")
        ax.axhline(rs[0]["roofline_bw_lower_bound_ms"], linestyle="--",
                   alpha=0.6, label=f"{m} roofline BW lower bound")
    ax.set_yscale("log")
    ax.set_ylabel("ms per forward")
    ax.set_xlabel("config index (batch × prompt_len × role)")
    ax.set_title("Observed forward vs HBM-BW roofline lower bound")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roofline.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True,
                   help="model paths (absolute or relative to repo root)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batches", type=int, nargs="+", default=[1])
    p.add_argument("--prompt_lens", type=int, nargs="+", default=[64])
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--num_blocks", type=int, default=8)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--hbm_bw_gb_s", type=float, default=2039.0,
                   help="HBM bandwidth (A800≈2039, H100≈3350). Used for roofline.")
    p.add_argument("--bytes_per_param", type=int, default=2,
                   help="bf16/fp16 = 2; fp32 = 4; fp8 = 1")
    # Mirror of sweep_forward.py's --target_eval_sdpa. Off by default so the
    # baseline (training branch / fused_flex_attention) stays byte-identical.
    p.add_argument("--target_eval_sdpa", action="store_true",
                   help="Run target role in eval mode with externally-built "
                        "4D bool mask  F.scaled_dot_product_attention "
                        "(same kernel path as draft).")
    # Additional eager-vs-graph comparison section. Runs alongside the existing
    # submodule breakdown (they don't share hooks, so both can be enabled).
    p.add_argument("--compare_cuda_graph", action="store_true",
                   help="Also measure single-forward latency eager vs CUDA "
                        "graph for each (model, role, batch, prompt_len). "
                        "Produces single_forward_graph_compare.{json,csv} and "
                        "per-role PNG. Use this to validate that once launch "
                        "overhead is removed, paramsBW becomes the dominant "
                        "term in single-forward latency (1.7B/4B launch-bound "
                        " big speedup; 8B already near BW wall  small).")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records, graph_records = run(args)
    write_json_csv(records, args.output_dir)
    plot_stacked(records, args.output_dir)
    plot_roofline(records, args.output_dir)
    if graph_records:
        write_graph_compare(graph_records, args.output_dir)
        plot_graph_compare(graph_records, args.output_dir)
        print(f"[breakdown] wrote {len(graph_records)} cuda-graph comparison "
              f"records to {args.output_dir}")
    print(f"[breakdown] wrote {len(records)} records + plots to {args.output_dir}")


if __name__ == "__main__":
    main()
