#!/usr/bin/env python3
"""SimSD: one-shot latency + quality benchmark for vanilla / vanilla+CG / ours.

An experiment is described by a YAML file under ``configs/experiments/``; this
script resolves it (YAML, then CLI overrides), fans out one subprocess per
(method, mode), and aggregates every ``SUMMARY.json`` into ``UNIFIED.json``.

  python main.py                                        # = configs/experiments/sdar.yaml
  python main.py -e configs/experiments/llada2.yaml     # LLaDA2 eager self-draft
  python main.py -e configs/experiments/sdar.yaml --num_samples 20 --datasets gsm8k
  python main.py -e configs/experiments/llada2.yaml --print_config

Two modes per method:
  latency : --no_eos_stop, short fixed-length gen (pure throughput; pass@1 is
            not meaningful on truncated output).
  quality : EOS stop on, larger budget (answer terminates naturally; pass@1 +
            acceptance rate reported).

Throughput is computed from ``gpu_event_ms`` (torch.cuda.Event), not wall clock.

Methods
  vanilla    : main_table/run_vanilla_tp2_cache.py   TP=2 multinomial
  vanilla_cg : main_table/run_native_tp2_cache.py    TP=2 argmax + cuda_graph
  ours       : main_table/run_ours_dual_gpu.py       dual-GPU SimSD spec + cg
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = REPO / "configs/experiments/sdar.yaml"

METHODS = {
    "vanilla":    {"runner": "main_table/run_vanilla_tp2_cache.py", "torchrun": True,  "use_cg": False, "needs_draft": False, "pipeline": False, "shards": False},
    "vanilla_cg": {"runner": "main_table/run_native_tp2_cache.py",  "torchrun": True,  "use_cg": True,  "needs_draft": False, "pipeline": False, "shards": False},
    # "ours" = SD+pipe per plan/NAMING.md (post 2026-05-06): pipelined cache_aware
    # with draft on cuda:0, target on cuda:1, overlap via cuda_graph.
    "ours":       {"runner": "main_table/run_ours_dual_gpu.py",     "torchrun": False, "use_cg": True,  "needs_draft": True,  "pipeline": True,  "shards": True},
    # Target-only reference for a target too large for one card. The TP=2
    # baselines need 95.8 GiB/rank on LLaDA2.0-flash against 95.6 GiB of usable
    # HBM, so they miss by a hair; this one shards the target
    # exactly the way `ours` does, so the comparison isolates speculation rather
    # than mixing in a placement difference.
    "native_sharded": {"runner": "main_table/run_native_sharded.py", "torchrun": False, "use_cg": False, "needs_draft": False, "pipeline": False, "shards": True},
}

#: Which decode knobs each runner actually accepts. Only `ours` has a draft, so
#: the acceptance-rule flags are meaningless (and rejected) elsewhere; only
#: `vanilla` exposes the sampling temperature.
SPEC_FLAGS = {
    "vanilla":    ("draft_sampling", "remasking_strategy", "confidence_threshold",
                   "temperature", "top_k", "top_p"),
    "vanilla_cg": ("remasking_strategy", "confidence_threshold"),
    "ours":       ("draft_sampling", "remasking_strategy", "confidence_threshold",
                   "speculative_branch", "partial_block_fill", "mrs_verify_order"),
    "native_sharded": ("draft_sampling", "remasking_strategy",
                       "confidence_threshold"),
}

#: Top-level YAML sections. Anything else is a typo — and a typo in an
#: experiment file silently produces a wrong table, so reject it loudly.
YAML_SECTIONS = {
    "name", "family", "models", "tokens", "block", "data", "methods", "modes",
    "mode_settings", "runtime", "spec", "method_spec", "output_dir",
}


# ─────────────────────────────────────────────────────────────────────
# Experiment resolution
# ─────────────────────────────────────────────────────────────────────

def load_experiment(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"[main] experiment file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(f"[main] {path}: top level must be a mapping")
    unknown = set(raw) - YAML_SECTIONS
    if unknown:
        raise SystemExit(
            f"[main] {path}: unknown section(s) {sorted(unknown)}; "
            f"known: {sorted(YAML_SECTIONS)}"
        )
    return raw


def _coerce(raw: str):
    """'0.5' -> 0.5, '8' -> 8, 'true' -> True, anything else stays a string."""
    low = raw.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def _pick(cli, yaml_value, fallback=None):
    """CLI wins when given, then YAML, then the hardcoded fallback."""
    if cli is not None:
        return cli
    if yaml_value is not None:
        return yaml_value
    return fallback


def resolve(args, exp: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten YAML + CLI overrides into the single dict the rest reads."""
    models = exp.get("models") or {}
    tokens = exp.get("tokens") or {}
    block = exp.get("block") or {}
    data = exp.get("data") or {}
    runtime = exp.get("runtime") or {}
    spec = exp.get("spec") or {}
    method_spec = exp.get("method_spec") or {}
    mode_settings = exp.get("mode_settings") or {}

    modes = _pick(args.modes, exp.get("modes"), ["latency", "quality"])
    methods = _pick(args.methods, exp.get("methods"), list(METHODS))

    # num_blocks per mode: from YAML mode_settings, overridable by the legacy
    # --{latency,quality}_num_blocks flags.
    legacy_nb = {"latency": args.latency_num_blocks,
                 "quality": args.quality_num_blocks}
    nb_fallback = {"latency": 32, "quality": 128}
    num_blocks = {}
    for m in modes:
        ms = mode_settings.get(m) or {}
        num_blocks[m] = _pick(legacy_nb.get(m), ms.get("num_blocks"),
                              nb_fallback.get(m, 32))

    S: Dict[str, Any] = {
        "experiment_file": str(args.experiment),
        "name": exp.get("name", args.experiment.stem),
        "family": exp.get("family", "sdar"),

        "target_model": _pick(args.target_model, models.get("target")),
        "draft_model": _pick(args.draft_model, models.get("draft")),
        "mask_token_id": _pick(args.mask_token_id, tokens.get("mask_token_id")),

        "block_length": _pick(args.block_length, block.get("block_length"), 4),
        "denoising_steps": _pick(args.denoising_steps,
                                 block.get("denoising_steps"), 4),

        "datasets": _pick(args.datasets, data.get("datasets"),
                          ["gsm8k", "mbpp", "triviaqa", "mmlu"]),
        "num_samples": _pick(args.num_samples, data.get("num_samples"), 200),

        "methods": methods,
        "modes": modes,
        "num_blocks": num_blocks,

        "gpus": str(_pick(args.gpus, runtime.get("gpus"), "0,1")),
        "draft_device": _pick(args.draft_device, runtime.get("draft_device"),
                              "cuda:0"),
        "target_device": _pick(args.target_device, runtime.get("target_device"),
                               "cuda:1"),
        # Shard the target across cards when it does not fit on one. LOCAL
        # indices, i.e. positions within runtime.gpus after CUDA_VISIBLE_DEVICES
        # is applied. None = load it whole onto target_device.
        "target_gpus": _pick(args.target_gpus, runtime.get("target_gpus")),
        "target_max_memory": _pick(args.target_max_memory,
                                   runtime.get("target_max_memory"), "88GiB"),
        "use_cuda_graph": bool(_pick(args.use_cuda_graph,
                                     runtime.get("use_cuda_graph"), True)),
        "pipeline": bool(_pick(args.pipeline, runtime.get("pipeline"), True)),
        "fused_denoise": bool(_pick(args.fused_denoise,
                                    runtime.get("fused_denoise"), True)),
        "speculative_target_extend": bool(_pick(
            args.speculative_target_extend,
            runtime.get("speculative_target_extend"), True)),
        "return_timings": bool(_pick(args.return_timings,
                                     runtime.get("return_timings"), False)),
        # Kernel-side knobs (see docs/kernel-optimization.md,
        # docs/optimize-extend.md). fold_draft_extend and fused_linear are
        # cuda_graph-path optimisations; kv_cache_max_len=0 means "size the
        # StaticBlockCache to max_prompt_len + gen_length" instead of the fixed
        # 1024, which every forward would otherwise scan in full.
        "fold_draft_extend": bool(_pick(args.fold_draft_extend,
                                        runtime.get("fold_draft_extend"), False)),
        "fused_linear": _pick(args.fused_linear, runtime.get("fused_linear"),
                              "off"),
        "kv_cache_max_len": _pick(args.kv_cache_max_len,
                                  runtime.get("kv_cache_max_len"), 1024),
        "seed": _pick(args.seed, runtime.get("seed"), 42),

        # Effective decode knobs per method: the shared `spec:` block, then the
        # method's own `method_spec:` entry on top. Only keys the runner accepts
        # are forwarded; a key absent from both falls through to the runner's
        # own default (which legitimately differs per method — `vanilla` is
        # multinomial by definition, `vanilla_cg` is argmax by definition).
        "spec": {
            m: {k: v for k, v in {**spec, **(method_spec.get(m) or {})}.items()
                if k in SPEC_FLAGS[m]}
            for m in methods if m in METHODS
        },

        "output_dir": _pick(args.output_dir, exp.get("output_dir"),
                            "runs/main_table"),
    }

    # --spec overrides, applied last so a sweep can vary one knob without
    # copying the whole experiment file.
    for item in (args.spec or []):
        if "=" not in item:
            raise SystemExit(
                f"[main] --spec expects [METHOD.]KEY=VALUE, got {item!r}")
        lhs, _, raw = item.partition("=")
        val = _coerce(raw)
        if "." in lhs:
            m, key = lhs.split(".", 1)
            if m not in METHODS:
                raise SystemExit(
                    f"[main] --spec {item}: unknown method {m!r}; "
                    f"known: {list(METHODS)}")
            if key not in SPEC_FLAGS[m]:
                raise SystemExit(
                    f"[main] --spec {item}: {METHODS[m]['runner']} does not "
                    f"accept {key!r}; allowed: {sorted(SPEC_FLAGS[m])}")
            if m in S["spec"]:
                S["spec"][m][key] = val
        else:
            hit = False
            for m in S["spec"]:
                if lhs in SPEC_FLAGS[m]:
                    S["spec"][m][lhs] = val
                    hit = True
            if not hit:
                raise SystemExit(
                    f"[main] --spec {item}: no selected method accepts "
                    f"{lhs!r} (methods: {list(S['spec'])})")

    # A shared `spec:` key that a given runner cannot take is fine (that is what
    # "shared" means), but a `method_spec:` key it cannot take was written for
    # that method specifically and would be silently dropped — reject it.
    for m, overrides in method_spec.items():
        if m not in METHODS:
            raise SystemExit(
                f"[main] method_spec has unknown method {m!r}; "
                f"known: {list(METHODS)}"
            )
        stray = sorted(set(overrides or {}) - set(SPEC_FLAGS[m]))
        if stray:
            raise SystemExit(
                f"[main] method_spec.{m} sets {stray}, which "
                f"{METHODS[m]['runner']} does not accept; allowed: "
                f"{sorted(SPEC_FLAGS[m])}"
            )

    bad = [m for m in S["methods"] if m not in METHODS]
    if bad:
        raise SystemExit(f"[main] unknown method(s) {bad}; known: {list(METHODS)}")
    bad = [m for m in S["modes"] if m not in ("latency", "quality")]
    if bad:
        raise SystemExit(f"[main] unknown mode(s) {bad}; known: latency, quality")
    if not S["target_model"]:
        raise SystemExit(
            "[main] no target model: set models.target in the experiment file "
            "or pass --target_model"
        )
    if any(METHODS[m]["needs_draft"] for m in S["methods"]) and not S["draft_model"]:
        raise SystemExit(
            "[main] method 'ours' needs a draft model: set models.draft in the "
            "experiment file or pass --draft_model"
        )

    # A cuda_graph-off experiment makes vanilla_cg identical to vanilla; say so
    # rather than silently reporting two identical rows.
    if not S["use_cuda_graph"] and "vanilla_cg" in S["methods"]:
        print("[main] WARNING: use_cuda_graph=false makes 'vanilla_cg' "
              "identical to 'vanilla' — the graph is the only difference.",
              flush=True)
    if S["target_gpus"]:
        # Normalise "1,2,3" and [1,2,3] to the same shape, then sanity-check it
        # against the other placement settings before a subprocess spends
        # minutes loading weights only to die.
        if isinstance(S["target_gpus"], str):
            S["target_gpus"] = [int(x) for x in S["target_gpus"].split(",") if x]
        S["target_gpus"] = [int(g) for g in S["target_gpus"]]
        visible = [int(x) for x in S["gpus"].split(",") if x != ""]
        n_visible = len(visible)
        out_of_range = [g for g in S["target_gpus"] if g >= n_visible]
        if out_of_range:
            raise SystemExit(
                f"[main] runtime.target_gpus {S['target_gpus']} references local "
                f"index {out_of_range} but runtime.gpus='{S['gpus']}' exposes "
                f"only {n_visible} ({list(range(n_visible))}). These are LOCAL "
                f"indices, applied after CUDA_VISIBLE_DEVICES."
            )
        # Only meaningful when something actually loads a draft: a target-only
        # run (native_sharded alone) is entitled to every card, including the
        # one the draft would have used.
        if any(METHODS[m]["needs_draft"] for m in S["methods"]):
            draft_idx = int(str(S["draft_device"]).rsplit(":", 1)[-1])
            if draft_idx in S["target_gpus"]:
                raise SystemExit(
                    f"[main] draft_device {S['draft_device']} is inside "
                    f"target_gpus {S['target_gpus']}; the sharded target would "
                    f"evict the draft. Give the draft a card of its own."
                )
        if S["use_cuda_graph"]:
            print("[main] target is sharded (target_gpus set); forcing "
                  "cuda_graph off — capture cannot span accelerate's "
                  "cross-device copies.", flush=True)
            S["use_cuda_graph"] = False

    if S["pipeline"] and S["draft_device"] == S["target_device"]:
        print(f"[main] pipeline needs two distinct devices but draft_device == "
              f"target_device == {S['draft_device']}; disabling pipeline.",
              flush=True)
        S["pipeline"] = False
    if S["pipeline"] and not S["use_cuda_graph"]:
        print("[main] pipeline needs cuda_graph; disabling pipeline.", flush=True)
        S["pipeline"] = False

    return S


def _build_cmd(method: str, mode: str, S: Dict[str, Any], out: Path,
               port: int) -> List[str]:
    meta = METHODS[method]
    runner = str(REPO / meta["runner"])
    cmd: List[str] = []
    if meta["torchrun"]:
        # Use `python -m torch.distributed.run` instead of bare `torchrun` so
        # the subprocess inherits whichever interpreter is running main.py
        # (no PATH dependency / no conda activate needed).
        cmd += [sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node=2", f"--master_port={port}", runner]
    else:
        cmd += [sys.executable, runner]

    eos = ["--no_eos_stop"] if mode == "latency" else []
    cmd += [
        "--target_model", str(S["target_model"]),
        "--output_dir", str(out),
        "--num_samples", str(S["num_samples"]),
        "--num_blocks", str(S["num_blocks"][mode]),
        "--block_length", str(S["block_length"]),
        "--denoising_steps", str(S["denoising_steps"]),
        "--seed", str(S["seed"]),
        "--datasets", *S["datasets"],
    ] + eos

    if S["mask_token_id"] is not None:
        cmd += ["--mask_token_id", str(S["mask_token_id"])]
    if S["kv_cache_max_len"] is not None:
        cmd += ["--kv_cache_max_len", str(S["kv_cache_max_len"])]

    # Per-method capability ANDed with the experiment-level gate: a method that
    # never uses the graph stays off, and an eager experiment turns it off
    # everywhere (e.g. LLaDA2, whose MoE dispatch cannot be captured).
    if meta["use_cg"] and S["use_cuda_graph"]:
        cmd += ["--use_cuda_graph"]

    for flag, val in (S["spec"].get(method) or {}).items():
        if val is not None:
            cmd += [f"--{flag}", str(val)]

    if meta["needs_draft"]:
        cmd += ["--draft_model", str(S["draft_model"]),
                "--draft_device", S["draft_device"]]
    if meta["shards"]:
        cmd += ["--target_device", S["target_device"]]
        if S["target_gpus"]:
            cmd += ["--target_gpus", ",".join(str(g) for g in S["target_gpus"]),
                    "--target_max_memory", str(S["target_max_memory"])]
    if method == "native_sharded" and S["return_timings"]:
        cmd += ["--return_timings"]
    if meta["pipeline"] and S["pipeline"]:
        cmd += ["--pipeline"]

    if method == "ours":
        if S["return_timings"]:
            cmd += ["--return_timings"]
        # Both are cuda_graph-only optimisations; passing them under an eager
        # experiment would trip the runner's own asserts.
        if S["use_cuda_graph"]:
            if S["fused_denoise"]:
                cmd += ["--fused_denoise"]
            if S["speculative_target_extend"]:
                cmd += ["--speculative_target_extend"]
            # Folding the draft extend into the fused denoise graph needs that
            # graph to exist; the runner asserts the same thing.
            if S["fold_draft_extend"] and S["fused_denoise"]:
                cmd += ["--fold_draft_extend"]
        if S["fused_linear"] and S["fused_linear"] != "off":
            cmd += ["--fused_linear", str(S["fused_linear"])]
    return cmd


# ─────────────────────────────────────────────────────────────────────
# Subprocess driving
# ─────────────────────────────────────────────────────────────────────

def _killpg(pid: int, sig: int) -> None:
    """Send `sig` to the entire process group whose leader is `pid`.

    `start_new_session=True` in Popen makes the child its own pgid leader, so
    every descendent (torchrun → worker ranks → any greenlet) is reachable.
    """
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError) as e:
        print(f"  [watchdog] killpg({pid}, {sig}) skipped: {e}", flush=True)


def _run(method: str, mode: str, S: Dict[str, Any], args, out_root: Path,
         port: int) -> Optional[Dict]:
    """Launch one (method, mode) subprocess with a watchdog that defeats the
    NCCL + cuda_graph teardown deadlock.

    Background: `run_native_tp2_cache.py` (and friends) write `SUMMARY.json`
    just before calling `dist.destroy_process_group()`. When cuda_graph is
    active, that call sometimes hangs forever (captured graphs hold stream
    resources that NCCL collective destroy waits on). The original
    `subprocess.run(...)` had no timeout, so the launcher would block
    indefinitely and `_kill_residuals` (run AFTER subprocess.run returns)
    would never fire. Observed wall-time stuck = 4 days.

    Watchdog protocol:
      1. Spawn child with start_new_session=True so we own its pgid.
      2. Poll proc.wait + summary_path.exists().
      3. Once SUMMARY.json appears the logical work is done; start grace timer.
      4. If proc still alive after grace_s seconds, SIGTERM the pgrp;
         SIGKILL after 30 s more.
      5. Absolute timeout (job_timeout_s) caps the whole job for safety.
    """
    out = out_root / f"{method}_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "SUMMARY.json"

    if args.skip_done and summary_path.exists():
        print(f"\n══ [{method}/{mode}] skip (SUMMARY.json present) ══", flush=True)
        return json.loads(summary_path.read_text())

    # Stale SUMMARY would confuse the watchdog → clear before launch.
    summary_path.unlink(missing_ok=True)

    cmd = _build_cmd(method, mode, S, out, port)
    log = out / "run.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = S["gpus"]

    print(f"\n══ [{method}/{mode}] launching ══", flush=True)
    print(f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    print(f"  log: {log}", flush=True)

    t0 = time.time()
    rc: int = -1
    forced = False
    with open(log, "w") as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env, cwd=str(REPO),
            start_new_session=True,
        )

        kill_deadline: Optional[float] = None
        job_deadline = t0 + args.job_timeout_s if args.job_timeout_s > 0 else None
        poll_interval = 5.0

        while True:
            try:
                rc = proc.wait(timeout=poll_interval)
                break
            except subprocess.TimeoutExpired:
                pass

            now = time.time()

            # Once logical work is done, give the child a bounded window to
            # finish teardown (NCCL destroy + cuda_graph cleanup).
            if kill_deadline is None and summary_path.exists():
                kill_deadline = now + args.cleanup_grace_s
                print(f"  [watchdog] SUMMARY.json written at "
                      f"+{now - t0:.0f}s; grace={args.cleanup_grace_s}s",
                      flush=True)

            if kill_deadline is not None and now > kill_deadline:
                print(f"  [watchdog] grace expired (likely NCCL/cuda_graph "
                      f"teardown hang); SIGTERM pgrp {proc.pid}", flush=True)
                _killpg(proc.pid, signal.SIGTERM)
                forced = True
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f"  [watchdog] SIGTERM ignored; SIGKILL pgrp",
                          flush=True)
                    _killpg(proc.pid, signal.SIGKILL)
                    try:
                        rc = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        rc = -signal.SIGKILL
                break

            if job_deadline is not None and now > job_deadline:
                print(f"  [watchdog] job timeout ({args.job_timeout_s}s) "
                      f"exceeded; SIGKILL pgrp {proc.pid}", flush=True)
                _killpg(proc.pid, signal.SIGKILL)
                forced = True
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    rc = -signal.SIGKILL
                break

    dt = time.time() - t0
    tag = " (forced)" if forced else ""
    print(f"  [{method}/{mode}] exit={rc}  elapsed={dt:.1f}s{tag}", flush=True)

    if not summary_path.exists():
        print(f"  [{method}/{mode}] !! no SUMMARY.json — see {log}", flush=True)
        return None
    return json.loads(summary_path.read_text())


def _print_table(combined: Dict[Tuple[str, str], Optional[Dict]],
                 methods: List[str], modes: List[str]) -> None:
    """Unified table; columns adapt to which fields are populated per mode."""
    rows: List[List[str]] = []
    header = ["method", "mode", "dataset", "n", "tok/s", "ms/tok", "pass@1", "α"]
    rows.append(header)
    rows.append(["-" * len(h) for h in header])
    for method in methods:
        for mode in modes:
            summary = combined.get((method, mode))
            if summary is None:
                rows.append([method, mode, "—", "—", "—", "—", "—", "—"])
                continue
            for run in summary.get("runs", []):
                tps = run.get("tokens_per_second")
                mpt = run.get("ms_per_token")
                pa1 = run.get("pass_at_1")
                alpha = run.get("mean_accept_rate")
                rows.append([
                    method, mode, run.get("dataset", "?"), str(run.get("n", "?")),
                    f"{tps:.1f}" if tps is not None else "—",
                    f"{mpt:.2f}" if mpt is not None else "—",
                    f"{pa1:.3f}" if pa1 is not None else "—",
                    f"{alpha:.3f}" if alpha is not None else "—",
                ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for r in rows:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)), flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--experiment", type=Path, default=DEFAULT_EXPERIMENT,
                   help="Experiment YAML. Default: configs/experiments/sdar.yaml")
    p.add_argument("--print_config", action="store_true",
                   help="Print the resolved experiment (YAML + CLI overrides) "
                        "and the commands that would run, then exit.")

    # ── overrides: None means "not given, take it from the YAML" ──
    p.add_argument("--target_model", help="Override models.target.")
    p.add_argument("--draft_model", help="Override models.draft.")
    p.add_argument("--mask_token_id", type=int,
                   help="Override tokens.mask_token_id.")
    p.add_argument("--output_dir", help="Override output_dir.")
    p.add_argument("--datasets", nargs="+", help="Override data.datasets.")
    p.add_argument("--num_samples", type=int, help="Override data.num_samples.")
    p.add_argument("--block_length", type=int, help="Override block.block_length.")
    p.add_argument("--denoising_steps", type=int,
                   help="Override block.denoising_steps.")
    p.add_argument("--latency_num_blocks", type=int,
                   help="Override mode_settings.latency.num_blocks.")
    p.add_argument("--quality_num_blocks", type=int,
                   help="Override mode_settings.quality.num_blocks.")
    p.add_argument("--seed", type=int, help="Override runtime.seed.")
    p.add_argument("--methods", nargs="+", choices=list(METHODS),
                   help="Override methods.")
    p.add_argument("--modes", nargs="+", choices=["latency", "quality"],
                   help="Override modes.")
    p.add_argument("--gpus", help="Override runtime.gpus (CUDA_VISIBLE_DEVICES).")
    p.add_argument("--draft_device", help="Override runtime.draft_device.")
    p.add_argument("--target_device", help="Override runtime.target_device.")
    p.add_argument("--target_gpus",
                   help="Override runtime.target_gpus: shard the target across "
                        "these LOCAL GPU indices, e.g. '1,2,3'. For targets too "
                        "big for one card.")
    p.add_argument("--target_max_memory",
                   help="Override runtime.target_max_memory (per-GPU cap for "
                        "the sharding planner, e.g. '88GiB').")
    p.add_argument("--spec", action="append", metavar="[METHOD.]KEY=VALUE",
                   help="Override one decode knob after the YAML; repeatable. "
                        "'ours.confidence_threshold=0.5' targets one method, "
                        "'confidence_threshold=0.5' every method that accepts "
                        "it. Lets a sweep vary one value without copying the "
                        "whole experiment file.")

    for name in ("use_cuda_graph", "pipeline", "fused_denoise",
                 "speculative_target_extend", "return_timings",
                 "fold_draft_extend"):
        p.add_argument(f"--{name}", dest=name, action="store_true", default=None,
                       help=f"Force runtime.{name} on.")
        p.add_argument(f"--no_{name}", dest=name, action="store_false",
                       default=None, help=f"Force runtime.{name} off.")

    p.add_argument("--fused_linear", choices=("off", "draft", "target", "both"),
                   help="Override runtime.fused_linear: route nn.Linear through "
                        "the fused split-K GEMV kernel. `ours` only.")
    p.add_argument("--kv_cache_max_len", type=int,
                   help="Override runtime.kv_cache_max_len; 0 = auto-size per "
                        "dataset to max_prompt_len + gen_length.")

    p.add_argument("--master_port_base", type=int, default=29571)
    p.add_argument("--skip_done", action="store_true",
                   help="If a job's SUMMARY.json already exists, skip rerunning "
                        "it (resume mode). Default: rerun and overwrite.")
    p.add_argument("--cleanup_grace_s", type=int, default=90,
                   help="After SUMMARY.json is written, wait this many seconds "
                        "for the child to exit cleanly before force-killing its "
                        "process group. Counters the NCCL + cuda_graph teardown "
                        "deadlock observed on 2026-05-17.")
    p.add_argument("--job_timeout_s", type=int, default=14400,
                   help="Absolute per-job wall-time cap (seconds). 0 = disabled. "
                        "Default 4h covers worst-case quality run @ N=200.")
    args = p.parse_args()

    exp = load_experiment(args.experiment)
    S = resolve(args, exp)

    out_root = Path(S["output_dir"]).resolve()
    bl = S["block_length"]

    print(f"[main] experiment = {S['name']}  ({args.experiment})", flush=True)
    print(f"[main] family     = {S['family']}  "
          f"mask_token_id={S['mask_token_id']}", flush=True)
    print(f"[main] target     = {S['target_model']}", flush=True)
    print(f"[main] draft      = {S['draft_model']}", flush=True)
    print(f"[main] output_dir = {out_root}", flush=True)
    tgt_place = (f"sharded over {S['target_gpus']} @ {S['target_max_memory']}"
                 if S["target_gpus"] else S["target_device"])
    print(f"[main] gpus = {S['gpus']}  draft_device={S['draft_device']}  "
          f"target={tgt_place}", flush=True)
    print(f"[main] cuda_graph={S['use_cuda_graph']}  pipeline={S['pipeline']}",
          flush=True)
    print(f"[main] methods = {S['methods']}", flush=True)
    print(f"[main] modes   = {S['modes']}", flush=True)
    print(f"[main] datasets = {S['datasets']}  N = {S['num_samples']}", flush=True)
    for m in S["modes"]:
        nb = S["num_blocks"][m]
        eos = "no EOS stop" if m == "latency" else "EOS stop"
        print(f"[main] {m}: num_blocks={nb} bl={bl} ({nb * bl} tokens, {eos})",
              flush=True)

    if args.print_config:
        print("\n══ resolved experiment ══", flush=True)
        print(json.dumps(S, indent=2, default=str), flush=True)
        print("\n══ commands ══", flush=True)
        for method in S["methods"]:
            for mode in S["modes"]:
                cmd = _build_cmd(method, mode, S,
                                 out_root / f"{method}_{mode}",
                                 args.master_port_base)
                print(f"  [{method}/{mode}] "
                      f"{' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        return 0

    out_root.mkdir(parents=True, exist_ok=True)

    combined: Dict[Tuple[str, str], Optional[Dict]] = {}
    port_offset = 0
    for method in S["methods"]:
        for mode in S["modes"]:
            combined[(method, mode)] = _run(
                method, mode, S, args, out_root,
                port=args.master_port_base + port_offset,
            )
            port_offset += 1

    unified = {
        "config": S,
        "per_method_mode": {f"{m}_{md}": v for (m, md), v in combined.items()},
    }
    (out_root / "UNIFIED.json").write_text(
        json.dumps(unified, indent=2, default=str))

    print("\n══ RESULTS ══", flush=True)
    _print_table(combined, S["methods"], S["modes"])
    print(f"\n[done] unified summary at {out_root / 'UNIFIED.json'}", flush=True)
    return 0 if all(v is not None for v in combined.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
