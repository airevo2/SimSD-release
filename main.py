#!/usr/bin/env python3
"""SimSD: one-shot latency + quality benchmark for vanilla / vanilla+CG / ours.

Two modes per method (6 subprocess runs total):
  latency : --no_eos_stop, num_blocks=32 → 128 token fixed-length gen
            (pure throughput; pass@1 not meaningful for truncated gen).
  quality : EOS stop on,   num_blocks=128 → up to 512 token budget
            (lets answer terminate naturally; pass@1 + α reported).

Throughput is computed from `gpu_event_ms` (torch.cuda.Event), not wall clock.

Methods
  vanilla    : main_table/run_vanilla_tp2_cache.py   TP=2 multinomial
  vanilla_cg : main_table/run_native_tp2_cache.py    TP=2 argmax + cuda_graph
  ours       : main_table/run_ours_dual_gpu.py       dual-GPU SimSD spec + cg

Usage
  python main.py \\
      --target_model JetLM/SDAR-8B-Chat \\
      --draft_model  JetLM/SDAR-1.7B-Chat \\
      --datasets gsm8k humaneval mbpp \\
      --num_samples 200 \\
      --output_dir runs/main
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
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent

METHODS = {
    "vanilla":    {"runner": "main_table/run_vanilla_tp2_cache.py", "torchrun": True,  "use_cg": False, "needs_draft": False, "pipeline": False},
    "vanilla_cg": {"runner": "main_table/run_native_tp2_cache.py",  "torchrun": True,  "use_cg": True,  "needs_draft": False, "pipeline": False},
    # "ours" = SD+pipe per plan/NAMING.md (post 2026-05-06): pipelined cache_aware
    # with draft on cuda:0, target on cuda:1, overlap via cuda_graph.
    "ours":       {"runner": "main_table/run_ours_dual_gpu.py",     "torchrun": False, "use_cg": True,  "needs_draft": True,  "pipeline": True},
}


def _build_cmd(method: str, mode: str, args, out: Path, port: int) -> List[str]:
    spec = METHODS[method]
    runner = str(REPO / spec["runner"])
    cmd: List[str] = []
    if spec["torchrun"]:
        # Use `python -m torch.distributed.run` instead of bare `torchrun` so
        # the subprocess inherits whichever interpreter is running main.py
        # (no PATH dependency / no conda activate needed).
        cmd += [sys.executable, "-m", "torch.distributed.run",
                "--nproc_per_node=2", f"--master_port={port}", runner]
    else:
        cmd += [sys.executable, runner]
    # Mode-specific knobs.
    if mode == "latency":
        nb = args.latency_num_blocks
        eos = ["--no_eos_stop"]
    elif mode == "quality":
        nb = args.quality_num_blocks
        eos = []
    else:
        raise ValueError(mode)
    cmd += [
        "--target_model", args.target_model,
        "--output_dir", str(out),
        "--num_samples", str(args.num_samples),
        "--num_blocks", str(nb),
        "--block_length", str(args.block_length),
        "--denoising_steps", str(args.denoising_steps),
        "--seed", str(args.seed),
        "--datasets", *args.datasets,
    ] + eos
    if spec["use_cg"]:
        cmd += ["--use_cuda_graph"]
    if spec["needs_draft"]:
        cmd += ["--draft_model", args.draft_model,
                "--draft_device", "cuda:0", "--target_device", "cuda:1"]
    if spec.get("pipeline"):
        cmd += ["--pipeline"]
    if method == "ours":
        cmd += ["--partial_block_fill", "truncate"]
        if args.return_timings:
            cmd += ["--return_timings"]
        if args.fused_denoise:
            cmd += ["--fused_denoise"]
        if args.speculative_target_extend:
            cmd += ["--speculative_target_extend"]
    return cmd


def _killpg(pid: int, sig: int) -> None:
    """Send `sig` to the entire process group whose leader is `pid`.

    `start_new_session=True` in Popen makes the child its own pgid leader, so
    every descendent (torchrun → worker ranks → any greenlet) is reachable.
    """
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError) as e:
        print(f"  [watchdog] killpg({pid}, {sig}) skipped: {e}", flush=True)


def _run(method: str, mode: str, args, out_root: Path, port: int) -> Optional[Dict]:
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

    cmd = _build_cmd(method, mode, args, out, port)
    log = out / "run.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus

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
    p.add_argument("--target_model", default="JetLM/SDAR-8B-Chat",
                   help="HF hub id or local path. Default: JetLM/SDAR-8B-Chat.")
    p.add_argument("--draft_model", default="JetLM/SDAR-1.7B-Chat",
                   help="HF hub id or local path. Used by 'ours'; ignored by "
                        "vanilla / vanilla_cg. Default: JetLM/SDAR-1.7B-Chat.")
    p.add_argument("--output_dir", default="runs/main_table",
                   help="Where SUMMARY.json / UNIFIED.json land. "
                        "Default: runs/main_table.")
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "mbpp", "triviaqa", "mmlu"])
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--latency_num_blocks", type=int, default=32,
                   help="num_blocks × block_length = fixed gen length for latency mode (default 32×4=128 tokens).")
    p.add_argument("--quality_num_blocks", type=int, default=128,
                   help="num_blocks × block_length = gen budget for quality mode (default 128×4=512 tokens, EOS stop on).")
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--methods", nargs="+",
                   default=["vanilla", "vanilla_cg", "ours"],
                   choices=list(METHODS),
                   help="Subset to run; default = all 3.")
    p.add_argument("--modes", nargs="+",
                   default=["latency", "quality"],
                   choices=["latency", "quality"],
                   help="Run latency, quality, or both; default = both.")
    p.add_argument("--gpus", default="0,1",
                   help="CUDA_VISIBLE_DEVICES for all subprocesses.")
    p.add_argument("--master_port_base", type=int, default=29571)
    p.add_argument("--skip_done", action="store_true",
                   help="If a job's SUMMARY.json already exists, skip rerunning "
                        "it (resume mode). Default: rerun and overwrite.")
    p.add_argument("--return_timings", action="store_true",
                   help="For 'ours' only: pass --return_timings to the runner "
                        "so per-stage latency breakdown (draft denoise / verify "
                        "/ extend / MRS-commit) is captured into SUMMARY.json.")
    p.add_argument("--fused_denoise", dest="fused_denoise",
                   action="store_true", default=True,
                   help="For 'ours' only: fuse 4 denoise steps into one "
                        "cuda_graph (default ON; use --no_fused_denoise to "
                        "disable for ablation).")
    p.add_argument("--no_fused_denoise", dest="fused_denoise",
                   action="store_false")
    p.add_argument("--speculative_target_extend",
                   dest="speculative_target_extend",
                   action="store_true", default=True,
                   help="For 'ours' only: speculatively extend target K/V "
                        "before MRS so target stream stays continuous "
                        "(default ON; use --no_speculative_target_extend "
                        "to disable for ablation).")
    p.add_argument("--no_speculative_target_extend",
                   dest="speculative_target_extend",
                   action="store_false")
    p.add_argument("--cleanup_grace_s", type=int, default=90,
                   help="After SUMMARY.json is written, wait this many seconds "
                        "for the child to exit cleanly before force-killing its "
                        "process group. Counters the NCCL + cuda_graph teardown "
                        "deadlock observed on 2026-05-17.")
    p.add_argument("--job_timeout_s", type=int, default=14400,
                   help="Absolute per-job wall-time cap (seconds). 0 = disabled. "
                        "Default 4h covers worst-case quality run @ N=200.")
    args = p.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[main] output_dir = {out_root}", flush=True)
    print(f"[main] gpus = {args.gpus}", flush=True)
    print(f"[main] methods = {args.methods}", flush=True)
    print(f"[main] modes   = {args.modes}", flush=True)
    print(f"[main] datasets = {args.datasets}  N = {args.num_samples}", flush=True)
    print(f"[main] latency: num_blocks={args.latency_num_blocks} bl={args.block_length} "
          f"({args.latency_num_blocks*args.block_length} tokens, no EOS stop)", flush=True)
    print(f"[main] quality: num_blocks={args.quality_num_blocks} bl={args.block_length} "
          f"({args.quality_num_blocks*args.block_length} token budget, EOS stop)", flush=True)

    combined: Dict[Tuple[str, str], Optional[Dict]] = {}
    port_offset = 0
    for method in args.methods:
        for mode in args.modes:
            combined[(method, mode)] = _run(
                method, mode, args, out_root,
                port=args.master_port_base + port_offset,
            )
            port_offset += 1

    unified = {
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "datasets": args.datasets,
            "num_samples": args.num_samples,
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "latency_num_blocks": args.latency_num_blocks,
            "quality_num_blocks": args.quality_num_blocks,
            "gpus": args.gpus,
            "methods": args.methods,
            "modes": args.modes,
        },
        "per_method_mode": {f"{m}_{md}": v for (m, md), v in combined.items()},
    }
    (out_root / "UNIFIED.json").write_text(json.dumps(unified, indent=2))

    print("\n══ RESULTS ══", flush=True)
    _print_table(combined, args.methods, args.modes)
    print(f"\n[done] unified summary at {out_root / 'UNIFIED.json'}", flush=True)
    return 0 if all(v is not None for v in combined.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
