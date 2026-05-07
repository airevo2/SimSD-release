"""Shared helpers for batch / K / dataset sweep drivers.

run_bench() wraps run_benchmark.py as a subprocess so a crash / OOM in one
configuration cannot take down the whole sweep. Stdout+stderr are tee'd to a
per-run log file for post-mortem. is_cuda_oom() scans the log to separate OOM
failures from other crashes so the driver can fallback gracefully.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "speculative_decoding/bench/run_benchmark.py"
EXP_ROOT = REPO / "speculative_decoding/Experiment_Backend"
RESULTS = EXP_ROOT / "results"
FIGURE = RESULTS / "figure"


def run_bench(
    *,
    draft: str,
    target: str = "SDAR-8B-Chat",
    batch: int,
    K: int,
    num_blocks: int = 8,
    block_length: int = 4,
    denoising_steps: int = 4,
    dataset: str,
    num_samples: int,
    compare: str,
    out_path: Path,
    log_path: Path,
    draft_device: str = "cuda:0",
    target_device: str = "cuda:1",
    extra_args=None,
    timeout_s: int = 1800,
) -> int:
    """Spawn run_benchmark.py with the given knobs; returns the subprocess rc.

    All sweep runs go through this helper so every run has:
      - identical CLI flags except the swept knobs (fair comparison),
      - a dedicated log file for post-mortem,
      - a consistent output JSON path naming scheme (owned by the caller).
    """
    cmd = [
        sys.executable, "-u", str(BENCH),
        "--compare", compare,
        "--use_cuda_graph",
        "--no_eos_stop",
        "--draft_model", f"inference/model/{draft}",
        "--target_model", f"inference/model/{target}",
        "--draft_device", draft_device,
        "--target_device", target_device,
        "--batch", str(batch),
        "--K", str(K),
        "--num_blocks", str(num_blocks),
        "--block_length", str(block_length),
        "--denoising_steps", str(denoising_steps),
        "--dataset", dataset,
        "--num_samples", str(num_samples),
        "--output", str(out_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        try:
            proc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(REPO), timeout=timeout_s, check=False,
            )
            return proc.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n# TIMEOUT after {timeout_s}s\n")
            return 124


def is_cuda_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        txt = log_path.read_text()
    except Exception:
        return False
    return ("CUDA out of memory" in txt) or ("OutOfMemoryError" in txt)


def append_oom(oom_csv: Path, tag: str) -> None:
    oom_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(oom_csv, "a") as f:
        f.write(f"{tag}\n")


def ensure_dirs() -> None:
    for p in [RESULTS, FIGURE, RESULTS / "batch_sweep",
              RESULTS / "k_batch_dataset_sweep"]:
        p.mkdir(parents=True, exist_ok=True)
