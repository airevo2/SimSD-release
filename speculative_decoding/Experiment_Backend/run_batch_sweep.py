"""Experiment 1  batch sweep @ cuda_graph=ON, K=1.

NOTE (2026-04-20): batch>1 is currently guarded by NotImplementedError in
``run_benchmark.py``  the CUDA-graph buffers in draft.py / verify.py and the
backend generate loop still assume bs=1. Until that lands, this driver runs
bs=1 only so the sweep completes without producing misleading bs>N logs. Once
batch support is wired, flip ``BATCHES`` back to ``[1, 2, 4, 8, 16, 32, 64]``
(preserved below as the intended matrix).

Axes covered right now:
  - 1.7B8B speculative at bs=1
  - 4B8B speculative at bs=1
  - Native 8B bs=1 × {humaneval, mbpp, ifeval}  used as the speedup
    denominator in Experiment 2's aggregate CSV.

Every run is one subprocess spawned via common.run_bench, so a crash in one
configuration only writes an oom_runs.csv line (for OOM) or a driver log line
and the sweep keeps marching.
"""
from __future__ import annotations

import sys
from pathlib import Path

from common import RESULTS, append_oom, ensure_dirs, is_cuda_oom, run_bench


# Intended once batch>1 support lands in draft.py / verify.py / backends:
#   BATCHES = [1, 2, 4, 8, 16, 32, 64]
#   PARTIAL_BATCHES = [16, 32, 64]
BATCHES = [1]
DRAFTS_FULL = ["SDAR-1_7B-Chat", "SDAR-4B-Chat"]
DRAFTS_PARTIAL: list = []
PARTIAL_BATCHES: list = []
DATASETS = ["humaneval", "mbpp", "ifeval"]
NUM_SAMPLES = 35


def main() -> int:
    ensure_dirs()
    out = RESULTS / "batch_sweep"
    oom_csv = RESULTS / "oom_runs.csv"
    rc_any_fail = 0

    # 1) speculative: 1.7B  8B full batch sweep; 4B  8B at large batches only.
    for draft in DRAFTS_FULL + DRAFTS_PARTIAL:
        bs_list = BATCHES if draft in DRAFTS_FULL else PARTIAL_BATCHES
        for bs in bs_list:
            tag = f"{draft}_to_8B_bs{bs}"
            log = out / f"log_{tag}.txt"
            print(f"[exp1] speculative draft={draft} bs={bs}", flush=True)
            rc = run_bench(
                draft=draft, batch=bs, K=1,
                dataset="humaneval", num_samples=NUM_SAMPLES,
                compare="both",
                out_path=out / f"bench_{tag}.json",
                log_path=log,
            )
            if rc != 0:
                rc_any_fail = rc
                if is_cuda_oom(log):
                    append_oom(oom_csv, f"exp1_spec,{tag}")
                else:
                    print(f"[exp1]  !! rc={rc} tag={tag} (see {log})", flush=True)

    # 2) native 8B baseline × 7 batches × 3 datasets (Exp 2 speedup denominator).
    for ds in DATASETS:
        for bs in BATCHES:
            tag = f"native_8B_{ds}_bs{bs}"
            log = out / f"log_{tag}.txt"
            print(f"[exp1] native 8B dataset={ds} bs={bs}", flush=True)
            rc = run_bench(
                draft="SDAR-8B-Chat", batch=bs, K=1,
                dataset=ds, num_samples=NUM_SAMPLES,
                compare="native",
                out_path=out / f"bench_{tag}.json",
                log_path=log,
            )
            if rc != 0:
                rc_any_fail = rc
                if is_cuda_oom(log):
                    append_oom(oom_csv, f"exp1_native,{tag}")
                else:
                    print(f"[exp1]  !! rc={rc} tag={tag} (see {log})", flush=True)

    print("[exp1] done")
    return 0 if rc_any_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
