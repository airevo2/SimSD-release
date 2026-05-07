"""Experiment 2  K × batch × dataset sweep @ cuda_graph=ON.

Goal: for each (draft, dataset), measure speculative TPS across
K ∈ {1, 2, 4} × batch ∈ {1, 4, 16, 32, 64}. Native denominators are reused
from Experiment 1 (run_batch_sweep.py writes them into batch_sweep/), so we
only run `--compare speculative` here to cut the wall-clock roughly in half.

1.7B main sweep: 3 K × 5 batch × 3 dataset = 45 runs.
4B sparse control: 3 K × 3 batch × 3 dataset = 27 runs.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

from common import RESULTS, append_oom, ensure_dirs, is_cuda_oom, run_bench


Ks = [1, 2, 4]
# Intended once batch>1 support lands:
#   BATCHES_MAIN = [1, 4, 16, 32, 64]; BATCHES_EXTRA = [1, 16, 64]
BATCHES_MAIN = [1]
BATCHES_EXTRA = [1]
DATASETS = ["humaneval", "mbpp", "ifeval"]
MAIN_DRAFTS = ["SDAR-1_7B-Chat"]
EXTRA_DRAFTS = ["SDAR-4B-Chat"]
NUM_SAMPLES = 35


def iter_runs():
    for draft in MAIN_DRAFTS:
        for ds in DATASETS:
            for K, bs in itertools.product(Ks, BATCHES_MAIN):
                yield draft, ds, K, bs
    for draft in EXTRA_DRAFTS:
        for ds in DATASETS:
            for K, bs in itertools.product(Ks, BATCHES_EXTRA):
                yield draft, ds, K, bs


def main() -> int:
    ensure_dirs()
    out = RESULTS / "k_batch_dataset_sweep"
    oom_csv = RESULTS / "oom_runs.csv"
    rc_any_fail = 0

    for draft, ds, K, bs in iter_runs():
        tag = f"{draft}_{ds}_K{K}_bs{bs}"
        log = out / f"log_{tag}.txt"
        print(f"[exp2] draft={draft} dataset={ds} K={K} bs={bs}", flush=True)
        rc = run_bench(
            draft=draft, batch=bs, K=K,
            dataset=ds, num_samples=NUM_SAMPLES,
            compare="speculative",
            out_path=out / f"bench_{tag}.json",
            log_path=log,
        )
        if rc != 0:
            rc_any_fail = rc
            if is_cuda_oom(log):
                append_oom(oom_csv, f"exp2,{tag}")
            else:
                print(f"[exp2]  !! rc={rc} tag={tag} (see {log})", flush=True)

    print("[exp2] done")
    return 0 if rc_any_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
