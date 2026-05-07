# Experiment_Backend

Drivers + aggregator for the batch / K / dataset sweep (2026-04-20).

## Current scope (2026-04-20)

**K × dataset sweep at bs=1.** `batch>1` is guarded by
`NotImplementedError` in `run_benchmark.py` because the CUDA-graph buffers in
`draft.py` / `verify.py` and the backend generate loops still assume bs=1.
Once batch support lands, flip the `BATCHES` lists in the two driver scripts
back to the full `[1, 2, 4, 8, 16, 32, 64]` matrix  the rest of the pipeline
(aggregate CSV, figures, OOM fallback) already handles variable bs.

Runs today:
- Exp 1: `{1.7B, 4B}  8B` speculative × 3 datasets × bs=1 (6 runs)
         + native 8B × 3 datasets × bs=1 (3 baselines)
- Exp 2: 1.7B  8B speculative × K ∈ {1,2,4} × 3 datasets × bs=1 (9 runs)
         + 4B  8B speculative × K ∈ {1,2,4} × 3 datasets × bs=1 (9 runs)

## Layout

```
Experiment_Backend/
├── common.py                        # run_bench() helper wrapping run_benchmark.py
├── run_batch_sweep.py               # Experiment 1: batch sweep (spec + native)
├── run_k_batch_dataset_sweep.py     # Experiment 2: K × batch × dataset sweep (spec only)
├── aggregate_sweep.py               # merge JSONs  CSV + figures
└── results/
    ├── batch_sweep/                 # Exp 1 JSONs + logs
    ├── k_batch_dataset_sweep/       # Exp 2 JSONs + logs
    ├── oom_runs.csv                 # one row per OOM fallback
    ├── tps_speedup_matrix.csv       # aggregated matrix
    └── figure/                      # PNG outputs
```

## Running

Both experiments must run on truly-idle GPUs  wrap with `gpu_grabber.py` so the
bench only starts after two cards are free:

```bash
# Experiment 1  batch sweep (3–5 h)
tmux new -d -s exp1 "python gpu_grabber.py --mode dual \
    --task 'python -u speculative_decoding/Experiment_Backend/run_batch_sweep.py' \
    2>&1 | tee speculative_decoding/Experiment_Backend/results/exp1_grabber.log"

# Experiment 2  K × batch × dataset sweep (after Exp 1)
tmux new -d -s exp2 "python gpu_grabber.py --mode dual \
    --task 'python -u speculative_decoding/Experiment_Backend/run_k_batch_dataset_sweep.py' \
    2>&1 | tee speculative_decoding/Experiment_Backend/results/exp2_grabber.log"

# Aggregate
python speculative_decoding/Experiment_Backend/aggregate_sweep.py
```

## Output shape

Per-run JSON is the standard `bench_latency_*.json` schema (see
`speculative_decoding/CLAUDE.md  Benchmark Output`). `aggregate_sweep.py`
reads:
- `results.hf_native.throughput_tokens_per_sec.mean_tok_per_s`
- `results.hf_speculative.throughput_tokens_per_sec.mean_tok_per_s`
- `results.hf_speculative.acceptance.accept_rate`
- `comparison.speedup_native_over_spec_ms_ratio`

Exp 2 reuses Exp 1's native TPS per (dataset, batch) instead of re-running
native, so filename conventions must be kept:
- Exp 1 native: `bench_native_8B_{dataset}_bs{N}.json`
- Exp 1 spec:   `bench_{draft}_to_8B_bs{N}.json` (humaneval only)
- Exp 2 spec:   `bench_{draft}_{dataset}_K{K}_bs{N}.json`

## OOM fallback

CUDA graph capture at high batch × large model can OOM. `run_bench` writes
subprocess stdout/stderr to `log_{tag}.txt`; on non-zero rc the driver checks
for `"CUDA out of memory"` and appends the tag to `results/oom_runs.csv`.
The sweep keeps marching  missing rows show up as blanks in the aggregated
CSV and as NaN cells in the speedup heatmaps.
