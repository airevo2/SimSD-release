"""v1.5 bench wrapper  applies the v1.5 draft_one_block patch as a side-effect,
then delegates CLI parsing to ``speculative_decoding.bench.run_benchmark``.

Also dumps a ``v1_5_timings.json`` next to ``--output`` with the per-call
CUDA-event breakdown collected from ``draft_probs_ablation_v1_5._V15_TIMINGS``.

Shape of v1_5_timings.json:
{
  "records": [{"denoising_ms": ..., "extra_forward_ms": ...,
               "softmax_cpu_ms": ..., "ctx_len": ..., "block_length": ...,
               "seq_len": ...}, ...],
  "aggregate": {
    "n_calls": int,
    "denoising_ms_mean": float,
    "extra_forward_ms_mean": float,
    "softmax_cpu_ms_mean": float,
    "extra_over_denoising_ratio": float,  # = extra / denoising
  }
}

Usage (example):
  python speculative_decoding/Experiment_Backend/bench_v1_5.py \\
      --compare both --runtime hf \\
      --draft_model inference/model/SDAR-1_7B-Chat \\
      --target_model inference/model/SDAR-8B-Chat \\
      --draft_device cuda:0 --target_device cuda:0 \\
      --dataset gsm8k --num_samples 10 --warmup 2 \\
      --num_blocks 32 --block_length 4 --denoising_steps 4 --K 8 \\
      --no_eos_stop --target_eval_sdpa --seed 42 \\
      --output fix_422/05_v1_5_latency/K8_1_7B/bench.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Apply the v1.5 draft_one_block patch BEFORE bench.run_benchmark imports it.
from speculative_decoding.Experiment_Backend import draft_probs_ablation_v1_5  # noqa: E402,F401
from speculative_decoding.bench import run_benchmark as _bench  # noqa: E402


def _dump_timings(output_path: str) -> None:
    records = draft_probs_ablation_v1_5.get_timings()
    if not records:
        return

    def _mean(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    mean_den = _mean("denoising_ms")
    mean_extra = _mean("extra_forward_ms")
    mean_softmax = _mean("softmax_cpu_ms")
    ratio = (mean_extra / mean_den) if mean_den > 0 else 0.0

    payload = {
        "records": records,
        "aggregate": {
            "n_calls": len(records),
            "denoising_ms_mean": mean_den,
            "extra_forward_ms_mean": mean_extra,
            "softmax_cpu_ms_mean": mean_softmax,
            "extra_over_denoising_ratio": ratio,
            "notes": (
                "CUDA-event timed; denoising_ms = original draft_one_block "
                "call (N denoising forwards); extra_forward_ms = v1.5 single "
                "intra_md-causal forward; softmax_cpu_ms = softmax(float) + "
                ".cpu() on block_length × V tensor."
            ),
        },
    }
    base = os.path.splitext(output_path)[0]
    timings_path = f"{os.path.dirname(output_path) or '.'}/v1_5_timings.json"
    # If --output lives in a directory, write sibling file. Otherwise use
    # current dir + basename-prefixed file.
    out_dir = os.path.dirname(output_path)
    if out_dir:
        timings_path = os.path.join(out_dir, "v1_5_timings.json")
    else:
        timings_path = f"v1_5_timings_{os.path.basename(base)}.json"
    os.makedirs(os.path.dirname(timings_path) or ".", exist_ok=True)
    with open(timings_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench_v1_5] wrote {timings_path} "
          f"(n={len(records)}, denoising={mean_den:.2f}ms, "
          f"extra={mean_extra:.2f}ms, ratio={ratio:.2%})",
          file=sys.stderr)


def _find_output_arg() -> str | None:
    for i, a in enumerate(sys.argv):
        if a == "--output" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith("--output="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    output_arg = _find_output_arg()
    try:
        _bench.main()
    finally:
        if output_arg:
            _dump_timings(output_arg)
