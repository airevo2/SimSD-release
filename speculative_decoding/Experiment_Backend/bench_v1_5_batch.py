"""v1.5 batched bench wrapper.

Applies BOTH the single-prompt v1.5 patch (so backend.generate keeps working
the same way it does in the existing v1.5 bench) AND the batched v1.5 patch
(so backend.generate_batch  newly added  gets the same semantics under
batch>1). Then delegates CLI to ``speculative_decoding.bench.run_benchmark``.

After the bench, dumps a ``v1_5_batch_timings.json`` next to ``--output`` with
the per-call CUDA-event breakdown of the batched extra forward.

Usage:
  python speculative_decoding/Experiment_Backend/bench_v1_5_batch.py \\
      --compare both --runtime hf \\
      --draft_model inference/model/SDAR-1_7B-Chat \\
      --target_model inference/model/SDAR-8B-Chat \\
      --draft_device cuda:0 --target_device cuda:0 \\
      --batch 8 --K 1 --num_blocks 32 --block_length 4 --denoising_steps 4 \\
      --use_cuda_graph --target_eval_sdpa --no_eos_stop --seed 42 \\
      --output fix_422/13_v1_5_batch/bs8_K1/bench.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Apply BOTH patches as side-effects (single + batch).
from speculative_decoding.Experiment_Backend import draft_probs_ablation_v1_5  # noqa: E402,F401
from speculative_decoding.Experiment_Backend import draft_probs_ablation_v1_5_batch  # noqa: E402
from speculative_decoding.bench import run_benchmark as _bench  # noqa: E402


def _dump_timings(output_path: str) -> None:
    # Read both single and batch v1.5 timings (only one will be populated for
    # a given run, depending on cfg.batch).
    single_records = draft_probs_ablation_v1_5.get_timings()
    batch_records = draft_probs_ablation_v1_5_batch.get_timings()

    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    def _summarize(records):
        if not records:
            return None

        def _mean(key: str) -> float:
            vals = [r[key] for r in records if key in r]
            return sum(vals) / len(vals) if vals else 0.0

        mean_den = _mean("denoising_ms")
        mean_extra = _mean("extra_forward_ms")
        mean_softmax = _mean("softmax_cpu_ms")
        ratio = (mean_extra / mean_den) if mean_den > 0 else 0.0
        return {
            "n_calls": len(records),
            "denoising_ms_mean": mean_den,
            "extra_forward_ms_mean": mean_extra,
            "softmax_cpu_ms_mean": mean_softmax,
            "extra_over_denoising_ratio": ratio,
        }

    payload = {
        "single": {"records": single_records, "aggregate": _summarize(single_records)},
        "batch":  {"records": batch_records,  "aggregate": _summarize(batch_records)},
        "notes": (
            "single = v1.5 single-prompt path (one entry per draft_one_block call). "
            "batch  = v1.5 batched path (one entry per draft_one_block_batch call, "
            "shared across B rows). For batch>1 runs only the 'batch' block is "
            "populated; for batch=1 runs only the 'single' block."
        ),
    }
    timings_path = os.path.join(out_dir, "v1_5_batch_timings.json")
    with open(timings_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench_v1_5_batch] wrote {timings_path}", file=sys.stderr)


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
