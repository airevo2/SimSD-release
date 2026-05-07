"""v1.5_graph bench wrapper  v1.5 patch with CUDA-graph-accelerated extra
forward. Same side-effects + timings dump as bench_v1_5.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding.Experiment_Backend import draft_probs_ablation_v1_5_graph  # noqa: E402,F401
from speculative_decoding.bench import run_benchmark as _bench  # noqa: E402


def _dump_timings(output_path: str) -> None:
    records = draft_probs_ablation_v1_5_graph.get_timings()
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
                "v1.5_graph: extra forward goes through CUDA graph replay. "
                "Expected extra_forward_ms_mean << v1.5 baseline (~44 ms on "
                "1.7B @ K=8) if launch-bound."
            ),
        },
    }
    out_dir = os.path.dirname(output_path) or "."
    timings_path = os.path.join(out_dir, "v1_5_graph_timings.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(timings_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench_v1_5_graph] wrote {timings_path} "
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
