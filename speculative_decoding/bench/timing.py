"""CUDA  +  bench """

from __future__ import annotations

import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence


def cuda_sync_if_available():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """p in [0, 100]. `sorted_vals` must be sorted ascending."""
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    """ end_to_end_ms / n_blocks"""
    if not values:
        return {
            "n": 0.0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    vs = [float(x) for x in values]
    vs_sorted = sorted(vs)
    return {
        "n": float(len(vs)),
        "mean_ms": float(statistics.mean(vs)),
        "std_ms": float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
        "p50_ms": percentile(vs_sorted, 50),
        "p95_ms": percentile(vs_sorted, 95),
    }


def summarize_scalar(values: Sequence[float], suffix: str = "") -> Dict[str, float]:
    """ tokens/s"""
    if not values:
        return {
            "n": 0.0,
            f"mean{suffix}": float("nan"),
            f"std{suffix}": float("nan"),
            f"p50{suffix}": float("nan"),
            f"p95{suffix}": float("nan"),
        }
    vs = sorted(float(x) for x in values)
    return {
        "n": float(len(vs)),
        f"mean{suffix}": float(statistics.mean(vs)),
        f"std{suffix}": float(statistics.pstdev(vs)) if len(vs) > 1 else 0.0,
        f"p50{suffix}": percentile(vs, 50),
        f"p95{suffix}": percentile(vs, 95),
    }


def summarize_latency_ms(seconds: Sequence[float]) -> Dict[str, float]:
    """ -> """
    if not seconds:
        return {
            "n": 0.0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
        }
    ms = [s * 1000.0 for s in seconds]
    ms_sorted = sorted(ms)
    return {
        "n": float(len(ms)),
        "mean_ms": float(statistics.mean(ms)),
        "std_ms": float(statistics.pstdev(ms)) if len(ms) > 1 else 0.0,
        "p50_ms": percentile(ms_sorted, 50),
        "p95_ms": percentile(ms_sorted, 95),
    }


def merge_timing_dicts(rows: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """ per-run timing  key  summarize_latency_ms"""
    out: Dict[str, List[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            if k in row and row[k] is not None:
                out[k].append(float(row[k]))
    return {k: summarize_latency_ms(v) for k, v in out.items()}
