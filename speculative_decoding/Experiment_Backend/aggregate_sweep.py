"""Aggregate bench JSONs from Experiment 1/2 into one CSV + figures.

Reads:
  results/batch_sweep/bench_*.json
  results/k_batch_dataset_sweep/bench_*.json

Writes:
  results/tps_speedup_matrix.csv
  results/figure/speedup_vs_K_batch_{dataset}.png
  results/figure/tps_vs_batch_native_vs_spec.png
  results/figure/accept_rate_vs_K.png

Fields per row: draft, dataset, K, batch, native_tps, spec_tps, speedup,
accept_rate, p50_ms_native, p50_ms_spec.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common import FIGURE, RESULTS


FNAME_EXP1_SPEC = re.compile(r"^bench_(?P<draft>SDAR-[^_]+)_to_8B_bs(?P<bs>\d+)\.json$")
FNAME_EXP1_NATIVE = re.compile(r"^bench_native_8B_(?P<ds>[a-z]+)_bs(?P<bs>\d+)\.json$")
FNAME_EXP2 = re.compile(
    r"^bench_(?P<draft>SDAR-[^_]+)_(?P<ds>[a-z]+)_K(?P<K>\d+)_bs(?P<bs>\d+)\.json$"
)


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"[aggregate] skip {path.name}: {e}", file=sys.stderr)
        return None


def _tps(doc: dict, key: str) -> Optional[float]:
    r = doc.get("results", {}).get(key) or {}
    return (r.get("throughput_tokens_per_sec") or {}).get("mean_tok_per_s")


def _p50_ms(doc: dict, key: str) -> Optional[float]:
    r = doc.get("results", {}).get(key) or {}
    return (r.get("end_to_end_ms") or {}).get("p50_ms")


def _accept_rate(doc: dict) -> Optional[float]:
    return (doc.get("results", {}).get("hf_speculative", {})
            .get("acceptance", {}) or {}).get("accept_rate")


def collect() -> Tuple[List[dict], Dict[Tuple[str, int], float],
                        Dict[Tuple[str, int], float]]:
    """Scan results/ and return (rows, native_tps_by[ds,bs], native_p50_by[ds,bs])."""
    rows: List[dict] = []
    native_tps: Dict[Tuple[str, int], float] = {}
    native_p50: Dict[Tuple[str, int], float] = {}

    batch_dir = RESULTS / "batch_sweep"
    k_dir = RESULTS / "k_batch_dataset_sweep"

    # Exp 1 native (the speedup denominators).
    for p in sorted(batch_dir.glob("bench_native_8B_*.json")):
        m = FNAME_EXP1_NATIVE.match(p.name)
        if not m:
            continue
        doc = _load(p)
        if doc is None:
            continue
        ds, bs = m["ds"], int(m["bs"])
        tps = _tps(doc, "hf_native")
        p50 = _p50_ms(doc, "hf_native")
        if tps is not None:
            native_tps[(ds, bs)] = tps
        if p50 is not None:
            native_p50[(ds, bs)] = p50

    # Exp 1 speculative (K=1, humaneval).
    for p in sorted(batch_dir.glob("bench_SDAR-*_to_8B_bs*.json")):
        m = FNAME_EXP1_SPEC.match(p.name)
        if not m:
            continue
        doc = _load(p)
        if doc is None:
            continue
        draft, bs = m["draft"], int(m["bs"])
        ds = "humaneval"  # Exp 1 spec is humaneval only
        spec_tps = _tps(doc, "hf_speculative")
        native_tps_here = _tps(doc, "hf_native") or native_tps.get((ds, bs))
        ratio = (doc.get("comparison") or {}).get("speedup_native_over_spec_ms_ratio")
        rows.append({
            "experiment": "exp1",
            "draft": draft,
            "dataset": ds,
            "K": 1,
            "batch": bs,
            "native_tps": native_tps_here,
            "spec_tps": spec_tps,
            "speedup": ratio,
            "accept_rate": _accept_rate(doc),
            "p50_ms_native": _p50_ms(doc, "hf_native") or native_p50.get((ds, bs)),
            "p50_ms_spec": _p50_ms(doc, "hf_speculative"),
            "source_file": str(p.relative_to(RESULTS)),
        })

    # Exp 2 speculative only  pull native from the (ds, bs) lookup.
    for p in sorted(k_dir.glob("bench_SDAR-*_*_K*_bs*.json")):
        m = FNAME_EXP2.match(p.name)
        if not m:
            continue
        doc = _load(p)
        if doc is None:
            continue
        draft, ds, K, bs = m["draft"], m["ds"], int(m["K"]), int(m["bs"])
        spec_tps = _tps(doc, "hf_speculative")
        nat_tps = native_tps.get((ds, bs))
        speedup = (spec_tps / nat_tps) if (spec_tps and nat_tps) else None
        rows.append({
            "experiment": "exp2",
            "draft": draft,
            "dataset": ds,
            "K": K,
            "batch": bs,
            "native_tps": nat_tps,
            "spec_tps": spec_tps,
            "speedup": speedup,
            "accept_rate": _accept_rate(doc),
            "p50_ms_native": native_p50.get((ds, bs)),
            "p50_ms_spec": _p50_ms(doc, "hf_speculative"),
            "source_file": str(p.relative_to(RESULTS)),
        })

    return rows, native_tps, native_p50


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        print(f"[aggregate] no rows  skipping {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["experiment", "draft", "dataset", "K", "batch", "native_tps",
            "spec_tps", "speedup", "accept_rate", "p50_ms_native",
            "p50_ms_spec", "source_file"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {path} ({len(rows)} rows)")


def _plot_speedup_heatmaps(rows: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[aggregate] matplotlib not available; skipping figures")
        return

    by_ds: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["experiment"] != "exp2":
            continue
        if r["speedup"] is None:
            continue
        by_ds[r["dataset"]].append(r)

    for ds, rs in by_ds.items():
        Ks = sorted({r["K"] for r in rs})
        bss = sorted({r["batch"] for r in rs})
        drafts = sorted({r["draft"] for r in rs})
        for draft in drafts:
            mat = np.full((len(Ks), len(bss)), np.nan)
            for r in rs:
                if r["draft"] != draft:
                    continue
                i, j = Ks.index(r["K"]), bss.index(r["batch"])
                mat[i, j] = r["speedup"]
            fig, ax = plt.subplots(figsize=(5, 3))
            im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdYlGn",
                           vmin=0.5, vmax=2.5)
            ax.set_xticks(range(len(bss)))
            ax.set_xticklabels(bss)
            ax.set_yticks(range(len(Ks)))
            ax.set_yticklabels(Ks)
            ax.set_xlabel("batch")
            ax.set_ylabel("K")
            ax.set_title(f"speedup  {draft}  8B  {ds}")
            for i, K in enumerate(Ks):
                for j, bs in enumerate(bss):
                    v = mat[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=8)
            fig.colorbar(im, ax=ax, label="TPS spec / TPS native")
            fig.tight_layout()
            out = FIGURE / f"speedup_vs_K_batch_{draft}_{ds}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"[aggregate] wrote {out}")


def _plot_tps_vs_batch(rows: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    by_ds: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    for ds, rs in by_ds.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        nat = sorted([r for r in rs if r["experiment"] == "exp1"
                      and r["draft"] == "SDAR-8B-Chat" or
                      r["native_tps"] is not None],
                     key=lambda r: r["batch"])
        # De-dup native series on (batch,) since Exp 1 both-runs and native-only
        # both carry a native_tps.
        seen = {}
        for r in rs:
            if r["native_tps"] is None:
                continue
            seen.setdefault(r["batch"], r["native_tps"])
        if seen:
            xs = sorted(seen)
            ax.plot(xs, [seen[x] for x in xs], "k-o", label="native 8B")
        drafts = sorted({r["draft"] for r in rs if r["spec_tps"] is not None})
        for draft in drafts:
            for K in sorted({r["K"] for r in rs if r["draft"] == draft}):
                pts = sorted(
                    [r for r in rs if r["draft"] == draft and r["K"] == K
                     and r["spec_tps"] is not None],
                    key=lambda r: r["batch"])
                if not pts:
                    continue
                ax.plot([r["batch"] for r in pts],
                        [r["spec_tps"] for r in pts],
                        marker="o", label=f"{draft} K={K}")
        ax.set_xscale("log")
        ax.set_xlabel("batch")
        ax.set_ylabel("TPS (mean tok/s)")
        ax.set_title(f"TPS vs batch  {ds}")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        out = FIGURE / f"tps_vs_batch_{ds}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[aggregate] wrote {out}")


def _plot_accept_rate_vs_K(rows: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    pts: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["accept_rate"] is None or r["K"] is None:
            continue
        pts[(r["draft"], r["dataset"])][r["K"]].append(r["accept_rate"])
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for (draft, ds), by_K in pts.items():
        xs = sorted(by_K)
        ys = [sum(by_K[k]) / len(by_K[k]) for k in xs]
        ax.plot(xs, ys, marker="o", label=f"{draft}  {ds}")
    ax.set_xlabel("K")
    ax.set_ylabel("accept_rate (mean over batches)")
    ax.set_title("Accept rate vs K")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGURE / "accept_rate_vs_K.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[aggregate] wrote {out}")


def main() -> int:
    rows, _, _ = collect()
    write_csv(rows, RESULTS / "tps_speedup_matrix.csv")
    _plot_speedup_heatmaps(rows)
    _plot_tps_vs_batch(rows)
    _plot_accept_rate_vs_K(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
