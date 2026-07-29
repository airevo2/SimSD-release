"""对拍: every candidate vs `F.linear`, on the persisted oracle AND fresh cases.

Coverage per CONTRACT.md:
  * all `oracle/*.pt` (real / fake / extreme) — the fixed ground truth that keeps
    candidates comparable across runs;
  * a fresh batch of **random** shapes and **special/edge** cases generated on the
    fly each run, with ground truth taken live from the reference. These are not
    persisted; their whole point is to differ every run.

A failure on *any* case fails the candidate. The tolerance is pinned in
CONTRACT.md and must never be loosened to make something pass.

Usage:
    CUDA_VISIBLE_DEVICES=4 python kernels/workspace/thin_linear/duipai.py            # promoted kernel
    CUDA_VISIBLE_DEVICES=4 python kernels/workspace/thin_linear/duipai.py -c v2_bn32
    CUDA_VISIBLE_DEVICES=4 python kernels/workspace/thin_linear/duipai.py --record baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from bench import graph_time_us, load_candidate  # noqa: E402

ORACLE = HERE / "oracle"
GPU_TAG = "RTX-PRO-6000-Blackwell"


def tol_for(ref: torch.Tensor) -> float:
    """CONTRACT.md: ~2-ULP-relative in bf16, ~1e-5-relative in fp32.

    Both bounds are *relative to* `ref.abs().max()`. For fp32 the repo convention
    reads "< 1e-5" and that was originally applied as an absolute bound, which no
    implementation can meet once the output is large: on a K=2175 case with
    |ref|max=378, cuBLAS itself lands 7.5e-5 from an fp64 ground truth and a
    plain chunked fp32 matmul differs from cuBLAS by 7.6e-5. Measured, in
    CONTRACT.md → "Tolerance". So the bound is scaled, not relaxed.
    """
    scale = max(1.0, float(ref.abs().max()))
    if ref.dtype == torch.float32:
        return 1e-5 * scale
    return 2 ** -7 * float(ref.abs().max()) + 1e-3


def check(fn, x, w, ref, label: str, rows: list[dict]) -> bool:
    try:
        with torch.inference_mode():
            got = fn(x, w)
    except Exception as e:                                  # noqa: BLE001
        rows.append({"case": label, "ok": False, "err": float("nan"),
                     "tol": float("nan"), "note": f"{type(e).__name__}: {e}"})
        return False

    if got.shape != ref.shape:
        rows.append({"case": label, "ok": False, "err": float("nan"),
                     "tol": float("nan"),
                     "note": f"shape {tuple(got.shape)} != {tuple(ref.shape)}"})
        return False
    if got.dtype != ref.dtype:
        rows.append({"case": label, "ok": False, "err": float("nan"),
                     "tol": float("nan"),
                     "note": f"dtype {got.dtype} != {ref.dtype}"})
        return False

    err = float((got.float() - ref.float()).abs().max())
    tol = tol_for(ref)
    # Bit-exactness vs cuBLAS is tracked separately from the tolerance: it is the
    # leading indicator for the byte-identical-text gate in
    # docs/kernel-optimization.md §4, which a 1-ULP difference can break via argmax.
    bitexact = 100 * int((got.view(-1) == ref.view(-1)).sum()) / max(1, ref.numel())
    rows.append({"case": label, "ok": err <= tol, "err": err, "tol": tol,
                 "bitexact_pct": round(bitexact, 3),
                 "note": f"bitexact {bitexact:.1f}%"})
    return err <= tol


# ── fresh (non-persisted) cases ───────────────────────────────────────────────
def fresh_cases(dev, n_random: int = 24):
    """Random shapes + hand-picked edges, ground truth taken live."""
    g = torch.Generator(device=dev).manual_seed(int(time.time()) & 0xFFFF)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=dev,
                            dtype=torch.float32) * scale).to(dtype)

    for i in range(n_random):
        M = int(torch.randint(1, 17, (1,), generator=g, device=dev))
        K = int(torch.randint(1, 4097, (1,), generator=g, device=dev))
        N = int(torch.randint(1, 4097, (1,), generator=g, device=dev))
        dt = torch.bfloat16 if i % 4 else torch.float32
        sc = float(10 ** int(torch.randint(-3, 3, (1,), generator=g, device=dev)))
        yield f"rand[{M}x{K}x{N},{str(dt).split('.')[-1]},s{sc:g}]", \
            rnd(M, K, dtype=dt, scale=sc), rnd(N, K, dtype=dt, scale=0.02)

    # special / edge
    yield "edge_single_element", rnd(1, 1), rnd(1, 1)
    yield "edge_N1", rnd(4, 2048), rnd(1, 2048, scale=0.02)
    yield "edge_K1", rnd(4, 1), rnd(2048, 1, scale=0.02)
    yield "edge_M1", rnd(1, 2048), rnd(2048, 2048, scale=0.02)
    yield "edge_K_odd_prime", rnd(4, 1021), rnd(769, 1021, scale=0.02)
    yield "edge_all_zeros", torch.zeros(4, 2048, device=dev, dtype=torch.bfloat16), \
        rnd(2048, 2048, scale=0.02)
    yield "edge_huge_mag", rnd(4, 2048, scale=1e3), rnd(2048, 2048, scale=1e2)
    yield "edge_denormal_ish", rnd(4, 2048, scale=1e-6), rnd(2048, 2048, scale=1e-6)

    # non-contiguous x (transposed view) and non-contiguous w (a slice of a
    # bigger buffer, which is what a merged qkv weight looks like)
    yield "edge_noncontig_x", rnd(2048, 4).t(), rnd(2048, 2048, scale=0.02)
    big_w = rnd(4096, 2048, scale=0.02)
    yield "edge_noncontig_w_slice", rnd(4, 2048), big_w[1024:3072]
    yield "edge_w_strided_view", rnd(4, 2048), big_w[::2]

    # deep K, so split-K has many tiles per split
    yield "edge_deep_K", rnd(4, 65536, scale=0.1), rnd(64, 65536, scale=0.005)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--candidate", default="promoted")
    ap.add_argument("--record", default=None,
                    help="append to candidates.jsonl/benchmark.csv under this id "
                         "(use 'baseline' for the kernel-fuse baseline)")
    ap.add_argument("--parent", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--no-bench", dest="bench", action="store_false")
    ap.add_argument("--n-random", type=int, default=24)
    args = ap.parse_args()

    dev = torch.device("cuda")
    mod = load_candidate(args.candidate)
    fn = mod.thin_linear
    print(f"[candidate] {args.candidate} -> {mod.__file__}")
    print(f"[gpu] {torch.cuda.get_device_name(0)}\n")

    rows: list[dict] = []
    n_fail = 0

    # 1. persisted oracle
    files = sorted(ORACLE.glob("*.pt"))
    print(f"--- persisted oracle ({len(files)} cases) ---")
    for p in files:
        case = torch.load(p, map_location=dev, weights_only=False)
        x, w = case["args"]
        ok = check(fn, x, w, case["ref_out"].to(dev), f"{case['source']}:{case['name']}", rows)
        n_fail += not ok
        del case, x, w
        torch.cuda.empty_cache()

    # 2. fresh random + edge, ground truth live
    print(f"--- fresh random + edge cases ---")
    n_fresh = 0
    for label, x, w in fresh_cases(dev, args.n_random):
        with torch.inference_mode():
            ref = F.linear(x, w)
        n_fail += not check(fn, x, w, ref, f"fresh:{label}", rows)
        n_fresh += 1

    # report
    worst = sorted(rows, key=lambda r: -(r["err"] / r["tol"] if r["tol"] and r["tol"] == r["tol"] else 9e9))
    print(f"\n{'case':>46s} {'max_err':>11s} {'tol':>11s} {'err/tol':>8s}  note")
    for r in worst[:14]:
        ratio = r["err"] / r["tol"] if r["tol"] and r["tol"] == r["tol"] else float("nan")
        flag = "" if r["ok"] else "  <-- FAIL"
        print(f"{r['case'][:46]:>46s} {r['err']:>11.3e} {r['tol']:>11.3e} "
              f"{ratio:>8.3f}  {r['note']}{flag}")

    total = len(rows)
    print(f"\n[对拍] {total - n_fail}/{total} passed "
          f"({len(files)} persisted + {n_fresh} fresh)")
    # A bare max-abs-error across cases whose outputs span 1e-4..1e4 is
    # meaningless, so report the worst error *relative to its own tolerance*
    # alongside it.
    max_err = max((r["err"] for r in rows if r["err"] == r["err"]), default=float("nan"))
    worst_ratio = max((r["err"] / r["tol"] for r in rows
                       if r["err"] == r["err"] and r["tol"]), default=float("nan"))
    real_be = [r["bitexact_pct"] for r in rows
               if r["case"].startswith("real:") and "bitexact_pct" in r]
    print(f"[max_err_fwd] {max_err:.3e}   [worst err/tol] {worst_ratio:.3f}")
    if real_be:
        print(f"[bitexact vs cuBLAS, real cases] min {min(real_be):.1f}%  "
              f"mean {sum(real_be) / len(real_be):.1f}%")

    # write the per-case table for the report
    with open(HERE / f"duipai_{args.candidate}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["case", "ok", "err", "tol",
                                           "bitexact_pct", "note"],
                           extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)

    if n_fail:
        print(f"\n[FAIL] {n_fail} case(s) exceeded tolerance -- fix the kernel, "
              f"do NOT loosen the tolerance (CONTRACT.md).")

    # 3. optional speed number, for the record
    wall = x_vs_base = float("nan")
    if args.bench and not n_fail:
        from shapes import draft_shapes
        tot_b = tot_c = 0.0
        for sh in draft_shapes(4):
            xx = torch.randn(sh.M, sh.K, device=dev, dtype=torch.bfloat16)
            pool = [torch.randn(sh.N, sh.K, device=dev, dtype=torch.bfloat16) * 0.02
                    for _ in range(max(2, -(-320 * 2**20 // sh.weight_bytes)))]
            b = graph_time_us(lambda i: F.linear(xx, pool[i]), len(pool))
            c = graph_time_us(mod.make_bench_fn(xx, pool, sh), len(pool))
            tot_b += b * sh.per_layer * sh.n_layers
            tot_c += c * sh.per_layer * sh.n_layers
            del pool
            torch.cuda.empty_cache()
        wall, x_vs_base = tot_c / 1e3, tot_b / tot_c
        print(f"\n[bench] whole-forward GEMV roll-up: cuBLAS {tot_b / 1e3:.2f} ms "
              f"-> candidate {wall:.2f} ms  [{x_vs_base:.2f}x]")

    if args.record:
        rec = {"id": args.record, "parent": args.parent,
               "status": "baseline" if args.record == "baseline"
                         else ("pass" if not n_fail else "fail"),
               "gpu": GPU_TAG, "wall_ms": wall, "x_vs_base": x_vs_base,
               "max_err_fwd": max_err, "max_err_bwd": None,
               "ncu": {"dram_pct": None, "sm_pct": None, "occ_pct": None},
               "note": args.note or f"{total - n_fail}/{total} 对拍 cases pass"}
        with open(HERE / "candidates.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        csv_path = HERE / "benchmark.csv"
        flat = {**{k: v for k, v in rec.items() if k != "ncu"},
                **{f"ncu_{k}": v for k, v in rec["ncu"].items()}}
        write_hdr = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(flat))
            if write_hdr:
                wr.writeheader()
            wr.writerow(flat)
        print(f"[record] appended '{args.record}' to candidates.jsonl / benchmark.csv")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
