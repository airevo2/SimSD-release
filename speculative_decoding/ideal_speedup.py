"""Ideal speedup predictor for speculative decoding.

Reads sweep_forward.py outputs (one JSON per (model, batch)) and computes the
upper-bound speedup under the standard formula:

    E[committed](K)   = (1 - a^K) / (1 - a)
    T_spec_iter(K, B) = K * T_draft_blk(B) + T_verify(K, B)
    speedup(K, a, B)  = E[committed] * T_native_blk(B) / T_spec_iter(K, B)

Inputs: per-forward times pulled from `draft_role.mean_ms` (block-causal =
native path) and `target_role.mean_ms` (multi-block verify).

Assumption: verify cost is approximated as T_verify(K, B) ~ T_verify(1, B),
which is what sweep_forward.py measures. Override via --verify_scale if you
have real K-sweep data.
"""
import argparse
import csv
import json
from pathlib import Path


def load_forward_json(path):
    d = json.load(open(path))
    cfg = d["config"]
    key = next(iter(d["results"]))  # e.g. "bl=4_ds=4"
    r = d["results"][key]
    # sweep_forward.py  `mean_ms` ** run** run
    # `n_forwards_per_run`  forwardper-forward  `per_step_latency_ms`
    # (= mean_ms / n_forwards_per_run)
    return {
        "batch": cfg["batch"],
        "block_length": r["block_length"],
        "denoising_steps": r["denoising_steps"],
        "draft_per_fwd_ms": r["draft_role"]["per_step_latency_ms"],
        "target_per_fwd_ms": r["target_role"]["per_step_latency_ms"],
    }


def collect(draft_files, target_files):
    by_batch = {}
    for f in draft_files:
        x = load_forward_json(f)
        by_batch.setdefault(x["batch"], {}).update(
            draft_per_fwd_ms=x["draft_per_fwd_ms"],
            block_length=x["block_length"],
            denoising_steps=x["denoising_steps"],
        )
    for f in target_files:
        x = load_forward_json(f)
        by_batch.setdefault(x["batch"], {}).update(
            native_per_fwd_ms=x["draft_per_fwd_ms"],
            verify_per_fwd_ms=x["target_per_fwd_ms"],
        )
    return dict(sorted(by_batch.items()))


def predict(t_draft_blk, t_native_blk, t_verify_1, K, alpha, verify_scale):
    E = K if alpha == 1.0 else (1 - alpha ** K) / (1 - alpha)
    t_verify = t_verify_1 * verify_scale(K)
    t_spec = K * t_draft_blk + t_verify
    return E * t_native_blk / t_spec, E, t_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_glob", required=True,
                   help="glob for draft-model sweep_forward JSONs, e.g. 'batch_sweep/sf_1_7B_bs*.json'")
    p.add_argument("--target_glob", required=True,
                   help="glob for target-model sweep_forward JSONs, e.g. 'batch_sweep/sf_8B_bs*.json'")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9])
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--verify_scale", choices=["flat", "linear"], default="flat",
                   help="flat: T_verify(K)=T_verify(1); linear: T_verify(K)=K*T_verify(1)")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    scale_fn = {"flat": lambda K: 1.0, "linear": lambda K: float(K)}[args.verify_scale]

    draft_files = sorted(Path().glob(args.draft_glob))
    target_files = sorted(Path().glob(args.target_glob))
    assert draft_files, f"no draft files match {args.draft_glob}"
    assert target_files, f"no target files match {args.target_glob}"

    by_batch = collect(draft_files, target_files)

    rows = []
    for B, v in by_batch.items():
        ds = v["denoising_steps"]
        t_draft_blk = ds * v["draft_per_fwd_ms"]
        t_native_blk = ds * v["native_per_fwd_ms"]
        t_verify_1 = v["verify_per_fwd_ms"]
        for K in args.ks:
            for a in args.alphas:
                sp, E, t_spec = predict(
                    t_draft_blk, t_native_blk, t_verify_1, K, a, scale_fn
                )
                rows.append({
                    "batch": B,
                    "K": K,
                    "alpha": a,
                    "T_draft_blk_ms": round(t_draft_blk, 3),
                    "T_native_blk_ms": round(t_native_blk, 3),
                    "T_verify_K_ms": round(t_verify_1 * scale_fn(K), 3),
                    "E_committed": round(E, 4),
                    "T_spec_iter_ms": round(t_spec, 3),
                    "speedup": round(sp, 4),
                })

    # Print: speedup table pivoted by (B, K) for each alpha
    print(f"verify_scale={args.verify_scale}, batches={list(by_batch)}, "
          f"Ks={args.ks}, alphas={args.alphas}\n")
    for a in args.alphas:
        print(f"=== alpha={a} ===")
        header = f"{'B':>3} | " + " ".join(f"K={K:<5}" for K in args.ks)
        print(header)
        print("-" * len(header))
        for B in by_batch:
            cells = []
            for K in args.ks:
                sp = next(r["speedup"] for r in rows
                          if r["batch"] == B and r["K"] == K and r["alpha"] == a)
                cells.append(f"{sp:<7.3f}")
            print(f"{B:>3} | " + " ".join(cells))
        print()

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
