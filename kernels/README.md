# kernels/

Fused GPU kernels for the SimSD inference path. All are **opt-in** — nothing here
runs unless a driver flag turns it on, and the reference implementation stays the
default.

| kernel | operator | mode | speedup | gpu | flag |
|---|---|---|---|---|---|
| [`thin_linear_fused.py`](thin_linear_fused.py) | `F.linear` at M = batch×block_length = 4 (the 197 projections per draft forward) | inference (fwd-only) | **1.51×** on the GEMV roll-up (4.06 → 2.70 ms), **+13.0%** end to end (79.21 → 89.54 tok/s) | `RTX PRO 6000 Blackwell` (sm_120) | `--fused_linear {off,draft,target,both}` |

Wall-clock and bandwidth figures are **specific to the GPU listed** and are not
portable — `thin_linear_fused.TUNED` was searched on that card and must be
regenerated (`kernels/workspace/thin_linear/search.py`) for any other.

## thin_linear

`nn.Linear` at M=4 gives cuBLAS too few output tiles to fill 188 SMs: `q_proj`
(N=2048) launches 16 CTAs. Per-CTA read rate saturates near 30 GB/s regardless of
shape, so runtime is (bytes per CTA)/30 GB/s and the only lever is more CTAs.
`lm_head` (1187 tiles) already hits 84% of DRAM peak under plain cuBLAS, which is
the proof that the shape — not the library — is the problem.

The kernel buys CTAs with a narrow `BLOCK_N` (16). A split-K path exists and is
tested but a 2160-config search chose `SPLIT_K=1` at every deployed shape: split-K
pays an fp32 staging round-trip through DRAM to buy what a narrower tile gives
free. Keeping the K reduction in order also keeps the result **bit-identical to
cuBLAS**, which is what lets the byte-identical-text gate in
`docs/kernel-optimization.md` §4 pass.

```bash
# use it
python main_table/run_ours_dual_gpu.py ... --fused_linear draft

# acceptance gate: byte-identical text vs runs/opt/v2_opt12_kvauto (20/20)
GPUS=4,5 bash scripts/37_fused_linear_quality.sh
# quality at scale (n=200, gsm8k+mbpp) -- use this if a future candidate gives up
# bit-exactness, since the target verifies every draft token
MODE=quality GPUS=4,5 bash scripts/37_fused_linear_quality.sh

# tests
CUDA_VISIBLE_DEVICES=4 python -m pytest kernels/thin_linear_fused_test.py -q

# where the forward's time goes, kernel by kernel
CUDA_VISIBLE_DEVICES=4 python scripts/35_kernel_profile.py --model draft --kv 256
CUDA_VISIBLE_DEVICES=4 python scripts/35_kernel_profile.py --model draft --fused_linear
```

**Dispatch is restricted to M ≤ 16** (`should_fuse`). Above that the kernel is
slower than cuBLAS — up to 0.18× on `down_proj` at prefill M=96 — and no longer
bit-exact. Since prefill builds the whole prompt's KV cache, ignoring this dropped
the text gate to 11/20 before the guard existed.

## Benchmarking rules

Violate these and the numbers lie; all three were hit during this work.

1. **Cycle distinct weights.** L2 is 128 MB. Replaying one 24 MB `gate_proj`
   measured 2194 GB/s — above the 1790 GB/s DRAM peak.
2. **Use an idle GPU.** Contention from other jobs on the node halves every
   number and looks like a clock-throttle effect. GPUs 0–3 are usually busy.
3. **Warm up every (shape, M).** Triton JIT inside the timed samples inverted a
   +11% result into −6%. `fused_toggle.warmup()` exists for this.

## Workspace

`kernels/workspace/<op>/` holds the tuning setup. It is **partially tracked**: the
correctness gate ships, the process residue does not.

| Tracked | Why |
|---|---|
| `CONTRACT.md` | The pinned mode (training / inference) and the I/O contract. Read this before touching the kernel. |
| `build_oracle.py` | Regenerates the oracle fixtures from scratch. |
| `oracle/manifest.jsonl` | What each fixture is (`T`, `k`, `E`, `H`, `I`, dtype, scale, concentration) — the input `build_oracle.py` needs. |
| `duipai.py`, `verify_model.py` | The correctness check against the reference. |

Not tracked: `oracle/*.pt` (44 GB — `flash_T32_fp32.pt` alone is 12 GB because it
holds all 256 experts' weights), plus `report.html`, `plan.md`, `draft.md`,
`benchmark*.csv`, `candidates/`, and `profile/*.ncu-rep` (a binary tied to the GPU
it was captured on, SM 12.0 here).

**After a fresh clone `oracle/` is empty.** Rebuild it before running any gate:

```bash
python kernels/workspace/moe_dispatch/build_oracle.py    # ~43 GB, needs a GPU
python kernels/workspace/thin_linear/build_oracle.py     # ~1.8 GB
```

The gate matters more than usual for `moe_dispatch_fused`: it is not bit-exact
against the stock `moe_infer`, and the divergence is load-bearing — fused MoE moved
α from 0.761 to 0.556 under the dynamic-remasking protocol, because draft and
target each flip tokens independently. A change that looks harmless can move the
acceptance rate, so re-run `duipai.py` rather than trusting a speed number.
