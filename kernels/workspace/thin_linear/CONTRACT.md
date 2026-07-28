# CONTRACT — `thin_linear` (thin-M bf16 GEMV)

Pinned at fuse time. Everything downstream (oracle, 对拍, embedding,
kernel-optimize) reads the mode, tolerance and constraints from here.
**Do not relax anything in this file to make a candidate pass.**

## Mode

**`inference`** — forward only. No backward, no `torch.autograd.Function`.
The operator only ever runs inside `torch.inference_mode()` in a CUDA-graph
replay; there is no training path in this repo that touches it.

## Hardware / software (stamp every record with this)

| | |
|---|---|
| `gpu` | `RTX PRO 6000 Blackwell Server Edition` (sm_120) |
| SMs | 188 |
| L2 | **128 MB** — big enough to hide a benchmarking mistake, see “Benchmarking” |
| DRAM | 96 GB GDDR7, peak **1790 GB/s** |
| torch / triton | 2.8.0+cu128 / 3.4.0 |
| CUDA toolkit | 13.0 (`ncu` 2025.3.1 present; **no `nsys` on this node**) |
| env | `source scripts/env.sh` (prefix `/mnt/home/haotian.ye/envs/simsd`) |
| node | already inside slurm alloc `510117` on `slurm-rtxp6000-195-037` |

## Operator + isolated signature

The reference is `torch.nn.functional.linear` with **M = batch × block_length = 4**
and no bias — i.e. a very thin GEMM that is really a batched GEMV.

```python
# reference (what every candidate must match)
def ref_thin_linear(x, w):            # x [M, K] bf16, w [N, K] bf16 (nn.Linear layout)
    return torch.nn.functional.linear(x, w)     # -> [M, N] bf16

# candidate contract — called exactly the same way
def thin_linear(x, w): ...            # -> [M, N] bf16
```

`bias` is always `None` in this model: `attention_bias=false` in
`models/SDAR-1_7B-Chat/config.json`, `SDARMLP` builds its three `nn.Linear` with
`bias=False`, and `lm_head` is `bias=False`. The signature still accepts an
optional `bias` for generality but the oracle pins `bias=None` as the deployed case.

## Call sites (the 197 GEMM launches per draft forward)

All in the **draft model’s** decoder stack. `models/SDAR-1_7B-Chat/modeling_sdar.py`
is loaded through `trust_remote_code` (so the live copy is
`~/hf_modules/transformers_modules/SDAR-1_7B-Chat/modeling_sdar.py`) — treat it as
**third-party**: do not edit it, monkeypatch from `kernels/fused_toggle.py`.

| call site | file | shape (K→N) | per fwd |
|---|---|---|---|
| `q_proj` | `SDARAttention` (patched by `patch_sdpa_static_cache`, `cache_aware.py:312`) | 2048→2048 | 28 |
| `k_proj` | ” `:313` | 2048→1024 | 28 |
| `v_proj` | ” `:314` | 2048→1024 | 28 |
| `o_proj` | ” `:404` | 2048→2048 | 28 |
| `gate_proj` | `SDARMLP.forward` | 2048→6144 | 28 |
| `up_proj` | ” | 2048→6144 | 28 |
| `down_proj` | ” | 6144→2048 | 28 |
| `lm_head` | `SDARForCausalLM.forward` | 2048→151936 | 1 |

Serving stack actually in force (measured, `scripts/35_kernel_profile.py`):
`patch_stock_rms_norm` (plain torch fp32-mean RMSNorm), `patch_sdpa_static_cache`
(SDPA + `index_copy_` into a `StaticBlockCache`), `patch_causallm_pass_cache`.
`liger_kernel` is **not installed**, so `SDARMLP` runs the eager
`down(silu(gate(x)) * up(x))` path.

## Serving-stack correctness constraints (not optional)

1. **CUDA-Graph-capture safe.** The op is captured by
   `torch.cuda.graph(...)` in `cache_aware.py:497` / `:920`. Therefore:
   no host synchronisation, no `.item()`/`.cpu()`, no value-dependent host
   control flow, and **no allocation on the hot path** — any scratch (e.g. a
   split-K staging buffer) must be allocated once and reused, keyed by shape.
2. **Deterministic across replays.** `atomic_add` reductions are ruled out:
   their summation order varies run to run, which would make the byte-identical
   text gate (below) flaky rather than merely different. Split-K must use a
   fixed-order two-stage reduction.
3. **No torch.compile graph break.** Not currently required — no compiled region
   wraps these call sites once `patch_sdpa_static_cache` replaces the
   `fused_flex_attention` path — but the entry point is registered as a
   `torch.library.custom_op` anyway so it stays safe if that changes.
4. **Stable output buffer identity is NOT required.** The op returns a fresh
   tensor during capture, as `F.linear` does; the graph freezes those addresses.

## Tolerance (forward only, this is inference mode)

* bf16: `max_err < 2**-7 * ref.abs().max() + 1e-3`
* fp32: `max_err < 1e-5 * max(1, ref.abs().max())`
* `max_err_bwd` is `null` on every record (field kept so the schema is stable).

The fp32 bound is **relative**, which is a correction to how the repo convention
("< 1e-5") was first applied here, not a relaxation. Measured on a
`M=14, K=2175, N=2560` fp32 case with `|ref|max = 377.8`, against an fp64
ground truth:

| fp32 implementation | vs fp64 | vs cuBLAS |
|---|---|---|
| cuBLAS `F.linear` (the reference itself) | 7.54e-5 | 0 |
| chunked fp32 matmul (256-wide, different order) | 6.18e-5 | 7.63e-5 |
| our split-K | 1.71e-4 | 1.83e-4 |

An **absolute** 1e-5 is unachievable at this magnitude by cuBLAS itself, so it
was a mis-specified bound rather than a failing kernel. All three sit within
~4 fp32 ULPs of fp64 (1.7e-4 / 378 = 4.5e-7 relative), which also confirms
`input_precision="ieee"` is honored on sm_120 — TF32 would show ~1e-3 relative.

This tolerance is *pointwise vs cuBLAS*, and cuBLAS is not itself exact.

## The lossless gate

> **RESOLVED — read the end of this section first.** What follows records the
> state at fuse time, when the kernel used split-K and was 99.6–99.9% bit-exact.
> The promoted kernel uses `SPLIT_K=1`, is **100% bit-exact on all 15 real cases**,
> and **passes the byte-identical gate 20/20**. The "needs a decision" framing
> below no longer applies; the fallback argument is retained because it is the
> right gate for any *future* candidate that gives up bit-exactness (e.g. a
> split-K config, or the merged qkv if it turns out not to preserve order).
> Note also that `docs/kernel-optimization.md` §4 has since been revised by the
> author to make statistical quality the required gate, with byte-identical
> explicitly *not* required.

### State at fuse time (split-K version)

`docs/kernel-optimization.md` §4 asks any kernel change to reproduce
`runs/opt/v2_opt12_kvauto/ours_dual_gpu_gsm8k.jsonl` **`text` 20/20 identical**.
Bit-exactness of this kernel vs cuBLAS on the 15 **real** captured cases:

| | bit-exact vs cuBLAS |
|---|---|
| `gate_proj`, `up_proj` (L0 & L27), `k_proj` L27, **`lm_head`** | **100%** |
| `q_proj`, `o_proj`, `v_proj`, `down_proj` | 99.6 – 99.9% |

So a per-forward drift of ~0.1–0.4% of elements by 1 bf16 ULP is expected, and
over 28 layers × 4 denoise steps that will occasionally flip a draft token.
Two things make this tolerable rather than fatal, but the call is the user's:

1. **The draft only proposes.** The target model verifies every token, so a
   flipped draft token cannot corrupt output — it can only change the accept
   rate. This repo already relies on that argument elsewhere:
   `scripts/36_fold_quality.sh` accepts exactly this trade for
   `--fold_draft_extend` ("bf16 reduction order differs and a draft token
   occasionally flips… that is not a correctness property").
2. Therefore the meaningful gate for a **draft-side** kernel is the one that
   script uses: accuracy + accept-rate unchanged at scale (n=200, gsm8k+mbpp),
   not byte-identical text.

Recorded here so it is not silently redefined. Byte-identical text is achievable
only by keeping cuBLAS's exact reduction order, which rules out split-K entirely.

**How it resolved:** the config search independently chose `SPLIT_K=1` at every
deployed shape on pure speed grounds (a narrow `BLOCK_N` buys CTAs without the
fp32 staging round-trip), which happens to keep exactly that in-order reduction.
So bit-exactness came for free rather than being traded for. The strict gate was
kept anyway and earned its place — it caught the split-K version at 16/20 and the
prefill dispatch bug at 11/20, neither of which an n=20 statistical gate would see.

## Measured baseline (2026-07-26, GPU 4 idle, this node)

`scripts/35_kernel_profile.py --model draft --kv 256` — one draft forward,
7.13 ms of GPU kernel time (7.76 ms wall; docs quote 7.48 ms):

| bucket | us/fwd | % | launches/fwd |
|---|---|---|---|
| GEMM (cutlass `s161616gemm`) | **4005** | 56% | 197 |
| attention (`fmha_cutlassF`) | 635 | 9% | 28 |
| elementwise | 1272 | 18% | 881 |
| KV-cache write (`index_copy_`/cat/copy) | 795 | 11% | 406 |
| norm + rope | 353 | 5% | 226 |
| sampling / other | 71 | 1% | 9 |

Per-op GEMV baseline (`bench.py`, L2-defeating weight pool):

| op | K | N | tiles@128 | MB | cuBLAS us | GB/s | % peak |
|---|---|---|---|---|---|---|---|
| q_proj | 2048 | 2048 | 16 | 8.0 | 13.43 | 624 | 35% |
| k_proj | 2048 | 1024 | 8 | 4.0 | 13.61 | 308 | 17% |
| v_proj | 2048 | 1024 | 8 | 4.0 | 13.60 | 308 | 17% |
| o_proj | 2048 | 2048 | 16 | 8.0 | 13.85 | 606 | 34% |
| gate_proj | 2048 | 6144 | 48 | 24.0 | 19.22 | 1310 | 73% |
| up_proj | 2048 | 6144 | 48 | 24.0 | 19.19 | 1311 | 73% |
| down_proj | 6144 | 2048 | 16 | 24.0 | 37.26 | 675 | 38% |
| lm_head | 2048 | 151936 | 1187 | 593.5 | 416.03 | 1496 | **84%** |
| *(qkv merged)* | 2048 | 4096 | 32 | 16.0 | 15.18 | 1105 | 62% |
| *(gate+up merged)* | 2048 | 12288 | 96 | 48.0 | 35.56 | 1416 | 79% |

Roll-up: **4.06 ms, 848 GB/s, 47% of peak** — within 1.5% of the 4005 us the
profiler attributes to GEMM, so the microbench is a faithful proxy.

**The diagnosis is confirmed and sharpened.** `lm_head` (1187 tiles) already runs
at 84% of peak, so cuBLAS is not the problem — *tile count* is. Every op with
≤48 N-tiles on 188 SMs is starved. Per-CTA read rate saturates near 28–31 GB/s
regardless of shape, so time ≈ (bytes per CTA)/30 GB/s and the only lever is
**more CTAs**: smaller `BLOCK_N`, split-K, or both. Merging q/k/v into one
N=4096 GEMM already buys 40.6 us → 15.2 us with plain cuBLAS.

## Benchmarking rules (learned the hard way — violate these and the numbers lie)

1. **Cycle distinct weights.** L2 is 128 MB. Replaying one 24 MB `gate_proj`
   measured **2194 GB/s**, above the 1790 GB/s DRAM peak. `bench.py` allocates a
   ≥320 MB pool of replicas and round-robins.
2. **Use an idle GPU.** GPUs 0–3 carry other jobs on this node; contention
   halves every number and looks like a real effect (it produced a bogus
   “burst 9.6 us vs sustained 19.0 us” clock-throttle story). Use **GPU 4 or 5**;
   `bench.py`/`35_kernel_profile.py` print per-GPU utilisation in the header.
3. **Time inside a CUDA graph.** At M=4 the kernel is under 10 us — a python
   loop measures launch overhead, and the real pipeline replays a graph anyway.

## kernel-fuse result (in situ, end to end)

`scripts/37_fused_linear_quality.sh` with `GPUS=4,5 N=20`, gsm8k, the full
`--use_cuda_graph --pipeline --fused_denoise --speculative_target_extend` stack:

| | tok/s | ms/tok | accept | pass@1 | text vs reference |
|---|---|---|---|---|---|
| reference (recorded) | 79.04 | 12.65 | 0.8955 | 0.250 | — |
| base re-run (cuBLAS) | **79.02** | 12.66 | 0.8955 | 0.250 | **20/20** |
| `--fused_linear draft` | **88.34** | 11.32 | 0.9001 | 0.250 | 16/20 |

+11.8% end to end from the *untuned* kernel. The base re-run reproducing the
recorded reference 20/20 is what makes the 16/20 meaningful: the gate works, and
the 4 divergences are the kernel, not run-to-run noise.

Per-forward kernel time (`scripts/35_kernel_profile.py`): 7132 → 6037 us, of
which the GEMV bucket is 4005 → 2860 us (2194 main + 249 reduce + 416 lm_head
still on cuBLAS).

### Warmup is load-bearing, not hygiene

The first measurement of this showed the fused run *slower* (74.55 tok/s). Cause:
Triton JIT + graph capture landed inside the timed samples — samples 0-3 carried
4.86 / 1.42 / 1.80 / 1.79 s of extra wall clock while samples 4-19 were already
uniformly ~11% faster. Over only 19 samples that one-off cost inverted the
headline. `fused_toggle.warmup()` now sweeps M ∈ {1, bl, 2·bl, 96, 97} per module
and the driver calls it right after patching. Any future benchmark of this kernel
must warm up the same way or it will measure the compiler.

## kernel-optimize result (final)

Search: 2160 configs per shape over BLOCK_N × BLOCK_K × SPLIT_K × warps × stages
(`search.py`). **SPLIT_K=1 won every shape**, BLOCK_N=16 all but `lm_head`.
A narrow N-tile buys CTAs for free; split-K pays an fp32 staging round-trip to buy
the same thing. Side effect that mattered more than the speed: SPLIT_K=1 makes the
K reduction one in-order fp32 accumulation like cuBLAS's, so the output is
**bit-identical to cuBLAS on 100% of elements** across all 15 real cases.

| op | cuBLAS | untuned | tuned | speedup | GB/s | ncu DRAM% | grid |
|---|---|---|---|---|---|---|---|
| q_proj | 13.43 | 9.45 | **7.62** | 1.76x | 1101 | 54% | 128 |
| k_proj | 13.61 | 6.59 | **5.19** | 2.62x | 808 | 37% | 64 |
| v_proj | 13.60 | 6.61 | **5.20** | 2.61x | 807 | 36% | 64 |
| o_proj | 13.85 | 9.45 | **7.63** | 1.82x | 1100 | 53% | 128 |
| gate_proj | 19.22 | 18.60 | **18.35** | 1.05x | 1372 | 72% | 384 |
| up_proj | 19.19 | 18.67 | **18.35** | 1.05x | 1372 | 72% | 384 |
| down_proj | 37.26 | 21.86 | **19.50** | 1.92x | 1291 | 70% | 128 |
| lm_head | 416.03 | 410.14 | **409.27** | 1.02x | 1521 | **95%** | 2374 |
| **roll-up** | **4058** | **2967** | **2700 us** | **1.51x** | 1274 | | |

Forward: 7132 → **5828 us**. End to end: 79.21 → **89.54 tok/s (+13.0%)**,
**gate A 20/20 byte-identical**, accept 0.8955 and pass@1 0.250 unchanged.
对拍 87/87, unit tests 54/54.

### The dispatch guard (`should_fuse`) — a bug fix, not a precaution

The kernel is dispatched **only where `search.py` measured it**, i.e. M ≤ 16.
Above that the unsearched heuristic path is a pessimisation *and* bit-divergent:

| M | q_proj | k_proj | down_proj | bit-exact |
|---|---|---|---|---|
| 4 | 1.76x | 2.63x | 1.92x | 100% |
| 16 | 1.83x | 2.59x | 1.91x | 100% |
| 32 | 0.76x | 0.67x | 0.94x | 60-63% |
| 96 | 0.70x | 0.70x | **0.18x** (132 vs 23 us) | 59-99.9% |

Prefill runs at M≈96. Before the guard existed it cost ~5.4 s of wall clock over
20 samples and — because prefill builds the whole prompt's KV cache — dropped
byte-identical text to **11/20**. `thin_linear` itself still accepts any shape, so
对拍 cannot hide a kernel bug behind the fallback; only the *dispatch* is limited.

### Not worth further work

`lm_head` is at 94.7% of DRAM peak. The GEMV roofline is 3.441 GB / 1521 GB/s =
**2.26 ms**, so only ~0.44 ms of matmul headroom remains. The doc's 3.6 ms target
needs the other 3.19 ms (881 elementwise + 406 KV-write + 226 norm/rope launches,
all latency-bound) folded into epilogues — `plan.md` v3–v5, not attempted here.
The measured next lever is the merged qkv: 13.07 us vs 18.01 us for the three
tuned kernels separately (−138 us/forward), which ncu justifies (k/v run 64 CTAs
on 188 SMs and `tl.dot` cannot go below BLOCK_N=16).

## Status

- [x] Step 1 workspace + contract
- [x] Step 2 locate & isolate
- [x] baseline measured (profiler + per-op bench agree within 1.5%)
- [x] Step 3 oracle — 51 cases (15 real / 29 fake / 7 extreme)
- [x] Step 4 fused kernel — `kernels/thin_linear_fused.py`
- [x] Step 5 对拍 87/87 (51 persisted + 36 fresh), worst err/tol 0.70
- [x] Step 6 embedded behind `--fused_linear {off,draft,target,both}`; 54/54 unit tests;
      verified under real CUDA-graph capture and end to end
- [x] Step 7 searched + profiled (ncu) + promoted + `report.html`
- [ ] optional: v3 merged qkv / v4-v5 epilogue fusion (see `plan.md`)
