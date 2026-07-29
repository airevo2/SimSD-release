"""Fused split-K thin-M GEMV — a drop-in for `F.linear(x, w)` when M is tiny.

Why this exists
---------------
The SDAR draft forward runs 197 `nn.Linear` calls at **M = batch × block_length = 4**.
At that shape the work per output column is a dot product of length K, and cuBLAS
parallelises only over N: with `BLOCK_N = 128`, `q_proj` (N=2048) launches 16 CTAs
onto a 188-SM card. Measured on this node, a CTA reads its slice of the weight at a
saturating ~30 GB/s no matter the shape, so runtime is simply
(bytes per CTA) / 30 GB/s and the *only* lever is more CTAs. `lm_head` (N=151936,
1187 tiles) already hits 84% of DRAM peak with plain cuBLAS, which is the proof
that the shape, not the library, is the problem.

This kernel adds the missing parallelism two ways, both tunable:
  * a smaller `BLOCK_N`, so a given N yields more output tiles, and
  * **split-K**: the reduction dimension is cut into `SPLIT_K` chunks computed by
    independent CTAs, then summed.

**Searched result: `BLOCK_N` wins, split-K does not.** A sweep of
BLOCK_N ∈ {16..256} × BLOCK_K ∈ {32..256} × SPLIT_K ∈ {1..32} × warps × stages
over every deployed shape (`workspace/thin_linear/search.py`) picked `SPLIT_K=1`
for *all* of them, and `BLOCK_N=16` for all but `lm_head`. That makes sense in
hindsight: a narrower N-tile buys CTAs for free, while split-K pays a full fp32
staging round-trip through DRAM to buy the same thing. The split-K path is kept —
it is tested, and it is the only remaining lever for a shape with tiny N and
enormous K — but `TUNED` does not use it at these shapes.

A welcome side effect: with `SPLIT_K=1` the K reduction is one in-order fp32
accumulation, as cuBLAS's is, and the result comes out **bit-identical to cuBLAS
on 100% of elements across all 15 real captured cases** (the split-K baseline was
99.6–99.9%). That is what lets the byte-identical-text gate in
`docs/kernel-optimization.md` §4 pass rather than merely come close.

Correctness notes (see kernels/workspace/thin_linear/CONTRACT.md)
----------------------------------------------------------------
* **Forward only.** This is an inference-mode kernel: no autograd wrapper. It
  raises if handed a tensor that requires grad, rather than silently
  detaching the graph.
* **Deterministic split-K.** The partials go to a staging buffer and are summed
  in a fixed index order (0..SPLIT_K-1). `atomic_add` would have been shorter but
  its summation order varies per run, and the pipeline's acceptance gate is
  byte-identical generated text — a nondeterministic last mantissa bit can flip
  an argmax and diverge the whole continuation.
* **CUDA-graph-capture safe.** No host sync, no `.item()`, no value-dependent
  host branching. The staging buffer comes from a module-level cache keyed by
  device, so the hot path performs no allocation; the pipeline's three warmup
  forwards (`cache_aware.py:480-492`) populate it before capture.
* Accumulation is fp32 regardless of input dtype, so on real data this is
  typically *more* accurate than the bf16 reference, not less.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Tuning knobs. `kernel-optimize` searches over these; the defaults here are
# chosen to be correct and reasonable, not optimal.
TARGET_CTAS = 2 * 188          # aim for >=2 waves on this card's 188 SMs
MAX_SPLIT_K = 16
MIN_K_PER_SPLIT = 256          # below this, split-K costs more than it buys


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _thin_gemv_kernel(
    X, W, Y, Part,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    stride_pk, stride_pm, stride_pn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, IEEE: tl.constexpr, LONG_IDX: tl.constexpr,
):
    """One CTA computes Y[:, n_tile] over the K-slice `pid_k`.

    SPLIT_K == 1 -> write the result straight to Y in the output dtype.
    SPLIT_K  > 1 -> write an fp32 partial to Part[pid_k], for `_thin_gemv_reduce`.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    if LONG_IDX:
        # A flat offset like n*K wraps past 2**31 for a big enough weight
        # (a 151936x2048 lm_head is 3.1e8, fine; a larger one is not), and the
        # wrap lands on a bad address instead of a wrong answer.
        offs_n = offs_n.to(tl.int64)
        offs_k = offs_k.to(tl.int64)
        offs_m = offs_m.to(tl.int64)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Split the K *tiles* (not raw K) into contiguous per-CTA ranges, so each
    # CTA streams a contiguous span of every weight row it touches.
    k_tiles = tl.cdiv(K, BLOCK_K)
    tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
    kt_begin = pid_k * tiles_per_split
    kt_end = tl.minimum(kt_begin + tiles_per_split, k_tiles)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(kt_begin, kt_end):
        k_now = kt * BLOCK_K + offs_k
        mask_k = k_now < K
        # Differentiated cache policy (KernelWiki `techniques/cache-policy`):
        # x is re-read by *every* CTA in the grid, so keep it resident; W is read
        # exactly once per forward (3.4 GB streaming through a 128 MB L2), so tell
        # the cache not to retain it and thereby evict x.
        # x: [BLOCK_M, BLOCK_K]
        x = tl.load(X + offs_m[:, None] * stride_xm + k_now[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0,
                    eviction_policy="evict_last")
        # w: [BLOCK_N, BLOCK_K] -- loaded in its native (N, K) layout so the
        # K axis stays contiguous and the reads coalesce; transposed in-register.
        w = tl.load(W + offs_n[:, None] * stride_wn + k_now[None, :] * stride_wk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0,
                    eviction_policy="evict_first")
        if IEEE:
            acc += tl.dot(x, tl.trans(w), input_precision="ieee")
        else:
            acc += tl.dot(x, tl.trans(w))

    if SPLIT_K == 1:
        tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
                 acc.to(Y.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])
    else:
        p = Part + pid_k * stride_pk
        tl.store(p + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn,
                 acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _thin_gemv_reduce(
    Part, Y,
    M, N,
    stride_pk, stride_pm, stride_pn,
    stride_ym, stride_yn,
    SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    LONG_IDX: tl.constexpr,
):
    """Sum the SPLIT_K fp32 partials in a fixed order and cast to Y's dtype.

    The fixed `range(SPLIT_K)` is the point: it makes the result bit-reproducible
    across replays, which `atomic_add` would not be.
    """
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if LONG_IDX:
        offs_m = offs_m.to(tl.int64)
        offs_n = offs_n.to(tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(Part + s * stride_pk + offs_m[:, None] * stride_pm
                       + offs_n[None, :] * stride_pn, mask=mask, other=0.0,
                       eviction_policy="evict_first")
    tl.store(Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(Y.dtype.element_ty), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Host side
# ─────────────────────────────────────────────────────────────────────────────
# device -> fp32 staging buffer for the split-K partials. Reused across calls so
# the hot path never allocates. Safe because a captured CUDA graph replays its
# calls serially on one stream (write-then-read, then the next call overwrites),
# and the draft/target models live on *different* devices, hence the device key.
_PARTIALS: dict[torch.device, torch.Tensor] = {}


def _partials(dev: torch.device, numel: int) -> torch.Tensor:
    buf = _PARTIALS.get(dev)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, device=dev, dtype=torch.float32)
        _PARTIALS[dev] = buf
    return buf


def reset_workspace() -> None:
    """Drop the staging buffers (call before re-capturing graphs in a new pool)."""
    _PARTIALS.clear()


# Searched winners, keyed by (K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K,
# num_warps, num_stages). Measured on `RTX PRO 6000 Blackwell` at serving
# conditions (M=4, bf16, inside a CUDA graph, over an L2-defeating weight pool)
# by `workspace/thin_linear/search.py`. Regenerate with that script if the shapes
# or the GPU change — these numbers are not portable across either.
#
#   op                us     GB/s   % of DRAM peak   ncu DRAM%   grid
#   q/o_proj         7.62    1101        61%            54%       128
#   k/v_proj         5.19     808        45%            37%        64  <- CTA-starved
#   gate/up_proj    18.35    1372        77%            72%       384
#   down_proj       19.50    1291        72%            70%       128
#   lm_head        409.27    1521        85%            95%      2374  <- at roofline
TUNED: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {
    (2048, 2048): (16, 16, 128, 1, 4, 4),     # q_proj, o_proj
    (2048, 1024): (16, 16, 256, 1, 2, 4),     # k_proj, v_proj
    (2048, 6144): (16, 16, 64, 1, 2, 4),      # gate_proj, up_proj
    (6144, 2048): (16, 16, 256, 1, 2, 4),     # down_proj
    (2048, 151936): (16, 64, 256, 1, 8, 2),   # lm_head
    (2048, 4096): (16, 16, 128, 1, 4, 4),     # qkv, if ever merged
    (2048, 12288): (16, 32, 64, 1, 4, 4),     # gate+up, if ever merged
    (4096, 12288): (16, 32, 64, 1, 4, 4),     # target-model gate/up
}


def pick_config(M: int, N: int, K: int, itemsize: int = 2) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K) — searched where known, else heuristic.

    The heuristic fallback aims for at least `TARGET_CTAS` CTAs, since the
    measured per-CTA read rate is ~30 GB/s regardless of shape. It prefers extra
    N-tiles (free) over split-K (which costs a staging round-trip through DRAM)
    and never splits so finely that a CTA reads less than `MIN_K_PER_SPLIT`
    elements of each row.
    """
    full = _tuned_for(M, N, K)
    if full is not None:
        return full[0], full[1], full[2], full[3]
    return _heuristic_config(M, N, K)


def _tuned_for(M: int, N: int, K: int):
    """The searched 6-tuple, or None when this (M, K, N) was not searched.

    The M guard matters: `TUNED` was searched at M=4 with BLOCK_M=16, so it is
    only valid while M still rounds up to 16. Prefill (M≈96) must fall through to
    the heuristic rather than inherit a config tuned for a 4-row activation.
    """
    c = TUNED.get((K, N))
    if c is None:
        return None
    return c if max(16, triton.next_power_of_2(max(M, 1))) == c[0] else None


def _heuristic_config(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    BLOCK_M = max(16, triton.next_power_of_2(M))     # tl.dot needs M >= 16
    BLOCK_K = 64 if K >= 64 else max(16, triton.next_power_of_2(K))

    # Smallest BLOCK_N (>=16, <=128) that alone reaches the CTA target.
    BLOCK_N = 128
    for cand in (16, 32, 64, 128):
        if cand < 16:
            continue
        if triton.cdiv(N, cand) >= TARGET_CTAS:
            BLOCK_N = cand
            break
    BLOCK_N = min(BLOCK_N, max(16, triton.next_power_of_2(N)))

    n_tiles = triton.cdiv(N, BLOCK_N)
    split = triton.cdiv(TARGET_CTAS, n_tiles)
    split = min(split, MAX_SPLIT_K, max(1, K // MIN_K_PER_SPLIT))
    return BLOCK_M, BLOCK_N, BLOCK_K, max(1, split)


def thin_linear(x: torch.Tensor, w: torch.Tensor,
                bias: torch.Tensor | None = None,
                config: tuple[int, int, int, int] | None = None,
                num_warps: int | None = None,
                num_stages: int | None = None) -> torch.Tensor:
    """`F.linear(x, w, bias)` for tiny M, via a split-K GEMV.

    x: [..., K]  (leading dims are flattened, as `nn.Linear` does)
    w: [N, K]    (`nn.Linear.weight` layout)
    -> [..., N]
    """
    # Forward-only: refuse exactly when autograd would try to record us. A loaded
    # `nn.Linear` keeps `weight.requires_grad=True` even after `.eval()`, so
    # testing `requires_grad` alone would reject every legitimate inference call —
    # what matters is whether grad mode is on at the same time.
    if torch.is_grad_enabled() and (x.requires_grad or w.requires_grad):
        raise RuntimeError(
            "thin_linear is a forward-only inference kernel (CONTRACT.md mode="
            "inference); it has no backward. Call it under torch.no_grad() / "
            "torch.inference_mode(), or use F.linear for training.")
    if x.dtype != w.dtype:
        raise TypeError(f"dtype mismatch: x={x.dtype} w={w.dtype}")
    if w.ndim != 2:
        raise ValueError(f"w must be 2-D [N, K], got {tuple(w.shape)}")

    out_shape = (*x.shape[:-1], w.shape[0])
    x2 = x.reshape(-1, x.shape[-1])
    M, K = x2.shape
    N = w.shape[0]
    if w.shape[1] != K:
        raise ValueError(f"K mismatch: x[..., {K}] vs w[{N}, {w.shape[1]}]")

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    if config is not None:
        BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K = config
        nw, ns = num_warps or 4, num_stages or 3
    else:
        tuned = _tuned_for(M, N, K)
        if tuned is not None:
            BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, nw, ns = tuned
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K = _heuristic_config(M, N, K)
            nw, ns = 4, 3
        nw, ns = num_warps or nw, num_stages or ns
    ieee = x.dtype == torch.float32
    long_idx = max(N * K, M * K, M * N) >= 2 ** 31

    if SPLIT_K == 1:
        part = y.new_empty(0, dtype=torch.float32)   # unused; keeps the signature fixed
        sp_k = sp_m = sp_n = 0
    else:
        part = _partials(x.device, SPLIT_K * M * N)[: SPLIT_K * M * N] \
            .view(SPLIT_K, M, N)
        sp_k, sp_m, sp_n = part.stride()

    _thin_gemv_kernel[(triton.cdiv(N, BLOCK_N), SPLIT_K)](
        x2, w, y, part if SPLIT_K > 1 else y,
        M, N, K,
        x2.stride(0), x2.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        sp_k, sp_m, sp_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K,
        IEEE=ieee, LONG_IDX=long_idx,
        num_warps=nw, num_stages=ns,
    )

    if SPLIT_K > 1:
        RED_N = 128
        _thin_gemv_reduce[(triton.cdiv(N, RED_N),)](
            part, y, M, N,
            sp_k, sp_m, sp_n,
            y.stride(0), y.stride(1),
            SPLIT_K=SPLIT_K, BLOCK_M=BLOCK_M, BLOCK_N=RED_N, LONG_IDX=long_idx,
            num_warps=4,
        )

    if bias is not None:
        y += bias
    return y.view(out_shape)


def should_fuse(M: int, N: int, K: int) -> bool:
    """Is the fused kernel the right choice for this shape in production?

    Only where it was actually searched. This is a measured boundary, not
    caution: the kernel exists because at M=4 cuBLAS has too few output tiles to
    fill 188 SMs, and that reason expires as M grows. Measured on this card
    (cuBLAS us -> ours us):

    | M | q_proj 2048x2048 | k_proj 2048x1024 | down_proj 6144x2048 |
    |---|---|---|---|
    | 4  | 13.43 -> 7.61 **1.76x** | 13.60 -> 5.17 **2.63x** | 37.25 -> 19.43 **1.92x** |
    | 16 | 13.92 -> 7.62 **1.83x** | 13.63 -> 5.27 **2.59x** | 37.33 -> 19.53 **1.91x** |
    | 32 | 10.71 -> 14.12   0.76x  |  7.68 -> 11.41   0.67x  | 24.47 -> 26.01   0.94x  |
    | 96 | 14.85 -> 21.31   0.70x  | 13.75 -> 19.69   0.70x  | 23.36 -> **132.19  0.18x** |

    So above M=16 the unsearched heuristic path is a *pessimisation*, badly so for
    `down_proj` at prefill sizes. It is also no longer bit-exact (59-63% of
    elements match cuBLAS, vs 100% at M<=16), which matters because prefill builds
    the KV cache for the whole prompt — a 1-ULP change there propagates into every
    subsequent block and diverges the generated text.

    Both were observed end to end before this guard existed: prefill running the
    heuristic path cost ~5.4 s of extra wall clock over 20 samples and dropped
    byte-identical text from 20/20 to 11/20.

    `thin_linear` itself still honours any shape (that is what the 对拍 harness
    exercises, so kernel bugs cannot hide behind a fallback); it is the *dispatch*
    that is restricted.
    """
    return _tuned_for(M, N, K) is not None


class ThinLinear(torch.nn.Module):
    """`nn.Linear` drop-in that routes the forward through `thin_linear`.

    Holds a *reference* to the original module's parameters (no copy), so
    swapping it in costs no memory and `state_dict()` round-trips unchanged.

    Falls back to `F.linear` for shapes outside the searched envelope — see
    `should_fuse`. That keeps a single patched model correct and fast across the
    pipeline's mix of prefill (M≈96), extend (M=8) and denoise (M=4) calls.
    """

    def __init__(self, linear: torch.nn.Linear,
                 config: tuple[int, int, int, int] | None = None,
                 always_fuse: bool = False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.config = config
        self.always_fuse = always_fuse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M = x.numel() // x.shape[-1]
        if (self.config is not None or self.always_fuse
                or should_fuse(M, self.out_features, self.in_features)):
            return thin_linear(x, self.weight, self.bias, config=self.config)
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, fused=thin-gemv "
                f"(M<=16 only unless always_fuse)")


# ─────────────────────────────────────────────────────────────────────────────
# Bench hook, consumed by kernels/workspace/thin_linear/bench.py
# ─────────────────────────────────────────────────────────────────────────────
def make_bench_fn(x, pool, sh):
    return lambda i: thin_linear(x, pool[i])
