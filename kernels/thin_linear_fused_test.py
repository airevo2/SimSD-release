"""Tests for the fused split-K thin GEMV.

The exhaustive 对拍 (51 persisted oracle cases + fresh random/edge cases each run)
lives in `kernels/workspace/thin_linear/duipai.py`, which needs the local oracle.
This file is the committed, self-contained subset: it regenerates its own inputs
and covers the shapes and the serving-stack properties that must not regress.

    CUDA_VISIBLE_DEVICES=4 python -m pytest kernels/thin_linear_fused_test.py -q
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import fused_toggle
from kernels.thin_linear_fused import ThinLinear, pick_config, thin_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
# (K, N) of the SDAR-1.7B draft projections, plus the target's widest.
DEPLOYED = [(2048, 2048), (2048, 1024), (2048, 6144), (6144, 2048),
            (2048, 151936), (4096, 12288)]


def tol(ref: torch.Tensor) -> float:
    """CONTRACT.md: bf16 ~2-ULP-relative, fp32 1e-5-relative."""
    if ref.dtype == torch.float32:
        return 1e-5 * max(1.0, float(ref.abs().max()))
    return 2 ** -7 * float(ref.abs().max()) + 1e-3


def assert_matches(x, w):
    with torch.inference_mode():
        ref = F.linear(x, w)
        got = thin_linear(x, w)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    err = float((got.float() - ref.float()).abs().max())
    assert err <= tol(ref), f"max_err {err:.3e} > tol {tol(ref):.3e}"
    return err


def rnd(*shape, dtype=torch.bfloat16, scale=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, generator=g, device=DEV,
                        dtype=torch.float32) * scale).to(dtype)


@pytest.mark.parametrize("K,N", DEPLOYED)
@pytest.mark.parametrize("M", [1, 2, 4, 8])
def test_deployed_shapes(M, K, N):
    assert_matches(rnd(M, K, seed=M), rnd(N, K, scale=0.02, seed=K + N))


@pytest.mark.parametrize("K,N", [(1000, 1531), (2049, 129), (6143, 2047),
                                 (127, 65), (1, 1), (2048, 1), (3, 4096),
                                 (1021, 769)])
def test_odd_dims(K, N):
    """Non-power-of-2 / degenerate dims must be masked, not rounded."""
    assert_matches(rnd(4, K, seed=K), rnd(N, K, scale=0.02, seed=N))


def test_fp32_ieee():
    """fp32 must use IEEE fp32 in tl.dot, not TF32 (which is ~1e-3 relative)."""
    x, w = rnd(4, 2048, dtype=torch.float32), rnd(2048, 2048, scale=0.02,
                                                  dtype=torch.float32, seed=3)
    err = assert_matches(x, w)
    ref_max = float(F.linear(x, w).abs().max())
    assert err / ref_max < 1e-5, f"relative {err / ref_max:.2e} smells like TF32"


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1.0, 1e2, 1e4])
def test_magnitudes(scale):
    assert_matches(rnd(4, 2048, scale=scale, seed=int(scale) + 1),
                   rnd(2048, 2048, scale=0.02, seed=9))


def test_zeros_and_empty_contribution():
    z = torch.zeros(4, 2048, device=DEV, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = thin_linear(z, rnd(2048, 2048, scale=0.02))
    assert torch.all(out == 0)


def test_noncontiguous_x_and_w():
    """A transposed activation view and a slice of a wider (merged) weight."""
    assert_matches(rnd(2048, 4, seed=11).t(), rnd(2048, 2048, scale=0.02, seed=12))
    big = rnd(4096, 2048, scale=0.02, seed=13)
    assert_matches(rnd(4, 2048, seed=14), big[1024:3072])   # offset view
    assert_matches(rnd(4, 2048, seed=15), big[::2])         # strided view


def test_leading_dims_preserved():
    """nn.Linear semantics: only the last dim is contracted."""
    x, w = rnd(2, 3, 2048, seed=16), rnd(512, 2048, scale=0.02, seed=17)
    with torch.inference_mode():
        got, ref = thin_linear(x, w), F.linear(x, w)
    assert got.shape == (2, 3, 512) == ref.shape
    assert float((got.float() - ref.float()).abs().max()) <= tol(ref)


def test_bias_applied():
    x, w = rnd(4, 512, seed=18), rnd(256, 512, scale=0.02, seed=19)
    b = rnd(256, scale=0.5, seed=20)
    with torch.inference_mode():
        got, ref = thin_linear(x, w, b), F.linear(x, w, b)
    assert float((got.float() - ref.float()).abs().max()) <= tol(ref)


def test_rejects_grad_mode():
    """Forward-only: must refuse when autograd would record, not silently detach."""
    x = rnd(4, 512, seed=21).requires_grad_(True)
    w = rnd(256, 512, scale=0.02, seed=22)
    with pytest.raises(RuntimeError, match="forward-only"):
        thin_linear(x, w)
    # ...but the very same tensors are fine with grad mode off, which is how the
    # pipeline calls it (loaded weights keep requires_grad=True after .eval()).
    with torch.no_grad():
        thin_linear(x, w)


def test_dtype_mismatch_rejected():
    with pytest.raises(TypeError):
        with torch.inference_mode():
            thin_linear(rnd(4, 512), rnd(256, 512, dtype=torch.float32))


@pytest.mark.parametrize("split_k", [1, 2, 8])
def test_reduction_is_deterministic(split_k):
    """Same inputs must give bit-identical outputs across calls.

    This is why the split-K reduction is a fixed-order two-stage sum rather than
    atomic_add: the pipeline's acceptance gate compares generated text, and a
    nondeterministic last mantissa bit would make it flaky.

    `split_k` is forced here rather than taken from `pick_config`, because the
    search found SPLIT_K=1 best at every deployed shape — but the split-K path is
    still reachable through the heuristic fallback for un-searched shapes, so its
    determinism still has to hold.
    """
    x, w = rnd(4, 6144, seed=23), rnd(2048, 6144, scale=0.02, seed=24)
    cfg = (16, 32, 64, split_k)
    with torch.inference_mode():
        first = thin_linear(x, w, config=cfg)
        for _ in range(8):
            assert torch.equal(thin_linear(x, w, config=cfg), first)
        # ...and it must still match the reference at every split count
        ref = F.linear(x, w)
        assert float((first.float() - ref.float()).abs().max()) <= tol(ref)


def test_dispatch_falls_back_above_searched_M():
    """`ThinLinear` must hand large-M shapes to cuBLAS, bit-exactly.

    Regression guard for a real end-to-end failure: with prefill (M≈96) running
    the unsearched heuristic path, `down_proj` took 132 us instead of 23 us and
    byte-identical text dropped from 20/20 to 11/20.
    """
    from kernels.thin_linear_fused import should_fuse
    lin = nn.Linear(2048, 2048, bias=False).to(DEV, torch.bfloat16).eval()
    tl_mod = ThinLinear(lin)
    assert should_fuse(4, 2048, 2048) and should_fuse(16, 2048, 2048)
    assert not should_fuse(32, 2048, 2048) and not should_fuse(96, 2048, 2048)

    with torch.inference_mode():
        for M, fused_expected in ((4, True), (16, True), (32, False), (96, False)):
            x = rnd(M, 2048, seed=M)
            got, ref = tl_mod(x), F.linear(x, lin.weight)
            if fused_expected:
                assert float((got.float() - ref.float()).abs().max()) <= tol(ref)
            else:
                # fell back to cuBLAS -> must be exactly the reference
                assert torch.equal(got, ref), f"M={M} should have used F.linear"
        # leading dims are flattened for the M decision, not just dim 0
        x3 = rnd(1, 96, 2048, seed=7)
        assert torch.equal(tl_mod(x3), F.linear(x3, lin.weight))

        # always_fuse overrides the guard (needed to benchmark the kernel at
        # large M); it must still be correct, just no longer bit-exact.
        x96 = rnd(96, 2048, seed=8)
        forced = ThinLinear(lin, always_fuse=True)(x96)
        ref96 = F.linear(x96, lin.weight)
        assert float((forced.float() - ref96.float()).abs().max()) <= tol(ref96)


def test_tuned_table_not_used_outside_its_M():
    """`TUNED` was searched at M=4/BLOCK_M=16; prefill-sized M must not inherit it."""
    from kernels.thin_linear_fused import TUNED, _tuned_for
    assert (2048, 2048) in TUNED
    assert _tuned_for(4, 2048, 2048) is not None
    assert _tuned_for(16, 2048, 2048) is not None      # still rounds to BLOCK_M=16
    assert _tuned_for(96, 2048, 2048) is None          # prefill -> heuristic
    assert _tuned_for(4, 999, 2048) is None            # unsearched shape


def test_cuda_graph_capture_and_replay():
    """The whole point: must capture and replay with no host sync or realloc."""
    x, w = rnd(4, 2048, seed=25), rnd(2048, 2048, scale=0.02, seed=26)
    with torch.inference_mode():
        ref = F.linear(x, w)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):          # warmup: JIT + workspace alloc
            for _ in range(3):
                thin_linear(x, w)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = thin_linear(x, w)
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    assert float((out.float() - ref.float()).abs().max()) <= tol(ref)

    # replay must track a mutated input, i.e. it really re-ran the kernel
    with torch.inference_mode():
        x.mul_(2.0)
        g.replay()
        torch.cuda.synchronize()
        assert float((out.float() - F.linear(x, w).float()).abs().max()) <= tol(ref) * 2


def test_graph_capture_with_multiple_shapes_shares_workspace():
    """Several split-K shapes in one graph reuse the staging buffer serially."""
    xs = [(rnd(4, 2048, seed=30), rnd(1024, 2048, scale=0.02, seed=31)),
          (rnd(4, 6144, seed=32), rnd(2048, 6144, scale=0.02, seed=33)),
          (rnd(4, 2048, seed=34), rnd(2048, 2048, scale=0.02, seed=35))]
    with torch.inference_mode():
        refs = [F.linear(x, w) for x, w in xs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                for x, w in xs:
                    thin_linear(x, w)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = [thin_linear(x, w) for x, w in xs]
        g.replay()
        torch.cuda.synchronize()
    for out, ref in zip(outs, refs):
        assert float((out.float() - ref.float()).abs().max()) <= tol(ref)


# ── the toggle ────────────────────────────────────────────────────────────────
class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512, bias=False)
        self.gate_proj = nn.Linear(512, 1024, bias=False)
        self.other = nn.Linear(512, 8, bias=False)     # not a target name

    def forward(self, x):
        return self.gate_proj(self.q_proj(x)), self.other(x)


def test_toggle_swaps_and_restores():
    m = _Tiny().to(DEV, torch.bfloat16).eval()
    x = rnd(4, 512, seed=40)
    with torch.inference_mode():
        ref = m(x)

    n = fused_toggle.apply_to_model(m, True)
    assert n == 2, "q_proj + gate_proj only; 'other' is not a target"
    assert isinstance(m.q_proj, ThinLinear) and isinstance(m.other, nn.Linear)
    assert fused_toggle.apply_to_model(m, True) == 0, "must be idempotent"
    with torch.inference_mode():
        got = m(x)
    for a, b in zip(got, ref):
        assert float((a.float() - b.float()).abs().max()) <= tol(b)

    assert fused_toggle.apply_to_model(m, False) == 2
    assert isinstance(m.q_proj, nn.Linear) and not fused_toggle.is_applied(m)


def test_toggle_preserves_state_dict_and_shares_params():
    m = _Tiny().to(DEV, torch.bfloat16).eval()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    w_ptr = m.q_proj.weight.data_ptr()
    with fused_toggle.fused_linear(m):
        after = m.state_dict()
        assert set(after) == set(before)
        for k in before:
            assert torch.equal(after[k], before[k]), k
        assert m.q_proj.weight.data_ptr() == w_ptr, "must alias, not copy"


def test_toggle_off_is_a_noop():
    m = _Tiny().to(DEV, torch.bfloat16).eval()
    assert fused_toggle.apply_to_model(m, False) == 0
    assert isinstance(m.q_proj, nn.Linear)
