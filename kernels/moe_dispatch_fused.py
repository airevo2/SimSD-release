"""Fused MoE expert dispatch for LLaDA2 — grouped GEMM, CUDA-graph safe.

Replaces ``LLaDA2MoeSparseMoeBlock.moe_infer``, whose per-expert Python loop plus
``tokens_per_expert.cpu().numpy()`` costs 65% of a forward (32.5 of 50.0 ms on
LLaDA2.0-mini) and, because of that host sync, makes ``torch.cuda.graph`` capture
impossible. See ``docs/llada2-plan.md`` and
``kernels/workspace/moe_dispatch/CONTRACT.md``.

Correctness-first (this is the kernel-fuse deliverable, not the tuned one):
three grouped-GEMM launches plus one elementwise, against the stock path's
~3 x n_active tiny ``nn.Linear`` calls. Fusing gate+up and the activation is
left to kernel-optimize.

CUDA-graph safety, which is the whole point:
  * no ``.item()`` / ``.cpu()`` / data-dependent Python control flow
  * every tensor shape is a function of (T, k, E, H, I) alone — only the
    *values* in ``slots`` / ``tile_expert`` depend on the routing
  * the Triton grid is sized for the worst case; tiles with no expert early-exit
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# BLOCK_M must be >= 16 for tl.dot. Routing is sparse (median 83 of 256 experts
# active for T=32, so ~3 slots per active expert), which means a tile is mostly
# padding — that is the price of turning ~250 launches into 4.
BLOCK_M = 16


@triton.jit
def _grouped_gemm_kernel(
    A_ptr, W_ptr, C_ptr,
    slots_ptr, tile_expert_ptr,
    M, N, K, E, ROWS_PER_SLOT,
    stride_am, stride_ak,
    stride_we, stride_wn, stride_wk,
    stride_cm, stride_cn,
    GATHER_ROWS: tl.constexpr,
    IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """C[slot] = A[row(slot)] @ W[e].T for every slot in this tile's expert group.

    ``slots`` holds, per padded row position, the original flat slot index
    (``token * k + j``) or -1 for padding. ``tile_expert`` holds this tile's
    expert id, or a value >= E when the tile is beyond the used range.

    GATHER_ROWS: A is indexed by ``slot // ROWS_PER_SLOT`` (the token row) when
    reading the hidden states, and by ``slot`` directly when reading an
    already-slot-indexed intermediate.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    e = tl.load(tile_expert_ptr + pid_m)
    if e >= E:                      # tile past the used range -> nothing to do
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    slot = tl.load(slots_ptr + offs_m, mask=offs_m < M, other=-1)
    valid = slot >= 0
    a_row = tl.where(valid, slot // ROWS_PER_SLOT if GATHER_ROWS else slot, 0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            A_ptr + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=valid[:, None] & k_mask[None, :], other=0.0)
        # W is (E, N, K) row-major, i.e. nn.Linear's (out, in): load W[e] as
        # (BLOCK_K, BLOCK_N) so the dot is a @ w without an explicit transpose.
        w = tl.load(
            W_ptr + e * stride_we + offs_n[None, :] * stride_wn
            + offs_k[:, None] * stride_wk,
            mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        if IEEE:
            acc += tl.dot(a, w, input_precision="ieee")
        else:
            acc += tl.dot(a, w)

    tl.store(
        C_ptr + slot[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C_ptr.dtype.element_ty),
        mask=valid[:, None] & n_mask[None, :])


def _align(topk_ids: torch.Tensor, E: int, num_tiles: int):
    """Sort slots by expert and lay each group out on BLOCK_M boundaries.

    Fully on-GPU and shape-static: only the *contents* of the two returned
    tensors depend on the routing. Mirrors vLLM's ``moe_align_block_size``.

    Returns
        slots       (num_tiles*BLOCK_M,) int32 — original flat slot index, or -1
        tile_expert (num_tiles,)         int32 — expert for the tile, or E (skip)
    """
    dev = topk_ids.device
    flat = topk_ids.reshape(-1)
    M = flat.numel()
    M_pad = num_tiles * BLOCK_M

    # counts per expert, then each group padded up to a BLOCK_M multiple.
    # scatter_add_ rather than torch.bincount: bincount reduces over the input
    # to size its output and syncs on CUDA even with minlength set, which alone
    # would keep this operator out of a CUDA graph.
    counts = torch.zeros(E, dtype=torch.long, device=dev).scatter_add_(
        0, flat, torch.ones_like(flat))
    padded = ((counts + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
    grp_pad_end = padded.cumsum(0)                       # (E,)
    grp_pad_start = grp_pad_end - padded
    grp_start = counts.cumsum(0) - counts                # unpadded group starts

    # stable sort keeps slots of one expert in ascending slot order, matching
    # the reference's argsort (which is stable for equal keys on CUDA).
    order = torch.argsort(flat, stable=True)
    rank = torch.arange(M, device=dev) - grp_start[flat[order]]
    dest = grp_pad_start[flat[order]] + rank

    slots = torch.full((M_pad,), -1, dtype=torch.int32, device=dev)
    slots[dest] = order.to(torch.int32)

    # tile t covers padded rows [t*BLOCK_M, ...): its expert is the group whose
    # padded range contains that row. Tiles past the end get E -> early exit.
    tile_row = torch.arange(num_tiles, device=dev) * BLOCK_M
    tile_expert = torch.searchsorted(grp_pad_end, tile_row, right=True)
    return slots, tile_expert.to(torch.int32)


def _num_tiles(M: int, E: int) -> int:
    """Worst-case tile count: every active group costs at least one tile."""
    return min(E, M) + (M + BLOCK_M - 1) // BLOCK_M


def _gemm(a, w, slots, tile_expert, E, M, N, gather_rows, rows_per_slot, out):
    K = a.shape[1]
    BLOCK_N = 64 if N >= 64 else triton.next_power_of_2(N)
    BLOCK_K = 64 if K >= 64 else triton.next_power_of_2(K)
    grid = (tile_expert.numel(), triton.cdiv(N, BLOCK_N))
    # Device guard is mandatory, not defensive. A raw Triton launch goes to
    # torch.cuda.current_device(), NOT to the tensors' device — unlike a PyTorch
    # op, which carries its own guard. With an accelerate-sharded target the
    # layer's weights live on cuda:1..4 while the current device is still cuda:0,
    # so without this the kernel reads another card's address space and the run
    # dies with an illegal memory access several layers later.
    with torch.cuda.device(a.device):
        _grouped_gemm_kernel[grid](
            a, w, out, slots, tile_expert,
            slots.numel(), N, K, E, rows_per_slot,
            a.stride(0), a.stride(1),
            w.stride(0), w.stride(1), w.stride(2),
            out.stride(0), out.stride(1),
            GATHER_ROWS=gather_rows,
            IEEE=(a.dtype == torch.float32),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )


def moe_dispatch(x, topk_ids, topk_weight, w_gate, w_up, w_down):
    """Drop-in for ``LLaDA2MoeSparseMoeBlock.moe_infer`` on stacked weights.

    x           (T, H)    topk_ids (T, k) int64    topk_weight (T, k) fp32
    w_gate/up   (E, I, H)  w_down  (E, H, I)   -> (T, H)
    """
    T, H = x.shape
    k = topk_ids.shape[1]
    E, I, _ = w_gate.shape
    M = T * k

    slots, tile_expert = _align(topk_ids, E, _num_tiles(M, E))

    g = torch.empty((M, I), device=x.device, dtype=x.dtype)
    u = torch.empty((M, I), device=x.device, dtype=x.dtype)
    _gemm(x, w_gate, slots, tile_expert, E, M, I, True, k, g)
    _gemm(x, w_up, slots, tile_expert, E, M, I, True, k, u)
    h = torch.nn.functional.silu(g) * u

    y = torch.empty((M, H), device=x.device, dtype=x.dtype)
    _gemm(h, w_down, slots, tile_expert, E, M, H, False, k, y)

    # Reduction over the k experts in fp32, cast back at the end — the
    # reference's dtype discipline, not an approximation of it.
    return (y.view(T, k, H).to(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(-1))
            .sum(dim=1)
            .to(x.dtype))


def stack_expert_weights(experts, free_originals: bool = True):
    """(E,I,H), (E,I,H), (E,H,I) from an ``nn.ModuleList`` of LLaDA2MoeMLP.

    Called once at patch time. Stacks **one projection at a time** and releases
    the per-expert tensors as it goes, so peak extra memory is a single
    projection (E*I*H) rather than the whole layer, and the model's total
    footprint is unchanged once done. Without the release this OOMs on
    LLaDA2.0-mini in fp32 (60.6 GiB of weights on a 95 GiB card).

    ``free_originals=True`` is **destructive**: the per-expert Linears are left
    with empty weights and the stock ``moe_infer`` can no longer run on this
    model object. Reload the checkpoint to get it back.
    """
    stacked = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        w = torch.stack([getattr(e, name).weight.data
                         for e in experts]).contiguous()
        if free_originals:
            empty = w.new_empty(0)
            for e in experts:
                getattr(e, name).weight.data = empty
        stacked.append(w)
    return tuple(stacked)


class FusedMoEDispatch(torch.nn.Module):
    """nn.Module drop-in mirroring the isolated signature."""

    def __init__(self, w_gate, w_up, w_down):
        super().__init__()
        self.register_buffer("w_gate", w_gate, persistent=False)
        self.register_buffer("w_up", w_up, persistent=False)
        self.register_buffer("w_down", w_down, persistent=False)

    def forward(self, x, topk_ids, topk_weight):
        return moe_dispatch(x, topk_ids, topk_weight,
                            self.w_gate, self.w_up, self.w_down)
