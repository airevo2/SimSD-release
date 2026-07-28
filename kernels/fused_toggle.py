"""Opt-in switch that routes a model's `nn.Linear` calls through the fused kernels.

The reference lives in third-party code — `models/SDAR-1_7B-Chat/modeling_sdar.py`
is loaded via `trust_remote_code`, so the module that actually executes is the
copy under `$HF_MODULES_CACHE/transformers_modules/SDAR-1_7B-Chat/` — and the
call sites are plain `self.q_proj(x)` module calls with no config to thread a flag
through. So the toggle patches the modules in place instead of editing that source.

Default is **off**: nothing here runs unless the driver passes the flag, and
`apply_to_model(m, False)` restores the original modules.

Usage from a driver:
    from kernels import fused_toggle
    fused_toggle.apply_to_model(draft_model, args.fused_linear)   # before graph capture

Must be called **before** CUDA graph capture: capture freezes whichever forward
is installed at capture time, so toggling afterwards silently has no effect on a
replayed graph.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .thin_linear_fused import ThinLinear, reset_workspace

# Which projections to swap. `lm_head` is excluded by default: measured on this
# card it already runs at 84% of DRAM peak under cuBLAS (N=151936 gives 1187
# output tiles, so it is not SM-starved) and the fused kernel only matches it.
DEFAULT_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
ALL_TARGETS = DEFAULT_TARGETS + ("lm_head",)

_ORIGINALS: dict[int, dict[str, nn.Module]] = {}


def _iter_targets(model: nn.Module, targets):
    for name, mod in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in targets and isinstance(mod, nn.Linear):
            yield name, mod


def _set_submodule(model: nn.Module, qualified: str, new: nn.Module) -> None:
    parent_path, _, leaf = qualified.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    setattr(parent, leaf, new)


def apply_to_model(model: nn.Module, enable: bool = True,
                   targets=DEFAULT_TARGETS,
                   config: tuple[int, int, int, int] | None = None) -> int:
    """Swap the target `nn.Linear` modules for `ThinLinear` (or restore them).

    Idempotent in both directions. Returns the number of modules swapped
    (or restored). `ThinLinear` aliases the original parameters rather than
    copying, so this costs no extra memory and `state_dict()` is unchanged.
    """
    key = id(model)
    if not enable:
        saved = _ORIGINALS.pop(key, None)
        if not saved:
            return 0
        for path, orig in saved.items():
            _set_submodule(model, path, orig)
        reset_workspace()
        return len(saved)

    if key in _ORIGINALS:
        return 0                                    # already patched
    saved: dict[str, nn.Module] = {}
    for path, mod in list(_iter_targets(model, targets)):
        saved[path] = mod
        _set_submodule(model, path, ThinLinear(mod, config=config))
    if saved:
        _ORIGINALS[key] = saved
    return len(saved)


def is_applied(model: nn.Module) -> bool:
    return id(model) in _ORIGINALS


class fused_linear:
    """Context manager form, for tests.

        with fused_toggle.fused_linear(model):
            ...
    """

    def __init__(self, model: nn.Module, enable: bool = True, **kw):
        self.model, self.enable, self.kw = model, enable, kw
        self.n = 0

    def __enter__(self):
        self.n = apply_to_model(self.model, self.enable, **self.kw)
        return self

    def __exit__(self, *exc):
        if self.enable:
            apply_to_model(self.model, False)
        return False


def warmup(model: nn.Module, block_length: int = 4, extra_M=()) -> None:
    """Trigger Triton JIT for every (shape, M) the pipeline will actually use.

    This is not optional bookkeeping — it is worth ~10 s of wall clock. Triton
    compiles per M (because `pick_config` picks `BLOCK_M` from it, and Triton
    additionally specialises on size divisibility), and the pipeline captures
    graphs at several `cur_len` values plus a long prefill. Without a sweep, the
    JIT cost lands *inside* the first few timed samples: measured on n=20 gsm8k,
    samples 0-3 carried 4.86 / 1.42 / 1.80 / 1.79 s of extra wall clock while
    samples 4-19 were uniformly ~11% faster than cuBLAS. Amortised over 19
    samples that one-off cost inverted the headline tok/s.

    M values covered: 1 (degenerate), `block_length` (denoise/verify),
    `2*block_length` (extend and the folded denoise), and two prefill-ish
    lengths on either side of a 16-boundary, since Triton's specialisation
    buckets sizes by divisibility rather than exact value.
    """
    Ms = {1, block_length, 2 * block_length, 96, 97, *extra_M}
    mods = [m for _, m in model.named_modules() if isinstance(m, ThinLinear)]
    with torch.inference_mode():
        for mod in mods:
            for M in sorted(Ms):
                # Goes through ThinLinear.forward, so the M values that dispatch
                # to F.linear cost nothing here and compile nothing.
                mod(torch.zeros(M, mod.in_features, device=mod.weight.device,
                                dtype=mod.weight.dtype))
    torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────
# LLaDA2 MoE expert dispatch  (kernels/moe_dispatch_fused.py)
# ─────────────────────────────────────────────────────────────────────
# The stock LLaDA2MoeSparseMoeBlock.moe_infer loops over active experts in
# Python and calls tokens_per_expert.cpu().numpy(). That costs 65% of a forward
# and, because of the host sync, makes torch.cuda.graph capture impossible.
# See docs/llada2-plan.md and kernels/workspace/moe_dispatch/CONTRACT.md.
#
# The stacked weights replace the per-expert parameters, so enabling this is not
# free in memory during the swap: peak overhead is one layer's experts
# (1.6 GiB for mini, 6.4 GiB for flash), released as each layer is converted.

_MOE_ATTR = "_simsd_fused_moe"


def apply_moe_to_model(model: nn.Module, enable: bool = True) -> int:
    """Swap LLaDA2 MoE blocks to the grouped-GEMM dispatch. Idempotent.

    Returns the number of blocks switched. Must run before any CUDA graph
    capture: capture freezes whichever forward is installed at capture time.
    """
    from .moe_dispatch_fused import moe_dispatch, stack_expert_weights

    n = 0
    for mod in model.modules():
        if mod.__class__.__name__ != "LLaDA2MoeSparseMoeBlock":
            continue
        if enable:
            if getattr(mod, _MOE_ATTR, None) is not None:
                continue
            wg, wu, wd = stack_expert_weights(mod.experts)
            setattr(mod, _MOE_ATTR, (wg, wu, wd))
            n += 1
        else:
            if getattr(mod, _MOE_ATTR, None) is None:
                continue
            raise NotImplementedError(
                "apply_moe_to_model(enable=False) is not supported: stacking "
                "releases the per-expert weights so the layer's footprint stays "
                "flat (keeping both copies OOMs mini in fp32). Reload the "
                "checkpoint to get the stock path back.")

    if n == 0:
        return 0

    cls = None
    for mod in model.modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            cls = mod.__class__
            break
    if enable and not getattr(cls, "_simsd_moe_patched", False):
        stock = cls.moe_infer

        def moe_infer(self, x, topk_ids, topk_weight):
            packed = getattr(self, _MOE_ATTR, None)
            if packed is None:
                return stock(self, x, topk_ids, topk_weight)
            return moe_dispatch(x, topk_ids, topk_weight, *packed)

        cls.moe_infer = moe_infer
        cls._simsd_moe_stock = stock
        cls._simsd_moe_patched = True
    return n


def is_moe_applied(model: nn.Module) -> bool:
    return any(getattr(m, _MOE_ATTR, None) is not None for m in model.modules())
