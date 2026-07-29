"""Build the `thin_linear` oracle: persisted (args, ref_out) cases.

Inference mode (CONTRACT.md), so a case carries no grad fields:

    {"name", "source": "real"|"fake"|"extreme",
     "args": [x, w], "kwargs": {}, "ref_out": F.linear(x, w)}

Three sources:
  * **real**   -- a `forward_pre_hook` on the live draft model's `nn.Linear`
                  modules during a real eager denoise forward: real activations
                  (which have bf16 outliers synthetic data does not) against the
                  real trained weights.
  * **fake**   -- synthetic sweep: the deployed shapes, tiny shapes,
                  non-power-of-2 dims, M in {1,2,4,8,16}, bf16 and fp32,
                  contiguous and transposed-view inputs.
  * **extreme** -- large / near-zero / mixed magnitudes, where a fp32-accumulating
                  split-K can visibly diverge from cuBLAS.

Usage:
    CUDA_VISIBLE_DEVICES=4 python kernels/workspace/thin_linear/build_oracle.py
    CUDA_VISIBLE_DEVICES=4 python kernels/workspace/thin_linear/build_oracle.py --no-real
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from shapes import draft_shapes  # noqa: E402

ORACLE = HERE / "oracle"
BL = 4
LAYER_OPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def save_case(name: str, source: str, x: torch.Tensor, w: torch.Tensor,
              manifest: list[dict]) -> None:
    with torch.inference_mode():
        ref = F.linear(x, w)
    case = {"name": name, "source": source, "args": [x, w], "kwargs": {},
            "ref_out": ref}
    torch.save(case, ORACLE / f"{name}.pt")
    manifest.append({
        "name": name, "source": source, "dtype": str(x.dtype).replace("torch.", ""),
        "M": x.shape[0], "K": x.shape[1], "N": w.shape[0],
        "x_contiguous": bool(x.is_contiguous()),
        "ref_absmax": float(ref.abs().max()), "x_absmax": float(x.abs().max()),
        "w_absmax": float(w.abs().max()),
    })
    print(f"  [{source:7s}] {name:44s} M={x.shape[0]:<3d} K={x.shape[1]:<6d} "
          f"N={w.shape[0]:<7d} {str(x.dtype).replace('torch.', ''):8s} "
          f"|ref|max={float(ref.abs().max()):.3e}")


# ── real I/O ──────────────────────────────────────────────────────────────────
def capture_real(dev, manifest: list[dict], layers=(0, 27), kv: int = 256) -> None:
    """Hook the real draft model during an eager denoise forward and stash the
    actual (activation, weight) pairs each `nn.Linear` was called with."""
    from transformers import AutoModelForCausalLM

    from sdar.run_native_tp2 import patch_stock_rms_norm
    from speculative_decoding.cache_aware import (
        StaticBlockCache, _prefill_prompt_static, patch_causallm_pass_cache,
        patch_sdpa_static_cache,
    )

    print(f"[real] loading draft model onto {dev} ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        str(REPO / "models/SDAR-1_7B-Chat"), torch_dtype=torch.bfloat16,
        trust_remote_code=True).to(dev).eval()
    patch_stock_rms_norm(m)
    patch_sdpa_static_cache(m)
    patch_causallm_pass_cache(m)

    cache = StaticBlockCache(m, kv, 2 * BL, dev, torch.bfloat16)
    m._kv_static_cache = cache
    ctx, _ = _prefill_prompt_static(m, list(range(100, 100 + 96)), cache, dev,
                                    block_length=BL)

    grabbed: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def mk_hook(tag):
        def hook(mod, inputs):
            if tag in grabbed:
                return
            x = inputs[0].detach()
            # nn.Linear flattens leading dims; the oracle contract is 2-D [M, K].
            x2 = x.reshape(-1, x.shape[-1])
            grabbed[tag] = (x2.clone(), mod.weight.detach().clone())
        return hook

    for li in layers:
        layer = m.model.layers[li]
        for op in LAYER_OPS:
            mod = getattr(layer.self_attn, op, None) or getattr(layer.mlp, op)
            handles.append(mod.register_forward_pre_hook(mk_hook(f"L{li}_{op}")))
    handles.append(m.lm_head.register_forward_pre_hook(mk_hook("lm_head")))

    # Eager forward with exactly the kwargs the captured graph uses
    # (cache_aware.py:499-509), so the activations are the deployed ones.
    B, cur_len, full_len = cache.batch_size, BL, cache.full_len
    with torch.no_grad():
        m(input_ids=torch.full((B, cur_len), 151669, dtype=torch.long, device=dev),
          attention_mask=torch.ones((B, 1, cur_len, full_len), dtype=torch.bool,
                                    device=dev),
          position_ids=torch.arange(ctx, ctx + cur_len, device=dev).unsqueeze(0)
                            .expand(B, -1).contiguous(),
          past_key_values=cache, use_cache=True, store_kv=False,
          cur_scratch_pos=torch.arange(cache.max_cache_len,
                                       cache.max_cache_len + cur_len,
                                       dtype=torch.long, device=dev),
          cache_position=torch.arange(cache.max_cache_len,
                                      cache.max_cache_len + cur_len,
                                      dtype=torch.long, device=dev),
          return_dict=True)
    for h in handles:
        h.remove()

    print(f"[real] captured {len(grabbed)} call sites")
    for tag, (x, w) in grabbed.items():
        save_case(f"real_{tag}", "real", x, w, manifest)

    del m, cache, grabbed
    torch.cuda.empty_cache()


# ── synthetic + extreme ───────────────────────────────────────────────────────
def build_fake(dev, manifest: list[dict]) -> None:
    g = torch.Generator(device=dev).manual_seed(1234)

    def rnd(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=g, device=dev,
                            dtype=torch.float32) * scale).to(dtype)

    # 1. deployed shapes, random values, both dtypes
    for sh in draft_shapes(4):
        for dt in (torch.bfloat16, torch.float32):
            if sh.tag == "lm_head" and dt is torch.float32:
                continue                      # 1.2 GB of fp32 weights, no new coverage
            save_case(f"fake_{sh.tag}_{str(dt).split('.')[-1]}", "fake",
                      rnd(sh.M, sh.K, dtype=dt), rnd(sh.N, sh.K, dtype=dt, scale=0.02),
                      manifest)

    # 2. M sweep -- bs=1 x block_length in {1,2,4,8}, plus 16 for the folded
    #    extend path (M = 2*block_length) and a spare power of two
    for M in (1, 2, 4, 8, 16):
        save_case(f"fake_M{M}_2048x2048", "fake", rnd(M, 2048),
                  rnd(2048, 2048, scale=0.02), manifest)

    # 3. odd / non-power-of-2 dims -- K not a multiple of any BLOCK_K,
    #    N not a multiple of any BLOCK_N, and a prime-ish K
    for K, N in ((1000, 1531), (2049, 129), (6143, 2047), (127, 65), (1, 1),
                 (2048, 1), (3, 4096)):
        save_case(f"fake_odd_{K}x{N}", "fake", rnd(4, K),
                  rnd(N, K, scale=0.02), manifest)

    # 4. tiny
    save_case("fake_tiny_64x64", "fake", rnd(1, 64), rnd(64, 64), manifest)

    # 5. non-contiguous x: a transposed view, which is what a fused caller that
    #    hands us `hidden.T` would produce
    xt = rnd(2048, 4).t()                     # [4, 2048], stride (1, 4)
    assert not xt.is_contiguous()
    save_case("fake_noncontig_x", "fake", xt, rnd(2048, 2048, scale=0.02), manifest)

    # 6. extremes -- these are where an fp32-accumulating split-K and cuBLAS
    #    diverge most visibly
    save_case("extreme_large_x", "extreme", rnd(4, 2048, scale=100.0),
              rnd(2048, 2048, scale=0.02), manifest)
    save_case("extreme_large_w", "extreme", rnd(4, 2048),
              rnd(2048, 2048, scale=10.0), manifest)
    save_case("extreme_tiny_vals", "extreme", rnd(4, 2048, scale=1e-3),
              rnd(2048, 2048, scale=1e-3), manifest)
    save_case("extreme_mixed_mag", "extreme",
              (rnd(4, 2048) * torch.logspace(-4, 4, 2048, device=dev)
               .to(torch.bfloat16)), rnd(2048, 2048, scale=0.02), manifest)
    save_case("extreme_zeros_x", "extreme",
              torch.zeros(4, 2048, device=dev, dtype=torch.bfloat16),
              rnd(2048, 2048, scale=0.02), manifest)
    # cancellation: alternating signs so the fp32 partial sums nearly cancel
    w_alt = rnd(2048, 2048, scale=0.02)
    w_alt[:, 1::2] *= -1
    save_case("extreme_cancellation", "extreme",
              rnd(4, 2048).abs(), w_alt, manifest)
    save_case("extreme_deep_K", "extreme", rnd(4, 32768),
              rnd(256, 32768, scale=0.005), manifest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-real", dest="real", action="store_false")
    ap.add_argument("--layers", default="0,27")
    args = ap.parse_args()

    ORACLE.mkdir(parents=True, exist_ok=True)
    for p in ORACLE.glob("*.pt"):
        p.unlink()

    dev = torch.device("cuda")
    print(f"[gpu] {torch.cuda.get_device_name(0)}\n")
    manifest: list[dict] = []

    build_fake(dev, manifest)
    if args.real:
        capture_real(dev, manifest,
                     layers=tuple(int(s) for s in args.layers.split(",")))

    with open(ORACLE / "manifest.jsonl", "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    n_by = {}
    for r in manifest:
        n_by[r["source"]] = n_by.get(r["source"], 0) + 1
    size = sum(p.stat().st_size for p in ORACLE.glob("*.pt")) / 2**20
    print(f"\n[oracle] {len(manifest)} cases ({n_by}) -> {ORACLE}  ({size:.0f} MB)")


if __name__ == "__main__":
    main()
