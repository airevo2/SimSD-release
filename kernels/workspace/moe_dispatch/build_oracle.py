#!/usr/bin/env python3
"""Build the moe_dispatch oracle: synthetic + extreme + real-weight cases.

Inference mode -> forward only, ground truth under torch.inference_mode().

Two ground-truth sources, both traced to the *real* reference:
  synthetic/extreme -- random inputs + random stacked weights, ground truth from
      `reference_moe_dispatch`, a line-by-line transcription of
      LLaDA2MoeSparseMoeBlock.moe_infer (verified against the real method below).
  real              -- one real MoE layer of LLaDA2.0-mini: its actual expert
      weights and a real (x, topk_ids, topk_weight) captured from a forward,
      ground truth from the module's own unmodified `moe_infer`.

Usage:
    python kernels/workspace/moe_dispatch/build_oracle.py            # synthetic only
    python kernels/workspace/moe_dispatch/build_oracle.py --real     # + real layer
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

WS = Path(__file__).resolve().parent
ORACLE = WS / "oracle"


# ─────────────────────────────────────────────────────────────────────
# Reference: a transcription of moe_infer against stacked weights.
# ─────────────────────────────────────────────────────────────────────
def reference_moe_dispatch(x, topk_ids, topk_weight, w_gate, w_up, w_down):
    """Line-for-line equivalent of LLaDA2MoeSparseMoeBlock.moe_infer.

    Kept deliberately naive (Python loop over experts, one matmul each) so it is
    obviously the same computation as the reference; speed is irrelevant here.
    """
    E = w_gate.shape[0]
    k = topk_ids.shape[1]
    cnts = topk_ids.new_zeros((topk_ids.shape[0], E))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // k]

    outputs = []
    start = 0
    for e in range(E):
        n = int(tokens_per_expert[e].item())
        if n == 0:
            continue
        toks = sorted_tokens[start:start + n]
        g = F.linear(toks, w_gate[e])
        u = F.linear(toks, w_up[e])
        outputs.append(F.linear(F.silu(g) * u, w_down[e]))
        start += n

    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype))


# ─────────────────────────────────────────────────────────────────────
def _routing(T, k, E, device, concentration=None, generator=None):
    """topk_ids/topk_weight with the reference's dtype discipline.

    `concentration`: None = uniform over experts; a float in (0,1] biases the
    draw toward a small subset, which is what the real router does (median 23 of
    256 active). Routing sparsity is the load-bearing property of this operator,
    so the oracle must cover both.
    """
    g = generator
    if concentration is None:
        logits = torch.randn(T, E, device=device, generator=g)
    else:
        pool = max(1, int(E * concentration))
        logits = torch.randn(T, E, device=device, generator=g) - 8.0
        hot = torch.randperm(E, device=device, generator=g)[:pool]
        logits[:, hot] += 8.0
    topk_ids = logits.topk(k, dim=-1).indices.contiguous()
    scores = torch.sigmoid(logits.float()).gather(1, topk_ids)
    w = scores / (scores.sum(-1, keepdim=True) + 1e-20) if k > 1 else scores
    return topk_ids, (w * 2.5).float()          # routed_scaling_factor = 2.5


def _case(name, source, T, k, E, H, I, dtype, device, scale=1.0,
          concentration=None, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn(T, H, device=device, dtype=torch.float32, generator=g)
         * scale).to(dtype)
    topk_ids, topk_weight = _routing(T, k, E, device, concentration, g)
    mk = lambda *s: (torch.randn(*s, device=device, dtype=torch.float32,
                                 generator=g) / (s[-1] ** 0.5)).to(dtype)
    w_gate, w_up, w_down = mk(E, I, H), mk(E, I, H), mk(E, H, I)
    with torch.inference_mode():
        ref = reference_moe_dispatch(x, topk_ids, topk_weight,
                                     w_gate, w_up, w_down)
    return {
        "name": name, "source": source,
        "args": [x, topk_ids, topk_weight, w_gate, w_up, w_down],
        "kwargs": {}, "ref_out": ref,
        "meta": {"T": T, "k": k, "E": E, "H": H, "I": I, "dtype": str(dtype),
                 "scale": scale, "concentration": concentration,
                 "n_active": int(topk_ids.unique().numel())},
    }


def build_synthetic(device):
    cases = []
    for dt in (torch.float32, torch.bfloat16):
        tag = "fp32" if dt is torch.float32 else "bf16"
        # 真实工作负载（mini / flash），以及 verify 前向的 T=64
        cases.append(_case(f"mini_T32_{tag}", "fake", 32, 8, 256, 2048, 512, dt,
                           device, concentration=0.15, seed=1))
        cases.append(_case(f"mini_T64_{tag}", "fake", 64, 8, 256, 2048, 512, dt,
                           device, concentration=0.20, seed=2))
        cases.append(_case(f"flash_T32_{tag}", "fake", 32, 8, 256, 4096, 1024, dt,
                           device, concentration=0.15, seed=3))
        # 均匀路由：活跃专家接近上限，tile 数最坏
        cases.append(_case(f"uniform_{tag}", "fake", 32, 8, 256, 2048, 512, dt,
                           device, concentration=None, seed=4))
        # 极端形状
        cases.append(_case(f"single_token_{tag}", "extreme", 1, 8, 256, 2048, 512,
                           dt, device, concentration=0.05, seed=5))
        cases.append(_case(f"tiny_{tag}", "extreme", 4, 2, 8, 64, 32, dt, device,
                           seed=6))
        cases.append(_case(f"odd_dims_{tag}", "extreme", 7, 3, 13, 130, 70, dt,
                           device, seed=7))
        cases.append(_case(f"k1_{tag}", "extreme", 16, 1, 32, 128, 64, dt, device,
                           seed=8))
        # 幅值极端
        cases.append(_case(f"large_mag_{tag}", "extreme", 16, 4, 64, 256, 128, dt,
                           device, scale=100.0, seed=9))
        cases.append(_case(f"tiny_mag_{tag}", "extreme", 16, 4, 64, 256, 128, dt,
                           device, scale=1e-3, seed=10))
        # 所有 token 撞同一个专家：单个 group 最长，其余 255 个 group 为空。
        # 必须用 k=1 —— 参考实现的 cnts.scatter_ 按 token 去重计数，隐含要求
        # 同一 token 的 top-k 专家互不相同（真实 topk 天然满足）。k>1 时把 ids
        # 全设成同一个专家会让 tokens_per_expert 之和 (T) 与 idxs 长度 (T*k)
        # 对不上，那是非法输入而不是边界情形。
        c = _case(f"all_one_expert_{tag}", "extreme", 32, 1, 256, 2048, 512, dt,
                  device, concentration=0.05, seed=11)
        c["args"][1] = torch.zeros_like(c["args"][1])
        with torch.inference_mode():
            c["ref_out"] = reference_moe_dispatch(*c["args"])
        c["meta"]["k"] = 1
        c["meta"]["n_active"] = 1
        cases.append(c)
    return cases


def build_real(device):
    """Capture one real MoE layer of LLaDA2.0-mini: its weights + a real routing."""
    import sys
    sys.path.insert(0, str(WS.parents[2]))
    from transformers import AutoModelForCausalLM
    from speculative_decoding.cache_aware import (
        patch_sdpa_with_cache, patch_family_plumbing)

    m = AutoModelForCausalLM.from_pretrained(
        "inclusionAI/LLaDA2.0-mini", torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": device}).eval()
    patch_sdpa_with_cache(m); patch_family_plumbing(m)

    blk = None
    for mod in m.modules():
        if mod.__class__.__name__ == "LLaDA2MoeSparseMoeBlock":
            blk = mod
            break
    grabbed = {}
    orig = type(blk).moe_infer

    def spy(self, x, topk_ids, topk_weight):
        out = orig(self, x, topk_ids, topk_weight)
        if self is blk and "x" not in grabbed:
            grabbed.update(x=x.clone(), ids=topk_ids.clone(),
                           w=topk_weight.clone(), out=out.clone())
        return out
    type(blk).moe_infer = spy

    from transformers.cache_utils import DynamicCache
    V = m.config.vocab_size
    torch.manual_seed(0)
    cur = torch.randint(0, V, (1, 32), device=device); cur[:, 16:] = 156895
    with torch.inference_mode():
        m(input_ids=cur,
          attention_mask=torch.ones(1, 1, 32, 32, dtype=torch.bool, device=device),
          position_ids=torch.arange(32, device=device).unsqueeze(0),
          past_key_values=DynamicCache(), use_cache=True, store_kv=False)
    type(blk).moe_infer = orig
    assert "x" in grabbed, "hook 没抓到 moe_infer 调用"

    w_gate = torch.stack([e.gate_proj.weight for e in blk.experts]).contiguous()
    w_up = torch.stack([e.up_proj.weight for e in blk.experts]).contiguous()
    w_down = torch.stack([e.down_proj.weight for e in blk.experts]).contiguous()
    case = {
        "name": "real_mini_layer", "source": "real",
        "args": [grabbed["x"], grabbed["ids"], grabbed["w"],
                 w_gate, w_up, w_down],
        "kwargs": {}, "ref_out": grabbed["out"],
        "meta": {"T": grabbed["x"].shape[0], "k": grabbed["ids"].shape[1],
                 "E": len(blk.experts), "H": w_gate.shape[2], "I": w_gate.shape[1],
                 "dtype": str(grabbed["x"].dtype), "scale": None,
                 "concentration": "real",
                 "n_active": int(grabbed["ids"].unique().numel())},
    }
    # 关键校验：我们的转写与模块自己的 moe_infer 必须一致（bf16 容差内）
    with torch.inference_mode():
        mine = reference_moe_dispatch(*case["args"])
    tol = 2 ** -7 * case["ref_out"].abs().max().item() + 1e-3
    d = (mine.float() - case["ref_out"].float()).abs().max().item()
    print(f"  [check] 转写 vs 真实 moe_infer: max|Δ|={d:.3e}  tol={tol:.3e}  "
          f"{'OK' if d <= tol else '**FAIL**'}")
    assert d <= tol, "reference_moe_dispatch 与真实 moe_infer 不一致"
    del m
    torch.cuda.empty_cache()
    return [case]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    ORACLE.mkdir(parents=True, exist_ok=True)

    cases = build_synthetic(args.device)
    if args.real:
        cases += build_real(args.device)

    man = []
    for c in cases:
        p = ORACLE / f"{c['name']}.pt"
        torch.save({k: v for k, v in c.items()}, p)
        man.append({"name": c["name"], "source": c["source"],
                    "file": p.name, **c["meta"]})
        print(f"  saved {c['name']:<26} {c['source']:<7} "
              f"T={c['meta']['T']:>3} E={c['meta']['E']:>3} "
              f"H={c['meta']['H']:>4} I={c['meta']['I']:>4} "
              f"active={c['meta']['n_active']:>3} {c['meta']['dtype']}")
    (ORACLE / "manifest.jsonl").write_text(
        "\n".join(json.dumps(x) for x in man) + "\n")
    print(f"\n{len(cases)} cases -> {ORACLE}")


if __name__ == "__main__":
    main()
