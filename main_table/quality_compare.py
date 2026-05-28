"""Generation-quality comparison: Native 8B vs Speculative (draft  8B).

Purpose: verify MRS is lossless in practice, not just theory. Runs a fixed set
of prompts through both backends, decodes to text, scores against ground truth,
and writes per-sample texts to results/ so both sides can be eyeballed side by
side. If spec pass@1 is significantly below native, indicates a bug in
verify-layout / step_map / probability alignment  not visible from latency alone.

Output: Experiment_Backend/results/quality/{dataset}_{tag}/
  - samples.jsonl : per-sample prompt / native_text / spec_text / pass flags
  - summary.json  : aggregate pass@1 for both sides + config
  - run.log       : stdout tee
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from speculative_decoding.bench.backends import (  # noqa: E402
    HFNativeBackend, HFSpeculativeBackend, compute_verify_padded_len,
)
from speculative_decoding.config import SpecConfig  # noqa: E402
from speculative_decoding.draft import patch_sdpa_eval_attention  # noqa: E402
from speculative_decoding.speculative_decode import load_prompts  # noqa: E402


# ─────────────────── scorers ───────────────────

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    return m.group(1) if m else text


def _exec_isolated(code: str, timeout_s: int = 5) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, "pass"
        return False, proc.stderr.decode("utf-8", "replace")[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"exc:{e!r}"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def score_gsm8k(gen_text: str, ref: dict) -> tuple[bool, str]:
    ref_ans = ref["answer"]
    m = re.search(r"####\s*([\-\d,\.]+)", ref_ans)
    if not m:
        return False, "no_ref"
    ref_num = m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", gen_text)
    if not nums:
        return False, f"no_gen_num ref={ref_num}"
    gen_num = nums[-1].replace(",", "").rstrip(".")
    try:
        ok = abs(float(gen_num) - float(ref_num)) < 1e-4
        return ok, f"ref={ref_num} gen={gen_num}"
    except ValueError:
        return False, f"parse_fail ref={ref_num} gen={gen_num}"


def score_humaneval(gen_text: str, ref: dict) -> tuple[bool, str]:
    code = _extract_code(gen_text)
    full = code + "\n" + ref["test"] + f"\ncheck({ref['entry_point']})\n"
    return _exec_isolated(full)


def score_mbpp(gen_text: str, ref: dict) -> tuple[bool, str]:
    code = _extract_code(gen_text)
    tests = "\n".join(ref.get("test_list", []))
    full = code + "\n" + tests + "\n"
    return _exec_isolated(full)


SCORERS = {"gsm8k": score_gsm8k, "humaneval": score_humaneval, "mbpp": score_mbpp}


# ─────────────────── main ───────────────────

def _load_model(path: str, device: torch.device, dtype: torch.dtype):
    m = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=dtype,
    ).to(device).eval()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draft_model", default="inference/model/SDAR-1_7B-Chat")
    p.add_argument("--target_model", default="inference/model/SDAR-8B-Chat")
    p.add_argument("--draft_device", default="cuda:0")
    p.add_argument("--target_device", default="cuda:0")
    p.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "humaneval", "mbpp"])
    p.add_argument("--dataset_split", default=None,
                   help="gsm8k: test / train (default test); ignored for others")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--num_blocks", type=int, default=64)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_eos_stop", action="store_true", default=False,
                   help="disable EOS stop (force full num_blocks gen). Off by default for quality eval.")
    p.add_argument("--tag", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--speculative_branch", default="mrs",
                   choices=["mrs", "greedy_match", "per_position_compare"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    draft_dev = torch.device(args.draft_device)
    target_dev = torch.device(args.target_device)

    draft_tag = Path(args.draft_model).name
    target_tag = Path(args.target_model).name
    tag = args.tag or f"{draft_tag}_to_{target_tag}"
    out_dir = Path(args.out_dir or (
        REPO / "speculative_decoding/Experiment_Backend/results/quality"
        / f"{args.dataset}_{tag}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {out_dir}")

    print(f"[load] tokenizer   {args.draft_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.draft_model, trust_remote_code=True)

    print(f"[load] draft  {args.draft_model}  {args.draft_device}")
    draft = _load_model(args.draft_model, draft_dev, dtype)
    print(f"[load] target {args.target_model}  {args.target_device}")
    target = _load_model(args.target_model, target_dev, dtype)

    # Build a cfg-like Namespace for load_prompts (uses dataset/dataset_split/num_samples)
    split = args.dataset_split or ("test" if args.dataset == "gsm8k" else "test")
    tmp_cfg = SimpleNamespace(
        dataset=args.dataset, dataset_split=split, num_samples=args.num_samples,
    )
    print(f"[data] {args.dataset}/{split}  num_samples={args.num_samples}")
    ds, prompt_ids = load_prompts(tokenizer, tmp_cfg)
    print(f"[data] loaded {len(prompt_ids)} prompts; lens "
          f"min={min(len(p) for p in prompt_ids)} max={max(len(p) for p in prompt_ids)}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id

    # ── NATIVE first (before any patch). Baseline = target model only.
    native_be = HFNativeBackend(
        model=target,
        num_blocks=args.num_blocks,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        mask_token_id=151669,
        eos_token_id=eos_id,
        use_cuda_graph=False,
    )
    print("[native] generating ...")
    native_out: list[tuple[int, list[int], str]] = []
    for i, pids in enumerate(prompt_ids):
        res = native_be.generate(pids, return_timings=False)
        txt = tokenizer.decode(res.generated_ids, skip_special_tokens=True)
        native_out.append((i, res.generated_ids, txt))
        snip = txt.replace("\n", "⏎")[:100]
        print(f"  [native {i+1:>3}/{len(prompt_ids)}] {len(res.generated_ids):>3}t  {snip!r}")

    # ── Patch for speculative path.
    # target_eval_sdpa=True path: skip patch_multi_block_mask_fn, only patch SDPA.
    # This matches bench_dual_gpu.yaml defaults and what all 2026-04-18+ runs used.
    patch_sdpa_eval_attention(target)
    patch_sdpa_eval_attention(draft)
    print("[patch] applied patch_sdpa_eval_attention on draft+target")

    spec_cfg = SpecConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_device=args.draft_device,
        target_device=args.target_device,
        mode="multi_block",
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        block_size=args.denoising_steps,
        num_blocks=args.num_blocks,
        K=args.K,
        dtype="bfloat16",
        mask_token_id=151669,
        seed=args.seed,
        batch=1,
        dataset=args.dataset,
        dataset_split=split,
        num_samples=args.num_samples,
        use_cuda_graph=False,
        target_eval_sdpa=True,
        pipeline=False,
        speculative_branch=args.speculative_branch,
    )
    max_prompt_len = max(len(p) for p in prompt_ids)
    padded_len = compute_verify_padded_len(
        max_prompt_len, spec_cfg.num_blocks, spec_cfg.block_length, K=spec_cfg.K,
    )
    print(f"[spec] padded_len={padded_len}  K={spec_cfg.K}  num_blocks={spec_cfg.num_blocks}")

    spec_be = HFSpeculativeBackend(
        draft_model=draft,
        target_model=target,
        cfg=spec_cfg,
        pad_token_id=pad_token_id,
        padded_len=padded_len,
        eos_token_id=eos_id,
    )

    print("[spec] generating ...")
    spec_out: list[tuple[int, list[int], str, dict]] = []
    for i, pids in enumerate(prompt_ids):
        res = spec_be.generate(pids, return_timings=False)
        txt = tokenizer.decode(res.generated_ids, skip_special_tokens=True)
        spec_out.append((i, res.generated_ids, txt, res.stats or {}))
        snip = txt.replace("\n", "⏎")[:100]
        a = (res.stats or {}).get("total_accepted_tokens", 0)
        d = (res.stats or {}).get("total_draft_tokens", 1) or 1
        alpha = (a / d) if d else 0.0
        print(f"  [spec   {i+1:>3}/{len(prompt_ids)}] {len(res.generated_ids):>3}t  α={alpha:.2f}  {snip!r}")

    # ── Score + write per-sample
    scorer = SCORERS.get(args.dataset)
    n_pass_native = 0
    n_pass_spec = 0
    jsonl_path = out_dir / "samples.jsonl"
    with open(jsonl_path, "w") as f:
        for (ni, nids, nt), (si, sids, st, stats) in zip(native_out, spec_out):
            assert ni == si
            ref = ds[ni]
            row: dict = {
                "idx": ni,
                "prompt_len": len(prompt_ids[ni]),
                "prompt": tokenizer.decode(prompt_ids[ni], skip_special_tokens=True),
                "native_text": nt,
                "spec_text": st,
                "native_n_tokens": len(nids),
                "spec_n_tokens": len(sids),
            }
            if scorer is not None:
                npass, ninfo = scorer(nt, ref)
                spass, sinfo = scorer(st, ref)
                row["native_pass"] = bool(npass)
                row["spec_pass"] = bool(spass)
                row["native_score_info"] = ninfo
                row["spec_score_info"] = sinfo
                n_pass_native += int(npass)
                n_pass_spec += int(spass)
            if args.dataset == "gsm8k":
                row["reference_answer"] = ref["answer"]
            elif args.dataset == "humaneval":
                row["entry_point"] = ref["entry_point"]
                row["canonical_solution"] = ref["canonical_solution"]
            elif args.dataset == "mbpp":
                row["test_list"] = ref.get("test_list", [])
                row["reference_code"] = ref.get("code", "")
            if stats:
                row["spec_acceptance"] = {
                    "total_accepted_tokens": int(stats.get("total_accepted_tokens", 0) or 0),
                    "total_draft_tokens": int(stats.get("total_draft_tokens", 0) or 0),
                    "total_bonus_tokens": int(stats.get("total_bonus_tokens", 0) or 0),
                    "per_block_accepted": stats.get("per_block_accepted", []) or [],
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "split": split,
        "num_samples": len(prompt_ids),
        "native_pass_at_1": n_pass_native / max(1, len(prompt_ids)),
        "spec_pass_at_1": n_pass_spec / max(1, len(prompt_ids)),
        "n_pass_native": n_pass_native,
        "n_pass_spec": n_pass_spec,
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "draft_device": args.draft_device,
            "target_device": args.target_device,
            "num_blocks": args.num_blocks,
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "K": args.K,
            "seed": args.seed,
            "no_eos_stop": args.no_eos_stop,
            "padded_len": padded_len,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] native pass@1 = {summary['native_pass_at_1']:.3f}  ({n_pass_native}/{len(prompt_ids)})")
    print(f"[done] spec   pass@1 = {summary['spec_pass_at_1']:.3f}  ({n_pass_spec}/{len(prompt_ids)})")
    print(f"[done] {jsonl_path}")
    print(f"[done] {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
