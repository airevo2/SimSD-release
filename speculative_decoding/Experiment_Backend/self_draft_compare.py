"""Self-draft correctness benchmark.

Runs three generation paths on the same prompts and compares them:
  1. native          target model only (single-model block diffusion baseline)
  2. self_draft      draft == target (same 8B), speculative path (α should ≈ 1.0)
  3. cross_draft     weaker draft (e.g. 4B / 1.7B) + target (8B), real spec path

Purpose:
  - **self_draft vs native**   correctness of verify/MRS implementation.
    When draft==target, q(x)==p(x) for every token, so MRS always accepts
    (α≈1.0) and the output distribution must equal native's. Any systematic
    divergence here is a bug in verify seq construction, causal mask, SDPA
    patch, position_ids, MRS ordering, step_map  not in the draft model.
  - **cross_draft vs self_draft**  pure effect of using a weaker draft.
    Any extra quality loss is attributable to draft/target distribution gap,
    *not* to the speculative machinery.
  - **cross_draft vs native**  end-to-end quality delta the user actually sees.

Metrics reported (per sample + aggregate):
  - α (acceptance rate) for each spec path
  - Longest common prefix (LCP) in token space vs native
  - Position-wise token agreement rate (for overlapping length)
  - Pass@1 on gsm8k / humaneval / mbpp (if dataset scorable)

Deterministic mode: with --branch greedy_match + fixed seed, native path uses
greedy argmax via draft.py (draft_one_block already samples top-confidence,
which is argmax-like). self_draft's greedy_match_verify should then reproduce
native exactly (modulo any residual non-determinism in attention kernels).

Output layout (results/{YYYY-MM-DD}/self_draft_{dataset}_{tag}/):
  - summary.json        aggregate pass@1, α, agreement stats
  - samples.jsonl       per-sample: prompt, 3× generated text + token ids, agreement lens
  - agreement.json      pairwise agreement matrices (N × 3)
  - run.log             stdout tee
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import transformers.utils as _tu_compat
if not hasattr(_tu_compat, "LossKwargs"):
    from typing import TypedDict as _TypedDict
    class LossKwargs(_TypedDict, total=False):
        pass
    _tu_compat.LossKwargs = LossKwargs
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding.bench.backends import (  # noqa: E402
    HFNativeBackend, HFSpeculativeBackend, compute_verify_padded_len,
)
from speculative_decoding.config import SpecConfig  # noqa: E402
from speculative_decoding.draft import patch_sdpa_eval_attention  # noqa: E402
from speculative_decoding.speculative_decode import load_prompts  # noqa: E402


# ─────────────────── scorers (copied from quality_compare.py) ───────────────────

def _extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    return m.group(1) if m else text


def _exec_isolated(code: str, timeout_s: int = 5) -> Tuple[bool, str]:
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


def score_gsm8k(gen_text: str, ref: dict) -> Tuple[bool, str]:
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


def score_humaneval(gen_text: str, ref: dict) -> Tuple[bool, str]:
    code = _extract_code(gen_text)
    full = code + "\n" + ref["test"] + f"\ncheck({ref['entry_point']})\n"
    return _exec_isolated(full)


def score_mbpp(gen_text: str, ref: dict) -> Tuple[bool, str]:
    code = _extract_code(gen_text)
    tests = "\n".join(ref.get("test_list", []))
    full = code + "\n" + tests + "\n"
    return _exec_isolated(full)


_IFEVAL_STRICT = None
_IFEVAL_INPUTEXAMPLE = None
def _load_ifeval():
    global _IFEVAL_STRICT, _IFEVAL_INPUTEXAMPLE
    if _IFEVAL_STRICT is None:
        import os as _os
        import sys as _sys
        _oc_path = _os.environ.get("OPENCOMPASS_PATH")
        if _oc_path:
            _sys.path.insert(0, _oc_path)
        from opencompass.datasets.IFEval.evaluation_main import (
            InputExample, test_instruction_following_strict)
        _IFEVAL_STRICT = test_instruction_following_strict
        _IFEVAL_INPUTEXAMPLE = InputExample
    return _IFEVAL_STRICT, _IFEVAL_INPUTEXAMPLE

def score_ifeval(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """IFEval prompt-level strict accuracy: every instruction in the prompt
    must be followed."""
    strict, InputExample = _load_ifeval()
    kwargs = list(ref.get("kwargs", []))
    for kw in kwargs:
        for k in list(kw.keys()):
            if kw[k] is None:
                kw.pop(k, None)
    inp = InputExample(
        key=ref.get("key", 0),
        instruction_id_list=ref["instruction_id_list"],
        prompt=ref["prompt"],
        kwargs=kwargs,
    )
    out = strict(inp, gen_text)
    flags = list(out.follow_instruction_list)
    ok = bool(flags) and all(flags)
    return ok, f"strict={flags}"


def _normalize_qa(s: str) -> str:
    """Standard SQuAD/TriviaQA normalization: lowercase, strip articles +
    punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())

def score_triviaqa(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """Lenient TriviaQA EM: any of the gold aliases (normalized) is a substring
    of the normalized gen_text. Uses official aliases from rc.nocontext split."""
    answer = ref.get("answer", {}) or {}
    aliases = answer.get("aliases") or answer.get("normalized_aliases") or []
    if not aliases and answer.get("value"):
        aliases = [answer["value"]]
    if not aliases:
        return False, "no_ref"
    norm_gen = _normalize_qa(gen_text)
    for a in aliases:
        norm_a = _normalize_qa(a)
        if norm_a and norm_a in norm_gen:
            return True, f"hit={a!r}"
    return False, f"miss aliases={aliases[:3]!r}..."


def score_mmlu(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """MMLU EM: extract the first standalone A/B/C/D letter from the
    generation and compare to the gold index.

    The scorer is permissive about format  it walks the generation looking
    for the first match of patterns like "A.", "(A)", " A ", "A\n", or a
    final "Answer: A". This handles both well-behaved single-letter answers
    and chain-of-thought outputs.
    """
    raw = ref.get("answer")
    if raw is None:
        return False, "no_ref"
    try:
        gold_idx = int(raw)
    except (TypeError, ValueError):
        return False, f"bad_ref={raw!r}"
    if gold_idx not in (0, 1, 2, 3):
        return False, f"bad_idx={gold_idx}"
    gold_letter = "ABCD"[gold_idx]
    # Prefer a final "Answer:" pattern when present
    m = re.search(r"[Aa]nswer\s*[:\-]?\s*\(?\s*([ABCD])\s*\)?", gen_text)
    if m:
        pred = m.group(1)
        return pred == gold_letter, f"answer_pat pred={pred} gold={gold_letter}"
    # Fall back to the first standalone letter token
    m = re.search(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)", gen_text)
    if m:
        pred = m.group(1)
        return pred == gold_letter, f"first_letter pred={pred} gold={gold_letter}"
    return False, f"no_letter_found gold={gold_letter}"


SCORERS = {"gsm8k": score_gsm8k, "humaneval": score_humaneval,
           "mbpp": score_mbpp, "ifeval": score_ifeval,
           "triviaqa": score_triviaqa, "mmlu": score_mmlu}


# ─────────────────── agreement metrics ───────────────────

def longest_common_prefix(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def positional_agreement(a: List[int], b: List[int]) -> Tuple[int, int]:
    """Return (n_match, n_overlap) where overlap = min(len(a), len(b))."""
    n = min(len(a), len(b))
    match = sum(1 for i in range(n) if a[i] == b[i])
    return match, n


# ─────────────────── model load ───────────────────

def _load_model(path: str, device: torch.device, dtype: torch.dtype):
    return (
        AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype,
        )
        .to(device)
        .eval()
    )


# ─────────────────── generate helpers ───────────────────

def _run_native(target, prompt_ids: List[int], args, eos_id: Optional[int]) -> Tuple[List[int], Dict[str, Any]]:
    be = HFNativeBackend(
        model=target,
        num_blocks=args.num_blocks,
        block_length=args.block_length,
        denoising_steps=args.denoising_steps,
        mask_token_id=151669,
        eos_token_id=eos_id,
        use_cuda_graph=getattr(args, "use_cuda_graph", False),
    )
    res = be.generate(prompt_ids, return_timings=False)
    return list(res.generated_ids), {"end_to_end_s": res.end_to_end_s}


def _run_spec(
    draft_model,
    target_model,
    prompt_ids: List[int],
    cfg: SpecConfig,
    padded_len: int,
    pad_token_id: int,
    eos_id: Optional[int],
) -> Tuple[List[int], Dict[str, Any]]:
    be = HFSpeculativeBackend(
        draft_model=draft_model,
        target_model=target_model,
        cfg=cfg,
        pad_token_id=pad_token_id,
        padded_len=padded_len,
        eos_token_id=eos_id,
    )
    res = be.generate(prompt_ids, return_timings=True)
    stats = dict(res.stats or {})
    stats["end_to_end_s"] = res.end_to_end_s
    if res.timing is not None:
        # Surface per-stage breakdown so the row + summary can aggregate.
        # In pipelined mode draft_s is reported as 0 (overlapped); use
        # block_wall_s for serial-equivalent comparison (see speculative_decode.py:1340-1348).
        stats["total_draft_s"] = float(res.timing.get("total_draft_s", 0.0))
        stats["total_target_verify_s"] = float(res.timing.get("total_target_verify_s", 0.0))
        stats["total_mrs_and_commit_s"] = float(res.timing.get("total_mrs_and_commit_s", 0.0))
        stats["total_block_wall_s"] = float(res.timing.get("total_block_wall_s", 0.0))
        stats["pipeline"] = bool(res.timing.get("pipeline", False))
        # cuda.Event-based pure GPU stream times (added 2026-05-05 plan/14 §6
        # option B). These are NOT bracketed by per-block _cuda_sync, so they
        # don't inflate end_to_end_s. Sum-of-stages > sample_wall when stages
        # overlap on different GPUs (pipeline)  see plan/14 for attribution.
        stats["total_verify_gpu_ms"] = float(res.timing.get("total_verify_gpu_ms", 0.0))
        stats["total_extend_gpu_ms"] = float(res.timing.get("total_extend_gpu_ms", 0.0))
        stats["total_denoise_gpu_ms"] = float(res.timing.get("total_denoise_gpu_ms", 0.0))
        stats["total_target_cpu_wait_ms"] = float(res.timing.get("total_target_cpu_wait_ms", 0.0))
    return list(res.generated_ids), stats


def _build_spec_cfg(
    draft_path: str,
    target_path: str,
    draft_dev: str,
    target_dev: str,
    args,
    split: str,
) -> SpecConfig:
    return SpecConfig(
        draft_model=draft_path,
        target_model=target_path,
        draft_device=draft_dev,
        target_device=target_dev,
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
        use_cuda_graph=getattr(args, "use_cuda_graph", False),
        target_eval_sdpa=True,
        pipeline=getattr(args, "pipeline", False),
        use_kv_cache=getattr(args, "use_kv_cache", False),
        speculative_branch=args.branch,
        draft_sampling=getattr(args, "draft_sampling", "argmax"),
        partial_block_fill=getattr(args, "partial_block_fill", "draft_argmax"),
        remasking_strategy=getattr(args, "remasking_strategy", "low_confidence_static"),
        confidence_threshold=getattr(args, "confidence_threshold", 0.9),
        kv_cache_max_len=getattr(args, "kv_cache_max_len", 1024),
    )


def _alpha(stats: Dict[str, Any]) -> float:
    a = int(stats.get("total_accepted_tokens", 0) or 0)
    d = int(stats.get("total_draft_tokens", 0) or 0)
    return a / d if d > 0 else 0.0


# ─────────────────── main ───────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model",
                   default="inference/model/SDAR-8B-Chat")
    p.add_argument("--cross_draft_model",
                   default="inference/model/SDAR-4B-Chat",
                   help="weaker draft for 'cross' path. Set to '' to skip cross path.")
    p.add_argument("--target_device", default="cuda:0")
    p.add_argument("--cross_draft_device", default="cuda:0")
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "humaneval", "mbpp", "ifeval", "triviaqa", "mmlu"])
    p.add_argument("--dataset_split", default=None)
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--num_blocks", type=int, default=32)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--branch", default="greedy_match",
                   choices=["mrs", "greedy_match"],
                   help="greedy_match is the deterministic correctness check; "
                        "mrs is the stochastic production path.")
    p.add_argument("--no_eos_stop", action="store_true", default=False)
    p.add_argument("--kv_cache_max_len", type=int, default=1024,
                   help="StaticBlockCache.max_cache_len. Smaller = less "
                        "redundant K/V scan, but caps prompt+gen length.")
    p.add_argument("--tag", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--skip_cross", action="store_true",
                   help="only run native + self_draft (faster, tests correctness only).")
    p.add_argument("--skip_native", action="store_true",
                   help="skip the native pass (use when native baseline is "
                        "already collected and you only need self_draft / cross).")
    p.add_argument("--skip_self", action="store_true",
                   help="skip the self_draft pass (use for stage A/B/C cross "
                        "bench; self_draft is a per-stage regression check, "
                        "not a per-(model,dataset) need).")
    p.add_argument("--draft_sampling", default="argmax",
                   choices=["argmax", "multinomial"],
                   help="Draft per-position pick rule. 'argmax' is the default "
                        "and matches existing native / self_draft behavior. "
                        "'multinomial' samples from softmax(logits)  required "
                        "for MRS's output-distribution-equals-target invariant "
                        "to hold under cross-draft (see plan/07 §2.A).")
    p.add_argument("--use_cuda_graph", action="store_true", default=False,
                   help="Enable CUDA graph capture/replay for both spec and "
                        "native backends  gives ~4× forward speedup by skipping "
                        "kernel launch overhead. Required for fair speedup comparison.")
    p.add_argument("--pipeline", action="store_true", default=False,
                   help="Enable dual-GPU pipelined spec (draft_{N+1} overlaps "
                        "verify_N on the other GPU). Requires draft_device != "
                        "target_device. Theoretical 1.5-2× extra speedup on top "
                        "of K=1 spec. Compatible with all partial_block_fill modes.")
    p.add_argument("--use_kv_cache", action="store_true", default=False,
                   help="Enable prefix-only KV cache for spec (à la "
                        "generate.py:block_diffusion_generate). Scaffolding only "
                        "for now  speculative_generate raises NotImplementedError. "
                        "Real impl in plan/10 §11.7 (~1 week eng).")
    p.add_argument("--partial_block_fill", default="target_argmax",
                   choices=["draft_argmax", "target_argmax", "redraft",
                            "truncate", "target_argmax_all",
                            "truncate_no_bonus"],
                   help="On MRS reject, how to handle the rejected tail of a "
                        "block before extending committed context. "
                        "'target_argmax' (DEFAULT post 2026-05-06): target verify's "
                        "argmax for the tail; bonus token at reject_pos extended "
                        "to [reject_pos+1, bl). +5-50% TPS over draft_argmax "
                        "(plan/18 §8). 'draft_argmax' (legacy): use draft's stale "
                        "predictions that target rejected (kept for reproducibility "
                        "of pre-2026-05-06 results); 'redraft' (Stage D, plan/08) "
                        "re-runs draft + verify on the post-reject prefix; "
                        "'truncate' (Stage E, plan/08 §5.3) commits a variable-"
                        "length partial block with no pad; 'target_argmax_all' "
                        "(Stage F) replaces the WHOLE block with target argmax; "
                        "'truncate_no_bonus' (Stage G) like truncate but also "
                        "drops the MRS bonus token before commit (1.7B8B "
                        "noise mitigation).")
    p.add_argument("--remasking_strategy", default="low_confidence_static",
                   choices=["low_confidence_static", "low_confidence_dynamic"],
                   help="Draft denoising remask picker. 'static' = topk by "
                        "confidence with deterministic schedule (0 host syncs "
                        "per step). 'dynamic' = unmask all positions above "
                        "confidence_threshold when count >= n_unmask, else "
                        "fall back to topk (1 host sync per step for the "
                        "high-conf count compare). Both work under "
                        "use_cuda_graph because picking is in eager.")
    p.add_argument("--confidence_threshold", type=float, default=0.9,
                   help="Threshold for low_confidence_dynamic. Ignored when "
                        "remasking_strategy=low_confidence_static.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16
    target_dev = torch.device(args.target_device)

    date_tag = datetime.now().strftime("%Y-%m-%d")
    target_name = Path(args.target_model).name
    cross_name = Path(args.cross_draft_model).name if args.cross_draft_model else "skip"
    tag = args.tag or f"{target_name}_vs_{cross_name}_{args.branch}"
    out_dir = Path(args.out_dir or (
        REPO / "speculative_decoding" / "results" / date_tag
        / f"self_draft_{args.dataset}_{tag}"
    ))
    out_dir.mkdir(parents=True, exist_ok=True)

    # tee stdout to run.log
    log_f = open(out_dir / "run.log", "w")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)
                st.flush()

        def flush(self):
            for st in self.streams:
                st.flush()

    sys.stdout = Tee(sys.__stdout__, log_f)
    print(f"[out] {out_dir}")
    print(f"[args] {vars(args)}")

    # ── Load tokenizer (any SDAR ckpt shares the same Qwen2 vocab)
    print(f"[load] tokenizer   {args.target_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)

    # ── Load target. Target doubles as self-draft (same object).
    print(f"[load] target  {args.target_model}  {args.target_device}")
    target = _load_model(args.target_model, target_dev, dtype)

    cross_draft = None
    cross_dev = None
    run_cross = bool(args.cross_draft_model) and not args.skip_cross
    if run_cross:
        cross_dev = torch.device(args.cross_draft_device)
        print(f"[load] cross_draft  {args.cross_draft_model}  {args.cross_draft_device}")
        cross_draft = _load_model(args.cross_draft_model, cross_dev, dtype)

    # The public JetLM/SDAR-*-Chat configs do not ship `block_size`, but
    # speculative_decoding/verify.py expects model.config.block_size to exist
    # so it can save/restore it around the multi-block forward. Seed it with
    # denoising_steps (the SpecConfig convention; see config.py:166).
    if not hasattr(target.config, "block_size"):
        target.config.block_size = args.denoising_steps
    if cross_draft is not None and not hasattr(cross_draft.config, "block_size"):
        cross_draft.config.block_size = args.denoising_steps

    # ── Load prompts
    split = args.dataset_split or "test"
    tmp_cfg = SimpleNamespace(
        dataset=args.dataset, dataset_split=split, num_samples=args.num_samples,
    )
    print(f"[data] {args.dataset}/{split}  num_samples={args.num_samples}")
    ds, prompt_ids = load_prompts(tokenizer, tmp_cfg)
    print(f"[data] loaded {len(prompt_ids)} prompts; "
          f"len min={min(len(p) for p in prompt_ids)} "
          f"max={max(len(p) for p in prompt_ids)}")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id

    # ── Apply SDPA patch BEFORE native so all three paths share the same
    #    attention implementation. The patched eval forward is numerically
    #    equivalent to the original (the original dispatches to SDPA anyway
    #    when the block-causal mask isn't all-ones, which is every iter here),
    #    but applying it uniformly eliminates the patch itself as a confound
    #    in the self_draft-vs-native correctness comparison.
    # When use_kv_cache=True, skip the SDPA patch  the patched eval forward
    # accepts but DOES NOT pass past_key_value through (draft.py:42), so the
    # cache would be silently dropped. The model's default attention forward
    # (modeling_sdar.py SDARAttention) handles past_key_value correctly.
    if getattr(args, "use_kv_cache", False):
        print("[patch] SDPA patch SKIPPED (use_kv_cache=True; default attn keeps cache)")
    else:
        patch_sdpa_eval_attention(target)
        if cross_draft is not None:
            patch_sdpa_eval_attention(cross_draft)
        print("[patch] applied patch_sdpa_eval_attention (target + cross_draft)")

    run_native_path = not args.skip_native
    run_self_path = not args.skip_self

    # ───── 1) NATIVE (target only) ─────
    native_out: List[Tuple[int, List[int], str, Dict[str, Any]]] = []
    if run_native_path:
        print("\n[1/3 native] generating ...")
        for i, pids in enumerate(prompt_ids):
            t0 = time.perf_counter()
            gen_ids, stats = _run_native(target, pids, args, eos_id)
            dt = time.perf_counter() - t0
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            native_out.append((i, gen_ids, txt, stats))
            print(f"  [native {i+1:>3}/{len(prompt_ids)}] {len(gen_ids):>3}t  {dt:.2f}s  "
                  f"{txt.replace(chr(10), '⏎')[:80]!r}")
    else:
        print("\n[1/3 native] skipped (--skip_native)")

    max_prompt_len = max(len(p) for p in prompt_ids)
    padded_len = compute_verify_padded_len(
        max_prompt_len, args.num_blocks, args.block_length, K=args.K,
    )
    print(f"[spec] padded_len={padded_len}  K={args.K}  num_blocks={args.num_blocks}")

    # ───── 2) SELF-DRAFT SPEC (draft == target) ─────
    self_out: List[Tuple[int, List[int], str, Dict[str, Any]]] = []
    if run_self_path:
        self_cfg = _build_spec_cfg(
            args.target_model, args.target_model,
            args.target_device, args.target_device, args, split,
        )
        print("\n[2/3 self_draft] generating (draft == target)...")
        for i, pids in enumerate(prompt_ids):
            t0 = time.perf_counter()
            gen_ids, stats = _run_spec(target, target, pids, self_cfg, padded_len,
                                       pad_token_id, eos_id)
            dt = time.perf_counter() - t0
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            self_out.append((i, gen_ids, txt, stats))
            a = _alpha(stats)
            print(f"  [self    {i+1:>3}/{len(prompt_ids)}] {len(gen_ids):>3}t  "
                  f"α={a:.3f}  {dt:.2f}s  "
                  f"{txt.replace(chr(10), '⏎')[:80]!r}")
    else:
        print("\n[2/3 self_draft] skipped (--skip_self)")

    # ───── 3) CROSS SPEC (weaker draft  target) ─────
    cross_out: List[Tuple[int, List[int], str, Dict[str, Any]]] = []
    if run_cross:
        cross_cfg = _build_spec_cfg(
            args.cross_draft_model, args.target_model,
            args.cross_draft_device, args.target_device, args, split,
        )
        print(f"\n[3/3 cross] generating ({cross_name}  {target_name})...")
        for i, pids in enumerate(prompt_ids):
            t0 = time.perf_counter()
            gen_ids, stats = _run_spec(cross_draft, target, pids, cross_cfg,
                                       padded_len, pad_token_id, eos_id)
            dt = time.perf_counter() - t0
            txt = tokenizer.decode(gen_ids, skip_special_tokens=True)
            cross_out.append((i, gen_ids, txt, stats))
            a = _alpha(stats)
            print(f"  [cross   {i+1:>3}/{len(prompt_ids)}] {len(gen_ids):>3}t  "
                  f"α={a:.3f}  {dt:.2f}s  "
                  f"{txt.replace(chr(10), '⏎')[:80]!r}")
    else:
        print("\n[3/3 cross] skipped")

    # ───── Score + aggregate ─────
    scorer = SCORERS.get(args.dataset)
    n_pass = {"native": 0, "self": 0, "cross": 0}
    alpha_sum = {"self": 0.0, "cross": 0.0}
    alpha_n = {"self": 0, "cross": 0}

    # Agreement stats: (lcp, n_match, n_overlap) vs native
    agree_self: List[Tuple[int, int, int]] = []
    agree_cross: List[Tuple[int, int, int]] = []

    # Latency aggregates  accumulated from per-sample end_to_end_s
    time_sum = {"native": 0.0, "self": 0.0, "cross": 0.0}
    tok_sum = {"native": 0, "self": 0, "cross": 0}
    # Spec breakdown aggregates (only populated when stats has timing)
    breakdown_sum = {
        k: {"draft_s": 0.0, "target_verify_s": 0.0, "mrs_and_commit_s": 0.0, "block_wall_s": 0.0, "n_samples": 0}
        for k in ("self", "cross")
    }

    jsonl_path = out_dir / "samples.jsonl"
    with open(jsonl_path, "w") as f:
        for idx in range(len(prompt_ids)):
            ni = idx
            nids: List[int] = []
            nt = ""
            nstats: Dict[str, Any] = {}
            sids: List[int] = []
            st = ""
            sstats: Dict[str, Any] = {}
            if run_native_path:
                ni, nids, nt, nstats = native_out[idx]
            if run_self_path:
                si, sids, st, sstats = self_out[idx]
                if run_native_path:
                    assert ni == si
                else:
                    ni = si

            row: Dict[str, Any] = {
                "idx": ni,
                "prompt_len": len(prompt_ids[ni]),
                "prompt": tokenizer.decode(prompt_ids[ni], skip_special_tokens=True),
            }
            if run_native_path:
                row["native_text"] = nt
                row["native_token_ids"] = nids
                row["native_n_tokens"] = len(nids)
                row["native_end_to_end_s"] = float(nstats.get("end_to_end_s", 0.0) or 0.0)
                time_sum["native"] += row["native_end_to_end_s"]
                tok_sum["native"] += len(nids)
            if run_self_path:
                row["self_text"] = st
                row["self_token_ids"] = sids
                row["self_n_tokens"] = len(sids)
                row["self_end_to_end_s"] = float(sstats.get("end_to_end_s", 0.0) or 0.0)
                time_sum["self"] += row["self_end_to_end_s"]
                tok_sum["self"] += len(sids)
                if "total_block_wall_s" in sstats:
                    row["self_breakdown_s"] = {
                        "draft_s": float(sstats.get("total_draft_s", 0.0) or 0.0),
                        "target_verify_s": float(sstats.get("total_target_verify_s", 0.0) or 0.0),
                        "mrs_and_commit_s": float(sstats.get("total_mrs_and_commit_s", 0.0) or 0.0),
                        "block_wall_s": float(sstats.get("total_block_wall_s", 0.0) or 0.0),
                        "pipeline": bool(sstats.get("pipeline", False)),
                    }
                    breakdown_sum["self"]["draft_s"] += row["self_breakdown_s"]["draft_s"]
                    breakdown_sum["self"]["target_verify_s"] += row["self_breakdown_s"]["target_verify_s"]
                    breakdown_sum["self"]["mrs_and_commit_s"] += row["self_breakdown_s"]["mrs_and_commit_s"]
                    breakdown_sum["self"]["block_wall_s"] += row["self_breakdown_s"]["block_wall_s"]
                    breakdown_sum["self"]["n_samples"] += 1
                row["self_acceptance"] = {
                    "alpha": _alpha(sstats),
                    "total_accepted_tokens": int(sstats.get("total_accepted_tokens", 0) or 0),
                    "total_draft_tokens": int(sstats.get("total_draft_tokens", 0) or 0),
                    "total_bonus_tokens": int(sstats.get("total_bonus_tokens", 0) or 0),
                }
                alpha_sum["self"] += _alpha(sstats)
                alpha_n["self"] += 1
                if run_native_path:
                    lcp_self = longest_common_prefix(nids, sids)
                    m_self, o_self = positional_agreement(nids, sids)
                    agree_self.append((lcp_self, m_self, o_self))
                    row["self_vs_native_lcp"] = lcp_self
                    row["self_vs_native_match"] = m_self
                    row["self_vs_native_overlap"] = o_self

            if run_cross:
                ci, cids, ct, cstats = cross_out[idx]
                assert ci == ni
                row["cross_text"] = ct
                row["cross_token_ids"] = cids
                row["cross_n_tokens"] = len(cids)
                row["cross_end_to_end_s"] = float(cstats.get("end_to_end_s", 0.0) or 0.0)
                time_sum["cross"] += row["cross_end_to_end_s"]
                tok_sum["cross"] += len(cids)
                if "total_block_wall_s" in cstats:
                    row["cross_breakdown_s"] = {
                        "draft_s": float(cstats.get("total_draft_s", 0.0) or 0.0),
                        "target_verify_s": float(cstats.get("total_target_verify_s", 0.0) or 0.0),
                        "mrs_and_commit_s": float(cstats.get("total_mrs_and_commit_s", 0.0) or 0.0),
                        "block_wall_s": float(cstats.get("total_block_wall_s", 0.0) or 0.0),
                        "pipeline": bool(cstats.get("pipeline", False)),
                        # Pure GPU stream times (cuda.Event, no sync inflation).
                        "verify_gpu_ms": float(cstats.get("total_verify_gpu_ms", 0.0) or 0.0),
                        "extend_gpu_ms": float(cstats.get("total_extend_gpu_ms", 0.0) or 0.0),
                        "denoise_gpu_ms": float(cstats.get("total_denoise_gpu_ms", 0.0) or 0.0),
                        "target_cpu_wait_ms": float(cstats.get("total_target_cpu_wait_ms", 0.0) or 0.0),
                    }
                    breakdown_sum["cross"]["draft_s"] += row["cross_breakdown_s"]["draft_s"]
                    breakdown_sum["cross"]["target_verify_s"] += row["cross_breakdown_s"]["target_verify_s"]
                    breakdown_sum["cross"]["mrs_and_commit_s"] += row["cross_breakdown_s"]["mrs_and_commit_s"]
                    breakdown_sum["cross"]["block_wall_s"] += row["cross_breakdown_s"]["block_wall_s"]
                    breakdown_sum["cross"]["n_samples"] += 1
                row["cross_acceptance"] = {
                    "alpha": _alpha(cstats),
                    "total_accepted_tokens": int(cstats.get("total_accepted_tokens", 0) or 0),
                    "total_draft_tokens": int(cstats.get("total_draft_tokens", 0) or 0),
                    "total_bonus_tokens": int(cstats.get("total_bonus_tokens", 0) or 0),
                }
                alpha_sum["cross"] += _alpha(cstats)
                alpha_n["cross"] += 1
                if run_native_path:
                    lcp_cross = longest_common_prefix(nids, cids)
                    m_cross, o_cross = positional_agreement(nids, cids)
                    agree_cross.append((lcp_cross, m_cross, o_cross))
                    row["cross_vs_native_lcp"] = lcp_cross
                    row["cross_vs_native_match"] = m_cross
                    row["cross_vs_native_overlap"] = o_cross
                if run_self_path:
                    lcp_sc = longest_common_prefix(sids, cids)
                    m_sc, o_sc = positional_agreement(sids, cids)
                    row["cross_vs_self_lcp"] = lcp_sc
                    row["cross_vs_self_match"] = m_sc
                    row["cross_vs_self_overlap"] = o_sc

            # Pass@1
            if scorer is not None:
                ref = ds[ni]
                if run_native_path:
                    np_ok, np_info = scorer(nt, ref)
                    row["native_pass"] = bool(np_ok)
                    row["native_score_info"] = np_info
                    n_pass["native"] += int(np_ok)
                if run_self_path:
                    sp_ok, sp_info = scorer(st, ref)
                    row["self_pass"] = bool(sp_ok)
                    row["self_score_info"] = sp_info
                    n_pass["self"] += int(sp_ok)
                if run_cross:
                    cp_ok, cp_info = scorer(cross_out[idx][2], ref)
                    row["cross_pass"] = bool(cp_ok)
                    row["cross_score_info"] = cp_info
                    n_pass["cross"] += int(cp_ok)

            if args.dataset == "gsm8k":
                row["reference_answer"] = ds[ni]["answer"]
            elif args.dataset == "humaneval":
                row["entry_point"] = ds[ni]["entry_point"]
            elif args.dataset == "mbpp":
                row["test_list"] = ds[ni].get("test_list", [])
            elif args.dataset == "ifeval":
                row["instruction_id_list"] = ds[ni].get("instruction_id_list", [])
            elif args.dataset == "triviaqa":
                row["triviaqa_answer"] = ds[ni].get("answer", {})

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ───── Aggregate ─────
    n = len(prompt_ids)

    def _mean_frac(pairs: List[Tuple[int, int, int]], which: str) -> float:
        """which in {'lcp_rate', 'match_rate'}. lcp_rate = lcp / overlap, etc."""
        vals = []
        for lcp, m, o in pairs:
            if o == 0:
                continue
            if which == "lcp_rate":
                vals.append(lcp / o)
            elif which == "match_rate":
                vals.append(m / o)
        return sum(vals) / len(vals) if vals else 0.0

    def _exact_count(pairs: List[Tuple[int, int, int]]) -> int:
        """Count samples where outputs are identical (lcp == both lengths)."""
        c = 0
        for lcp, _m, o in pairs:
            if lcp == o and o > 0:
                c += 1
        return c

    summary: Dict[str, Any] = {
        "dataset": args.dataset,
        "split": split,
        "num_samples": n,
        "branch": args.branch,
        "seed": args.seed,
        "config": {
            "target_model": args.target_model,
            "cross_draft_model": args.cross_draft_model if run_cross else None,
            "num_blocks": args.num_blocks,
            "block_length": args.block_length,
            "denoising_steps": args.denoising_steps,
            "K": args.K,
            "no_eos_stop": args.no_eos_stop,
            "padded_len": padded_len,
        },
        "acceptance_rate": {
            "self_draft_mean_alpha": (alpha_sum["self"] / alpha_n["self"]) if alpha_n["self"] else None,
            "cross_mean_alpha": (alpha_sum["cross"] / alpha_n["cross"]) if alpha_n["cross"] else None,
        },
        "agreement_self_vs_native": {
            "mean_lcp_rate": _mean_frac(agree_self, "lcp_rate"),
            "mean_positional_match_rate": _mean_frac(agree_self, "match_rate"),
            "n_exact_match": _exact_count(agree_self),
            "n_exact_match_frac": _exact_count(agree_self) / n if n else 0.0,
        },
    }
    if run_cross:
        summary["agreement_cross_vs_native"] = {
            "mean_lcp_rate": _mean_frac(agree_cross, "lcp_rate"),
            "mean_positional_match_rate": _mean_frac(agree_cross, "match_rate"),
            "n_exact_match": _exact_count(agree_cross),
            "n_exact_match_frac": _exact_count(agree_cross) / n if n else 0.0,
        }

    def _latency_block(key: str) -> Optional[Dict[str, float]]:
        if tok_sum[key] <= 0 or time_sum[key] <= 0:
            return None
        return {
            "total_s": time_sum[key],
            "total_tokens": tok_sum[key],
            "ms_per_token": time_sum[key] * 1000.0 / tok_sum[key],
            "tokens_per_second": tok_sum[key] / time_sum[key],
        }

    summary["latency"] = {
        "native": _latency_block("native"),
        "self_draft": _latency_block("self"),
        "cross": _latency_block("cross"),
    }

    def _breakdown_block(key: str) -> Optional[Dict[str, Any]]:
        b = breakdown_sum[key]
        if b["n_samples"] <= 0 or tok_sum[key] <= 0:
            return None
        # Per-token ms (totals divided by all output tokens for this path).
        total_tok = tok_sum[key]
        return {
            "n_samples": b["n_samples"],
            "total_block_wall_s": b["block_wall_s"],
            "total_draft_s": b["draft_s"],
            "total_target_verify_s": b["target_verify_s"],
            "total_mrs_and_commit_s": b["mrs_and_commit_s"],
            "ms_per_token": {
                "block_wall": b["block_wall_s"] * 1000.0 / total_tok,
                "draft": b["draft_s"] * 1000.0 / total_tok,
                "target_verify": b["target_verify_s"] * 1000.0 / total_tok,
                "mrs_and_commit": b["mrs_and_commit_s"] * 1000.0 / total_tok,
            },
            "fraction_of_block_wall": (
                {
                    "draft": b["draft_s"] / b["block_wall_s"] if b["block_wall_s"] else 0.0,
                    "target_verify": b["target_verify_s"] / b["block_wall_s"] if b["block_wall_s"] else 0.0,
                    "mrs_and_commit": b["mrs_and_commit_s"] / b["block_wall_s"] if b["block_wall_s"] else 0.0,
                }
            ),
        }

    summary["latency_breakdown"] = {
        "self_draft": _breakdown_block("self"),
        "cross": _breakdown_block("cross"),
    }
    lat_native = summary["latency"]["native"]
    lat_cross = summary["latency"]["cross"]
    if lat_native and lat_cross and lat_cross["ms_per_token"] > 0:
        summary["speedup_native_vs_cross"] = lat_native["ms_per_token"] / lat_cross["ms_per_token"]

    if scorer is not None:
        summary["pass_at_1"] = {
            "native": (n_pass["native"] / n) if run_native_path else None,
            "self_draft": (n_pass["self"] / n) if run_self_path else None,
            "cross": (n_pass["cross"] / n) if run_cross else None,
            "n_pass_native": n_pass["native"] if run_native_path else None,
            "n_pass_self": n_pass["self"] if run_self_path else None,
            "n_pass_cross": n_pass["cross"] if run_cross else None,
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # agreement.json (raw per-sample triples for downstream plotting)
    with open(out_dir / "agreement.json", "w") as f:
        json.dump(
            {
                "self_vs_native": [
                    {"idx": i, "lcp": lcp, "match": m, "overlap": o}
                    for i, (lcp, m, o) in enumerate(agree_self)
                ],
                "cross_vs_native": [
                    {"idx": i, "lcp": lcp, "match": m, "overlap": o}
                    for i, (lcp, m, o) in enumerate(agree_cross)
                ] if run_cross else [],
            },
            f, indent=2,
        )

    # ───── Pretty print ─────
    print("\n" + "=" * 70)
    print(f"SUMMARY  {args.dataset} / {args.branch}  (N={n})")
    print("=" * 70)
    a_self = summary["acceptance_rate"]["self_draft_mean_alpha"]
    a_cross = summary["acceptance_rate"]["cross_mean_alpha"]
    print(f"  α (self_draft, draft=target): {a_self:.4f}" if a_self is not None else "  α (self_draft): n/a")
    if a_cross is not None:
        print(f"  α (cross,      weakertarget): {a_cross:.4f}")

    if run_native_path and run_self_path:
        ag_s = summary["agreement_self_vs_native"]
        print(f"\n  self_draft vs native:")
        print(f"    mean LCP rate          = {ag_s['mean_lcp_rate']:.4f}  "
              f"(1.0 = every sample shares full prefix)")
        print(f"    mean positional match  = {ag_s['mean_positional_match_rate']:.4f}")
        print(f"    # exact-match samples  = {ag_s['n_exact_match']}/{n}  "
              f"({ag_s['n_exact_match_frac']:.2%})")

    if run_cross and run_native_path:
        ag_c = summary["agreement_cross_vs_native"]
        print(f"\n  cross vs native:")
        print(f"    mean LCP rate          = {ag_c['mean_lcp_rate']:.4f}")
        print(f"    mean positional match  = {ag_c['mean_positional_match_rate']:.4f}")
        print(f"    # exact-match samples  = {ag_c['n_exact_match']}/{n}  "
              f"({ag_c['n_exact_match_frac']:.2%})")

    if scorer is not None:
        pa = summary["pass_at_1"]
        print(f"\n  pass@1:")
        if pa.get("native") is not None:
            print(f"    native     = {pa['native']:.3f}  ({pa['n_pass_native']}/{n})")
        if pa.get("self_draft") is not None:
            print(f"    self_draft = {pa['self_draft']:.3f}  ({pa['n_pass_self']}/{n})")
        if run_cross:
            print(f"    cross      = {pa['cross']:.3f}  ({pa['n_pass_cross']}/{n})")

    print("\n  Interpretation:")
    print("    - self_draft should match native exactly under greedy_match branch.")
    print("      If n_exact_match / n < 1.0  verify/MRS implementation bug.")
    print("      Any α < 1.0 in self_draft is the same bug signal (q==p always accepts).")
    print("    - cross vs native gap above self_draft's gap = draft-model quality cost.")
    print("=" * 70)
    print(f"\n[done] {out_dir / 'summary.json'}")
    print(f"[done] {jsonl_path}")


if __name__ == "__main__":
    main()
