"""TP=2 native 8B benchmark on gsm8k / humaneval / mbpp (N=40, seed=42).

Run with:
  CUDA_VISIBLE_DEVICES=<a>,<b> torchrun --nproc_per_node=2 --master_port=29520 \
      run_native_tp2.py --output_dir <path>

Each rank loads SDAR-8B-Chat with tp_plan='auto' (DTensor-sharded). Rank 0 drives
the decoding loop and broadcasts the next input_ids to other ranks each forward,
so every rank executes the same model forward in lock-step (required for TP
collectives). Only rank 0 does sampling + writes outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from transformers import AutoModelForCausalLM, AutoTokenizer
from speculative_decoding.speculative_decode import load_prompts
from speculative_decoding.Experiment_Backend.self_draft_compare import SCORERS
from speculative_decoding.draft import patch_sdpa_eval_attention


def patch_stock_rms_norm(model):
    cls_to_patch = None
    for m in model.modules():
        if m.__class__.__name__ == "SDARRMSNorm":
            cls_to_patch = m.__class__
            break
    if cls_to_patch is None:
        return 0

    def stock_forward(self, x):
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.variance_epsilon)
        return self.weight * x32.to(in_dtype)

    cls_to_patch.forward = stock_forward
    return sum(1 for m in model.modules() if isinstance(m, cls_to_patch))


def broadcast_ids(ids: torch.Tensor, device, src=0):
    """Broadcast variable-length int64 ids tensor from rank 0 to all ranks.

    First broadcast the length, then the buffer.
    """
    rank = dist.get_rank()
    shape_t = torch.zeros(2, dtype=torch.long, device=device)
    if rank == src and ids is not None:
        shape_t[0] = ids.shape[0]
        shape_t[1] = ids.shape[1]
    dist.broadcast(shape_t, src=src)
    B, L = shape_t[0].item(), shape_t[1].item()
    if rank == src:
        out = ids.to(device)
    else:
        out = torch.zeros((B, L), dtype=torch.long, device=device)
    dist.broadcast(out, src=src)
    return out


def block_diffusion_native(model, prompt_ids, num_blocks, block_length, denoising_steps,
                           mask_token_id, device, eos_id, no_eos_stop, rank, is_main):
    """All ranks call collectively. Rank 0 drives + samples; others get ids broadcast."""
    print(f"[rank {rank}] block_diffusion_native enter", flush=True)
    if is_main:
        ctx = list(prompt_ids)
        generated = []
    for blk in range(num_blocks):
        if is_main:
            block = [mask_token_id] * block_length
            stop_signal = torch.tensor([0], dtype=torch.long, device=device)
        else:
            block = None
            stop_signal = torch.tensor([0], dtype=torch.long, device=device)

        for step in range(denoising_steps):
            # Build ids on rank 0; broadcast shape + content to other ranks
            if is_main:
                ids_local = torch.tensor([ctx + block], dtype=torch.long, device=device)
            else:
                ids_local = None
            print(f"[rank {rank}] blk={blk} step={step} before broadcast_ids", flush=True)
            ids = broadcast_ids(ids_local, device, src=0)
            print(f"[rank {rank}] blk={blk} step={step} after broadcast_ids ids.shape={ids.shape}", flush=True)
            with torch.no_grad():
                out = model(input_ids=ids)
            print(f"[rank {rank}] blk={blk} step={step} after model forward", flush=True)
            # logits should be replicated across ranks (lm_head is colwise_rep), but
            # to be safe materialize the full tensor on rank 0 only.
            if is_main:
                logits = out.logits
                if hasattr(logits, "full_tensor"):
                    logits = logits.full_tensor()
                blk_logits = logits[0, len(ctx):len(ctx) + block_length, :]
                probs = torch.softmax(blk_logits.float(), dim=-1)

                # Find masked positions
                mask_positions = [i for i, t in enumerate(block) if t == mask_token_id]
                if not mask_positions:
                    pass
                else:
                    n_to_unmask = max(1, len(mask_positions) // max(1, denoising_steps - step))
                    confs = []
                    for pos in mask_positions:
                        p_top, t_top = probs[pos].max(dim=-1)
                        confs.append((p_top.item(), pos, int(t_top.item())))
                    confs.sort(reverse=True, key=lambda x: x[0])
                    for _, pos, tok in confs[:n_to_unmask]:
                        block[pos] = tok

        # Force-fill any leftover masks (shouldn't happen with our schedule)
        if is_main:
            for i in range(block_length):
                if block[i] == mask_token_id:
                    block[i] = int(probs[i].argmax().item())
            ctx.extend(block)
            generated.extend(block)
            # EOS check
            if (not no_eos_stop) and eos_id is not None and eos_id in block:
                stop_signal[0] = 1
        # Broadcast stop signal so all ranks exit together
        dist.broadcast(stop_signal, src=0)
        if stop_signal.item() == 1:
            break

    return generated if is_main else []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_model", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--num_blocks", type=int, default=32)
    p.add_argument("--block_length", type=int, default=4)
    p.add_argument("--denoising_steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151669)
    p.add_argument("--no_eos_stop", action="store_true", default=True)
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "humaneval", "mbpp"])
    args = p.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    is_main = (rank == 0)

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)

    def log(s):
        if is_main:
            print(s, flush=True)

    def log_all(s):
        print(f"[rank {rank}] {s}", flush=True)

    log(f"[init] world_size={world}  cuda:{rank}")
    log(f"[load] {args.target_model}  with tp_plan='auto'")
    t0 = time.time()
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        tp_plan="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    log(f"[load] done in {time.time()-t0:.1f}s")

    n_rms = patch_stock_rms_norm(model)
    log(f"[patch] {n_rms} SDARRMSNorm replaced with stock impl")
    sdpa_ok = patch_sdpa_eval_attention(model)
    log(f"[patch] patch_sdpa_eval_attention: {sdpa_ok}")
    if not hasattr(model.config, "block_size"):
        model.config.block_size = args.denoising_steps

    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    eos_id = None if args.no_eos_stop else tokenizer.eos_token_id
    device = torch.device(f"cuda:{rank}")

    summary = {"runs": []}
    for dataset in args.datasets:
        # Load prompts on rank 0; broadcast count + lengths to other ranks
        if is_main:
            cfg = SimpleNamespace(dataset=dataset, dataset_split="test", num_samples=args.num_samples)
            ds, prompt_ids = load_prompts(tokenizer, cfg)
            log(f"\n[{dataset}] loaded {len(prompt_ids)} prompts")
        else:
            ds, prompt_ids = None, None

        n_prompts_t = torch.tensor([0 if not is_main else len(prompt_ids)], dtype=torch.long, device=device)
        log_all(f"[{dataset}] before broadcast n_prompts; tensor={n_prompts_t.item()}")
        dist.broadcast(n_prompts_t, src=0)
        n_prompts = n_prompts_t.item()
        log_all(f"[{dataset}] after broadcast n_prompts = {n_prompts}")

        results = []
        for i in range(n_prompts):
            pids = prompt_ids[i] if is_main else None
            log_all(f"[{dataset} {i+1}/{n_prompts}] entering block_diffusion_native")
            torch.cuda.synchronize()
            t_start = time.time()
            gen = block_diffusion_native(
                model, pids, args.num_blocks, args.block_length, args.denoising_steps,
                args.mask_token_id, device, eos_id, args.no_eos_stop, rank, is_main,
            )
            torch.cuda.synchronize()
            dt = time.time() - t_start
            if is_main:
                txt = tokenizer.decode(gen, skip_special_tokens=True)
                results.append({"idx": i, "prompt_len": len(pids), "n_tokens": len(gen),
                                "end_to_end_s": dt, "text": txt})
                preview = txt.replace("\n", "⏎")[:80]
                print(f"  [{dataset} {i+1:>3}/{n_prompts}] {len(gen):>3}t  {dt:.2f}s  {preview!r}", flush=True)

        if is_main:
            total_t = sum(r["end_to_end_s"] for r in results)
            total_tok = sum(r["n_tokens"] for r in results)
            ms_per_tok = total_t * 1000.0 / total_tok if total_tok else None
            tok_per_s = total_tok / total_t if total_t else None

            scorer = SCORERS.get(dataset)
            n_pass = 0
            for r, ref in zip(results, ds):
                if scorer is not None:
                    ok, _ = scorer(r["text"], ref)
                    n_pass += int(ok)
            pass_at_1 = n_pass / len(results) if results else None

            run = {
                "dataset": dataset, "n": len(results),
                "total_s": total_t, "total_tokens": total_tok,
                "ms_per_token": ms_per_tok, "tokens_per_second": tok_per_s,
                "pass_at_1": pass_at_1,
            }
            summary["runs"].append(run)
            (out_dir / f"native_tp2_{dataset}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in results) + "\n"
            )
            print(f"\n  [{dataset}] ms/tok={ms_per_tok:.2f}  tok/s={tok_per_s:.1f}  pass@1={pass_at_1}", flush=True)

    if is_main:
        (out_dir / "SUMMARY_TP2.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[done] summary at {out_dir / 'SUMMARY_TP2.json'}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
