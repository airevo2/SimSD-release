"""Config loader for speculative decoding experiments."""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class SpecConfig:
    # ── Model ──
    draft_model: str = os.path.join(REPO_ROOT, "inference/model/SDAR-4B-Chat")
    target_model: str = os.path.join(REPO_ROOT, "inference/model/SDAR-8B-Chat")
    dtype: str = "bfloat16"
    mask_token_id: int = 151669

    # ── Block diffusion ──
    block_length: int = 4
    denoising_steps: int = 4
    block_size: int = 4
    block_causal_prompt: bool = True  # True=SDAR block-level causal; False=legacy prompt sees mask

    # ── Draft early-stop ablation ──
    # Controls how many denoising steps the draft model runs per block.
    # -1 (default): run the full ``denoising_steps`` schedule  identical to
    #               legacy behaviour.
    # k > 0: stop draft after k denoising steps. Positions still MASK at
    #        step k get ``step_map[i] = -1`` and remain ``mask_token_id`` in
    #        ``draft_ids``. Target verify treats these as still-mask
    #        (same token_label as the MASK segment), and at MRS time we
    #        substitute ``draft_ids[i] = target_probs[i].argmax()`` and
    #        ``draft_probs[i] = target_probs[i]`` so MRS auto-accepts them
    #        (they are committed via target argmax, no MRS rejection on
    #        non-drafted positions). Saves (denoising_steps - k) draft
    #        forwards per block at the cost of giving up MRS quality control
    #        for (block_length - k) positions per block. ONLY honoured on
    #        the simple ``speculative_generate`` path (no KV cache, no
    #        pipeline, K=1, no batched). All other paths ignore the flag.
    draft_steps_per_block: int = -1

    # ── RoPE ablation: mask-token position sharing ──
    # True (default, matches training-time SDAR layout): each draft block's MASK
    #     positions reuse the same RoPE position_ids as the corresponding DATA
    #     positions. Mask token at within-block rank k and data token at rank k
    #     get the SAME position id, so RoPE rotation is identical and the
    #     training-time semantics ("mask is the placeholder for the data token
    #     to be predicted at the same position") hold.
    # False (ablation): mask positions are assigned distinct sequential ids
    #     immediately after their data positions. Each draft block then occupies
    #     2*block_length unique positions (data: [p, p+bl), mask: [p+bl, p+2bl))
    #     instead of bl. This deliberately breaks training-inference RoPE
    #     alignment and is intended ONLY for ablation studies; quality is
    #     expected to drop. The flag does not change the attention-mask
    #     construction (visibility rules are unchanged)  only position_ids.
    share_mask_position: bool = True

    # ── Generation ──
    num_blocks: int = 8
    gen_length: int = 512
    # Draft lookahead per speculative iteration. Each spec iter, draft speculates
    # K blocks forward; target verifies all K in one forward. Must divide
    # num_blocks. K=1 reproduces legacy one-block-at-a-time behavior.
    K: int = 1

    # ── Batch ──
    # Number of prompts processed together per bench call. bs>1 re-captures a
    # new CUDA graph per (batch, padded_len) pair. Static buffers grow as
    # (batch, padded_len); memory cost ≈ linear in batch.
    batch: int = 1

    # ── Dataset ──
    # Supported: gsm8k, humaneval, mbpp, ifeval.
    dataset: str = "gsm8k"
    dataset_split: str = "train"
    num_samples: int = 50

    # ── Evaluation mode ──
    #   "single_block"   draft+verify only 1 block per prompt (for alignment test)
    #   "multi_block"    full sequential generation over num_blocks
    mode: str = "multi_block"

    # ── Speculative branch (single_block ) ──
    #   "mrs"   MRS /
    #   "per_position_compare"   MRS draft token / target
    #        batch_inference_multi_block  position
    speculative_branch: str = "mrs"
    # MRS  speculative_branch=mrs :
    #   "position"   0L-1
    #   "step_map"   unmask ** bug** stepposition  identity
    #                mrs.py:88  `range(first_reject_pos)`
    #                 token Experiment_Backend/results/quality
    #                /SUMMARY.md  2026-04-21
    mrs_verify_order: str = "position"
    #  jsonl  per_position
    per_position_breakdown: bool = False

    # ── Output ──
    #  output_dir / {YYYY-MM-DD} / {prefix}{tag}/ report_paths.report_date_str
    output_dir: str = os.path.join(REPO_ROOT, "speculative_decoding/results")
    config_name: Optional[str] = None  # yaml stem as file prefix (e.g. multi_block_default)
    tag: Optional[str] = None
    save_plots: bool = True
    ascii_demo: bool = False

    # ── Device ──
    draft_device: str = "cuda:1"
    target_device: str = "cuda:1"

    # ── Target attention backend ──
    # True: target verify  eval + externally-built 4D bool mask  SDPA,
    #        patch_multi_block_mask_fn ( sweep_forward  --target_eval_sdpa).
    # False:  (patched create_multi_block_causal_mask + flex_attention).
    target_eval_sdpa: bool = False

    # ── CUDA graph for draft denoising forwards ──
    # True: draft_one_block  replay CUDA graph (per (model, seq_len,
    #       block_length, device) ).  ~99%  kernel launch overhead,
    #       launch-bound  (1.7B/4B)  forward ~4-5x .
    #        seq_len  ~3  forward  capture/warmup,  replay
    #       . SDPA  patch_sdpa_eval_attention ,
    #       graph capture  attention  GPUCPU sync .
    # False:  eager forward .
    use_cuda_graph: bool = False

    # ── Pipeline overlap (dual-GPU) ──
    # True: speculative_generate  pipelined :  verify_N  draft_{N+1}
    #        sync barrier,  draft GPU  target GPU
    #        speculative_generate_pipelined : ""
    #        draft_{N+1},  MRS_N  or
    #        dual-GPU (draft_device != target_device)
    # False:  (draftverifyMRS )
    pipeline: bool = False

    # ── Draft sampling ──
    # "argmax"       top-1 (default; matches all existing native / self_draft
    #                 behavior bit-for-bit)
    # "multinomial"  sample from softmax(probs); required for the MRS
    #                 "output ≡ target distribution" theorem to hold under
    #                 cross-draft. argmax breaks the theorem because the
    #                 draft_id distribution under argmax is NOT p, so MRS's
    #                 min(1, q/p) acceptance rule produces a biased mixture
    #                 instead of q. See plan/07_v3_quality_alignment.md §2.A.
    draft_sampling: str = "argmax"

    # ── Remasking strategy for draft denoising ──
    # Picks which masked positions to commit each denoising step. Lives outside
    # the cuda_graph capture (only model.forward is captured) so dynamic shapes
    # are fine.
    # "low_confidence_static"   every step picks exactly topk(n_unmask) by
    #                            confidence; n_unmask precomputed from
    #                            ceil(remaining / remaining_steps). Fully
    #                            deterministic schedule, 0 host syncs/step.
    # "low_confidence_dynamic"  if #(conf > threshold) >= n_unmask, unmask
    #                            ALL high-conf positions (may exceed n_unmask);
    #                            otherwise fall back to topk. Costs 1 host
    #                            sync/step (the high_conf count compare).
    # Both share the same captured forward graph; switching strategy does not
    # invalidate any graph entry.
    remasking_strategy: str = "low_confidence_static"
    confidence_threshold: float = 0.9

    # ── Partial-block fill on reject ──
    # When MRS rejects mid-block, the tail of the block needs to be filled
    # before extending committed_context (we do not commit MASKs  they put
    # target attention OOD on the next block).
    # "draft_argmax"   pad with draft's own samples for those positions
    #                   (default; preserves legacy behavior). Under cross-
    #                   draft this leaks the draft's distribution into the
    #                   committed context, causing K-cascade quality drift.
    # "target_argmax"  pad with target's argmax at those positions (uses
    #                   target_probs already in scope; no extra forward).
    #                   Cuts root-cause B in plan/07 §2.B; but target_probs
    #                   were computed conditioned on the draft tokens that
    #                   MRS just rejected  "stale-pad" (plan/08 §2.3).
    # "redraft"        Stage D fix: re-draft + re-verify the remaining
    #                   positions on the post-reject prefix (committed +
    #                   final_ids). Pads with argmax of the new target_probs,
    #                   which are conditioned on the actually-committed
    #                   prefix. Costs +1 draft + 1 target verify per reject
    #                   (≈ 9–22% overhead at α=0.87). See plan/08 §4.
    # "truncate"       Stage E (plan/08 §5.3 reverse-proof): commit
    #                   final_ids as a variable-length partial block; do not
    #                   pad. build_verify_sequence handles via running offset
    #                   so subsequent block target_probs aren't conditioned
    #                   on any rejected/discarded tail. If this materially
    #                   recovers code-task quality, partial-pad mechanism
    #                   itself is the bug; if not, the residual is upstream
    #                   (draft quality / MRS bonus token).
    # "target_argmax_all"  Stage F (aggressive fallback): on any reject,
    #                       replace the WHOLE block with target_probs.argmax,
    #                       discarding both the MRS bonus and the accepted
    #                       draft prefix. Probes whether keeping draft tokens
    #                       in the committed context (even if accepted) is
    #                       itself harmful for cross-draft on code.
    # "truncate_no_bonus"  Stage G (strict-truncate, 1.7B8B ): like
    #                       truncate but ALSO drops the MRS bonus token before
    #                       commit. Reasoning: bonus = sample from
    #                       norm(max(0, q − p)) = mass "8B prefers but 1.7B
    #                       doesn't propose". For 1.7B8B these are often
    #                       jarring isolated tokens (e.g. 'List' without
    #                       preceding 'from typing import'). Dropping the
    #                       bonus + truncating lets the next block's diffusion
    #                       fill position j naturally from a clean prefix.
    #                       Trade-off: deviates from MRS theory's "output ≡
    #                       target distribution" property.
    # 2026-05-06: default switched draft_argmax  target_argmax. Empirically
    # 50% TPS gain on bl=8 gsm8k (from 49.5 to 74.1 tok/s) and ~5-35% across
    # other (bl, dataset) combos. Theoretically more correct: at reject
    # positions [reject_pos+1, bl) we use target verify's argmax (bonus token
    # extended) instead of draft's stale predictions that target rejected.
    # plan/18 §8 has full data.
    partial_block_fill: str = "target_argmax"

    # ── KV cache for spec (prefix-only, à la generate.py:block_diffusion_generate) ──
    # True:  spec generate  draft + target  DynamicCache,prefill  store_kv=True
    #         prompt  K/V , spec iter  draft denoising  target verify
    #         forward  block(K=1  forward 4  draft tokens + 4  mask tokens),
    #        attend  cached prefix block  store_kv=True forward
    #        committed block  K/V  cache GPU bs=1 spec/native
    #        2-3×, ratio ( native  cache)
    # False (default): , forward  seq(, bench
    #        HFNativeBackend / HFSpeculativeBackend )
    # NOTE:  dispatch  use_kv_cache=True  raise NotImplementedError;
    #       speculative_decoding/draft.py:draft_one_block( prefix  prefill, forward
    #       current block + cross-attend cache) speculative_decoding/verify.py:
    #       target_verify_forward( prefix prefill +  forward draft_data + draft_mask)
    #       bench/backends/hf/{native,speculative}.py  cache state
    #        1 ( cuda_graph SDPA patch  K  block )
    use_kv_cache: bool = False

    # StaticBlockCache.max_cache_len (= permanent K/V slot size). Smaller is
    # faster (less redundant K/V scan in mem_efficient SDPA, since the bool
    # 4D mask doesn't physically skip masked-off slots) but caps prompt+gen
    # length. For prompt_len ≤ 200 and num_blocks*block_length ≤ 128, set to
    # 256-384. Default 1024 is the safe oversized buffer.
    kv_cache_max_len: int = 1024

    # ── Misc ──
    seed: int = 42
    log_interval: int = 10

    def __post_init__(self):
        if self.block_size != self.denoising_steps:
            self.block_size = self.denoising_steps
        if self.K < 1:
            raise ValueError(f"K must be >= 1, got {self.K}")
        if self.num_blocks % self.K != 0:
            raise ValueError(
                f"num_blocks={self.num_blocks} must be divisible by K={self.K}"
            )
        if self.batch < 1:
            raise ValueError(f"batch must be >= 1, got {self.batch}")
        if self.tag is None:
            self.tag = f"{self.mode}_bl{self.block_length}_ds{self.denoising_steps}_nb{self.num_blocks}"

    def to_dict(self):
        return asdict(self)


def load_config(yaml_path: str, overrides: Optional[dict] = None) -> SpecConfig:
    """Load config from YAML, with optional dict overrides."""
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    # Use yaml filename (stem) as output file prefix
    raw["config_name"] = os.path.splitext(os.path.basename(yaml_path))[0]

    if overrides:
        raw.update({k: v for k, v in overrides.items() if v is not None})

    return SpecConfig(**{k: v for k, v in raw.items() if hasattr(SpecConfig, k)})


def default_config(overrides: Optional[dict] = None) -> SpecConfig:
    """Create config from defaults + overrides (no YAML file needed)."""
    if overrides:
        filtered = {k: v for k, v in overrides.items() if v is not None and hasattr(SpecConfig, k)}
        return SpecConfig(**filtered)
    return SpecConfig()
