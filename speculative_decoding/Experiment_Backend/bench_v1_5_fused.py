"""v1.5_fused bench wrapper  applies the v1.5_fused draft patch (last
denoising step uses intra_md mask; no extra forward) as a side-effect, then
delegates CLI parsing to ``speculative_decoding.bench.run_benchmark``.

Usage (example):
  python speculative_decoding/Experiment_Backend/bench_v1_5_fused.py \\
      --compare both --runtime hf \\
      --draft_model inference/model/SDAR-1_7B-Chat \\
      --target_model inference/model/SDAR-8B-Chat \\
      --draft_device cuda:0 --target_device cuda:0 \\
      --dataset gsm8k --num_samples 10 --warmup 2 \\
      --num_blocks 32 --block_length 4 --denoising_steps 4 --K 8 \\
      --no_eos_stop --target_eval_sdpa --seed 42 \\
      --output fix_422/07_v1_5_fused/K8_1_7B/bench.json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from speculative_decoding.Experiment_Backend import draft_probs_ablation_v1_5_fused  # noqa: E402,F401
from speculative_decoding.bench import run_benchmark as _bench  # noqa: E402


if __name__ == "__main__":
    _bench.main()
