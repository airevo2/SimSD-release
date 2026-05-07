"""Runtime=JetEngine, Mode=native:  SDAR  (jetengine.LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from speculative_decoding.bench.backends.base import (
    BaseGenerationBackend,
    GenerationResult,
)


@dataclass
class JetEngineNativeBackend(BaseGenerationBackend):
    """
     jetengine.LLM

     HFNativeBackend
      - JetEngine  scheduler / KV-cache / runner request
        block-diffusion `SamplingParams.block_length / denoising_steps`
      -  `generate([prompt], sp)`  prompt batched
         prompt  `generate` prompt
        backend  wall-clock
      -  per-block  timingJetEngine  end_to_end_s

    `llm`  `jetengine.LLM` model / KV cache
    prompt
    """

    llm: Any = None
    max_tokens: int = 256
    block_length: int = 4
    denoising_steps: int = 4
    temperature: float = 0.0
    ignore_eos: bool = False
    runtime: str = "jetengine"
    mode: str = "native"

    def reset_runtime_state(self) -> None:
        # JetEngine scheduler  torch._dynamo
        pass

    def generate(self, prompt_ids: List[int], *, return_timings: bool = True) -> GenerationResult:
        import time
        from jetengine import SamplingParams
        from speculative_decoding.bench.timing import cuda_sync_if_available

        sp = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
            block_length=self.block_length,
            denoising_steps=self.denoising_steps,
        )

        cuda_sync_if_available()
        t0 = time.perf_counter()
        outs = self.llm.generate([prompt_ids], sp, use_tqdm=False)
        cuda_sync_if_available()
        t1 = time.perf_counter()

        gen_ids = outs[0]["token_ids"] if outs else []
        stats = {
            "num_blocks_run": 0,  # JetEngine  block
            "total_draft_tokens": len(gen_ids),
        }
        timing = {
            "per_block": [],
            "total_block_wall_s": t1 - t0,
        } if return_timings else None
        return GenerationResult(
            generated_ids=list(gen_ids),
            stats=stats,
            timing=timing,
            end_to_end_s=t1 - t0,
        )
