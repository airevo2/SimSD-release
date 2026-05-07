"""Runtime=HF, Mode=speculative: draft + target verify + MRS.

 `speculative_decoding.speculative_decode.speculative_generate` prompt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from speculative_decoding.config import SpecConfig
from speculative_decoding.speculative_decode import (
    speculative_generate,
    speculative_generate_batch,
)

from speculative_decoding.bench.backends.base import (
    BaseGenerationBackend,
    GenerationResult,
)


@dataclass
class HFSpeculativeBackend(BaseGenerationBackend):
    """Draft + Target verify + MRS"""

    draft_model: Any = None
    target_model: Any = None
    cfg: Optional[SpecConfig] = None
    pad_token_id: int = 0
    padded_len: int = 512
    eos_token_id: Optional[int] = None
    runtime: str = "hf"
    mode: str = "speculative"

    def reset_runtime_state(self) -> None:
        # NOTE: torch._dynamo.reset() removed  invalidates Inductor-compiled
        # flex_attention Triton kernels captured by the patched multi-block
        # mask path, leading to "invalid resource handle" / "cpu tensor?" on
        # the next launch. See matching comment in native.py and the per-sample
        # loops in speculative_decode.py.
        pass

    def generate(self, prompt_ids: List[int], *, return_timings: bool = True) -> GenerationResult:
        import time
        from speculative_decoding.bench.timing import cuda_sync_if_available

        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else None
        cuda_sync_if_available()
        t0 = time.perf_counter()
        if return_timings:
            gen, stats, timing = speculative_generate(
                self.draft_model,
                self.target_model,
                prompt_ids,
                self.cfg,
                self.pad_token_id,
                self.padded_len,
                eos_ids,
                return_timings=True,
            )
        else:
            gen, stats = speculative_generate(
                self.draft_model,
                self.target_model,
                prompt_ids,
                self.cfg,
                self.pad_token_id,
                self.padded_len,
                eos_ids,
                return_timings=False,
            )
            timing = None
        cuda_sync_if_available()
        t1 = time.perf_counter()
        return GenerationResult(
            generated_ids=gen,
            stats=stats,
            timing=timing,
            end_to_end_s=t1 - t0,
        )

    def generate_batch(self, prompt_ids_batch: List[List[int]], *,
                       return_timings: bool = True) -> List[GenerationResult]:
        import time
        from speculative_decoding.bench.timing import cuda_sync_if_available

        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else None
        cuda_sync_if_available()
        t0 = time.perf_counter()
        out_rows = speculative_generate_batch(
            self.draft_model,
            self.target_model,
            prompt_ids_batch,
            self.cfg,
            self.pad_token_id,
            self.padded_len,
            eos_ids,
            return_timings=return_timings,
        )
        cuda_sync_if_available()
        t1 = time.perf_counter()
        wall = t1 - t0  # batch wall-clock (shared across rows)

        results: List[GenerationResult] = []
        for r in out_rows:
            if return_timings:
                gen, stats, timing = r
            else:
                gen, stats = r
                timing = None
            results.append(GenerationResult(
                generated_ids=gen,
                stats=stats,
                timing=timing,
                end_to_end_s=wall,
            ))
        return results
