"""Runtime=HF, Mode=native:  SDAR  (draft_one_block loop)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from speculative_decoding.bench.backends.base import (
    BaseGenerationBackend,
    GenerationResult,
)
from speculative_decoding.bench.native_generate import (
    native_multi_block_generate,
    native_multi_block_generate_batch,
)


@dataclass
class HFNativeBackend(BaseGenerationBackend):
    """
     block SDAR  draft_one_block
    `model`  SDAR Chat checkpoint1.7B / 4B / 8B /
    """

    model: Any = None
    num_blocks: int = 8
    block_length: int = 4
    denoising_steps: int = 4
    mask_token_id: int = 151669
    eos_token_id: Optional[int] = None
    pad_token_id: int = 0
    use_cuda_graph: bool = False
    runtime: str = "hf"
    mode: str = "native"

    def reset_runtime_state(self) -> None:
        # NOTE: torch._dynamo.reset() was here, but it invalidates the
        # Inductor-compiled flex_attention Triton kernels captured by the
        # patched multi-block mask path, causing "Triton [CUDA]: invalid
        # resource handle" or "Pointer argument (at 0) cannot be accessed
        # from Triton (cpu tensor?)" on the next launch. Same fix applied in
        # speculative_decode.py's per-sample loops.
        pass

    def generate(self, prompt_ids: List[int], *, return_timings: bool = True) -> GenerationResult:
        import time
        from speculative_decoding.bench.timing import cuda_sync_if_available

        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else None
        # Align thread-local current device with the model's device. Required
        # because Liger SwiGLU (and other non-dispatcher Triton kernels) pick
        # up torch.cuda.current_stream() off the current device, which defaults
        # to cuda:0 regardless of where the model lives.
        model_device = next(self.model.parameters()).device
        cuda_sync_if_available()
        t0 = time.perf_counter()
        with torch.cuda.device(model_device):
            gen, stats = native_multi_block_generate(
                self.model,
                prompt_ids,
                self.num_blocks,
                self.block_length,
                self.denoising_steps,
                self.mask_token_id,
                eos_ids=eos_ids,
                return_timings=return_timings,
                use_cuda_graph=self.use_cuda_graph,
            )
        cuda_sync_if_available()
        t1 = time.perf_counter()
        timing = None
        if return_timings:
            timing = {
                "per_block": stats.get("per_block_timings", []),
                "total_block_wall_s": stats.get("total_block_wall_s", 0.0),
            }
        return GenerationResult(
            generated_ids=gen,
            stats={k: v for k, v in stats.items() if k != "per_block_timings"},
            timing=timing,
            end_to_end_s=t1 - t0,
        )

    def generate_batch(self, prompt_ids_batch: List[List[int]], *,
                       return_timings: bool = True) -> List[GenerationResult]:
        import time
        from speculative_decoding.bench.timing import cuda_sync_if_available

        eos_ids = [self.eos_token_id] if self.eos_token_id is not None else None
        model_device = next(self.model.parameters()).device
        cuda_sync_if_available()
        t0 = time.perf_counter()
        with torch.cuda.device(model_device):
            out_rows = native_multi_block_generate_batch(
                self.model,
                prompt_ids_batch,
                self.num_blocks,
                self.block_length,
                self.denoising_steps,
                self.mask_token_id,
                pad_token_id=self.pad_token_id,
                eos_ids=eos_ids,
                return_timings=return_timings,
                use_cuda_graph=self.use_cuda_graph,
            )
        cuda_sync_if_available()
        t1 = time.perf_counter()
        wall = t1 - t0  # batch wall-clock (shared across rows)

        results: List[GenerationResult] = []
        for gen, stats in out_rows:
            timing = None
            if return_timings:
                timing = {
                    "per_block": stats.get("per_block_timings", []),
                    "total_block_wall_s": stats.get("total_block_wall_s", 0.0),
                }
            results.append(GenerationResult(
                generated_ids=gen,
                stats={k: v for k, v in stats.items() if k != "per_block_timings"},
                timing=timing,
                end_to_end_s=wall,
            ))
        return results
