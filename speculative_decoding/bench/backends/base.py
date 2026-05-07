"""Base backend abstraction + shared helpers.

Backend  (runtime, mode)
  runtime: "hf" / "jetengine" / ...
  mode:    "native" / "speculative"
 BaseGenerationBackend `name`
`f"{runtime}_{mode}"` `hf_native``hf_speculative``jetengine_native`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def compute_verify_padded_len(
    max_prompt_len: int,
    num_blocks: int,
    block_length: int,
    K: int = 1,
    align: int = 128,
) -> int:
    """ eval_multi_block  target  forward  padding

     verify (verify.py build_verify_sequence_multi):
      prompt + [accepted_i × bl]         data, mask
             + [current_draft × 2bl]    K  draft block, data + mask
    accepted_blocks  iter  num_blocks-K, K  draft block,
    raw_max = prompt + (num_blocks-K)bl + K2bl = prompt + (num_blocks+K)bl

    K  1( block speculative)K>1
    """
    raw_max = max_prompt_len + (num_blocks - K) * block_length + K * 2 * block_length
    return ((raw_max + align - 1) // align) * align


@dataclass
class GenerationResult:
    """ generate  + """

    generated_ids: List[int]
    stats: Dict[str, Any]
    timing: Optional[Dict[str, Any]] = None
    end_to_end_s: Optional[float] = None


class BaseGenerationBackend(ABC):
    """ prompt token ids"""

    runtime: str = "base"
    mode: str = "base"

    @property
    def name(self) -> str:
        return f"{self.runtime}_{self.mode}"

    @abstractmethod
    def generate(
        self,
        prompt_ids: List[int],
        *,
        return_timings: bool = True,
    ) -> GenerationResult:
        ...

    def generate_batch(
        self,
        prompt_ids_batch: List[List[int]],
        *,
        return_timings: bool = True,
    ) -> List[GenerationResult]:
        """Batched variant. Default impl: fall back to looping ``generate``.

        Subclasses with a real batched implementation should override this.
        Timing of fallback path is per-row (no overlap), so don't use it
        for batch latency benches.
        """
        return [
            self.generate(p, return_timings=return_timings)
            for p in prompt_ids_batch
        ]

    def reset_runtime_state(self) -> None:
        """ torch dynamo"""
        pass
