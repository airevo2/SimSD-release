"""Multi-block speculative vs native SDAR"""

from speculative_decoding.bench.backends import (
    BaseGenerationBackend,
    NativeSdarBackend,
    SpeculativeBackend,
    compute_verify_padded_len,
    GenerationResult,
)
from speculative_decoding.bench.native_generate import native_multi_block_generate
from speculative_decoding.bench.timing import summarize_latency_ms, summarize_ms, cuda_sync_if_available

__all__ = [
    "BaseGenerationBackend",
    "NativeSdarBackend",
    "SpeculativeBackend",
    "GenerationResult",
    "compute_verify_padded_len",
    "native_multi_block_generate",
    "summarize_latency_ms",
    "summarize_ms",
    "cuda_sync_if_available",
]
