"""Generation backends, organized by (runtime, mode).

Layout:
    backends/
        base.py          - BaseGenerationBackend, GenerationResult, compute_verify_padded_len
        hf/
            native.py        - HFNativeBackend
            speculative.py   - HFSpeculativeBackend
        jet/
            native.py        - JetEngineNativeBackend
            # speculative.py - future

All public symbols are re-exported here so existing imports
`from speculative_decoding.bench.backends import HFNativeBackend` keep working.
Old class names `NativeSdarBackend` / `SpeculativeBackend` stay as deprecated
aliases for backward compatibility.
"""

from speculative_decoding.bench.backends.base import (
    BaseGenerationBackend,
    GenerationResult,
    compute_verify_padded_len,
)
from speculative_decoding.bench.backends.hf import (
    HFNativeBackend,
    HFSpeculativeBackend,
)
from speculative_decoding.bench.backends.jet import JetEngineNativeBackend

# Legacy aliases  do not use in new code, kept so old import sites still work.
NativeSdarBackend = HFNativeBackend
SpeculativeBackend = HFSpeculativeBackend

__all__ = [
    "BaseGenerationBackend",
    "GenerationResult",
    "compute_verify_padded_len",
    "HFNativeBackend",
    "HFSpeculativeBackend",
    "JetEngineNativeBackend",
    # Legacy:
    "NativeSdarBackend",
    "SpeculativeBackend",
]
