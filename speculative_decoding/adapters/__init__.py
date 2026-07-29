"""Model-family adapters — the only layer in SimSD that knows architecture names.

Selection is by **inspection of the loaded model**, not by a flag, so a
checkpoint can never be paired with the wrong adapter:

    >>> ad = detect(model)          # -> SdarAdapter() | Llada2Adapter()
    >>> ad.family
    'llada2'

``get(family)`` is available for the places that need an adapter before a model
exists (e.g. resolving a default ``mask_token_id`` from a YAML experiment file).

Adding a family means adding one module here — nothing in ``cache_aware.py``,
``draft.py``, ``verify.py`` or the runners should need to grow a branch.
"""

from __future__ import annotations

from typing import Optional

from .base import (
    ModelAdapter,
    is_sharded,
    make_eager_cache_forward,
    make_eval_forward,
    make_static_cache_forward,
    rebind_hooked_forwards,
)
from .llada2 import Llada2Adapter
from .sdar import SdarAdapter

#: Registry, in detection order.
ADAPTERS = (SdarAdapter(), Llada2Adapter())

FAMILIES = tuple(a.family for a in ADAPTERS)

__all__ = [
    "ADAPTERS", "FAMILIES", "ModelAdapter", "SdarAdapter", "Llada2Adapter",
    "detect", "get", "is_sharded", "make_static_cache_forward",
    "make_eager_cache_forward", "make_eval_forward", "rebind_hooked_forwards",
]


def get(family: str) -> ModelAdapter:
    """Adapter by family name. Raises on unknown names (typo in YAML)."""
    for a in ADAPTERS:
        if a.family == family:
            return a
    raise KeyError(
        f"unknown model family {family!r}; known: {', '.join(FAMILIES)}"
    )


def detect(model) -> Optional[ModelAdapter]:
    """Adapter matching a loaded model, or None if no family recognises it.

    Returning None (rather than raising) keeps the patch functions' historical
    contract: they report "nothing patched" and the caller decides whether that
    is fatal.
    """
    for a in ADAPTERS:
        if a.matches(model):
            return a
    return None


def detect_or_raise(model) -> ModelAdapter:
    got = detect(model)
    if got is None:
        names = sorted({m.__class__.__name__ for m in model.modules()
                        if "ttention" in m.__class__.__name__})
        raise RuntimeError(
            f"no SimSD adapter matches {model.__class__.__name__}. Attention "
            f"classes found: {names or 'none'}. Known families: "
            f"{', '.join(FAMILIES)}. Add one in speculative_decoding/adapters/."
        )
    return got
