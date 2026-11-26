"""Per-model fact files for causal tracing (stage 4 of dataset construction)."""

from .builder import (
    build_facts,
    continuation_ids,
    screen_candidates,
    single_token_candidates,
    to_records,
)
from .config import FactConfig

__all__ = [
    "FactConfig",
    "build_facts",
    "continuation_ids",
    "screen_candidates",
    "single_token_candidates",
    "to_records",
]
