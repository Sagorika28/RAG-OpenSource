"""
src/rerank/none.py — No-op reranker (passthrough).

Returns hits as-is without any reranking.  Used when reranking
is disabled in config.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.core.types import Hit
from src.rerank.base import BaseReranker


class NoReranker(BaseReranker):
    """Passthrough reranker — returns hits unchanged."""

    def rerank(self, query: str, hits: List[Hit]) -> List[Hit]:
        """Return hits as-is (no reranking)."""
        return hits
