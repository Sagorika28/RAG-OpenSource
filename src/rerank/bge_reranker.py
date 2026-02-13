"""
src/rerank/bge_reranker.py — BGE cross-encoder reranker.

Uses sentence-transformers CrossEncoder with:
  - CPU baseline:  BAAI/bge-reranker-base
  - GPU upgrade:   BAAI/bge-reranker-large
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.core.types import Hit
from src.rerank.base import BaseReranker

logger = logging.getLogger(__name__)


class BGEReranker(BaseReranker):
    """Cross-encoder reranker using a BGE reranker model."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._model = None
        self._model_name = self.config.get(
            "model_name", "BAAI/bge-reranker-base"
        )
        self._device = self.config.get("device", "cpu")
        self._top_k = self.config.get("top_k", 5)

    def _load_model(self):
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self._model_name}")
            self._model = CrossEncoder(
                self._model_name, device=self._device
            )
            logger.info(f"Loaded reranker: {self._model_name}")

    def rerank(self, query: str, hits: List[Hit]) -> List[Hit]:
        """
        Rerank hits using the cross-encoder model.

        Scores each (query, chunk_text) pair, then sorts descending.
        Returns at most top_k hits.
        """
        if not hits:
            return hits

        self._load_model()

        # Build (query, passage) pairs
        pairs = [(query, hit.chunk.raw_text or hit.chunk.text) for hit in hits]

        # Score all pairs
        scores = self._model.predict(pairs)

        # Attach rerank scores and sort
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)

        reranked = sorted(hits, key=lambda h: h.rerank_score or 0, reverse=True)

        logger.debug(
            f"Reranked {len(hits)} hits → returning top {self._top_k}"
        )
        return reranked[: self._top_k]
