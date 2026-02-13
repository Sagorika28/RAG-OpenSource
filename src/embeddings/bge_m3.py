"""
src/embeddings/bge_m3.py — BGE-M3 embedder (GPU upgrade path).

Uses sentence-transformers to load BAAI/bge-m3 (1024-dim).
This is the GPU-optimized embedding model for production scaling.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BGEM3Embedder(BaseEmbedder):
    """GPU-optimized embedder using BGE-M3 (1024-dim)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._model = None
        self._dimension = self.config.get("dimension", 1024)
        self._batch_size = self.config.get("batch_size", 64)
        self._device = self.config.get("device", "cuda")
        self._model_name = self.config.get("model_name", "BAAI/bge-m3")

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
            logger.info(
                f"Loaded {self._model_name} "
                f"(dim={self._model.get_sentence_embedding_dimension()})"
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using BGE-M3.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape (len(texts), 1024).
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.  BGE-M3 does not require a query prefix.

        Args:
            query: The search query string.

        Returns:
            np.ndarray of shape (1, 1024).
        """
        return self.embed([query])

    @property
    def dimension(self) -> int:
        return self._dimension
