"""
src/embeddings/bge_small.py — BGE-small-en-v1.5 embedder (CPU baseline).

Uses sentence-transformers to load BAAI/bge-small-en-v1.5 (384-dim).
This is the default CPU-friendly embedding model.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from src.core.device import resolve_torch_device
from src.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class BGESmallEmbedder(BaseEmbedder):
    """CPU-friendly embedder using BGE-small-en-v1.5 (384-dim)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._model = None
        self._dimension = self.config.get("dimension", 384)
        self._batch_size = self.config.get("batch_size", 32)
        self._device = resolve_torch_device(
            self.config.get("device", "auto"),
            component="embeddings",
        )
        self._model_name = self.config.get(
            "model_name", "BAAI/bge-small-en-v1.5"
        )
        self._query_prefix = self.config.get(
            "query_prefix",
            "Represent this sentence for searching relevant passages: ",
        )

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
        Embed a list of texts using BGE-small.

        Args:
            texts: List of text strings.

        Returns:
            np.ndarray of shape (len(texts), 384).
        """
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # BGE models work best with L2-normed vectors
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query, prepending the BGE query prefix.

        Args:
            query: The search query string.

        Returns:
            np.ndarray of shape (1, 384).
        """
        prefixed = f"{self._query_prefix}{query}"
        return self.embed([prefixed])

    @property
    def dimension(self) -> int:
        return self._dimension
