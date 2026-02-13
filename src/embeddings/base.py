"""
src/embeddings/base.py — Abstract base class for embedding models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class BaseEmbedder(ABC):
    """Interface that all embedding models must implement."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            np.ndarray of shape (len(texts), dimension).
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    def name(self) -> str:
        return self.__class__.__name__
