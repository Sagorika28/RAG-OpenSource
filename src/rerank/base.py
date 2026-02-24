"""
src/rerank/base.py — Abstract base class for rerankers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.core.types import Hit


class BaseReranker(ABC):
    """Interface that all rerankers must implement."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def rerank(self, query: str, hits: List[Hit], top_k: Optional[int] = None) -> List[Hit]:
        """
        Rerank a list of search hits given the query.

        Args:
            query: The original search query.
            hits:  List of Hit objects from vector search.
            top_k: Optional override for the number of chunks returned.

        Returns:
            Reranked list of Hit objects (sorted by rerank_score desc).
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__
