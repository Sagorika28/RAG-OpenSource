"""
src/generation/base.py — Abstract base class for LLM generators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.core.types import Hit


class BaseGenerator(ABC):
    """Interface that all generators must implement."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def generate(self, query: str, hits: List[Hit]) -> str:
        """
        Generate an answer using the query and retrieved context chunks.

        Args:
            query: The user's question.
            hits:  List of Hit objects providing context.

        Returns:
            Generated answer string with citations.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__
