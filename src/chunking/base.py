"""
src/chunking/base.py — Abstract base class for document chunkers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.core.types import Chunk, Document


class BaseChunker(ABC):
    """Interface that all chunkers must implement."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a Document into a list of Chunks.

        Args:
            document: Parsed Document with pages and elements.

        Returns:
            List of Chunk objects with text, context_header, and metadata.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__
