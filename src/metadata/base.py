"""
src/metadata/base.py — Abstract base class for metadata enrichers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from src.core.types import Document


class BaseMetadataEnricher(ABC):
    """Interface for metadata enrichment strategies."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def enrich(self, document: Document) -> Document:
        """
        Enrich a Document with additional metadata (e.g. doc_type).

        Args:
            document: The parsed Document.

        Returns:
            The same Document with updated metadata / doc_type fields.
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__
