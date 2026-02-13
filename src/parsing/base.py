"""
src/parsing/base.py — Abstract base class for PDF parsers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from src.core.types import Document


class BaseParser(ABC):
    """Interface that all PDF parsers must implement."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abstractmethod
    def parse(self, pdf_path: str | Path) -> Document:
        """
        Parse a PDF file and return a Document.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A Document with pages, raw_text, and source populated.
        """
        ...

    def name(self) -> str:
        """Human-readable parser name."""
        return self.__class__.__name__
