"""
src/generation/none.py — No-op generator (retrieval-only mode).

Returns an empty answer, allowing the pipeline to be used
for retrieval-only evaluation without an LLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.core.types import Hit
from src.generation.base import BaseGenerator


class NoGenerator(BaseGenerator):
    """No-op generator — retrieval-only mode."""

    def generate(
        self,
        query: str,
        hits: List[Hit],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Return empty string (no generation)."""
        return ""
