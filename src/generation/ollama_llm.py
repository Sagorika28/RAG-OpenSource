"""
src/generation/ollama_llm.py — Ollama LLM generator via HTTP.

Calls Ollama's /api/generate endpoint with a prompt that includes
retrieved context chunks.  Uses Qwen2.5:3b-instruct by default.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import requests

from src.core.types import Hit
from src.generation.base import BaseGenerator

logger = logging.getLogger(__name__)


class OllamaGenerator(BaseGenerator):
    """Generate answers using a locally-running Ollama model."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.model = self.config.get("model", "qwen2.5:3b")
        self.base_url = self.config.get(
            "base_url", "http://localhost:11434"
        )
        self.temperature = self.config.get("temperature", 0.1)
        self.max_tokens = self.config.get("max_tokens", 1024)
        self.system_prompt = self.config.get(
            "system_prompt",
            (
                "You are a helpful technical assistant. Answer the question "
                "using ONLY the provided context. Cite sources using "
                "[Source: filename, Page: N]. If the context does not "
                "contain enough information, say so."
            ),
        )

    def generate(self, query: str, hits: List[Hit]) -> str:
        """
        Generate an answer with citations from retrieved chunks.

        Args:
            query: User question.
            hits:  Retrieved (and optionally reranked) chunks.

        Returns:
            Generated answer string.
        """
        if not hits:
            return "No relevant context was found to answer this question."

        # Build context block from hits
        context = self._build_context(hits)

        # Construct the full prompt
        prompt = (
            f"{self.system_prompt}\n\n"
            f"### Context\n{context}\n\n"
            f"### Question\n{query}\n\n"
            f"### Answer"
        )

        try:
            response = self._call_ollama(prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"[Error: LLM generation failed — {e}]"

    def _build_context(self, hits: List[Hit]) -> str:
        """Format retrieved chunks as numbered context for the prompt."""
        context_parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            source = hit.chunk.metadata.get("source", "unknown")
            page_start = hit.chunk.metadata.get("page_start", "?")
            page_end = hit.chunk.metadata.get("page_end", "?")
            section = hit.chunk.metadata.get("section_path", "")

            header = f"[{i}] Source: {source}"
            if section:
                header += f" | Section: {section}"
            header += f" | Pages: {page_start}-{page_end}"

            # Use raw_text (without context header) to avoid duplication
            text = hit.chunk.raw_text or hit.chunk.text
            context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def _call_ollama(self, prompt: str) -> str:
        """
        POST to Ollama's /api/generate endpoint.

        Uses stream=false for simplicity.  Returns the full response text.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        logger.debug(f"Calling Ollama: model={self.model}")
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()

        data = resp.json()
        answer = data.get("response", "").strip()

        logger.debug(f"Ollama response length: {len(answer)} chars")
        return answer

    def is_available(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = requests.get(
                f"{self.base_url}/api/tags", timeout=5
            )
            return resp.status_code == 200
        except Exception:
            return False
