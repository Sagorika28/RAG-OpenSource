from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from groq import Groq

from src.core.types import Hit
from src.generation.base import BaseGenerator

logger = logging.getLogger(__name__)


class GroqGenerator(BaseGenerator):
    """
    Generator using Groq API (e.g. Llama 3 via Groq).
    Requires GROQ_API_KEY environment variable.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment variables.")

        # Initialize Groq client
        # If api_key is None, Groq() might check env automatically, 
        # but better to provide explicitly if we have it or let it fail gracefully.
        self.client = Groq(api_key=self.api_key or "missing_key")
        
        self.model = config.get("model", "llama3-8b-8192")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.system_prompt = config.get("system_prompt", "You are a helpful assistant.")

    def generate(
        self,
        query: str,
        context_hits: List[Hit],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Generate answer using Groq API.
        """
        if not self.api_key:
            return "Error: GROQ_API_KEY not set. Please add it to your environment."

        context_str = self._build_context(context_hits)
        history_str = self._build_history(conversation_history)
        
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_prompt,
                        },
                    {
                        "role": "user",
                        "content": (
                            f"Conversation History:\n{history_str}\n\n"
                            f"Context:\n{context_str}\n\n"
                            f"Question: {query}"
                        ),
                    },
                ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                answer = chat_completion.choices[0].message.content
                return answer

            except Exception as e:
                # Handle rate limits (429) with retry
                if "429" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"Groq Rate Limit hit (429). Retrying in {retry_delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        return (
                            "⚠️ **Groq Rate Limit Reached.** You've hit the daily token limit for your account "
                            "(likely 100k tokens/day). Please try again later or switch to a high-quota model like `llama-3.1-8b-instant` in your config."
                        )
                
                logger.error(f"Groq generation failed: {e}")
                return "❌ **AI Generation Error.** The AI provider (Groq) encountered an issue. Please check your API key and connection."
        
        return "⚠️ **Request Timed Out.** The AI didn't respond after several retries. This usually happens during high-traffic periods."

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
            # header += f" | Pages: {page_start}-{page_end}"

            # Use raw_text (without context header) to avoid duplication
            text = hit.chunk.raw_text or hit.chunk.text
            context_parts.append(f"{header}\n{text}")

        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _build_history(
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Format recent user/assistant turns for chat continuity."""
        if not conversation_history:
            return "(none)"

        recent = conversation_history[-6:]
        lines: list[str] = []
        for turn in recent:
            role = str(turn.get("role", "")).lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines) if lines else "(none)"
