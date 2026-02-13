"""
src/eval/llm_judge.py — LLM-as-a-Judge for answer quality evaluation.

Uses Groq API to score generated answers on three dimensions:
  - Faithfulness: Is the answer grounded in the provided context?
  - Relevance:    Does the answer address the user's question?
  - Completeness: Does the answer cover key information from context?

Each dimension is scored 0-5. Returns an overall average score.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from src.core.types import Hit

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

You will be given:
1. A user QUESTION
2. CONTEXT (retrieved source chunks)
3. An AI-generated ANSWER

IMPORTANT:
- Evaluate the ANSWER using ONLY the provided CONTEXT.
- Do NOT use outside knowledge.
- Do NOT assume missing information.
- If the ANSWER contains any claim not explicitly supported by the CONTEXT, it must reduce the Faithfulness score.

Score the ANSWER on three dimensions (0-5 scale):

Faithfulness (0-5):
5 = Every factual claim is directly supported by the CONTEXT.
3 = Minor unsupported details or mild inference.
1 = Significant unsupported claims.
0 = Mostly or entirely hallucinated.

Relevance (0-5):
5 = Directly and precisely answers the QUESTION.
3 = Partially answers but includes unnecessary or tangential information.
1 = Barely addresses the question.
0 = Completely off-topic.

Completeness (0-5):
5 = Covers all key information present in the CONTEXT that is needed to answer the QUESTION.
3 = Covers some but misses important details.
1 = Very incomplete.
0 = Fails to provide meaningful coverage.

Respond ONLY with valid JSON in this exact format (no extra text, no markdown):

{"faithfulness": <int>, "relevance": <int>, "completeness": <int>, "justification": "<one concise sentence explaining the scores>"}
"""


class LLMJudge:
    """
    Evaluates generated answers using an LLM (Groq API).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found — LLM Judge will be disabled")
            self.client = None
            return

        from groq import Groq
        self.client = Groq(api_key=self.api_key)
        # Use a fast model for judging (cost-effective)
        self.model = (config or {}).get("judge_model", "llama-3.1-8b-instant")

    def evaluate(
        self,
        question: str,
        context_hits: List[Hit],
        answer: str,
    ) -> Dict[str, Any]:
        """
        Score a generated answer.

        Returns:
            Dict with faithfulness, relevance, completeness (0-5), overall (avg), justification.
        """
        if not self.client or not answer.strip():
            return self._empty_scores()

        # Build context string
        context_str = self._format_context(context_hits)

        user_content = (
            f"QUESTION:\n{question}\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"ANSWER:\n{answer}"
        )

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=self.model,
                temperature=0.0,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()
            scores = self._parse_scores(raw)
            return scores

        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            return self._empty_scores()

    def evaluate_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate a batch of results and return averaged scores.

        Args:
            results: List of dicts with keys: question, hits, answer.

        Returns:
            Averaged scores: {faithfulness, relevance, completeness, overall}.
        """
        all_scores: List[Dict[str, Any]] = []

        for r in results:
            scores = self.evaluate(
                question=r["question"],
                context_hits=r["hits"],
                answer=r["answer"],
            )
            all_scores.append(scores)

        return self._aggregate_scores(all_scores)

    @staticmethod
    def _format_context(hits: List[Hit]) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            text = hit.chunk.raw_text or hit.chunk.text
            parts.append(f"[{i}] {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_scores(raw: str) -> Dict[str, Any]:
        """Parse JSON scores from LLM response."""
        # Try to extract JSON from the response
        try:
            # Handle cases where LLM wraps JSON in markdown code blocks
            json_match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(raw)

            faithfulness = int(data.get("faithfulness", 0))
            relevance = int(data.get("relevance", 0))
            completeness = int(data.get("completeness", 0))

            # Clamp to 0-5
            faithfulness = max(0, min(5, faithfulness))
            relevance = max(0, min(5, relevance))
            completeness = max(0, min(5, completeness))

            overall = round((faithfulness + relevance + completeness) / 3, 2)

            return {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness,
                "overall": overall,
                "justification": data.get("justification", ""),
            }

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse judge scores: {e} | raw: {raw[:200]}")
            return LLMJudge._empty_scores()

    @staticmethod
    def _empty_scores() -> Dict[str, Any]:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "overall": 0.0,
            "justification": "evaluation skipped",
        }

    @staticmethod
    def _aggregate_scores(all_scores: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average scores across all evaluations."""
        n = len(all_scores)
        if n == 0:
            return {"faithfulness": 0, "relevance": 0, "completeness": 0, "overall": 0}

        return {
            "faithfulness": round(sum(s["faithfulness"] for s in all_scores) / n, 2),
            "relevance": round(sum(s["relevance"] for s in all_scores) / n, 2),
            "completeness": round(sum(s["completeness"] for s in all_scores) / n, 2),
            "overall": round(sum(s["overall"] for s in all_scores) / n, 2),
        }
