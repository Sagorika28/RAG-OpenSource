"""
src/pipeline/query.py — Query pipeline.

Orchestrates: embed query → search Qdrant → (optional) rerank →
(optional) generate → return results with citations and timings.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from src.core.config import get_section
from src.core.types import Hit, QueryResult
from src.core.utils import timer

logger = logging.getLogger(__name__)


class QueryPipeline:
    """
    Full query pipeline: question → embed → search → rerank → generate.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._embedder = None
        self._store = None
        self._reranker = None
        self._generator = None

    # ------------------------------------------------------------------ #
    #  Component factories                                                #
    # ------------------------------------------------------------------ #

    def _get_embedder(self):
        if self._embedder is None:
            cfg = get_section(self.config, "embeddings")
            model_name = cfg.get("model_name", "")
            if "bge-m3" in model_name.lower():
                from src.embeddings.bge_m3 import BGEM3Embedder
                self._embedder = BGEM3Embedder(cfg)
            else:
                from src.embeddings.bge_small import BGESmallEmbedder
                self._embedder = BGESmallEmbedder(cfg)
        return self._embedder

    def _get_store(self):
        if self._store is None:
            from src.index.qdrant_store import QdrantStore
            self._store = QdrantStore(get_section(self.config, "qdrant"))
        return self._store

    def _get_reranker(self):
        if self._reranker is None:
            cfg = get_section(self.config, "reranker")
            if cfg.get("enabled", False):
                from src.rerank.bge_reranker import BGEReranker
                self._reranker = BGEReranker(cfg)
            else:
                from src.rerank.none import NoReranker
                self._reranker = NoReranker(cfg)
        return self._reranker

    def _get_generator(self):
        if self._generator is None:
            cfg = get_section(self.config, "generation")
            if not cfg.get("enabled", True):
                from src.generation.none import NoGenerator
                self._generator = NoGenerator(cfg)
            else:
                provider = cfg.get("provider", "ollama").lower()
                if provider == "groq":
                    from src.generation.groq_llm import GroqGenerator
                    self._generator = GroqGenerator(cfg)
                else:
                    from src.generation.ollama_llm import OllamaGenerator
                    self._generator = OllamaGenerator(cfg)
        return self._generator

    # ------------------------------------------------------------------ #
    #  Query rewriting                                                     #
    # ------------------------------------------------------------------ #

    REWRITE_PROMPT = (
        "You are a query expansion engine for a technical HVAC/refrigerant knowledge base. "
        "A field technician has typed a short, vague query. Your job is to rewrite it into "
        "a richer search query that will retrieve the most relevant technical procedures, "
        "safety guidelines, and specifications.\n\n"
        "Rules:\n"
        "- Keep the original intent.\n"
        "- Add related technical terms: procedures, tools, refrigerant types, safety steps.\n"
        "- Output ONLY the rewritten query (one paragraph, no explanation).\n"
        "- Do NOT answer the question, just expand the search terms.\n"
    )

    def _rewrite_query(self, query: str) -> str:
        """
        Use LLM to expand a vague user query into a richer search query.

        Uses the fast 8B model for speed (~200ms via Groq).
        Falls back to original query on any error.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return query

        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.REWRITE_PROMPT},
                    {"role": "user", "content": f"Technician query: {query}"},
                ],
                model="llama-3.1-8b-instant",  # Fast model for rewriting
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = resp.choices[0].message.content.strip()
            # Sanity check: if rewrite is too long or empty, use original
            if not rewritten or len(rewritten) > 500:
                return query
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            return query

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        query: str,
        top_k: Optional[int] = None,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
        skip_generation: bool = False,
    ) -> QueryResult:
        """
        Run the full query pipeline.

        Args:
            query:           User's question.
            top_k:           Override retrieval top_k from config.
            doc_type:        Optional metadata filter by doc_type.
            source:          Optional metadata filter by source filename.
            skip_generation: If True, skip the LLM generation step.

        Returns:
            QueryResult with hits, answer, citations, and timings.
        """
        timings: Dict[str, float] = {}
        retrieval_k = top_k or self.config.get("retrieval", {}).get("top_k", 10)

        # 0. Query rewriting (expand vague queries for better retrieval)
        search_query = query
        if self.config.get("retrieval", {}).get("query_rewrite", False):
            with timer("rewrite", timings):
                search_query = self._rewrite_query(query)
                if search_query != query:
                    logger.info(f"Query rewritten: '{query}' → '{search_query}'")

        # 1. Embed query (use rewritten version for search)
        embedder = self._get_embedder()
        with timer("embed", timings):
            if hasattr(embedder, "embed_query"):
                query_vector = embedder.embed_query(search_query)
            else:
                query_vector = embedder.embed([search_query])

        # 2. Search Qdrant
        store = self._get_store()
        with timer("search", timings):
            hits: List[Hit] = store.search(
                query_vector=query_vector,
                top_k=retrieval_k,
                doc_type=doc_type,
                source=source,
            )

        # 3. Rerank (optional) — use rewritten query for better ranking
        reranker = self._get_reranker()
        with timer("rerank", timings):
            hits = reranker.rerank(search_query, hits)

        # 3.5 Deduplicate near-identical chunks
        dedup_threshold = self.config.get("retrieval", {}).get(
            "dedup_threshold", 0.9
        )
        if dedup_threshold < 1.0:
            pre_dedup = len(hits)
            hits = self._deduplicate_hits(hits, dedup_threshold)
            if len(hits) < pre_dedup:
                logger.info(
                    f"Dedup removed {pre_dedup - len(hits)} near-duplicate chunks"
                )

        # 4. Generate answer (optional)
        answer = ""
        if not skip_generation:
            generator = self._get_generator()
            with timer("generate", timings):
                answer = generator.generate(query, hits)

        # 5. Build citations
        citations = self._build_citations(hits)

        return QueryResult(
            query=query,
            hits=hits,
            answer=answer,
            citations=citations,
            timings=timings,
        )

    @staticmethod
    def _deduplicate_hits(
        hits: List[Hit], threshold: float = 0.9
    ) -> List[Hit]:
        """
        Remove near-duplicate chunks using Jaccard word-set similarity.

        Keeps the first (highest-scored) chunk and drops later chunks
        whose word overlap with any already-kept chunk exceeds `threshold`.
        """
        if not hits:
            return hits

        def _word_set(text: str) -> set:
            return set(text.lower().split())

        def _jaccard(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        kept: List[Hit] = []
        kept_words: List[set] = []

        for hit in hits:
            text = hit.chunk.raw_text or hit.chunk.text
            words = _word_set(text)
            is_dup = any(
                _jaccard(words, kw) >= threshold for kw in kept_words
            )
            if not is_dup:
                kept.append(hit)
                kept_words.append(words)

        return kept

    @staticmethod
    def _build_citations(hits: List[Hit]) -> List[Dict[str, Any]]:
        """Extract citation info from hits for display."""
        citations: List[Dict[str, Any]] = []
        for i, hit in enumerate(hits, 1):
            citations.append({
                "rank": i,
                "source": hit.chunk.metadata.get("source", "unknown"),
                "doc_type": hit.chunk.metadata.get("doc_type", ""),
                "section": hit.chunk.metadata.get("section_path", ""),
                "pages": f"{hit.chunk.metadata.get('page_start', '?')}-{hit.chunk.metadata.get('page_end', '?')}",
                "chunk_id": hit.chunk.chunk_id,
                "score": round(hit.score, 4),
                "rerank_score": (
                    round(hit.rerank_score, 4)
                    if hit.rerank_score is not None
                    else None
                ),
            })
        return citations


# --------------------------------------------------------------------- #
#  CLI entry point                                                       #
# --------------------------------------------------------------------- #

def main():
    """Run a single query from the command line."""
    import argparse
    import json

    from src.core.config import load_config
    from src.core.utils import setup_logging
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query the RAG-OS pipeline"
    )
    parser.add_argument("query", help="Question to ask")
    parser.add_argument(
        "--config", "-c",
        default="configs/cpu.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--top-k", "-k", type=int, default=None)
    parser.add_argument("--no-generate", action="store_true")
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    pipeline = QueryPipeline(config)
    result = pipeline.run(
        query=args.query,
        top_k=args.top_k,
        skip_generation=args.no_generate,
    )

    print(f"\n{'='*60}")
    print(f"Query: {result.query}")
    print(f"{'='*60}")

    print(f"\n--- Retrieved Chunks ({len(result.hits)}) ---")
    for cit in result.citations:
        print(
            f"  [{cit['rank']}] {cit['source']} | "
            f"{cit['section']} | pp {cit['pages']} | "
            f"score={cit['score']}"
        )

    if result.answer:
        print(f"\n--- Answer ---\n{result.answer}")

    print(f"\n--- Timings ---")
    for k, v in result.timings.items():
        print(f"  {k}: {v:.1f}")


if __name__ == "__main__":
    main()
