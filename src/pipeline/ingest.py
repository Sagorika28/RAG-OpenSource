"""
src/pipeline/ingest.py — Ingestion pipeline.

Orchestrates: parse → enrich metadata → chunk → embed → upsert to Qdrant.
Processes all PDFs in a directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.core.config import load_config, get_section
from src.core.types import Chunk, Document
from src.core.utils import setup_logging, timer

logger = logging.getLogger(__name__)


class IngestPipeline:
    """
    Full ingestion pipeline: PDF → parse → metadata → chunk → embed → store.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._parser = None
        self._enricher = None
        self._chunker = None
        self._embedder = None
        self._store = None

    # ------------------------------------------------------------------ #
    #  Component factories (lazy initialization)                          #
    # ------------------------------------------------------------------ #

    def _get_parser(self):
        if self._parser is None:
            cfg = get_section(self.config, "parsing")
            primary = cfg.get("primary", "docling")
            if primary == "docling":
                from src.parsing.docling_parser import DoclingParser
                self._parser = DoclingParser(cfg)
            else:
                from src.parsing.pymupdf_fallback import PyMuPDFParser
                self._parser = PyMuPDFParser(cfg)
        return self._parser

    def _get_enricher(self):
        if self._enricher is None:
            from src.metadata.rules_metadata import RulesMetadataEnricher
            self._enricher = RulesMetadataEnricher()
        return self._enricher

    def _get_chunker(self):
        if self._chunker is None:
            from src.chunking.adaptive_chunker import AdaptiveChunker
            self._chunker = AdaptiveChunker(
                get_section(self.config, "chunking")
            )
        return self._chunker

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
            self._store = QdrantStore(
                get_section(self.config, "qdrant")
            )
        return self._store

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def run(
        self,
        pdf_dir: str | Path | None = None,
        recreate_collection: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest all PDFs from a directory.

        Args:
            pdf_dir: Directory containing PDFs.  Falls back to
                     config["data"]["pdf_dir"] if not specified.
            recreate_collection: If True, drop and recreate the Qdrant
                                 collection before ingesting.

        Returns:
            Stats dict with doc count, chunk count, timings, etc.
        """
        pdf_dir = Path(
            pdf_dir or self.config.get("data", {}).get("pdf_dir", "./data/pdfs")
        )
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {"docs": 0, "chunks": 0}

        logger.info(f"Found {len(pdf_files)} PDFs in {pdf_dir}")

        # Initialize components
        parser = self._get_parser()
        enricher = self._get_enricher()
        chunker = self._get_chunker()
        embedder = self._get_embedder()
        store = self._get_store()

        # Create / recreate collection
        store.create_collection(
            dimension=embedder.dimension,
            recreate=recreate_collection,
        )

        # Process each PDF
        stats: Dict[str, Any] = {
            "docs_processed": 0,
            "docs_failed": 0,
            "total_chunks": 0,
            "total_chars": 0,
            "timings": {},
        }
        all_timings: Dict[str, float] = {}

        for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs"):
            try:
                self._ingest_one(
                    pdf_path, parser, enricher, chunker, embedder, store,
                    all_timings, stats,
                )
                stats["docs_processed"] += 1
            except Exception as e:
                logger.error(f"Failed to ingest {pdf_path.name}: {e}")
                stats["docs_failed"] += 1

        stats["timings"] = all_timings
        stats["index_stats"] = store.get_stats()

        logger.info(
            f"Ingestion complete: {stats['docs_processed']} docs, "
            f"{stats['total_chunks']} chunks"
        )
        return stats

    def _ingest_one(
        self,
        pdf_path: Path,
        parser,
        enricher,
        chunker,
        embedder,
        store,
        timings: Dict[str, float],
        stats: Dict[str, Any],
    ) -> None:
        """Process a single PDF through the full pipeline."""
        # 1. Parse
        with timer("parse", timings):
            doc: Document = parser.parse(pdf_path)

        # 2. Enrich metadata
        with timer("enrich", timings):
            doc = enricher.enrich(doc)

        # 3. Chunk
        with timer("chunk", timings):
            chunks: List[Chunk] = chunker.chunk(doc)

        if not chunks:
            logger.warning(f"No chunks produced for {pdf_path.name}")
            return

        # 4. Embed
        chunk_texts = [c.text for c in chunks]
        with timer("embed", timings):
            vectors = embedder.embed(chunk_texts)

        # 5. Upsert to Qdrant
        with timer("upsert", timings):
            store.upsert_chunks(chunks, vectors)

        stats["total_chunks"] += len(chunks)
        stats["total_chars"] += sum(len(c.text) for c in chunks)

        logger.info(
            f"  {pdf_path.name}: type={doc.doc_type}, "
            f"chunks={len(chunks)}"
        )


# --------------------------------------------------------------------- #
#  CLI entry point                                                       #
# --------------------------------------------------------------------- #

def main():
    """Run the ingest pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the RAG-OS vector store"
    )
    parser.add_argument(
        "--config", "-c",
        default="configs/cpu.yaml",
        help="Path to config YAML (default: configs/cpu.yaml)",
    )
    parser.add_argument(
        "--pdf-dir", "-d",
        default=None,
        help="Directory containing PDFs (overrides config)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    pipeline = IngestPipeline(config)
    stats = pipeline.run(
        pdf_dir=args.pdf_dir,
        recreate_collection=args.recreate,
    )

    import json
    print("\n=== Ingestion Stats ===")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
