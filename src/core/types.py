"""
src/core/types.py — Core data classes used throughout the RAG-OS pipeline.

These are plain dataclasses (no heavy dependencies) so every module can
import them without pulling in torch / qdrant / etc.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Page:
    """Represents a single page extracted from a PDF."""
    page_number: int
    text: str
    # Structured elements extracted by Docling (headings, lists, tables, etc.)
    elements: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Document:
    """A parsed PDF document with its raw content and metadata."""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""                    # Original filename / path
    doc_type: str = "unknown"           # manual | standard | newsletter | booklet | flyer | slides | unknown
    pages: List[Page] = field(default_factory=list)
    raw_text: str = ""                  # Full concatenated text (convenience)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk of text ready for embedding and storage."""
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str = ""
    text: str = ""                      # Chunk text WITH context header prepended
    raw_text: str = ""                  # Chunk text WITHOUT context header
    context_header: str = ""            # e.g. "[Manual | Safety Guide | §3.2 | pp 4-5]"
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Expected metadata keys:
    #   source, doc_type, section_path, page_start, page_end,
    #   chunk_index, chunk_type


@dataclass
class Hit:
    """A search result from the vector store, optionally reranked."""
    chunk: Chunk
    score: float = 0.0                  # Vector similarity score
    rerank_score: Optional[float] = None


@dataclass
class QueryResult:
    """Full result from the query pipeline."""
    query: str = ""
    hits: List[Hit] = field(default_factory=list)
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)  # embed_ms, search_ms, rerank_ms, gen_ms


@dataclass
class EvalExample:
    """A single evaluation entry (from JSONL)."""
    question: str = ""
    gold_doc_ids: List[str] = field(default_factory=list)     # Ground-truth doc IDs
    gold_sources: List[str] = field(default_factory=list)     # Ground-truth source filenames
    gold_chunk_ids: List[str] = field(default_factory=list)   # Optional ground-truth chunk IDs
    gold_answer: Optional[str] = None                          # Optional reference answer
    metadata: Dict[str, Any] = field(default_factory=dict)    # Extra info (page, section)
