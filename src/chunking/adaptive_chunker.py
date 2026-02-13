"""
src/chunking/adaptive_chunker.py — Adaptive structural chunking.

Strategy varies by document type:
  - manual / standard  → section-aware chunking with no-split rules
  - slides / newsletter → block/page chunking, keep bullet groups
  - flyer / one-pager   → page-level chunks with optional sub-splits
  - unknown             → fallback to fixed-size with overlap

Context headers are prepended to each chunk so the embedding model
gets richer context: [doc_title | doc_type | section_path | pp X-Y].
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from src.core.types import Chunk, Document, Page
from src.core.utils import generate_id
from src.chunking.base import BaseChunker

logger = logging.getLogger(__name__)

# Patterns for "no-split" blocks (warnings, cautions, numbered steps)
_WARNING_RE = re.compile(
    r"^(WARNING|CAUTION|DANGER|NOTE|IMPORTANT)\s*[:!]",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBERED_STEP_RE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


class AdaptiveChunker(BaseChunker):
    """
    Chunks documents adaptively based on document type and structure.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.max_chunk_size: int = self.config.get("max_chunk_size", 512)
        self.min_chunk_size: int = self.config.get("min_chunk_size", 100)
        self.overlap: int = self.config.get("overlap", 50)
        self.use_context_header: bool = self.config.get("context_header", True)

    # --------------------------------------------------------------------- #
    #  Public API                                                            #
    # --------------------------------------------------------------------- #

    def chunk(self, document: Document) -> List[Chunk]:
        """Route to the appropriate chunking strategy based on doc_type."""
        doc_type = document.doc_type.lower()

        if doc_type in ("manual", "standard"):
            raw_chunks = self._chunk_structured(document)
        elif doc_type in ("slides", "newsletter", "booklet"):
            raw_chunks = self._chunk_block(document)
        elif doc_type in ("flyer",):
            raw_chunks = self._chunk_page_level(document)
        else:
            # Unknown → use fallback fixed-size chunking
            raw_chunks = self._chunk_fixed_size(document)

        logger.info(
            f"Chunked {document.source} ({doc_type}): "
            f"{len(raw_chunks)} chunks"
        )
        return raw_chunks

    # --------------------------------------------------------------------- #
    #  Strategy 1: Section-aware (manuals, standards)                        #
    # --------------------------------------------------------------------- #

    def _chunk_structured(self, doc: Document) -> List[Chunk]:
        """
        Section-aware chunking for structured docs (manuals, standards).

        Groups elements by their heading hierarchy.  Keeps warnings,
        cautions, numbered step sequences, and tables intact (no-split).
        Merges small sections up to max_chunk_size.
        """
        sections = self._split_by_headings(doc)

        # If only 1 section found (no real headings detected), fall through
        # to fixed-size chunking — it's more reliable than one giant chunk.
        if len(sections) <= 1:
            text = doc.raw_text or "\n".join(p.text for p in doc.pages)
            if text.strip():
                return self._chunk_fixed_size(doc)
            return []

        chunks: List[Chunk] = []

        for section_path, section_texts, page_range in sections:
            combined = "\n".join(section_texts)
            # If small enough, emit as a single chunk
            if len(combined) <= self.max_chunk_size:
                if combined.strip():
                    chunks.append(self._make_chunk(
                        doc=doc,
                        text=combined,
                        section_path=section_path,
                        page_start=page_range[0],
                        page_end=page_range[1],
                        chunk_index=len(chunks),
                        chunk_type="section",
                    ))
            else:
                # Sub-split respecting no-split blocks
                sub_parts = self._split_respecting_blocks(combined)
                for part in sub_parts:
                    if part.strip():
                        chunks.append(self._make_chunk(
                            doc=doc,
                            text=part.strip(),
                            section_path=section_path,
                            page_start=page_range[0],
                            page_end=page_range[1],
                            chunk_index=len(chunks),
                            chunk_type="section_part",
                        ))

        return chunks if chunks else self._chunk_fixed_size(doc)

    # Regex for markdown headings (# Title, ## Subtitle, etc.)
    _MD_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    def _split_by_headings(
        self, doc: Document
    ) -> List[tuple[str, List[str], tuple[int, int]]]:
        """
        Split document into sections by headings.

        Uses two strategies:
          1. Structured elements with type=heading (from parser).
          2. Markdown heading patterns (# Title) in raw_text (fallback).
        """
        # --- Strategy 1: element-level headings ---
        has_heading_elements = any(
            elem.get("type") == "heading"
            for page in doc.pages
            for elem in page.elements
        )

        if has_heading_elements:
            return self._split_by_element_headings(doc)

        # --- Strategy 2: markdown heading patterns in raw_text ---
        if doc.raw_text:
            md_sections = self._split_by_markdown_headings(doc)
            if len(md_sections) > 1:
                return md_sections

        # --- No headings found at all → return single section ---
        all_texts = [
            elem.get("text", "")
            for page in doc.pages
            for elem in page.elements
        ]
        p_start = doc.pages[0].page_number if doc.pages else 1
        p_end = doc.pages[-1].page_number if doc.pages else 1
        return [("Introduction", all_texts, (p_start, p_end))]

    def _split_by_element_headings(
        self, doc: Document
    ) -> List[tuple[str, List[str], tuple[int, int]]]:
        """Split by parsed heading elements."""
        sections: List[tuple[str, List[str], tuple[int, int]]] = []
        current_path = "Introduction"
        current_texts: List[str] = []
        page_start = 1
        page_end = 1

        for page in doc.pages:
            for elem in page.elements:
                if elem.get("type") == "heading":
                    if current_texts:
                        sections.append(
                            (current_path, current_texts, (page_start, page_end))
                        )
                    current_path = elem["text"]
                    current_texts = []
                    page_start = page.page_number
                    page_end = page.page_number
                else:
                    current_texts.append(elem.get("text", ""))
                    page_end = page.page_number

        if current_texts:
            sections.append(
                (current_path, current_texts, (page_start, page_end))
            )
        return sections

    def _split_by_markdown_headings(
        self, doc: Document
    ) -> List[tuple[str, List[str], tuple[int, int]]]:
        """
        Split raw_text (markdown) by # heading lines.
        Returns (heading_text, [body_lines], (page_start, page_end)) tuples.
        """
        lines = doc.raw_text.split("\n")
        sections: List[tuple[str, List[str], tuple[int, int]]] = []
        current_heading = "Introduction"
        current_body: List[str] = []
        total_pages = len(doc.pages) or 1

        for line in lines:
            match = self._MD_HEADING_RE.match(line)
            if match:
                # Flush previous
                if current_body:
                    body_text = "\n".join(current_body)
                    if body_text.strip():
                        sections.append(
                            (current_heading, [body_text], (1, total_pages))
                        )
                current_heading = match.group(2).strip()
                current_body = []
            else:
                current_body.append(line)

        if current_body:
            body_text = "\n".join(current_body)
            if body_text.strip():
                sections.append(
                    (current_heading, [body_text], (1, total_pages))
                )
        return sections

    def _split_respecting_blocks(self, text: str) -> List[str]:
        """
        Split text into chunks of ~max_chunk_size, but never split inside
        WARNING/CAUTION blocks, numbered-step sequences, or table rows.

        Handles text with both single and double newline separators.
        """
        # Try double-newline split first; fall back to single newline
        paragraphs = re.split(r"\n{2,}", text)
        if len(paragraphs) <= 1 and len(text) > self.max_chunk_size:
            # Text has no double-newlines — split on single newlines
            paragraphs = text.split("\n")

        chunks: List[str] = []
        buffer: List[str] = []
        buffer_len = 0
        in_protected_block = False

        for para in paragraphs:
            if not para.strip():
                continue

            # Detect protected blocks
            is_protected = bool(
                _WARNING_RE.search(para) or
                _NUMBERED_STEP_RE.match(para) or
                "|" in para  # crude table detection
            )

            if is_protected:
                in_protected_block = True

            if (
                buffer_len + len(para) > self.max_chunk_size
                and buffer
                and not in_protected_block
            ):
                chunks.append("\n\n".join(buffer))
                # Keep overlap
                overlap_text = buffer[-1] if buffer else ""
                buffer = [overlap_text] if len(overlap_text) <= self.overlap else []
                buffer_len = sum(len(b) for b in buffer)

            buffer.append(para)
            buffer_len += len(para)

            # End protected block after adding it
            if is_protected and not _NUMBERED_STEP_RE.match(para):
                in_protected_block = False

        if buffer:
            chunks.append("\n\n".join(buffer))

        return chunks

    # --------------------------------------------------------------------- #
    #  Strategy 2: Block/page chunking (slides, newsletters, booklets)       #
    # --------------------------------------------------------------------- #

    def _chunk_block(self, doc: Document) -> List[Chunk]:
        """
        Block/page chunking for slides and newsletters.
        Keeps bullet groups together per page.
        """
        chunks: List[Chunk] = []

        for page in doc.pages:
            # Group elements into blocks: headings start new blocks
            blocks: List[List[dict]] = []
            current_block: List[dict] = []

            for elem in page.elements:
                if elem.get("type") == "heading" and current_block:
                    blocks.append(current_block)
                    current_block = [elem]
                else:
                    current_block.append(elem)

            if current_block:
                blocks.append(current_block)

            # If no element-level structure, use full page text
            if not blocks:
                if page.text.strip():
                    chunks.append(self._make_chunk(
                        doc=doc,
                        text=page.text.strip(),
                        section_path="",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_index=len(chunks),
                        chunk_type="page",
                    ))
                continue

            # Merge small adjacent blocks
            merged = self._merge_small_blocks(blocks)
            for block_elems in merged:
                block_text = "\n".join(e.get("text", "") for e in block_elems)
                section = next(
                    (e["text"] for e in block_elems if e.get("type") == "heading"),
                    "",
                )
                if block_text.strip():
                    chunks.append(self._make_chunk(
                        doc=doc,
                        text=block_text.strip(),
                        section_path=section,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_index=len(chunks),
                        chunk_type="block",
                    ))

        return chunks if chunks else self._chunk_fixed_size(doc)

    def _merge_small_blocks(
        self, blocks: List[List[dict]]
    ) -> List[List[dict]]:
        """Merge adjacent blocks that are smaller than min_chunk_size."""
        merged: List[List[dict]] = []
        buffer: List[dict] = []
        buffer_len = 0

        for block in blocks:
            block_len = sum(len(e.get("text", "")) for e in block)
            if buffer_len + block_len <= self.max_chunk_size:
                buffer.extend(block)
                buffer_len += block_len
            else:
                if buffer:
                    merged.append(buffer)
                buffer = list(block)
                buffer_len = block_len

        if buffer:
            merged.append(buffer)

        return merged

    # --------------------------------------------------------------------- #
    #  Strategy 3: Page-level (flyers, one-pagers)                           #
    # --------------------------------------------------------------------- #

    def _chunk_page_level(self, doc: Document) -> List[Chunk]:
        """
        Page-level chunks for flyers/one-pagers.
        Optionally sub-splits if page text exceeds max_chunk_size.
        """
        chunks: List[Chunk] = []

        for page in doc.pages:
            text = page.text.strip()
            if not text:
                continue

            if len(text) <= self.max_chunk_size:
                chunks.append(self._make_chunk(
                    doc=doc,
                    text=text,
                    section_path="",
                    page_start=page.page_number,
                    page_end=page.page_number,
                    chunk_index=len(chunks),
                    chunk_type="page",
                ))
            else:
                # Sub-split by paragraphs
                parts = self._fixed_split(text)
                for part in parts:
                    chunks.append(self._make_chunk(
                        doc=doc,
                        text=part,
                        section_path="",
                        page_start=page.page_number,
                        page_end=page.page_number,
                        chunk_index=len(chunks),
                        chunk_type="page_part",
                    ))

        return chunks if chunks else self._chunk_fixed_size(doc)

    # --------------------------------------------------------------------- #
    #  Fallback: fixed-size chunking                                         #
    # --------------------------------------------------------------------- #

    def _chunk_fixed_size(self, doc: Document) -> List[Chunk]:
        """
        Simple fixed-size chunking with overlap.
        Used as a last resort when no structural info is available.
        """
        text = doc.raw_text or "\n".join(p.text for p in doc.pages)
        if not text.strip():
            return []

        parts = self._fixed_split(text)
        chunks: List[Chunk] = []
        for part in parts:
            chunks.append(self._make_chunk(
                doc=doc,
                text=part,
                section_path="",
                page_start=1,
                page_end=len(doc.pages) or 1,
                chunk_index=len(chunks),
                chunk_type="fixed",
            ))
        return chunks

    def _fixed_split(self, text: str) -> List[str]:
        """Split text into chunks of max_chunk_size with overlap."""
        parts: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_size
            chunk_text = text[start:end]
            # Try to break at a sentence/paragraph boundary
            if end < len(text):
                last_break = max(
                    chunk_text.rfind("\n\n"),
                    chunk_text.rfind(". "),
                    chunk_text.rfind(".\n"),
                )
                if last_break > self.min_chunk_size:
                    chunk_text = chunk_text[: last_break + 1]
                    end = start + last_break + 1

            if chunk_text.strip():
                parts.append(chunk_text.strip())
            start = max(end - self.overlap, start + 1)

        return parts

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #

    def _make_chunk(
        self,
        doc: Document,
        text: str,
        section_path: str,
        page_start: int,
        page_end: int,
        chunk_index: int,
        chunk_type: str,
    ) -> Chunk:
        """Build a Chunk object, optionally prepending a context header."""
        # Build context header
        header_parts = []
        if doc.source:
            header_parts.append(doc.source)
        if doc.doc_type and doc.doc_type != "unknown":
            header_parts.append(doc.doc_type)
        if section_path:
            header_parts.append(section_path)
        if page_start:
            page_str = (
                f"p{page_start}" if page_start == page_end
                else f"pp{page_start}-{page_end}"
            )
            header_parts.append(page_str)

        context_header = " | ".join(header_parts)
        full_text = f"[{context_header}]\n{text}" if (
            self.use_context_header and context_header
        ) else text

        return Chunk(
            chunk_id=generate_id(),
            doc_id=doc.doc_id,
            text=full_text,
            raw_text=text,
            context_header=context_header,
            metadata={
                "source": doc.source,
                "doc_type": doc.doc_type,
                "section_path": section_path,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_index": chunk_index,
                "chunk_type": chunk_type,
            },
        )
