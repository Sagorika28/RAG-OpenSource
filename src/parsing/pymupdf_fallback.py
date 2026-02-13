"""
src/parsing/pymupdf_fallback.py — PyMuPDF fallback parser.

Uses PyMuPDF (fitz) to extract text page-by-page.  Simpler than Docling
but more robust for edge-case PDFs.  AGPL-licensed — keep optional.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.core.types import Document, Page
from src.parsing.base import BaseParser

logger = logging.getLogger(__name__)


class PyMuPDFParser(BaseParser):
    """Fallback PDF parser using PyMuPDF (fitz)."""

    def parse(self, pdf_path: str | Path) -> Document:
        """
        Parse a PDF using PyMuPDF — extracts text per page.

        Tries to detect headings via font-size heuristics so downstream
        chunking still gets some structural signals.
        """
        import fitz  # PyMuPDF

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf = fitz.open(str(pdf_path))
        pages: list[Page] = []
        full_text_parts: list[str] = []

        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_number = page_idx + 1

            # Extract text blocks for structural hints
            text = page.get_text("text")
            elements = self._extract_elements(page, page_number)

            pages.append(Page(
                page_number=page_number,
                text=text,
                elements=elements,
            ))
            if text.strip():
                full_text_parts.append(text.strip())

        pdf.close()

        raw_text = "\n\n".join(full_text_parts)
        doc = Document(
            source=pdf_path.name,
            pages=pages,
            raw_text=raw_text,
            metadata={"parser": "pymupdf", "num_pages": len(pages)},
        )

        logger.info(
            f"PyMuPDF parsed {pdf_path.name}: {len(pages)} pages, "
            f"{len(raw_text)} chars"
        )
        return doc

    @staticmethod
    def _extract_elements(page, page_number: int) -> list[dict]:
        """
        Extract structured elements from a PyMuPDF page using text blocks
        and font-size heuristics to identify potential headings.
        """
        elements: list[dict] = []

        # Use get_text("dict") for richer block info
        try:
            page_dict = page.get_text("dict")
        except Exception:
            # Fallback: just return plain text as a single element
            text = page.get_text("text")
            if text.strip():
                elements.append({
                    "type": "paragraph",
                    "text": text.strip(),
                    "page": page_number,
                })
            return elements

        # Collect font sizes to determine "large" (heading) thresholds
        font_sizes: list[float] = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("text", "").strip():
                        font_sizes.append(span["size"])

        if not font_sizes:
            return elements

        median_size = sorted(font_sizes)[len(font_sizes) // 2]
        heading_threshold = median_size * 1.15  # 15% larger → likely heading

        # Build elements from blocks
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            block_text_parts: list[str] = []
            max_font_size = 0.0
            is_bold = False

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    line_text += span_text
                    if span.get("size", 0) > max_font_size:
                        max_font_size = span["size"]
                    if "bold" in span.get("font", "").lower():
                        is_bold = True
                if line_text.strip():
                    block_text_parts.append(line_text.strip())

            block_text = "\n".join(block_text_parts)
            if not block_text.strip():
                continue

            # Classify element type
            if max_font_size >= heading_threshold or (
                is_bold and len(block_text) < 120
            ):
                element_type = "heading"
            elif block_text.strip().startswith(("•", "-", "–", "▪", "●")):
                element_type = "list"
            elif any(
                block_text.strip().startswith(f"{i}.")
                for i in range(1, 20)
            ):
                element_type = "list"
            else:
                element_type = "paragraph"

            elements.append({
                "type": element_type,
                "text": block_text.strip(),
                "page": page_number,
            })

        return elements
