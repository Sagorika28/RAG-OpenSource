"""
src/parsing/docling_parser.py — Docling-based PDF parser (primary).

Uses the Docling SDK to extract structured content (headings, paragraphs,
tables, lists) from PDFs.  Falls back to PyMuPDF if configured and
Docling fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from src.core.types import Document, Page
from src.parsing.base import BaseParser

logger = logging.getLogger(__name__)


class DoclingParser(BaseParser):
    """Primary PDF parser using Docling."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self._fallback_parser = None

        # Lazy-load fallback if enabled
        if self.config.get("fallback_enabled", False):
            try:
                from src.parsing.pymupdf_fallback import PyMuPDFParser
                self._fallback_parser = PyMuPDFParser(config)
                logger.info("PyMuPDF fallback parser loaded")
            except ImportError:
                logger.warning("PyMuPDF not installed — fallback disabled")


    def parse(self, pdf_path: str | Path) -> Document:
        """
        Parse a PDF using Docling. Falls back to PyMuPDF on failure
        OR if Docling returns empty content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            doc = self._parse_with_docling(pdf_path)
            # If Docling returned empty content, try fallback
            if not doc.raw_text.strip() and self._fallback_parser is not None:
                logger.warning(
                    f"Docling returned empty content for {pdf_path.name} "
                    f"— falling back to PyMuPDF"
                )
                return self._fallback_parser.parse(pdf_path)
            return doc
        except Exception as e:
            logger.warning(f"Docling failed on {pdf_path.name}: {e}")
            if self._fallback_parser is not None:
                logger.info(f"Falling back to PyMuPDF for {pdf_path.name}")
                return self._fallback_parser.parse(pdf_path)
            raise

    def _parse_with_docling(self, pdf_path: Path) -> Document:
        """
        Core Docling parsing logic.

        Docling v2's DocumentConverter produces a structured representation.
        iterate_items() yields (item, level) tuples.  We also use
        export_to_markdown() as a reliable full-text source.
        """
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc_result = result.document

        # --- Primary: get full markdown text (most reliable) ---
        try:
            raw_text = doc_result.export_to_markdown()
        except Exception:
            raw_text = ""

        # --- Extract structured elements via iterate_items() ---
        pages: list[Page] = []
        elements_by_page: dict[int, list[dict]] = {}

        try:
            for item_tuple in doc_result.iterate_items():
                # Docling v2 yields (item, level) tuples
                if isinstance(item_tuple, tuple):
                    element_obj, level = item_tuple
                else:
                    element_obj = item_tuple
                    level = 0

                # Extract text content
                text = ""
                if hasattr(element_obj, "text"):
                    text = element_obj.text or ""

                # Determine element type from class name
                class_name = type(element_obj).__name__.lower()
                if "heading" in class_name or "title" in class_name:
                    element_type = "heading"
                elif "list" in class_name:
                    element_type = "list"
                elif "table" in class_name:
                    element_type = "table"
                    # For tables, try markdown representation
                    if hasattr(element_obj, "export_to_markdown"):
                        try:
                            text = element_obj.export_to_markdown()
                        except Exception:
                            pass
                elif "caption" in class_name:
                    element_type = "caption"
                else:
                    element_type = "paragraph"

                # Get page number from provenance
                page_no = 0
                if hasattr(element_obj, "prov") and element_obj.prov:
                    for prov in element_obj.prov:
                        if hasattr(prov, "page_no"):
                            page_no = prov.page_no
                            break

                if text.strip():
                    element_dict = {
                        "type": element_type,
                        "text": text.strip(),
                        "page": page_no,
                    }
                    if element_type == "heading":
                        element_dict["level"] = (
                            getattr(element_obj, "level", level) or level
                        )
                    elements_by_page.setdefault(page_no, []).append(element_dict)

        except Exception as e:
            logger.warning(f"iterate_items() failed for {pdf_path.name}: {e}")

        # --- Build Page objects from structured elements ---
        if elements_by_page:
            for page_num in sorted(elements_by_page.keys()):
                elems = elements_by_page[page_num]
                page_text = "\n".join(e["text"] for e in elems)
                pages.append(Page(
                    page_number=page_num,
                    text=page_text,
                    elements=elems,
                ))
        elif raw_text.strip():
            # Fallback: no structured elements but we have markdown text.
            # Split by rough page markers or treat as single page.
            pages.append(Page(
                page_number=1,
                text=raw_text,
                elements=[{"type": "paragraph", "text": raw_text, "page": 1}],
            ))

        doc = Document(
            source=pdf_path.name,
            pages=pages,
            raw_text=raw_text,
            metadata={"parser": "docling", "num_pages": len(pages)},
        )

        logger.info(
            f"Docling parsed {pdf_path.name}: {len(pages)} pages, "
            f"{len(raw_text)} chars"
        )
        return doc
