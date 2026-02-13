"""
src/metadata/rules_metadata.py — Rule-based metadata enricher.

Classifies document type using heuristics on:
  1. Filename patterns (e.g. "manual", "std", "newsletter")
  2. Page count
  3. Content patterns (heading density, bullet density, etc.)

Sets doc.doc_type to one of:
  manual | standard | newsletter | booklet | flyer | slides | unknown
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict

from src.core.types import Document
from src.metadata.base import BaseMetadataEnricher

logger = logging.getLogger(__name__)

# Filename-based patterns → doc_type mapping
_FILENAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"manual|technician|service|repair|maintenance|guide|handbook", re.I), "manual"),
    (re.compile(r"standard|compliance|regulation|spec|code|iso|ansi|nfpa|ashrae", re.I), "standard"),
    (re.compile(r"newsletter|bulletin|update|digest", re.I), "newsletter"),
    (re.compile(r"booklet|brochure|catalog|catalogue", re.I), "booklet"),
    (re.compile(r"flyer|flysheet|leaflet|one[_-]?pager|fact[_-]?sheet", re.I), "flyer"),
    (re.compile(r"slide|presentation|ppt|deck", re.I), "slides"),
]


class RulesMetadataEnricher(BaseMetadataEnricher):
    """Classify document type and enrich metadata using heuristics."""

    def enrich(self, document: Document) -> Document:
        """
        Classify doc_type and add metadata to the document.
        """
        # --- 1. Try filename pattern matching ---
        doc_type = self._classify_by_filename(document.source)

        # --- 2. Fallback to content-based heuristics ---
        if doc_type == "unknown":
            doc_type = self._classify_by_content(document)

        document.doc_type = doc_type
        document.metadata["doc_type"] = doc_type
        document.metadata["source"] = document.source
        document.metadata["num_pages"] = len(document.pages)

        logger.info(f"Classified {document.source} as: {doc_type}")
        return document

    @staticmethod
    def _classify_by_filename(filename: str) -> str:
        """Match filename against known patterns."""
        for pattern, dtype in _FILENAME_PATTERNS:
            if pattern.search(filename):
                return dtype
        return "unknown"

    @staticmethod
    def _classify_by_content(doc: Document) -> str:
        """
        Heuristic classification based on page count and content structure.
        """
        num_pages = len(doc.pages)

        # Count structural elements across all pages
        heading_count = 0
        list_count = 0
        total_elements = 0

        for page in doc.pages:
            for elem in page.elements:
                total_elements += 1
                etype = elem.get("type", "")
                if etype == "heading":
                    heading_count += 1
                elif etype == "list":
                    list_count += 1

        # One-pager / flyer
        if num_pages <= 2:
            return "flyer"

        # Slides: high heading density, low text per page
        if total_elements > 0:
            heading_ratio = heading_count / total_elements
            if heading_ratio > 0.3 and num_pages >= 5:
                return "slides"

        # Manuals: many headings, moderate page count
        if heading_count >= 5 and num_pages >= 10:
            return "manual"

        # Standards: many headings + numbered lists
        if heading_count >= 5 and list_count >= 5:
            return "standard"

        # Newsletters/booklets: short-ish, varied content
        if 3 <= num_pages <= 15:
            return "newsletter" if num_pages <= 8 else "booklet"

        # Long docs default to manual
        if num_pages > 15:
            return "manual"

        return "unknown"
