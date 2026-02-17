"""
src/parsing/selective_ocr.py - Mixed-page selective OCR using OCRmyPDF.

Heuristic per page:
  - OCR if text_chars < 40
    OR
  - OCR if image_area_ratio >= 0.6 and text_chars <= 200

Only selected pages are OCR'd via OCRmyPDF --pages.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _safe_image_area_ratio(page) -> float:
    """Estimate image area ratio on a page using PyMuPDF image rectangles."""
    page_area = float(page.rect.width * page.rect.height) or 1.0
    total_image_area = 0.0

    # Iterate by image xref and sum drawn image rectangles on the page.
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            total_image_area += max(0.0, float(rect.width * rect.height))

    # Cap at 1.0 in case overlapping images inflate the estimate.
    return min(1.0, total_image_area / page_area)


def _select_pages_for_ocr(pdf_in: Path) -> List[int]:
    """Return 1-based page indices selected by the mixed-page OCR heuristic."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError(
            "PyMuPDF is required for selective OCR page analysis."
        ) from e

    selected_pages: List[int] = []
    doc = fitz.open(str(pdf_in))
    try:
        for idx in range(len(doc)):
            page = doc[idx]
            text_chars = len(page.get_text("text") or "")
            image_area_ratio = _safe_image_area_ratio(page)

            needs_ocr = (
                text_chars < 40
                or (image_area_ratio >= 0.6 and text_chars <= 200)
            )
            if needs_ocr:
                selected_pages.append(idx + 1)  # OCRmyPDF expects 1-based pages

            logger.debug(
                "Selective OCR page=%s text_chars=%s image_area_ratio=%.3f needs_ocr=%s",
                idx + 1,
                text_chars,
                image_area_ratio,
                needs_ocr,
            )
    finally:
        doc.close()

    return selected_pages


def ocr_pdf_selective(pdf_in: Path, pdf_out: Path) -> Path:
    """
    Run selective page OCR with OCRmyPDF and return output path.

    OCR command:
      ocrmypdf --skip-text --deskew --rotate-pages --optimize 3
               --pages "<1-based page list>" input.pdf output.pdf
    """
    pdf_in = Path(pdf_in)
    pdf_out = Path(pdf_out)
    if not pdf_in.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_in}")
    pdf_out.parent.mkdir(parents=True, exist_ok=True)

    pages = _select_pages_for_ocr(pdf_in)
    if not pages:
        shutil.copy2(pdf_in, pdf_out)
        logger.info(
            "Selective OCR: no pages selected for %s; copied original PDF",
            pdf_in.name,
        )
        return pdf_out

    if shutil.which("ocrmypdf") is None:
        raise RuntimeError(
            "ocrmypdf executable not found in PATH. Install OCRmyPDF first."
        )

    page_spec = ",".join(str(p) for p in pages)
    cmd = [
        "ocrmypdf",
        "--skip-text",
        "--deskew",
        "--rotate-pages",
        "--optimize",
        "3",
        "--pages",
        page_spec,
        str(pdf_in),
        str(pdf_out),
    ]

    logger.info(
        "Selective OCR: %s pages selected for %s (%s)",
        len(pages),
        pdf_in.name,
        page_spec,
    )
    subprocess.run(cmd, check=True)
    return pdf_out

