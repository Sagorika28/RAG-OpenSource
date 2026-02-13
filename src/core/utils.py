"""
src/core/utils.py — Shared utilities.

Provides ID generation, timing helpers, and logging setup used across
the entire RAG-OS pipeline.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Generator


def generate_id() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


@contextmanager
def timer(label: str, store: Dict[str, float] | None = None) -> Generator[None, None, None]:
    """
    Context manager that measures elapsed wall-clock time in milliseconds.

    Usage:
        timings: dict[str, float] = {}
        with timer("embed", timings):
            embeddings = model.encode(texts)
        print(timings["embed_ms"])   # e.g. 123.4

    Args:
        label: Name prefix for the timing key (stored as ``{label}_ms``).
        store: Optional dict to write the elapsed time into.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if store is not None:
            store[f"{label}_ms"] = round(elapsed_ms, 2)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for display/logging purposes."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
