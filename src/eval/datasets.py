"""
src/eval/datasets.py — Evaluation dataset loader.

Loads a JSONL file of evaluation examples into a list of EvalExample
objects.  Each line is a JSON object with:
  - question (str, required)
  - gold_sources (list[str], required) — expected source filenames
  - gold_doc_ids (list[str], optional)
  - gold_chunk_ids (list[str], optional)
  - gold_answer (str, optional)
  - metadata (dict, optional) — extra info like page, section
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from src.core.types import EvalExample

logger = logging.getLogger(__name__)


def load_eval_dataset(path: str | Path) -> List[EvalExample]:
    """
    Load evaluation examples from a JSONL file.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of EvalExample objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    examples: List[EvalExample] = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                example = EvalExample(
                    question=data["question"],
                    gold_sources=data.get("gold_sources", []),
                    gold_doc_ids=data.get("gold_doc_ids", []),
                    gold_chunk_ids=data.get("gold_chunk_ids", []),
                    gold_answer=data.get("gold_answer"),
                    metadata=data.get("metadata", {}),
                )
                examples.append(example)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(
                    f"Skipping line {line_no} in {path.name}: {e}"
                )

    logger.info(f"Loaded {len(examples)} eval examples from {path.name}")
    return examples
