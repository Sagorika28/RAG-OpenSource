"""
src/core/config.py — YAML configuration loader.

Loads a config YAML (e.g. configs/cpu.yaml) and returns a plain dict.
Supports environment-variable overrides for sensitive values.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


# Keys that MUST be present at the top level of every config file
_REQUIRED_SECTIONS = [
    "parsing",
    "chunking",
    "embeddings",
    "qdrant",
    "reranker",
    "retrieval",
    "generation",
    "data",
]


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Load and validate a YAML configuration file.

    Args:
        path: Path to the YAML file (absolute or relative to CWD).

    Returns:
        Parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required sections are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)

    # --- Validate required sections ---
    missing = [s for s in _REQUIRED_SECTIONS if s not in config]
    if missing:
        raise ValueError(
            f"Config {path.name} is missing required sections: {missing}"
        )

    # --- Apply environment-variable overrides ---
    # Example: RAG_OS_QDRANT_URL overrides config["qdrant"]["url"]
    env_overrides = {
        "RAG_OS_QDRANT_URL": ("qdrant", "url"),
        "RAG_OS_QDRANT_MODE": ("qdrant", "mode"),
        "RAG_OS_OLLAMA_URL": ("generation", "base_url"),
        "RAG_OS_OLLAMA_MODEL": ("generation", "model"),
    }
    for env_key, (section, key) in env_overrides.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            config[section][key] = env_val

    return config


def get_section(config: Dict[str, Any], section: str) -> Dict[str, Any]:
    """Convenience accessor — returns a config sub-section or empty dict."""
    return config.get(section, {})
