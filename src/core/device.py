"""
src/core/device.py - Cross-platform torch device resolution helpers.

Allows configs to use `device: auto` while safely falling back when a
requested accelerator (cuda/mps) is not available on the current system.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    if not _has_torch():
        return False
    import torch
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    if not _has_torch():
        return False
    import torch
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_built = getattr(mps_backend, "is_built", lambda: False)()
    is_available = getattr(mps_backend, "is_available", lambda: False)()
    return bool(is_built and is_available)


def best_available_device() -> str:
    """Return best available torch device in priority: cuda -> mps -> cpu."""
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_torch_device(
    requested: str | None,
    component: str = "model",
) -> str:
    """
    Resolve a requested device into a safe runtime device.

    Supported requests: auto, cuda, mps, cpu.
    Unknown or unavailable requests fall back to best available device.
    """
    req = (requested or "auto").strip().lower()

    if req in ("auto", ""):
        resolved = best_available_device()
        logger.info(f"{component}: device=auto -> resolved '{resolved}'")
        return resolved

    if req == "cpu":
        return "cpu"

    if req == "cuda":
        if _cuda_available():
            return "cuda"
        fallback = best_available_device()
        logger.warning(
            f"{component}: requested device 'cuda' is unavailable; "
            f"falling back to '{fallback}'"
        )
        return fallback

    if req == "mps":
        if _mps_available():
            return "mps"
        fallback = best_available_device()
        logger.warning(
            f"{component}: requested device 'mps' is unavailable; "
            f"falling back to '{fallback}'"
        )
        return fallback

    fallback = best_available_device()
    logger.warning(
        f"{component}: unknown device '{requested}'; "
        f"falling back to '{fallback}'"
    )
    return fallback
