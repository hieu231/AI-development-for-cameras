"""Helpers for selecting a writable evidence image directory."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_WRITE_PROBE_FILE = ".write_probe"


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / _WRITE_PROBE_FILE
        probe.touch(exist_ok=True)
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _iter_candidate_paths(base_path: str) -> Iterable[Path]:
    base = Path(base_path).expanduser()
    yield base

    fallback_env = os.getenv("EVIDENCE_IMAGE_FALLBACK_BASE_PATH", "").strip()
    if fallback_env:
        yield Path(fallback_env).expanduser()

    yield Path.cwd() / "evidence_image"
    yield Path(tempfile.gettempdir()) / "evidence_image"


def resolve_writable_evidence_base_path() -> Path:
    """
    Resolve a writable evidence directory.

    Preference order:
    1. EVIDENCE_IMAGE_BASE_PATH
    2. EVIDENCE_IMAGE_FALLBACK_BASE_PATH (optional)
    3. <cwd>/evidence_image
    4. <tempdir>/evidence_image
    """
    configured = os.getenv("EVIDENCE_IMAGE_BASE_PATH", "evidence_image").strip() or "evidence_image"
    seen: set[Path] = set()

    for candidate in _iter_candidate_paths(configured):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate

        if resolved in seen:
            continue
        seen.add(resolved)

        if _is_writable_dir(resolved):
            if str(resolved) != configured:
                logger.warning(
                    "Evidence path '%s' is unavailable. Falling back to '%s'.",
                    configured,
                    resolved,
                )
            os.environ["EVIDENCE_IMAGE_BASE_PATH"] = str(resolved)
            return resolved

    raise PermissionError(
        "No writable directory found for evidence images. "
        "Set EVIDENCE_IMAGE_BASE_PATH to a writable path."
    )
