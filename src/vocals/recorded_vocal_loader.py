"""
Utility helpers for locating recorded vocal tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

DEFAULT_RECORDED_DIR = Path("data/recorded")


def find_latest_recorded_vocal(root: Optional[Path | str] = None) -> Optional[Path]:
    """
    Return the newest .wav inside the recorded directory (defaults to data/recorded).
    """
    search_root = Path(root) if root is not None else DEFAULT_RECORDED_DIR
    if not search_root.exists():
        return None

    candidates = [
        path
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)
