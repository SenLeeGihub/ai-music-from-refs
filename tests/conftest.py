"""
Test configuration utilities.
Ensures the project source tree is importable when running pytest from the repo root.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    repo_root = Path(__file__).resolve().parent.parent
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
