"""Make the repository root importable from the legacy ``thesis_final`` scripts.

Import this module first (``import _bootstrap``) in a legacy script so that the
``vfoa`` package can be imported. Adding the repo root to ``sys.path`` is the
only side effect.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_root() -> Path:
    """Insert the repository root at ``sys.path[0]`` if it is not already present."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT


ensure_repo_root()
