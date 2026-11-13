"""Locate the thesis third-party source checkouts and their weights.

The root comes from the config key ``models.third_party_root`` or, when that is
null, from the environment variable ``VFOA_THIRD_PARTY`` (thesis layout: a
detectron2 checkout that also holds ``projects/DensePose``, ``Pose_Estimation``,
``gaze360`` and ``headpose-fsanet-pytorch``). No path is hard-coded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from vfoa.backends.base import BackendUnavailable

__all__ = ["ENV_VAR", "third_party_root", "resolve_path", "add_sys_paths"]

ENV_VAR = "VFOA_THIRD_PARTY"


def third_party_root(cfg: Mapping[str, Any]) -> Optional[Path]:
    """Root of the third-party checkouts (config first, then env), or None."""
    models = cfg.get("models") or {}
    value = models.get("third_party_root") or os.environ.get(ENV_VAR)
    if not value:
        return None
    return Path(str(value)).expanduser()


def resolve_path(cfg: Mapping[str, Any], value: Union[str, os.PathLike]) -> Path:
    """Absolute paths are returned as-is; relative ones are joined to the third-party root.

    Raises BackendUnavailable when a relative path is given and no root is configured.
    """
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    root = third_party_root(cfg)
    if root is None:
        raise BackendUnavailable(
            f"Cannot resolve relative model path {str(value)!r}: set the config key "
            f"models.third_party_root or the environment variable {ENV_VAR} to the "
            f"directory holding the thesis third-party checkouts"
        )
    return root / path


def add_sys_paths(cfg: Mapping[str, Any], *subdirs: str) -> list[Path]:
    """Insert ``<root>/<sub>`` (or the root itself when no subdir is given) into sys.path once.

    Raises BackendUnavailable when the root or a directory is missing.
    """
    root = third_party_root(cfg)
    if root is None:
        raise BackendUnavailable(
            f"Third-party checkouts not configured: set models.third_party_root or {ENV_VAR}"
        )
    if not root.is_dir():
        raise BackendUnavailable(
            f"Third-party root {str(root)!r} (models.third_party_root / {ENV_VAR}) does not exist"
        )
    added: list[Path] = []
    for sub in subdirs or ("",):
        path = (root / sub) if sub else root
        if not path.is_dir():
            raise BackendUnavailable(
                f"Third-party directory {str(path)!r} is missing (root from "
                f"models.third_party_root / {ENV_VAR})"
            )
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        added.append(path)
    return added
