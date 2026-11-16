"""Thin, lazy wrapper around the optional ``ultralytics`` package.

This is the ONLY module of vfoa that imports ultralytics, and it does so
inside :func:`import_yolo` (called from a backend's ``load()``), never at
import time. ultralytics is licensed AGPL-3.0, so it is an optional
dependency (``pip install -r requirements-fast.txt``).

Rules enforced here:

* ``YOLO_AUTOINSTALL`` is set to ``"false"`` before ultralytics is imported,
  so it never pip-installs packages behind our back.
* Only ``model.predict(...)`` is used; ``model.track(...)`` is never called
  (its trackers need the ``lap`` package and would try to auto-install it).
  Tracking is done by :mod:`vfoa.tracking`.
* Weights resolve to an existing path, else ``<weights_dir>/<name>``
  (``models.weights_dir`` or env ``VFOA_WEIGHTS_DIR``); a missing file is
  downloaded by ultralytics to exactly that path only when
  ``models.allow_download`` is true, else :class:`BackendUnavailable`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from vfoa.backends.base import BackendUnavailable, weights_dir

__all__ = [
    "INSTALL_HINT",
    "import_yolo",
    "resolve_weights",
    "load_yolo",
    "apply_torch_threads",
    "predict",
    "to_numpy",
]

log = logging.getLogger(__name__)

INSTALL_HINT = (
    "The 'fast' backends need the optional package 'ultralytics' (licensed AGPL-3.0, "
    "therefore not a hard dependency of vfoa). Install it with "
    "`pip install -r requirements-fast.txt`, or choose other backends "
    "(e.g. --profile accurate or persons.backend/segmentation.backend in the config)."
)


def import_yolo() -> type:
    """Import and return ``ultralytics.YOLO`` (lazily).

    Sets ``YOLO_AUTOINSTALL=false`` (if unset) BEFORE the import. Any import
    failure raises :class:`BackendUnavailable` with :data:`INSTALL_HINT`.
    """
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO  # noqa: PLC0415 (optional AGPL dependency, lazy)
    except Exception as exc:  # ImportError, or a broken install
        raise BackendUnavailable(f"{INSTALL_HINT} (import failed: {exc!r})") from exc
    return YOLO


def resolve_weights(cfg: Mapping[str, Any], model: str, config_key: str) -> Path:
    """Path of the weights file ``model`` (e.g. ``yolo11n-pose.pt``).

    Order: ``model`` itself if it is an existing file; else
    ``weights_dir(cfg) / basename(model)`` if it exists; else, when
    ``models.allow_download`` is true, that target path (its parent is
    created; ultralytics downloads known assets to exactly that path);
    otherwise :class:`BackendUnavailable` naming the file and the config keys.
    """
    if not model:
        raise BackendUnavailable(f"No YOLO model configured; set {config_key} (e.g. yolo11n-pose.pt).")
    path = Path(str(model)).expanduser()
    if path.is_file():
        return path
    wdir = weights_dir(cfg)
    target = wdir / path.name
    if target.is_file():
        return target
    models = cfg.get("models") or {}
    if models.get("allow_download", True):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise BackendUnavailable(
                f"Cannot create the weights directory {target.parent} for {path.name!r} "
                f"(models.weights_dir / VFOA_WEIGHTS_DIR): {exc}"
            ) from exc
        log.info("YOLO weights %s not found locally; ultralytics will download them to %s", path.name, target)
        return target
    raise BackendUnavailable(
        f"YOLO weights {path.name!r} not found (looked for {path} and {target}). "
        f"Set {config_key} to an existing file, place the file in models.weights_dir "
        f"(or env VFOA_WEIGHTS_DIR), or set models.allow_download=true."
    )


def load_yolo(cfg: Mapping[str, Any], model: str, config_key: str, **warmup_kwargs: Any) -> Any:
    """Build an ultralytics ``YOLO`` model and run one warm-up prediction.

    ``warmup_kwargs`` are passed to :func:`predict` for the warm-up (e.g.
    ``imgsz``, ``device``). Construction errors (including failed downloads)
    raise :class:`BackendUnavailable`.

    After the warm-up, torch's intra-op thread count is set by
    :func:`apply_torch_threads`. ultralytics' ``select_device()`` calls
    ``torch.set_num_threads(min(8, cpu_count - 1))`` at the first predict;
    with OpenMP's default active waiting on a shared CPU (and ONNX Runtime
    threads spinning beside it) that made each YOLO11n call 25x slower
    (2-3 s instead of about 0.1 s) on an 8-vCPU WSL2 VM. Setting the
    variable ``OMP_WAIT_POLICY`` here would be too late, because
    :func:`vfoa.backends.base.resolve_device` has already imported torch.
    """
    YOLO = import_yolo()
    path = resolve_weights(cfg, model, config_key)
    try:
        yolo = YOLO(str(path))
    except Exception as exc:
        raise BackendUnavailable(
            f"Cannot load YOLO weights {path} ({config_key}): {exc!r}. "
            f"If the file must be downloaded, check the network or set models.allow_download."
        ) from exc
    try:
        predict(yolo, np.zeros((64, 64, 3), dtype=np.uint8), **warmup_kwargs)
    except Exception as exc:  # warm-up is best effort; real errors surface on use
        log.warning("YOLO warm-up of %s failed: %r", path, exc)
    apply_torch_threads(cfg, warmup_kwargs.get("device"))
    return yolo


def apply_torch_threads(cfg: Mapping[str, Any], device: Optional[str] = None) -> Optional[int]:
    """Set torch's intra-op thread count after ultralytics has changed it.

    Rules: ``runtime.num_threads`` set -> ``torch.set_num_threads(n)``; else,
    on CPU (``device`` is ``None`` or starts with ``"cpu"``) ->
    ``max(1, min(torch.get_num_threads(), cpu_count // 2))``, because
    YOLO11n gains almost nothing beyond 2-4 threads while more threads
    spin-wait against the other models; else (a GPU device) nothing.
    torch is taken from ``sys.modules`` only (never imported here); without
    it nothing happens. Returns the value set, or ``None``.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    n = (cfg.get("runtime") or {}).get("num_threads")
    try:
        if n:
            value = int(n)
            reason = "runtime.num_threads"
        elif device is None or str(device).lower().startswith("cpu"):
            value = max(1, min(int(torch.get_num_threads()), (os.cpu_count() or 2) // 2))
            reason = "CPU default: half the logical CPUs (runtime.num_threads is null)"
        else:
            return None
        torch.set_num_threads(value)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Cannot set torch threads: %r", exc)
        return None
    log.info("YOLO: torch intra-op threads set to %d (%s)", value, reason)
    return value


def predict(model: Any, image: np.ndarray, device: Optional[str] = None, **kwargs: Any) -> Any:
    """``model.predict(image, verbose=False, **kwargs)`` -> first result or ``None``.

    ``image`` is passed through unchanged (uint8 BGR, as ultralytics expects
    for NumPy input). ``device`` is only forwarded when not ``None``.
    """
    if device is not None:
        kwargs["device"] = device
    results = model.predict(image, verbose=False, **kwargs)
    if results is None:
        return None
    try:
        return results[0] if len(results) else None
    except TypeError:
        return results


def to_numpy(x: Any) -> np.ndarray:
    """Convert a tensor-like (``.detach().cpu().numpy()``, duck-typed) or array to NumPy."""
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    for attr in ("detach", "cpu", "numpy"):
        fn = getattr(x, attr, None)
        if callable(fn) and not isinstance(x, np.ndarray):
            x = fn()
    return np.asarray(x)
