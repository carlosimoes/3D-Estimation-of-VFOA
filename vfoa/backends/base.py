"""Backend base classes: one interface per model kind.

A backend's ``__init__`` must be cheap (store the config, no heavy import, no
file access). Heavy imports and weights are loaded once in ``load()``, which
calls the subclass hook ``_load()`` exactly once. Missing optional packages or
weights raise :class:`BackendUnavailable` naming the package/weights and the
config key to change.
"""

from __future__ import annotations

import abc
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from vfoa.types import (
        DepthFrame,
        DirectionEstimate,
        Frame,
        Person,
        PersonDetection,
        Segmentation,
    )

__all__ = [
    "KINDS",
    "BackendUnavailable",
    "Backend",
    "DepthBackend",
    "PersonBackend",
    "HeadPoseBackend",
    "GazeBackend",
    "SegmentationBackend",
    "resolve_device",
    "weights_dir",
]

#: Backend kinds; each is also the config section holding ``backend: <name>``.
KINDS: tuple[str, ...] = ("depth", "persons", "headpose", "gaze", "segmentation")


class BackendUnavailable(RuntimeError):
    """A backend cannot run here (missing package, weights or hardware)."""


class Backend:
    """Base of every backend.

    Subclasses set ``kind`` and ``name`` and implement ``_load()`` for heavy
    initialisation. ``is_loaded`` tells whether ``load()`` has completed.
    """

    kind: str = ""
    name: str = ""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def section(self) -> Mapping[str, Any]:
        """The config section of this backend's kind (``cfg[kind]``)."""
        return self.cfg[self.kind]

    def load(self) -> None:
        """Load heavy dependencies and weights; idempotent (``_load`` runs once)."""
        if not self._loaded:
            self._load()
            self._loaded = True

    def _load(self) -> None:
        """Subclass hook for heavy imports/weights. Default: nothing."""

    def reset(self) -> None:
        """Clear per-stream state (e.g. temporal buffers). Default: nothing."""

    def close(self) -> None:
        """Release resources. Default: nothing."""


class DepthBackend(Backend, abc.ABC):
    kind = "depth"

    @abc.abstractmethod
    def compute(self, frame: "Frame") -> "DepthFrame":
        """Depth of ``frame`` at full frame resolution (metres, camera frame)."""


class PersonBackend(Backend, abc.ABC):
    kind = "persons"

    @abc.abstractmethod
    def detect(self, frame: "Frame") -> "list[PersonDetection]":
        """Untracked person detections of ``frame``."""


class HeadPoseBackend(Backend, abc.ABC):
    kind = "headpose"

    @abc.abstractmethod
    def estimate(self, frame: "Frame", persons: "Sequence[Person]") -> "dict[int, DirectionEstimate]":
        """Head direction per track id; persons without a usable head box are absent."""


class GazeBackend(Backend, abc.ABC):
    kind = "gaze"

    def observe(self, frame: "Frame", persons: "Sequence[Person]") -> None:
        """Called on every frame for cheap buffering (temporal models). Default: no-op."""

    @abc.abstractmethod
    def estimate(self, frame: "Frame", persons: "Sequence[Person]") -> "dict[int, DirectionEstimate]":
        """Gaze direction per track id; persons already carry ``.head``."""


class SegmentationBackend(Backend, abc.ABC):
    kind = "segmentation"

    @abc.abstractmethod
    def segment(self, frame: "Frame") -> "Segmentation":
        """Panoptic-style segmentation of ``frame``."""


def resolve_device(cfg: Mapping[str, Any]) -> str:
    """Torch device from ``models.device``: explicit values pass through;
    ``auto`` -> ``cuda`` if available, else ``cpu`` (also when torch is missing).
    torch is imported lazily."""
    models = cfg.get("models") or {}
    device = models.get("device", "auto") or "auto"
    if device != "auto":
        return str(device)
    try:
        import torch  # noqa: PLC0415 (lazy heavy import)
    except Exception:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def weights_dir(cfg: Mapping[str, Any]) -> Path:
    """Directory for downloadable weights: env ``VFOA_WEIGHTS_DIR`` wins over ``models.weights_dir``."""
    env = os.environ.get("VFOA_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser()
    models = cfg.get("models") or {}
    return Path(models.get("weights_dir") or "weights").expanduser()
