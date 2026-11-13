"""Simple depth backends: Sintel ground truth and a constant depth.

Both return :class:`vfoa.types.DepthFrame` at full frame resolution, depth in
metres (camera frame Z, positive), 0 where invalid.

Also provides :func:`frame_intrinsics`, the shared rule used by the depth
backends to find the intrinsics of a frame.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from vfoa.backends.base import BackendUnavailable, DepthBackend
from vfoa.io.sintel import depth_read, intrinsics_from_cam
from vfoa.types import DepthFrame, Frame, Intrinsics

__all__ = ["frame_intrinsics", "ConstantDepth", "SintelGTDepth"]

logger = logging.getLogger(__name__)


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return cfg.get(name) or {}


def frame_intrinsics(cfg: Mapping[str, Any], frame: Frame) -> Intrinsics:
    """Intrinsics (pixels; baseline metres) matching ``frame.image``.

    First match wins: ``frame.intrinsics``; ``<input.camera_dir>/frame_{index+1:04d}.cam``;
    ``camera.fx`` (with ``fy``/``cx``/``cy`` defaults); ``Intrinsics.from_hfov``.
    The result is rescaled when its size differs from the image, and gets the
    ``camera.baseline_m`` baseline when it has none.
    """
    h, w = frame.image.shape[:2]
    cam = _section(cfg, "camera")
    inp = _section(cfg, "input")
    baseline = cam.get("baseline_m")
    baseline = float(baseline) if baseline is not None else None

    intr: Optional[Intrinsics] = frame.intrinsics
    if intr is None:
        cam_dir = inp.get("camera_dir")
        if cam_dir:
            path = Path(cam_dir) / f"frame_{int(frame.index) + 1:04d}.cam"
            if path.is_file():
                intr = intrinsics_from_cam(path, w, h, baseline=baseline)
    if intr is None and cam.get("fx") is not None:
        fx = float(cam["fx"])
        fy = float(cam["fy"]) if cam.get("fy") is not None else fx
        cx = float(cam["cx"]) if cam.get("cx") is not None else (w - 1) / 2.0
        cy = float(cam["cy"]) if cam.get("cy") is not None else (h - 1) / 2.0
        intr = Intrinsics(fx, fy, cx, cy, w, h, baseline)
    if intr is None:
        intr = Intrinsics.from_hfov(w, h, float(cam.get("hfov_deg", 60.0) or 60.0), baseline)

    if (intr.width, intr.height) != (w, h):
        intr = intr.scaled(w / intr.width, h / intr.height)
        intr = dataclasses.replace(intr, width=w, height=h)
    if intr.baseline is None and baseline is not None:
        intr = dataclasses.replace(intr, baseline=baseline)
    return intr


class ConstantDepth(DepthBackend):
    """Every pixel at ``depth.constant_depth_m`` metres (monocular fallback)."""

    name = "constant"

    def _load(self) -> None:
        value = self.section.get("constant_depth_m", 2.5)
        if value is None or not float(value) > 0:
            raise ValueError(f"depth.constant_depth_m must be > 0, got {value!r}")
        self._value = float(value)

    def compute(self, frame: Frame) -> DepthFrame:
        self.load()
        h, w = frame.image.shape[:2]
        depth = np.full((h, w), self._value, dtype=np.float32)
        valid = np.ones((h, w), dtype=bool)
        return DepthFrame(depth, valid, frame_intrinsics(self.cfg, frame), disparity=None)


class SintelGTDepth(DepthBackend):
    """MPI-Sintel ground-truth depth: ``<input.depth_dir>/frame_{index+1:04d}.dpt`` (metres)."""

    name = "sintel_gt"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._warned_missing = False

    def _load(self) -> None:
        depth_dir = _section(self.cfg, "input").get("depth_dir")
        if not depth_dir or not os.path.isdir(depth_dir):
            raise BackendUnavailable(
                f"sintel_gt depth needs a directory of .dpt files in input.depth_dir (got {depth_dir!r})"
            )
        self._dir = Path(depth_dir)

    def compute(self, frame: Frame) -> DepthFrame:
        self.load()
        h, w = frame.image.shape[:2]
        intr = frame_intrinsics(self.cfg, frame)
        path = self._dir / f"frame_{int(frame.index) + 1:04d}.dpt"
        if not path.is_file():
            if not self._warned_missing:
                logger.warning("sintel_gt: missing depth file %s; returning invalid depth", path)
                self._warned_missing = True
            return DepthFrame(
                np.zeros((h, w), np.float32), np.zeros((h, w), bool), intr, disparity=None
            )
        depth = depth_read(path)
        if depth.shape != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = np.array(depth, dtype=np.float32, copy=True)
        lo = float(self.section.get("min_depth_m", 0.05))
        hi = float(self.section.get("max_depth_m", 100.0))
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(depth) & (depth >= lo) & (depth <= hi)
        if not valid.all():
            depth[~valid] = 0.0
        return DepthFrame(depth, valid, intr, disparity=None)
