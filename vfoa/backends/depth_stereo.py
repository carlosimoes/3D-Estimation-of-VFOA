"""Stereo depth backend (OpenCV block matching) and a right-view synthesiser.

:class:`StereoBMDepth` computes a REAL disparity (float32 pixels,
``d = fx * B / Z``) from a rectified left/right pair and converts it to depth
in metres in the left camera frame. The matcher (and the optional WLS filter
from ``cv2.ximgproc``) are built once in ``load()``; each frame costs exactly
one left-matcher ``compute`` (plus one right-matcher ``compute`` when WLS is on).

:func:`synthesize_right_view` warps a left image to the right camera with a
known depth; tests and benchmarks use it because the repository ships no
stereo pairs.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from vfoa.backends.base import BackendUnavailable, DepthBackend
from vfoa.backends.depth_simple import frame_intrinsics
from vfoa.types import DepthFrame, Frame, Intrinsics

__all__ = ["StereoBMDepth", "synthesize_right_view"]

logger = logging.getLogger(__name__)


def _parse_wls(value: Any) -> Optional[bool]:
    """``None`` for auto, else a bool."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "auto":
        return None
    if text in ("true", "yes", "on", "1"):
        return True
    if text in ("false", "no", "off", "0"):
        return False
    raise ValueError(f"depth.wls must be auto, true or false, got {value!r}")


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return image


class StereoBMDepth(DepthBackend):
    """Block-matching stereo depth (needs ``frame.right`` and a positive baseline).

    Config (``depth`` section): ``scale`` (0 < s <= 1, matching resolution),
    ``num_disparities`` (at full resolution), ``block_size``, ``wls``
    (auto|true|false), ``wls_lambda``, ``wls_sigma``, ``min_depth_m``,
    ``max_depth_m``; optional ``speckle_window`` (default 100, 0 = off) and
    ``speckle_range`` (default 2).
    """

    name = "stereo_bm"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._left = None
        self._right = None
        self._wls = None
        self._scale = 1.0

    def _load(self) -> None:
        sec = self.section
        s = float(sec.get("scale", 1.0))
        if not 0.0 < s <= 1.0:
            raise ValueError(f"depth.scale must be in (0, 1], got {s}")
        nd = max(16, int(math.ceil(float(sec.get("num_disparities", 160)) * s / 16.0)) * 16)
        bs = max(5, int(sec.get("block_size", 15)) | 1)
        wls = _parse_wls(sec.get("wls", "auto"))
        xi = getattr(cv2, "ximgproc", None)
        has_xi = xi is not None and hasattr(xi, "createDisparityWLSFilter")
        if wls is True and not has_xi:
            raise BackendUnavailable(
                "depth.wls is true but cv2.ximgproc is not available; install "
                "opencv-contrib-python or set depth.wls to auto/false"
            )

        left = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
        sw = int(sec.get("speckle_window", 100))
        if sw > 0:
            left.setSpeckleWindowSize(sw)
            left.setSpeckleRange(int(sec.get("speckle_range", 2)))
        self._left = left
        self._scale = s

        if wls is None and not has_xi:
            logger.info("cv2.ximgproc not available: WLS disparity filtering disabled (depth.wls=auto)")
        if has_xi and wls is not False:
            self._right = xi.createRightMatcher(left)
            self._wls = xi.createDisparityWLSFilter(left)
            self._wls.setLambda(float(sec.get("wls_lambda", 8000000.0)))
            self._wls.setSigmaColor(float(sec.get("wls_sigma", 1.5)))

    def reset(self) -> None:
        """No per-stream state."""

    def disparity(self, left: np.ndarray, right: Optional[np.ndarray]) -> np.ndarray:
        """Disparity (float32 pixels at the input resolution, 0 = invalid) of a rectified pair."""
        self.load()
        if right is None:
            raise ValueError(
                "stereo_bm depth needs the rectified right view (frame.right); set input.right_path"
            )
        if left.shape[:2] != right.shape[:2]:
            raise ValueError(f"left/right size mismatch: {left.shape} vs {right.shape}")
        h, w = left.shape[:2]
        gl = _to_gray(left)
        gr = _to_gray(right)
        s = self._scale
        scaled = s != 1.0
        if scaled:
            sw, sh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            gl = cv2.resize(gl, (sw, sh), interpolation=cv2.INTER_AREA)
            gr = cv2.resize(gr, (sw, sh), interpolation=cv2.INTER_AREA)
        raw = self._left.compute(gl, gr)
        if self._wls is not None:
            raw_right = self._right.compute(gr, gl)
            raw = self._wls.filter(raw, gl, None, raw_right)
        d = np.asarray(raw).astype(np.float32) * np.float32(1.0 / 16.0)
        if scaled:
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST) * np.float32(w / sw)
        d[~(d > 0)] = 0.0
        return d

    def compute(self, frame: Frame) -> DepthFrame:
        """Depth (metres) from ``frame.image`` and ``frame.right``; ``depth * disparity = fx * B``."""
        self.load()
        intr = frame_intrinsics(self.cfg, frame)
        if intr.baseline is None or not intr.baseline > 0:
            raise ValueError("stereo_bm depth needs a positive stereo baseline (camera.baseline_m)")
        d = self.disparity(frame.image, frame.right)
        valid = d > 0
        depth = np.zeros(d.shape, dtype=np.float32)
        np.divide(np.float32(intr.fx * intr.baseline), d, out=depth, where=valid)
        lo = float(self.section.get("min_depth_m", 0.05))
        hi = float(self.section.get("max_depth_m", 100.0))
        valid &= (depth >= lo) & (depth <= hi)
        depth[~valid] = 0.0
        d[~valid] = 0.0
        return DepthFrame(depth, valid, intr, disparity=d)

    def close(self) -> None:
        self._left = self._right = self._wls = None
        self._loaded = False


def synthesize_right_view(left_bgr: np.ndarray, depth: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Right view of a rectified pair rendered from the left image and its depth.

    Forward warp ``u_r = rint(u - fx * B / Z)`` with a z-buffer (nearer pixels
    win) and hole filling from the nearest filled pixel to the right (background
    side of a disocclusion), else to the left. Same shape/dtype as ``left_bgr``.
    """
    if intrinsics.baseline is None or not intrinsics.baseline > 0:
        raise ValueError("synthesize_right_view needs intrinsics with a positive baseline")
    left = np.asarray(left_bgr)
    h, w = left.shape[:2]
    z = np.asarray(depth, dtype=np.float64)
    if z.shape != (h, w):
        raise ValueError(f"depth shape {z.shape} does not match image {(h, w)}")
    ok = np.isfinite(z) & (z > 0)
    d = np.zeros((h, w), dtype=np.float64)
    d[ok] = intrinsics.fx * intrinsics.baseline / z[ok]
    vv, uu = np.nonzero(ok)
    ur = np.rint(uu - d[vv, uu]).astype(np.int64)
    inside = (ur >= 0) & (ur < w)
    vv, uu, ur = vv[inside], uu[inside], ur[inside]
    order = np.argsort(d[vv, uu], kind="stable")  # far first, near last -> near wins
    vv, uu, ur = vv[order], uu[order], ur[order]

    right = np.zeros_like(left)
    filled = np.zeros((h, w), dtype=bool)
    right[vv, ur] = left[vv, uu]
    filled[vv, ur] = True
    if filled.all() or not filled.any():
        return right

    cols = np.broadcast_to(np.arange(w), (h, w))
    # nearest filled column to the right (inclusive): reverse running minimum
    nxt = np.where(filled, cols, w)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    # nearest filled column to the left (inclusive): running maximum
    prv = np.where(filled, cols, -1)
    prv = np.maximum.accumulate(prv, axis=1)
    src = np.where(nxt < w, nxt, prv)
    rows = np.broadcast_to(np.arange(h)[:, None], (h, w))
    holes = ~filled & (src >= 0)
    right[holes] = right[rows[holes], src[holes]]
    return right
