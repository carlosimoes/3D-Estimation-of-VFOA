"""Streaming frame sources: video files, image directories, cameras.

Every source yields :class:`vfoa.types.Frame` objects one at a time (frames are
never accumulated in RAM). ``Frame.index`` is the absolute 0-based index in the
source (``input.start`` included) and ``Frame.t = index / fps`` seconds as a
Python float computed per frame (never accumulated). Images are uint8 BGR.
``Frame.intrinsics`` is always set (Sintel ``.cam`` directory, ``camera.*``
config values, or a horizontal-field-of-view fallback).
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import cv2
import numpy as np

from vfoa.io.sintel import intrinsics_from_cam
from vfoa.types import Frame, Intrinsics

__all__ = [
    "FrameSource",
    "IntrinsicsProvider",
    "VideoSource",
    "ImageDirSource",
    "CameraSource",
    "IMAGE_EXTENSIONS",
    "SOURCE_KINDS",
    "open_source",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm")
SOURCE_KINDS: tuple[str, ...] = ("auto", "video", "images", "camera", "synthetic")
_DEFAULT_FPS = 30.0


def _section(cfg: Any, name: str) -> dict:
    try:
        sec = cfg.get(name) if hasattr(cfg, "get") else None
    except Exception:  # pragma: no cover - defensive
        sec = None
    return sec if sec is not None else {}


def _valid_fps(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


class FrameSource:
    """Iterable (and context-managed) stream of :class:`Frame`.

    Attributes: ``fps`` (frames per second), ``width``/``height`` (pixels of the
    left image), ``n_frames`` (number of frames that iteration yields, ``None``
    when unknown) and ``fingerprint`` (string identifying the input for stage
    cache keys).
    """

    fps: float = _DEFAULT_FPS
    width: int = 0
    height: int = 0
    n_frames: Optional[int] = None
    fingerprint: str = ""

    def __iter__(self) -> Iterator[Frame]:  # pragma: no cover - abstract
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (idempotent)."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class IntrinsicsProvider:
    """Per-frame intrinsics from the config.

    Priority: ``input.camera_dir`` (Sintel ``frame_{index+1:04d}.cam`` files),
    ``camera.fx/fy/cx/cy``, then ``Intrinsics.from_hfov(camera.hfov_deg)``.
    The baseline (metres) is ``camera.baseline_m``.
    """

    def __init__(self, cfg: Any) -> None:
        inp = _section(cfg, "input")
        cam = _section(cfg, "camera")
        b = cam.get("baseline_m", None)
        self.baseline: Optional[float] = None if b is None else float(b)
        self.hfov_deg = float(cam.get("hfov_deg", 60.0) or 60.0)
        self.fx = cam.get("fx", None)
        self.fy = cam.get("fy", None)
        self.cx = cam.get("cx", None)
        self.cy = cam.get("cy", None)
        cam_dir = inp.get("camera_dir", None)
        self.camera_dir: Optional[Path] = None
        if cam_dir is not None:
            self.camera_dir = Path(cam_dir)
            if not self.camera_dir.is_dir():
                raise FileNotFoundError(f"input.camera_dir not found: {self.camera_dir}")
        self._last: Optional[Intrinsics] = None
        self._warned_missing = False

    def _static(self, width: int, height: int) -> Intrinsics:
        if self.fx is not None:
            fx = float(self.fx)
            fy = fx if self.fy is None else float(self.fy)
            cx = (width - 1) / 2.0 if self.cx is None else float(self.cx)
            cy = (height - 1) / 2.0 if self.cy is None else float(self.cy)
            return Intrinsics(fx, fy, cx, cy, width, height, self.baseline)
        return Intrinsics.from_hfov(width, height, self.hfov_deg, self.baseline)

    def get(self, index: int, width: int, height: int) -> Intrinsics:
        """Intrinsics of frame ``index`` (absolute, 0-based) at ``width x height`` pixels."""
        if self.camera_dir is not None:
            path = self.camera_dir / f"frame_{index + 1:04d}.cam"
            if path.is_file():
                intr = intrinsics_from_cam(path, width, height, self.baseline)
                self._last = intr
                return intr
            if not self._warned_missing:
                logger.warning(
                    "camera file %s missing; reusing the last available intrinsics", path
                )
                self._warned_missing = True
            if self._last is not None and (self._last.width, self._last.height) == (width, height):
                return self._last
        return self._static(width, height)


# ---------------------------------------------------------------- readers


class _VideoReader:
    """Sequential cv2.VideoCapture reader (one frame in memory at a time)."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"video not found: {self.path}")
        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            self.cap.release()
            raise OSError(f"cv2.VideoCapture cannot open {self.path}")
        count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.count: Optional[int] = count if count > 0 else None
        self.fps: Optional[float] = _valid_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.size = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )

    def skip(self, n: int) -> None:
        for _ in range(int(n)):
            if not self.cap.grab():
                break

    def read(self) -> Optional[np.ndarray]:
        ok, img = self.cap.read()
        return img if ok and img is not None else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _list_images(directory: Path) -> list[Path]:
    files = [
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: _natural_key(p.name))


class _ImageReader:
    """Reads a directory of images in natural sort order."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"image directory not found: {self.path}")
        self.files = _list_images(self.path)
        if not self.files:
            raise ValueError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) in {self.path}")
        self.count: Optional[int] = len(self.files)
        self.fps: Optional[float] = None
        first = self._load(self.files[0])
        self.size = (int(first.shape[1]), int(first.shape[0]))
        self._pos = 0

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"cannot read image {path}")
        return img

    def skip(self, n: int) -> None:
        self._pos += int(n)

    def read(self) -> Optional[np.ndarray]:
        if self._pos >= len(self.files):
            return None
        img = self._load(self.files[self._pos])
        self._pos += 1
        return img

    def close(self) -> None:
        pass


def _make_reader(path: PathLike) -> Any:
    return _ImageReader(path) if Path(path).is_dir() else _VideoReader(path)


def _path_token(path: Optional[PathLike]) -> str:
    if path is None:
        return "-"
    p = Path(path)
    ap = os.path.abspath(p)
    try:
        if p.is_dir():
            files = _list_images(p)
            stats = [f.stat() for f in files]
            total = sum(s.st_size for s in stats)
            mt = max((s.st_mtime_ns for s in stats), default=0)
            return f"{ap}:{len(files)}:{total}:{mt}"
        st = p.stat()
        return f"{ap}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{ap}:?"


def _abs_or_dash(path: Optional[PathLike]) -> str:
    return "-" if path is None else os.path.abspath(path)


# ---------------------------------------------------------------- sources


class _PairedSource(FrameSource):
    """Left (video or image dir) plus optional right view, streamed lazily."""

    kind = "paired"

    def __init__(self, cfg: Any) -> None:
        inp = _section(cfg, "input")
        path = inp.get("path", None)
        if path is None:
            raise ValueError("input.path is not set")
        self.path = Path(path)
        rp = inp.get("right_path", None)
        self.right_path: Optional[Path] = None if rp is None else Path(rp)
        self.start = max(int(inp.get("start", 0) or 0), 0)
        mf = inp.get("max_frames", None)
        self.max_frames: Optional[int] = None if mf is None else max(int(mf), 0)
        self._check_left(self.path)
        self.intrinsics_provider = IntrinsicsProvider(cfg)

        left = _make_reader(self.path)
        try:
            left_count, left_fps, size = left.count, left.fps, left.size
        finally:
            left.close()
        right_count: Optional[int] = None
        if self.right_path is not None:
            right = _make_reader(self.right_path)
            try:
                right_count = right.count
            finally:
                right.close()

        self.fps = _valid_fps(inp.get("fps", None)) or left_fps or _DEFAULT_FPS
        self.width, self.height = int(size[0]), int(size[1])
        if left_count is not None and right_count is not None and left_count != right_count:
            logger.warning(
                "left (%d) and right (%d) views have different frame counts; stopping at the shorter",
                left_count, right_count,
            )
        if left_count is None:
            n: Optional[int] = None
        else:
            total = left_count if right_count is None else min(left_count, right_count)
            n = max(total - self.start, 0)
        if self.max_frames is not None:
            n = self.max_frames if n is None else min(n, self.max_frames)
        self.n_frames = n
        self.fingerprint = "|".join(
            [
                self.kind,
                _path_token(self.path),
                f"start={self.start}",
                f"max={self.max_frames}",
                f"fps={self.fps!r}",
                "right=" + _path_token(self.right_path),
                "cam=" + _abs_or_dash(inp.get("camera_dir", None)),
                "depth=" + _abs_or_dash(inp.get("depth_dir", None)),
            ]
        )

    def _check_left(self, path: Path) -> None:  # overridden
        pass

    def __iter__(self) -> Iterator[Frame]:
        return self._generate()

    def _generate(self) -> Iterator[Frame]:
        left = _make_reader(self.path)
        right = None
        try:
            if self.right_path is not None:
                right = _make_reader(self.right_path)
            left.skip(self.start)
            if right is not None:
                right.skip(self.start)
            warned_resize = False
            i = 0
            while self.max_frames is None or i < self.max_frames:
                image = left.read()
                if image is None:
                    break
                right_img = None
                if right is not None:
                    right_img = right.read()
                    if right_img is None:
                        logger.warning(
                            "right view ended before the left view at frame %d; stopping",
                            self.start + i,
                        )
                        break
                    if right_img.shape[:2] != image.shape[:2]:
                        if not warned_resize:
                            logger.warning(
                                "right view size %s differs from left %s; resizing",
                                right_img.shape[:2], image.shape[:2],
                            )
                            warned_resize = True
                        right_img = cv2.resize(right_img, (image.shape[1], image.shape[0]))
                index = self.start + i
                h, w = image.shape[:2]
                yield Frame(
                    index=index,
                    t=float(index) / float(self.fps),
                    image=image,
                    right=right_img,
                    intrinsics=self.intrinsics_provider.get(index, w, h),
                )
                i += 1
        finally:
            left.close()
            if right is not None:
                right.close()


class VideoSource(_PairedSource):
    """Streams a video file (``input.path``) with ``cv2.VideoCapture``."""

    kind = "video"

    def _check_left(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"video not found: {path}")


class ImageDirSource(_PairedSource):
    """Streams a directory of images (``input.path``) in natural sort order."""

    kind = "images"

    def _check_left(self, path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"image directory not found: {path}")


class CameraSource(FrameSource):
    """Live camera (``input.path`` = device index). Ignores ``input.start``.

    The fingerprint is unique per instance so a live stream never hits a cache.
    """

    kind = "camera"

    def __init__(self, cfg: Any) -> None:
        inp = _section(cfg, "input")
        self.device = int(str(inp.get("path", 0)))
        self.cap = cv2.VideoCapture(self.device)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise OSError(f"cannot open camera {self.device}")
        self.fps = (
            _valid_fps(inp.get("fps", None))
            or _valid_fps(self.cap.get(cv2.CAP_PROP_FPS))
            or _DEFAULT_FPS
        )
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        mf = inp.get("max_frames", None)
        self.max_frames: Optional[int] = None if mf is None else max(int(mf), 0)
        self.n_frames = self.max_frames
        self.intrinsics_provider = IntrinsicsProvider(cfg)
        self.fingerprint = f"camera|{self.device}|{time.time_ns()}"
        self._next_index = 0

    def __iter__(self) -> Iterator[Frame]:
        return self._generate()

    def _generate(self) -> Iterator[Frame]:
        produced = 0
        while self.cap is not None and (self.max_frames is None or produced < self.max_frames):
            ok, image = self.cap.read()
            if not ok or image is None:
                break
            index = self._next_index
            self._next_index += 1
            h, w = image.shape[:2]
            yield Frame(
                index=index,
                t=float(index) / float(self.fps),
                image=image,
                intrinsics=self.intrinsics_provider.get(index, w, h),
            )
            produced += 1

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_source(cfg: Any) -> FrameSource:
    """Create the frame source described by ``cfg.input``.

    ``input.source``: ``auto`` (``"synthetic"`` -> synthetic world, directory ->
    images, digits -> camera, otherwise video), ``video``, ``images``,
    ``camera`` or ``synthetic``.
    """
    inp = _section(cfg, "input")
    kind = str(inp.get("source", "auto") or "auto")
    path = inp.get("path", None)
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown input.source {kind!r}; valid: {', '.join(SOURCE_KINDS)}")
    if kind == "auto":
        if path is None:
            raise ValueError("input.path is not set (use a video, an image directory, a camera index or 'synthetic')")
        sp = str(path)
        if sp == "synthetic":
            kind = "synthetic"
        elif os.path.isdir(sp):
            kind = "images"
        elif sp.isdigit():
            kind = "camera"
        else:
            kind = "video"
    if kind == "synthetic":
        module = importlib.import_module("vfoa.backends.synthetic")
        return module.SyntheticWorld.from_config(cfg).source()
    if kind == "images":
        return ImageDirSource(cfg)
    if kind == "camera":
        return CameraSource(cfg)
    return VideoSource(cfg)
