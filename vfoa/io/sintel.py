"""Python-3 readers/writers for MPI-Sintel camera (.cam) and depth (.dpt) files.

File formats (little-endian):

* ``.dpt``: tag float32 202021.25 (bytes ``PIEH``), int32 width, int32 height,
  then ``height * width`` float32 depths in metres, row-major.
* ``.cam``: the same tag, then 9 float64 (intrinsic matrix K, row-major) and
  12 float64 (extrinsic 3x4 matrix [R|t], world -> camera).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from vfoa.types import Intrinsics

__all__ = [
    "TAG_FLOAT",
    "TAG_CHAR",
    "depth_read",
    "depth_write",
    "cam_read",
    "cam_write",
    "intrinsics_from_cam",
]

PathLike = Union[str, os.PathLike]

TAG_FLOAT = 202021.25
TAG_CHAR = b"PIEH"
_MAX_SIZE = 100_000_000


def _read_exact(fh, n: int, path: PathLike) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError(f"{os.fspath(path)}: truncated file (expected {n} bytes, got {len(data)})")
    return data


def _check_tag(raw: bytes, path: PathLike) -> None:
    tag = np.frombuffer(raw, dtype="<f4", count=1)[0]
    if tag != np.float32(TAG_FLOAT):
        raise ValueError(f"{os.fspath(path)}: wrong Sintel tag {tag!r} (expected {TAG_FLOAT})")


def depth_read(path: PathLike) -> np.ndarray:
    """Read a Sintel ``.dpt`` file; returns an (H, W) native float32 array in metres."""
    with open(path, "rb") as fh:
        _check_tag(_read_exact(fh, 4, path), path)
        width, height = (int(v) for v in np.frombuffer(_read_exact(fh, 8, path), dtype="<i4"))
        size = width * height
        if width <= 0 or height <= 0 or size <= 1 or size >= _MAX_SIZE:
            raise ValueError(f"{os.fspath(path)}: invalid size width={width} height={height}")
        raw = _read_exact(fh, 4 * size, path)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32).reshape(height, width)


def depth_write(path: PathLike, depth: np.ndarray) -> None:
    """Write an (H, W) depth map (metres) as a Sintel ``.dpt`` file."""
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth.shape}")
    height, width = depth.shape
    with open(path, "wb") as fh:
        fh.write(TAG_CHAR)
        fh.write(np.array([width, height], dtype="<i4").tobytes())
        fh.write(np.ascontiguousarray(depth, dtype="<f4").tobytes())


def cam_read(path: PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Read a Sintel ``.cam`` file; returns (K 3x3, Rt 3x4) float64."""
    with open(path, "rb") as fh:
        _check_tag(_read_exact(fh, 4, path), path)
        K = np.frombuffer(_read_exact(fh, 72, path), dtype="<f8").astype(np.float64).reshape(3, 3)
        Rt = np.frombuffer(_read_exact(fh, 96, path), dtype="<f8").astype(np.float64).reshape(3, 4)
    return K, Rt


def cam_write(path: PathLike, K: np.ndarray, Rt: np.ndarray) -> None:
    """Write intrinsic K (3x3) and extrinsic Rt (3x4) as a Sintel ``.cam`` file."""
    K = np.asarray(K, dtype="<f8")
    Rt = np.asarray(Rt, dtype="<f8")
    if K.shape != (3, 3) or Rt.shape != (3, 4):
        raise ValueError(f"expected K 3x3 and Rt 3x4, got {K.shape} and {Rt.shape}")
    with open(path, "wb") as fh:
        fh.write(TAG_CHAR)
        fh.write(np.ascontiguousarray(K).tobytes())
        fh.write(np.ascontiguousarray(Rt).tobytes())


def intrinsics_from_cam(
    path: PathLike, width: int, height: int, baseline: Optional[float] = 0.1
) -> Intrinsics:
    """Intrinsics from a Sintel ``.cam`` file (Sintel stereo baseline: 0.1 m)."""
    K, _ = cam_read(path)
    return Intrinsics.from_K(K, width, height, baseline=baseline)
