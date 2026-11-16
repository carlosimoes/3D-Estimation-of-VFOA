"""Structured outputs: JSONL/CSV sinks, overlay drawing, streaming video, mosaics.

A sink implements ``write(frame, result)`` and ``close()``. Timestamps are
frame based (``result.t = index / fps`` seconds). Drawing functions never draw
in place: they return a NEW uint8 BGR image. No ffmpeg, matplotlib or torch.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

from vfoa.types import Frame, FrameResult, Intrinsics

__all__ = [
    "CSV_COLUMNS",
    "result_to_dict",
    "JsonlSink",
    "CsvSink",
    "head_axes_endpoints",
    "draw_head_axes",
    "draw_overlay",
    "VideoSink",
    "compose_mosaic",
    "DisplaySink",
]

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]

CSV_COLUMNS: tuple[str, ...] = (
    "frame", "t", "person_id", "object_id", "label",
    "score_gaze", "score_head", "score_total", "occluded", "is_target",
)

# BGR palette for track ids.
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 200, 255), (255, 128, 0), (60, 220, 60), (200, 60, 255),
    (255, 255, 0), (0, 128, 255), (180, 180, 255), (128, 255, 200),
)


# ---------------------------------------------------------------- serialisation


def _jsonable(x: Any) -> Any:
    """Convert numpy types/arrays to JSON-safe Python values (non-finite -> None)."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    return x


def _vec(v: Any) -> Optional[list]:
    if v is None:
        return None
    return _jsonable(np.asarray(v, dtype=np.float64).reshape(-1))


def result_to_dict(result: FrameResult) -> dict:
    """JSON-serialisable dict of a :class:`FrameResult` (section 12 keys)."""
    labels = {int(o.object_id): o.label for o in result.objects}
    persons = []
    for p in result.persons:
        tid = int(p.track_id)
        target = result.targets.get(tid)
        det = p.detection
        persons.append(
            {
                "id": tid,
                "head_box": _vec(det.head_box if det is not None else None),
                "eyes_3d": _vec(p.eyes_3d),
                "gaze": _vec(p.gaze.direction) if p.gaze is not None else None,
                "gaze_conf": _jsonable(p.gaze.confidence) if p.gaze is not None else None,
                "head_dir": _vec(p.head.direction) if p.head is not None else None,
                "head_conf": _jsonable(p.head.confidence) if p.head is not None else None,
                "target": _jsonable(target),
                "target_label": labels.get(int(target)) if target is not None else None,
            }
        )
    objects = [
        {
            "id": int(o.object_id),
            "segment_id": int(o.segment_id),
            "label": str(o.label),
            "is_thing": bool(o.is_thing),
            "aabb_min": _vec(o.aabb_min),
            "aabb_max": _vec(o.aabb_max),
            "centroid": _vec(o.centroid),
        }
        for o in result.objects
    ]
    attention = [
        {
            "person_id": int(s.person_id),
            "object_id": int(s.object_id),
            "label": str(s.label),
            "score_gaze": _jsonable(s.score_gaze),
            "score_head": _jsonable(s.score_head),
            "score_total": _jsonable(s.score_total),
            "occluded": bool(s.occluded),
        }
        for s in result.scores
    ]
    return {
        "frame": int(result.index),
        "t": _jsonable(float(result.t)),
        "persons": persons,
        "objects": objects,
        "attention": attention,
        "timings_ms": _jsonable(dict(result.timings_ms)),
    }


class JsonlSink:
    """Writes one JSON object per frame (``result_to_dict``) to a ``.jsonl`` file."""

    def __init__(self, path: PathLike, flush_every: int = 1) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
        self.flush_every = max(int(flush_every), 1)
        self._n = 0

    def write(self, frame: Optional[Frame], result: FrameResult) -> None:
        if self._fh is None:
            raise ValueError("JsonlSink is closed")
        d = result_to_dict(result)
        self._fh.write(json.dumps(d, allow_nan=False, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % self.flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JsonlSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _csv_value(v: Any) -> Any:
    v = _jsonable(v)
    return "" if v is None else v


class CsvSink:
    """One CSV row per (person, object) attention pair; header ``CSV_COLUMNS``."""

    def __init__(self, path: PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh, lineterminator="\n")
        self._writer.writerow(CSV_COLUMNS)

    def write(self, frame: Optional[Frame], result: FrameResult) -> None:
        if self._fh is None:
            raise ValueError("CsvSink is closed")
        t = _csv_value(float(result.t))
        for s in result.scores:
            pid, oid = int(s.person_id), int(s.object_id)
            self._writer.writerow(
                [
                    int(result.index), t, pid, oid, str(s.label),
                    _csv_value(s.score_gaze), _csv_value(s.score_head), _csv_value(s.score_total),
                    int(bool(s.occluded)), int(result.targets.get(pid) == oid),
                ]
            )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "CsvSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------- drawing


def head_axes_endpoints(
    yaw_deg: float, pitch_deg: float, roll_deg: float, origin: Sequence[float], size: float = 50.0
) -> np.ndarray:
    """Pixel endpoints (3, 2) of the head X, Y, Z axes (legacy ``draw_axis`` maths).

    Angles are FSA-Net (yaw, pitch, roll) degrees; ``origin`` is (u, v) pixels.
    The Z row equals ``origin + size * forward.xy`` (section 2 convention).
    """
    p = math.radians(float(pitch_deg))
    y = -math.radians(float(yaw_deg))
    r = math.radians(float(roll_deg))
    axes = np.array(
        [
            [math.cos(y) * math.cos(r), math.cos(p) * math.sin(r) + math.cos(r) * math.sin(p) * math.sin(y)],
            [-math.cos(y) * math.sin(r), math.cos(p) * math.cos(r) - math.sin(p) * math.sin(y) * math.sin(r)],
            [math.sin(y), -math.cos(y) * math.sin(p)],
        ],
        dtype=np.float64,
    )
    o = np.asarray(origin, dtype=np.float64).reshape(2)
    return o[None, :] + float(size) * axes


def _copy_image(image: np.ndarray) -> np.ndarray:
    out = np.array(image, dtype=np.uint8, order="C", copy=True)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


def _make_pt(width: int, height: int):
    lim = 10.0 * max(width, height, 1)

    def _pt(xy: Any) -> Optional[tuple[int, int]]:
        if xy is None:
            return None
        a = np.asarray(xy, dtype=np.float64).reshape(-1)
        if a.size < 2 or not np.isfinite(a[:2]).all():
            return None
        a = np.clip(a[:2], -lim, lim)
        return int(round(a[0])), int(round(a[1]))

    return _pt


def _draw_axes_on(out: np.ndarray, yaw: float, pitch: float, roll: float, origin: Any,
                  size: float, thickness: int) -> None:
    pt = _make_pt(out.shape[1], out.shape[0])
    o = pt(origin)
    if o is None:
        return
    ends = head_axes_endpoints(yaw, pitch, roll, o, size)
    for end, color in zip(ends, ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        e = pt(end)
        if e is not None:
            cv2.line(out, o, e, color, thickness)


def draw_head_axes(
    image: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    origin: Optional[Sequence[float]] = None,
    size: float = 50.0,
    thickness: int = 2,
) -> np.ndarray:
    """Return a COPY of ``image`` with the head axes drawn (X red, Y green, Z blue)."""
    out = _copy_image(image)
    if origin is None:
        origin = ((out.shape[1] - 1) / 2.0, (out.shape[0] - 1) / 2.0)
    _draw_axes_on(out, yaw_deg, pitch_deg, roll_deg, origin, size, int(thickness))
    return out


def _eye_pixel_from_box(box: Any) -> Optional[np.ndarray]:
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.size != 4 or not np.isfinite(b).all():
        return None
    return np.array([(b[0] + b[2]) / 2.0, 0.65 * b[1] + 0.35 * b[3]])


def _finite_vec(v: Any, n: int) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != n or not np.isfinite(a).all():
        return None
    return a


def _cue_end(eye: np.ndarray, eyes_3d: Optional[np.ndarray], d: np.ndarray,
             intrinsics: Optional[Intrinsics], ray_length_m: float, arrow_px: float) -> np.ndarray:
    if intrinsics is not None and eyes_3d is not None and eyes_3d[2] > 0:
        L = float(ray_length_m)
        if d[2] < 0:
            L = min(L, 0.9 * eyes_3d[2] / -d[2])
        end = np.asarray(intrinsics.project(eyes_3d + L * d), dtype=np.float64)
        if np.isfinite(end).all():
            return end
    return eye + float(arrow_px) * d[:2]


def draw_overlay(
    image: np.ndarray,
    result: FrameResult,
    intrinsics: Optional[Intrinsics] = None,
    *,
    ray_length_m: float = 0.5,
    arrow_px: float = 60.0,
) -> np.ndarray:
    """Return a NEW image with persons, cues, targets and objects drawn.

    Never modifies ``image``. Robust to missing boxes, cues, eyes, objects and
    intrinsics. Gaze arrows are yellow-ish (track colour), head arrows white.
    """
    out = _copy_image(image)
    h, w = out.shape[:2]
    pt = _make_pt(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    objects = {int(o.object_id): o for o in result.objects}
    targeted = {int(v) for v in result.targets.values() if v is not None}

    # Objects (need intrinsics to project).
    if intrinsics is not None:
        for oid, o in objects.items():
            c = pt(intrinsics.project(np.asarray(o.centroid, dtype=np.float64)))
            if c is not None:
                cv2.circle(out, c, 4, (0, 255, 255), -1)
                cv2.putText(out, f"{oid}:{o.label}", (c[0] + 5, c[1] - 5), font, 0.4,
                            (0, 255, 255), 1, cv2.LINE_AA)
            if oid in targeted:
                lo = np.asarray(o.aabb_min, dtype=np.float64)
                hi = np.asarray(o.aabb_max, dtype=np.float64)
                corners = np.array(
                    [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
                )
                uv = intrinsics.project(corners)
                if np.isfinite(uv).all():
                    p1 = pt(uv.min(axis=0))
                    p2 = pt(uv.max(axis=0))
                    if p1 is not None and p2 is not None:
                        cv2.rectangle(out, p1, p2, (0, 255, 255), 1)

    for person in result.persons:
        tid = int(person.track_id)
        color = _PALETTE[tid % len(_PALETTE)]
        det = person.detection
        box = det.head_box if det is not None else None
        target = result.targets.get(tid)
        tlabel = objects[target].label if target is not None and target in objects else "-"
        b = _finite_vec(box, 4)
        if b is not None:
            p1, p2 = pt(b[:2]), pt(b[2:])
            if p1 is not None and p2 is not None:
                cv2.rectangle(out, p1, p2, color, 2)
                cv2.putText(out, f"{tid} -> {tlabel}", (p1[0], max(p1[1] - 5, 10)), font, 0.45,
                            color, 1, cv2.LINE_AA)

        eye = _finite_vec(person.eyes_px, 2)
        if eye is None:
            eye = _eye_pixel_from_box(b)
        if eye is None:
            continue
        eye_pt = pt(eye)
        if eye_pt is None:
            continue
        eyes_3d = _finite_vec(person.eyes_3d, 3)

        if person.gaze is not None:
            d = _finite_vec(person.gaze.direction, 3)
            if d is not None:
                e = pt(_cue_end(eye, eyes_3d, d, intrinsics, ray_length_m, arrow_px))
                if e is not None:
                    cv2.arrowedLine(out, eye_pt, e, color, 2, cv2.LINE_AA, tipLength=0.2)
        if person.head is not None:
            eul = _finite_vec(person.head.euler_deg, 3)
            if eul is not None:
                _draw_axes_on(out, eul[0], eul[1], eul[2], eye, 40.0, 2)
            else:
                d = _finite_vec(person.head.direction, 3)
                if d is not None:
                    e = pt(_cue_end(eye, eyes_3d, d, intrinsics, ray_length_m, arrow_px))
                    if e is not None:
                        cv2.arrowedLine(out, eye_pt, e, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2)
        if target is not None and target in objects and intrinsics is not None:
            c = pt(intrinsics.project(np.asarray(objects[target].centroid, dtype=np.float64)))
            if c is not None:
                cv2.line(out, eye_pt, c, color, 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- video


def _even(n: int) -> int:
    n = int(n)
    return max(n - (n % 2), 2)


class VideoSink:
    """Streaming mp4 (``mp4v``) writer; the writer is created on the first frame.

    ``size`` = (width, height) or None (taken from the first image); both are
    forced even. Every image is resized to the writer size (cv2 silently drops
    frames of a different size).
    """

    def __init__(self, path: PathLike, fps: float, size: Optional[Sequence[int]] = None,
                 overlay: bool = True) -> None:
        self.path = Path(path)
        f = None
        try:
            f = float(fps)
        except (TypeError, ValueError):
            pass
        self.fps = f if f is not None and math.isfinite(f) and f > 0 else 30.0
        self.size: Optional[tuple[int, int]] = (
            None if size is None else (_even(size[0]), _even(size[1]))
        )
        self.overlay = bool(overlay)
        self.frames_written = 0
        self._writer: Optional[cv2.VideoWriter] = None
        self._closed = False

    def _open(self, image: np.ndarray) -> None:
        if self.size is None:
            self.size = (_even(image.shape[1]), _even(image.shape[0]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"cv2.VideoWriter cannot open {self.path}")
        self._writer = writer

    def write_image(self, image: np.ndarray) -> None:
        """Append one BGR (or gray) image."""
        if self._closed:
            raise ValueError("VideoSink is closed")
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._writer is None:
            self._open(img)
        assert self.size is not None
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        self._writer.write(np.ascontiguousarray(img))
        self.frames_written += 1

    def write(self, frame: Optional[Frame], result: Optional[FrameResult]) -> None:
        if frame is None:
            return
        if self.overlay and result is not None:
            image = draw_overlay(frame.image, result, frame.intrinsics)
        else:
            image = frame.image
        self.write_image(image)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._closed = True

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def compose_mosaic(
    tiles: Sequence[Optional[np.ndarray]],
    grid: Sequence[int],
    size: Sequence[int],
    keep_aspect: bool = True,
) -> np.ndarray:
    """Compose tiles into one (H, W, 3) uint8 image, in-process.

    ``grid`` = (rows, cols), ``size`` = (width, height); tiles fill cells
    row-major, ``None`` = black cell, gray tiles become BGR; with
    ``keep_aspect`` each tile is letterboxed and centred in its cell.
    """
    rows, cols = int(grid[0]), int(grid[1])
    W, H = int(size[0]), int(size[1])
    if rows <= 0 or cols <= 0 or W <= 0 or H <= 0:
        raise ValueError(f"invalid grid {grid} or size {size}")
    if len(tiles) > rows * cols:
        raise ValueError(f"{len(tiles)} tiles do not fit a {rows}x{cols} grid")
    xs = np.linspace(0, W, cols + 1).round().astype(int)
    ys = np.linspace(0, H, rows + 1).round().astype(int)
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for k, tile in enumerate(tiles):
        if tile is None:
            continue
        r, c = divmod(k, cols)
        x0, x1, y0, y1 = xs[c], xs[c + 1], ys[r], ys[r + 1]
        cw, ch = x1 - x0, y1 - y0
        if cw <= 0 or ch <= 0:
            continue
        t = np.asarray(tile)
        if t.dtype != np.uint8:
            t = np.clip(t, 0, 255).astype(np.uint8)
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        elif t.ndim == 3 and t.shape[2] == 4:
            t = cv2.cvtColor(t, cv2.COLOR_BGRA2BGR)
        th, tw = t.shape[:2]
        if th == 0 or tw == 0:
            continue
        if keep_aspect:
            s = min(cw / tw, ch / th)
            nw, nh = max(int(round(tw * s)), 1), max(int(round(th * s)), 1)
            nw, nh = min(nw, cw), min(nh, ch)
        else:
            nw, nh = cw, ch
        resized = cv2.resize(np.ascontiguousarray(t), (nw, nh), interpolation=cv2.INTER_AREA)
        ox = x0 + (cw - nw) // 2
        oy = y0 + (ch - nh) // 2
        out[oy:oy + nh, ox:ox + nw] = resized
    return out


class DisplaySink:
    """Shows frames with ``cv2.imshow``; disables itself on headless OpenCV.

    Pressing ``q`` or Esc sets ``stop_requested``.
    """

    def __init__(self, window: str = "vfoa", overlay: bool = True, wait_ms: int = 1) -> None:
        self.window = str(window)
        self.overlay = bool(overlay)
        self.wait_ms = max(int(wait_ms), 1)
        self.enabled = True
        self.stop_requested = False
        self._shown = False

    def write(self, frame: Optional[Frame], result: Optional[FrameResult]) -> None:
        if not self.enabled or frame is None:
            return
        image = (
            draw_overlay(frame.image, result, frame.intrinsics)
            if self.overlay and result is not None else frame.image
        )
        try:
            cv2.imshow(self.window, image)
            self._shown = True
            key = cv2.waitKey(self.wait_ms)
        except cv2.error as exc:
            logger.warning("display unavailable (%s); disabling DisplaySink", exc)
            self.enabled = False
            return
        if key is not None and key >= 0 and (key & 0xFF) in (ord("q"), 27):
            self.stop_requested = True

    def close(self) -> None:
        if self._shown:
            try:
                cv2.destroyWindow(self.window)
            except cv2.error:
                pass
            self._shown = False

    def __enter__(self) -> "DisplaySink":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
