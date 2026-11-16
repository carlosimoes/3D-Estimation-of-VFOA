"""Persistent multi-person tracker (Hungarian IoU association, ByteTrack-style).

Boxes are ``[x1, y1, x2, y2]`` in pixels of the image they were detected in.
Track ids start at 0, increase monotonically, are never reused or duplicated
within a frame and persist for the whole stream: only :meth:`PersonTracker.reset`
restarts them (the cross-batch id collisions of issue #12 and the
previous-frame-only matching of issue #18 are gone).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from vfoa.types import Person, PersonDetection

logger = logging.getLogger(__name__)

__all__ = ["iou_matrix", "box_iou", "PersonTracker"]

_DEFAULTS: dict[str, Any] = {
    "type": "byte",
    "box": "body",
    "iou_thr": 0.3,
    "max_age": 30,
    "high_thr": 0.5,
    "low_thr": 0.1,
    "motion": True,
    "max_predict_frames": 5,
}
_TYPES = ("byte", "greedy_iou")
_NO_MATCH_COST = 1e6


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of boxes ``a`` (N,4) and ``b`` (M,4) ``[x1,y1,x2,y2]`` px -> (N, M) float64.

    Zero-area boxes or a zero union give 0. Empty inputs give an empty matrix.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    valid = (union > 0) & (area_a[:, None] > 0) & (area_b[None, :] > 0)
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=valid)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two ``[x1,y1,x2,y2]`` boxes (0 for degenerate boxes)."""
    return float(iou_matrix(np.asarray(a)[None], np.asarray(b)[None])[0, 0])


def _finite_box(box: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if box is None:
        return None
    arr = np.asarray(box, dtype=np.float64).reshape(-1)
    if arr.shape != (4,) or not np.all(np.isfinite(arr)):
        return None
    return arr


@dataclass
class _Track:
    track_id: int
    head: Optional[np.ndarray]
    body: Optional[np.ndarray]
    vel: np.ndarray
    age: int
    last_frame: int
    hits: int
    has_vel: bool = False

    def box(self, kind: str) -> Optional[np.ndarray]:
        return self.head if kind == "head" else self.body


def _centre(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])


class PersonTracker:
    """Associates per-frame :class:`PersonDetection` lists into persistent tracks.

    Parameters come from defaults <- ``cfg.tracking`` <- keyword overrides, so the
    legacy wrappers can build a tracker without a :class:`vfoa.config.Config`.
    Optional keys not in the config DEFAULTS: ``motion`` (constant-velocity box
    prediction, byte only) and ``max_predict_frames``.

    ``box`` (``tracking.box``) picks the box kind used for IoU matching; the
    default is ``"body"``. A small head that moves more than about a third of
    its width per frame has IoU below ``iou_thr`` with its previous box, so no
    velocity is ever learnt and the person gets a new id on every frame (the
    fast profile on museum.mp4: 258 ids over 552 frames on head boxes, 47 on
    body boxes). When the preferred kind is missing on the track or on the
    detection, the other kind is used, so head-only detections (for example
    from thesis_final/BoxTracker.py) are still tracked. The thesis profile
    keeps ``"head"``.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **overrides: Any) -> None:
        params = dict(_DEFAULTS)
        if cfg is not None:
            section = cfg.get("tracking", {}) if hasattr(cfg, "get") else {}
            params.update(dict(section or {}))
        params.update(overrides)
        if params["type"] not in _TYPES:
            raise ValueError(f"tracking.type must be one of {_TYPES}, got {params['type']!r}")
        if params["box"] not in ("head", "body"):
            raise ValueError(f"tracking.box must be 'head' or 'body', got {params['box']!r}")
        self.params = params
        self.type: str = params["type"]
        self.box_kind: str = params["box"]
        self.iou_thr = float(params["iou_thr"])
        self.max_age = int(params["max_age"])
        self.high_thr = float(params["high_thr"])
        self.low_thr = float(params["low_thr"])
        self.motion = bool(params["motion"]) and self.type == "byte"
        self.max_predict_frames = max(int(params["max_predict_frames"]), 1)
        self._tracks: list[_Track] = []
        self._next_id = 0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget every track and restart ids at 0 (start of a new stream)."""
        self._tracks = []
        self._next_id = 0

    @property
    def next_id(self) -> int:
        return self._next_id

    @property
    def active_ids(self) -> list[int]:
        return [t.track_id for t in self._tracks]

    def _new_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # ---------------------------------------------------------------- scoring
    def _track_box(self, track: _Track, kind: str, frame_index: int) -> Optional[np.ndarray]:
        box = track.box(kind)
        if box is None:
            return None
        if self.motion and track.has_vel:
            steps = float(np.clip(frame_index - track.last_frame, 1, self.max_predict_frames))
            shift = np.array([track.vel[0], track.vel[1], track.vel[0], track.vel[1]]) * steps
            box = box + shift
        return box

    def _iou(self, tracks: list[_Track], dets: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]],
             frame_index: int) -> np.ndarray:
        """IoU (T, D) using tracking.box when both sides have it, else the other kind."""
        out = np.zeros((len(tracks), len(dets)), dtype=np.float64)
        if not tracks or not dets:
            return out
        other = "body" if self.box_kind == "head" else "head"
        for kind_idx, kind in enumerate((self.box_kind, other)):
            t_boxes = [self._track_box(t, kind, frame_index) for t in tracks]
            d_boxes = [d[0] if kind == "head" else d[1] for d in dets]
            ti = [i for i, b in enumerate(t_boxes) if b is not None]
            di = [j for j, b in enumerate(d_boxes) if b is not None]
            if not ti or not di:
                continue
            m = iou_matrix(np.stack([t_boxes[i] for i in ti]), np.stack([d_boxes[j] for j in di]))
            sub = out[np.ix_(ti, di)]
            if kind_idx == 0:
                out[np.ix_(ti, di)] = m
            else:
                # Only fill pairs where the preferred kind was not available on both sides.
                pref_t = np.array([tracks[i].box(self.box_kind) is not None for i in ti])
                pref_d = np.array([
                    (dets[j][0] if self.box_kind == "head" else dets[j][1]) is not None for j in di
                ])
                use = ~(pref_t[:, None] & pref_d[None, :])
                out[np.ix_(ti, di)] = np.where(use, m, sub)
        return out

    def _assign(self, iou: np.ndarray) -> list[tuple[int, int]]:
        """Matched (row, col) pairs with IoU > iou_thr."""
        if iou.size == 0:
            return []
        gate = iou > self.iou_thr
        if not gate.any():
            return []
        if self.type == "byte":
            cost = np.where(gate, 1.0 - iou, _NO_MATCH_COST)
            rows, cols = linear_sum_assignment(cost)
            return [(int(r), int(c)) for r, c in zip(rows, cols) if gate[r, c]]
        # greedy_iou: global greedy by descending IoU, each row/col used once.
        flat = np.argsort(-iou, axis=None, kind="stable")
        used_r: set[int] = set()
        used_c: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for k in flat:
            r, c = divmod(int(k), iou.shape[1])
            if not gate[r, c]:
                break
            if r in used_r or c in used_c:
                continue
            used_r.add(r)
            used_c.add(c)
            pairs.append((r, c))
        return pairs

    # ----------------------------------------------------------------- update
    def update(self, detections: Sequence[PersonDetection], frame_index: int) -> list[Person]:
        """Associate ``detections`` of frame ``frame_index`` with the existing tracks.

        Returns one :class:`Person` per kept detection, in detection order, each
        holding the SAME detection object. Low-score detections (byte) only
        continue existing tracks; detections below ``low_thr`` are dropped.
        """
        frame_index = int(frame_index)
        dets = list(detections or [])
        boxes = [(_finite_box(d.head_box), _finite_box(d.body_box)) for d in dets]
        has_box = [h is not None or b is not None for h, b in boxes]

        if self.type == "byte":
            scores = [float(getattr(d, "score", 1.0)) for d in dets]
            high = [j for j, s in enumerate(scores) if s >= self.high_thr]
            low = [j for j, s in enumerate(scores) if self.low_thr <= s < self.high_thr]
        else:
            high = list(range(len(dets)))
            low = []

        assigned: dict[int, int] = {}  # detection index -> track index
        high_b = [j for j in high if has_box[j]]
        low_b = [j for j in low if has_box[j]]

        # Stage 1: high-score detections against every track.
        iou1 = self._iou(self._tracks, [boxes[j] for j in high_b], frame_index)
        for r, c in self._assign(iou1):
            assigned[high_b[c]] = r
        # Stage 2: low-score detections against the still unmatched tracks.
        if low_b:
            free = [i for i in range(len(self._tracks)) if i not in assigned.values()]
            iou2 = self._iou([self._tracks[i] for i in free], [boxes[j] for j in low_b], frame_index)
            for r, c in self._assign(iou2):
                assigned[low_b[c]] = free[r]

        matched_tracks = set(assigned.values())
        high_set = set(high)
        persons: list[Person] = []
        new_tracks: list[_Track] = []
        for j, det in enumerate(dets):
            if j in assigned:
                track = self._tracks[assigned[j]]
                self._apply_match(track, boxes[j], frame_index)
                persons.append(Person(track_id=track.track_id, detection=det))
            elif j in high_set:
                tid = self._new_id()
                if has_box[j]:
                    new_tracks.append(_Track(
                        track_id=tid, head=boxes[j][0], body=boxes[j][1], vel=np.zeros(2),
                        age=0, last_frame=frame_index, hits=1,
                    ))
                persons.append(Person(track_id=tid, detection=det))
            # unmatched low-score (or dropped) detections produce no Person

        survivors: list[_Track] = []
        for i, track in enumerate(self._tracks):
            if i not in matched_tracks:
                track.age += 1
                if track.age > self.max_age:
                    continue
            survivors.append(track)
        self._tracks = survivors + new_tracks
        return persons

    def _apply_match(self, track: _Track, boxes: tuple[Optional[np.ndarray], Optional[np.ndarray]],
                     frame_index: int) -> None:
        head, body = boxes
        kind = self.box_kind if (track.box(self.box_kind) is not None
                                 and (head if self.box_kind == "head" else body) is not None) else None
        if kind is None:
            other = "body" if self.box_kind == "head" else "head"
            if track.box(other) is not None and (head if other == "head" else body) is not None:
                kind = other
        if kind is not None:
            old = track.box(kind)
            new = head if kind == "head" else body
            gap = max(frame_index - track.last_frame, 1)
            v = (_centre(new) - _centre(old)) / gap
            track.vel = v if not track.has_vel else 0.5 * track.vel + 0.5 * v
            track.has_vel = True
        # Store the boxes as observed (a missing kind is not kept stale).
        track.head = head
        track.body = body
        track.age = 0
        track.last_frame = frame_index
        track.hits += 1
