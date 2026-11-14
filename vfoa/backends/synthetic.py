"""Synthetic world and model-free backends (ground truth for tests and benchmarks).

:class:`SyntheticWorld` is a deterministic scripted scene rendered with exact,
self-consistent geometry:

* Persons stand BEHIND a table (far, upper image band, torso cut at the table
  line); objects lie ON the table (near, lower band, outer objects deeper); a
  back wall at :data:`Z_WALL` metres fills the background (no segment).
* Every entity is a fronto-parallel billboard: integer half-open pixel
  rectangles ``[x1, x2) x [y1, y2)`` at one constant depth. 3D positions are
  DERIVED from pixel rectangle + depth, so depth, label map, boxes, eye points
  and directions agree exactly. Depth is valid everywhere.
* Camera frame: OpenCV pinhole of the left camera (+X right, +Y down, +Z
  forward, metres). Intrinsics = ``Intrinsics.from_hfov(W, H, 60, baseline=0.1)``;
  the ``camera`` config section is deliberately ignored.

Segment ids: object ``j`` -> ``j + 1`` (unique labels), person ``k`` ->
``n_objects + 1 + k`` (label ``"person"``), optional occluder panel ->
``n_objects + n_persons + 1`` (label ``"panel"``).

Script: the sequence is split in 4 phases of ``max(n_frames // 4, 1)`` frames.
In phase ``p`` person ``k`` looks at object ``(k + p) % n_objects + 1``, except
in phase 2 where even persons look at a neighbour PERSON. With
``synthetic.occluder`` a panel stands between person 0 and object 1 during
phase 0: person 0 still aims at object 1 (gaze/head point there) but
``truth()`` reports the panel (first surface along the gaze);
:attr:`SyntheticWorld.occlusion` documents that pair.

The layout constants are a verbatim port of the lead-validated prototype
(0 target mismatches from 160x90 to 1280x720); supported: 0..3 persons and
0..6 objects. The five ``Synthetic*`` backends serve the world through the
standard backend interfaces with ``calls``/``loaded`` counters and optional
emulated model cost (``synthetic.cost_ms[kind]`` milliseconds of sleep).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from vfoa.backends.base import (
    DepthBackend,
    GazeBackend,
    HeadPoseBackend,
    PersonBackend,
    SegmentationBackend,
)
from vfoa.config import config_hash
from vfoa.types import (
    DepthFrame,
    DirectionEstimate,
    Frame,
    Intrinsics,
    Person,
    PersonDetection,
    SegmentInfo,
    Segmentation,
)

__all__ = [
    "SyntheticWorld",
    "SyntheticSource",
    "SyntheticDepth",
    "SyntheticPersons",
    "SyntheticHeadPose",
    "SyntheticGaze",
    "SyntheticSegmentation",
    "clear_world_cache",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- layout constants
HFOV_DEG = 60.0
BASELINE_M = 0.1
Z_WALL = 7.0
OBJ_Z0, OBJ_DZ = 2.3, 0.5          # object depth = OBJ_Z0 + OBJ_DZ * |xn| / OBJ_XLIM
OBJ_XLIM = 0.78                    # objects spread over xn in [-0.78, 0.78] (NDC)
OBJ_V0, OBJ_DV = 0.64, 0.115       # object centre row (NDC) = OBJ_V0 - OBJ_DV * |xn| / OBJ_XLIM
OBJ_W_M, OBJ_H_M, OBJ_HH_NDC = 0.30, 0.24, 0.14
PER_Z = (3.6, 3.9)                 # person k depth = PER_Z[k % 2]
PER_XLIM = {1: 0.0, 2: 0.45}       # else 0.7
HEAD_V = -0.52                     # head-box centre row (NDC)
TORSO_V = 0.26                     # bottom row of the torso (NDC)
HEAD_W_M, HEAD_H_M, BODY_W_M = 0.22, 0.26, 0.50
SWAY_M, SWAY_PERIOD = 0.05, 60.0   # lateral sway of persons (metres, frames)
N_PHASES = 4
MAX_PERSONS, MAX_OBJECTS = 3, 6
OBJECT_LABELS = ("cup", "book", "laptop", "bottle", "vase", "clock")
PERSON_LABEL = "person"
PANEL_LABEL = "panel"
DETECTION_SCORE = 0.95
_RASTER_CACHE = 8

_Rect = tuple[int, int, int, int]


def _sleep(seconds: float) -> None:
    """Sleep wrapper (module level so tests can monkeypatch it)."""
    time.sleep(seconds)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not n > 1e-12:
        return np.array([0.0, 0.0, -1.0])
    return v / n


def _basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (u, v) perpendicular to unit ``d`` (valid for any direction)."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(d, a))
    v = np.cross(d, u)
    return u, v


def _world_params(**params: Any) -> dict:
    return {
        "width": int(params["width"]),
        "height": int(params["height"]),
        "fps": float(params["fps"]),
        "n_frames": int(params["n_frames"]),
        "n_persons": int(params["n_persons"]),
        "n_objects": int(params["n_objects"]),
        "seed": int(params["seed"]),
        "noise_deg": float(params["noise_deg"]),
        "occluder": bool(params["occluder"]),
        "sway_m": float(params["sway_m"]),
    }


class _Raster:
    """Rasterised frame: label map, depth and lazily computed per-segment stats."""

    __slots__ = ("label", "depth", "stats")

    def __init__(self, label: np.ndarray, depth: np.ndarray) -> None:
        self.label = label
        self.depth = depth
        self.stats: dict[int, Optional[tuple]] = {}


class SyntheticWorld:
    """Deterministic scripted scene (see the module docstring).

    Parameters: image ``width``/``height`` (pixels), ``fps``, ``n_frames`` of
    the scripted sequence, ``n_persons`` (0..3), ``n_objects`` (0..6), ``seed``
    (texture, palette and direction noise), ``noise_deg`` (angular noise of the
    gaze/head directions; deviation ``|N(0,1)| * noise_deg``), ``occluder``
    (panel scenario) and ``sway_m`` (lateral sway amplitude of persons, metres).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        fps: float = 30.0,
        n_frames: int = 120,
        n_persons: int = 2,
        n_objects: int = 4,
        seed: int = 0,
        noise_deg: float = 0.0,
        occluder: bool = False,
        sway_m: float = SWAY_M,
    ) -> None:
        p = _world_params(
            width=width, height=height, fps=fps, n_frames=n_frames, n_persons=n_persons,
            n_objects=n_objects, seed=seed, noise_deg=noise_deg, occluder=occluder, sway_m=sway_m,
        )
        if not 0 <= p["n_persons"] <= MAX_PERSONS:
            raise ValueError(f"synthetic.n_persons must be in 0..{MAX_PERSONS}, got {p['n_persons']}")
        if not 0 <= p["n_objects"] <= MAX_OBJECTS:
            raise ValueError(f"synthetic.n_objects must be in 0..{MAX_OBJECTS}, got {p['n_objects']}")
        if p["width"] < 16 or p["height"] < 16:
            raise ValueError(f"synthetic resolution too small: {p['width']}x{p['height']}")
        if not p["fps"] > 0:
            raise ValueError(f"synthetic.fps must be positive, got {p['fps']}")
        if p["n_frames"] < 1:
            raise ValueError(f"synthetic.n_frames must be >= 1, got {p['n_frames']}")
        if p["seed"] < 0:
            raise ValueError(f"synthetic.seed must be >= 0, got {p['seed']}")
        self.params = p
        self.width, self.height = p["width"], p["height"]
        self.fps, self.n_frames = p["fps"], p["n_frames"]
        self.n_persons, self.n_objects = p["n_persons"], p["n_objects"]
        self.seed, self.noise_deg = p["seed"], p["noise_deg"]
        self.occluder, self.sway_m = p["occluder"], p["sway_m"]
        #: Hash of the world parameters (used in the source fingerprint).
        self.key = config_hash({"synthetic": p}, ["synthetic"])
        self.intrinsics = Intrinsics.from_hfov(self.width, self.height, HFOV_DEG, baseline=BASELINE_M)
        self._T = math.tan(math.radians(HFOV_DEG) / 2.0)
        self._f = self.intrinsics.fx
        self._cx, self._cy = self.intrinsics.cx, self.intrinsics.cy
        #: Frames per script phase.
        self.phase_len = max(self.n_frames // N_PHASES, 1)

        W, H, nO = self.width, self.height, self.n_objects
        xs = [0.0] if nO == 1 else [float(x) for x in np.linspace(-OBJ_XLIM, OBJ_XLIM, nO)]
        self._obj: list[tuple[_Rect, float]] = []
        for j in range(nO):
            a = abs(xs[j]) / OBJ_XLIM
            z = OBJ_Z0 + OBJ_DZ * a
            uc = int(round(self._cx + xs[j] * W / 2))
            vc = int(round(self._cy + (OBJ_V0 - OBJ_DV * a) * H / 2))
            w_m = OBJ_W_M * min(1.0, 4.0 / nO)
            hw = max(int(round(self._f * w_m / 2 / z)), 3)
            hh = max(min(int(round(self._f * OBJ_H_M / 2 / z)), int(round(OBJ_HH_NDC * H / 2))), 3)
            self._obj.append(((uc - hw, vc - hh, uc + hw, vc + hh), z))
        lim = PER_XLIM.get(self.n_persons, 0.7)
        nP = self.n_persons
        self._px = [0.0] if nP == 1 else [float(x) for x in np.linspace(-lim, lim, nP)]

        self._labels: dict[int, str] = {j + 1: OBJECT_LABELS[j] for j in range(nO)}
        for k in range(nP):
            self._labels[self.person_segment_id(k)] = PERSON_LABEL
        self.panel_id: Optional[int] = None
        #: Documented occlusion: None, or {"person": 0, "target": 1, "occluder": panel id}
        #: (person 0's aim at object 1 is blocked by the panel during phase 0).
        self.occlusion: Optional[dict[str, int]] = None
        if self.occluder and nP >= 1 and nO >= 1:
            self.panel_id = nO + nP + 1
            self._labels[self.panel_id] = PANEL_LABEL
            self.occlusion = {"person": 0, "target": 1, "occluder": self.panel_id}

        rng = np.random.default_rng([self.seed, 1])
        n_ids = nO + nP + 2
        self._palette = rng.integers(50, 220, size=(n_ids, 3)).astype(np.uint8)
        self._palette[0] = (150, 160, 170)  # wall
        self._texture: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._rasters: "OrderedDict[int, _Raster]" = OrderedDict()

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SyntheticWorld":
        """World for ``cfg.synthetic``, memoised on the world parameters.

        The memo key is the hash of the synthetic section without ``cost_ms``
        (which does not change the world); other sections are ignored.
        """
        s = cfg.get("synthetic") or {}
        p = _world_params(
            width=s.get("width", 640), height=s.get("height", 360), fps=s.get("fps", 30.0),
            n_frames=s.get("n_frames", 120), n_persons=s.get("n_persons", 2),
            n_objects=s.get("n_objects", 4), seed=s.get("seed", 0),
            noise_deg=s.get("noise_deg", 0.0), occluder=s.get("occluder", False),
            sway_m=s.get("sway_m", SWAY_M),
        )
        key = config_hash({"synthetic": p}, ["synthetic"])
        with _WORLDS_LOCK:
            world = _WORLDS.get(key)
            if world is None:
                world = cls(**p)
                _WORLDS[key] = world
            return world

    # ------------------------------------------------------------ ids / labels
    def object_segment_id(self, j: int) -> int:
        """Segment id of object ``j`` (``j + 1``)."""
        return j + 1

    def person_segment_id(self, k: int) -> int:
        """Segment id of person ``k`` (``n_objects + 1 + k``)."""
        return self.n_objects + 1 + k

    def label(self, sid: int) -> str:
        """Label of segment ``sid``."""
        return self._labels[sid]

    # ------------------------------------------------------------ geometry
    def phase(self, i: int) -> int:
        """Script phase (0..3) of frame ``i``."""
        return (i // self.phase_len) % N_PHASES

    def _person(self, i: int, k: int, sway: Optional[float] = None):
        """-> head rect, body rect, eye pixel (2,), eye point (3,), depth."""
        sway = self.sway_m if sway is None else sway
        z = PER_Z[k % 2]
        X = self._px[k] * z * self._T + sway * math.sin(2 * math.pi * i / SWAY_PERIOD + 1.7 * k)
        uc = int(round(self._cx + self._f * X / z))
        vc = int(round(self._cy + HEAD_V * self.height / 2))
        hw = max(int(round(self._f * HEAD_W_M / 2 / z)), 4)
        hh = max(int(round(self._f * HEAD_H_M / 2 / z)), 4)
        bw = max(int(round(self._f * BODY_W_M / 2 / z)), hw + 1)
        head = (uc - hw, vc - hh, uc + hw, vc + hh)
        yb = max(int(round(self._cy + TORSO_V * self.height / 2)), vc + hh + 4)
        body = (uc - bw, vc + hh, uc + bw, yb)
        e = np.array([(head[0] + head[2]) / 2.0, 0.65 * head[1] + 0.35 * head[3]])
        E = np.array([(e[0] - self._cx) * z / self._f, (e[1] - self._cy) * z / self._f, z])
        return head, body, e, E, z

    def _center_static(self, sid: int) -> np.ndarray:
        (x1, y1, x2, y2), z = self._obj[sid - 1]
        u, v = (x1 + x2 - 1) / 2.0, (y1 + y2 - 1) / 2.0
        return np.array([(u - self._cx) * z / self._f, (v - self._cy) * z / self._f, z])

    def occluder_active(self, i: int) -> bool:
        """True when the occluder panel is present in frame ``i`` (phase 0 only)."""
        return self.panel_id is not None and self.phase(i) == 0

    def _occ(self, i: int) -> Optional[tuple[_Rect, float]]:
        """Occluder panel (rect, depth) between person 0 and object 1 during phase 0."""
        if not self.occluder_active(i):
            return None
        head, _body, _e, E, _z = self._person(0, 0, sway=0.0)
        C = self._center_static(1)
        zo = float(E[2] + 0.75 * (C[2] - E[2]))  # pierced at fraction 0.75

        def proj(t: float) -> tuple[float, float]:
            S = E + t * (C - E)
            return self._cx + self._f * S[0] / S[2], self._cy + self._f * S[1] / S[2]

        us = [proj(t)[0] for t in (0.15, 0.75)]
        half = self._f * 0.22 / zo
        x1 = int(math.floor(min(us) - half))
        x2 = int(math.ceil(max(us) + half))
        y1 = head[3] + 2
        y2 = int(round(proj(0.75)[1] + self._f * 0.10 / zo))
        return (x1, y1, x2, y2), zo

    def _raster(self, i: int) -> _Raster:
        """Painter rasterisation (far -> near, ties by id), LRU-cached (internal arrays)."""
        i = int(i)
        if i < 0:
            raise ValueError(f"frame index must be >= 0, got {i}")
        with self._lock:
            r = self._rasters.get(i)
            if r is not None:
                self._rasters.move_to_end(i)
                return r
        W, H = self.width, self.height
        lab = np.zeros((H, W), np.int32)
        dep = np.full((H, W), Z_WALL, np.float32)
        items = []
        for k in range(self.n_persons):
            head, body, _e, _E, z = self._person(i, k)
            items.append((z, self.person_segment_id(k), [head, body]))
        for j, (rect, z) in enumerate(self._obj):
            items.append((z, j + 1, [rect]))
        occ = self._occ(i)
        if occ is not None:
            items.append((occ[1], self.panel_id, [occ[0]]))
        for z, sid, rects in sorted(items, key=lambda t: (-t[0], t[1])):
            for (x1, y1, x2, y2) in rects:
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
                if x2 > x1 and y2 > y1:
                    lab[y1:y2, x1:x2] = sid
                    dep[y1:y2, x1:x2] = z
        r = _Raster(lab, dep)
        with self._lock:
            existing = self._rasters.get(i)
            if existing is not None:
                return existing
            self._rasters[i] = r
            while len(self._rasters) > _RASTER_CACHE:
                self._rasters.popitem(last=False)
        return r

    def _stats(self, i: int, sid: int) -> Optional[tuple]:
        """(median u, median v, umin, umax, vmin, vmax, z) of the visible mask, or None."""
        r = self._raster(i)
        if sid in r.stats:
            return r.stats[sid]
        vs, us = np.nonzero(r.label == sid)
        if len(vs) == 0:
            st = None
        else:
            st = (
                float(np.median(us)), float(np.median(vs)),
                int(us.min()), int(us.max()), int(vs.min()), int(vs.max()),
                float(r.depth[vs[0], us[0]]),
            )
        with self._lock:
            r.stats[sid] = st
        return st

    def _unproject(self, u: float, v: float, z: float) -> np.ndarray:
        return np.array([(u - self._cx) * z / self._f, (v - self._cy) * z / self._f, z], dtype=np.float64)

    # ------------------------------------------------------------ public rendering
    def frame(self, i: int) -> Frame:
        """BGR uint8 frame ``i``: segment palette + static texture + skin-tone heads."""
        r = self._raster(i)
        if self._texture is None:
            tex = np.random.default_rng([self.seed, 2]).integers(
                -6, 7, size=(self.height, self.width, 3)
            ).astype(np.int16)
            self._texture = tex
        img = self._palette[r.label].astype(np.int16) + self._texture
        for k in range(self.n_persons):
            x1, y1, x2, y2 = self._person(i, k)[0]
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, self.width), min(y2, self.height)
            if x2 > x1 and y2 > y1:
                img[y1:y2, x1:x2] = (125, 160, 215)
        image = np.clip(img, 0, 255).astype(np.uint8)
        return Frame(index=int(i), t=int(i) / self.fps, image=image, intrinsics=self.intrinsics)

    def depth(self, i: int) -> DepthFrame:
        """Depth of frame ``i``: float32 metres (wall at 7 m), valid everywhere."""
        r = self._raster(i)
        return DepthFrame(
            depth=r.depth.copy(), valid=np.ones(r.depth.shape, dtype=bool), intrinsics=self.intrinsics
        )

    def segmentation(self, i: int) -> Segmentation:
        """Label map (int32) and the segments visible in frame ``i`` (sorted by id)."""
        r = self._raster(i)
        ids = [int(s) for s in np.unique(r.label) if s > 0]
        segments = [SegmentInfo(id=s, label=self._labels[s], is_thing=True, score=1.0) for s in ids]
        return Segmentation(label_map=r.label.copy(), segments=segments, frame_index=int(i))

    def head_box(self, i: int, k: int) -> np.ndarray:
        """Head box of person ``k`` (float32 [x1, y1, x2, y2], half-open pixel rect)."""
        return np.asarray(self._person(i, k)[0], dtype=np.float32)

    def eye_px(self, i: int, k: int) -> np.ndarray:
        """Eye pixel (u, v) of person ``k``: ((x1+x2)/2, 0.65*y1 + 0.35*y2) of the head box."""
        return self._person(i, k)[2].copy()

    def eye_point(self, i: int, k: int) -> np.ndarray:
        """3D eye point (camera frame, metres) of person ``k``; depth there equals its Z."""
        return self._person(i, k)[3].copy()

    def detections(self, i: int) -> list[PersonDetection]:
        """Person detections of frame ``i`` in person-index order."""
        out = []
        W, H = self.width, self.height
        for k in range(self.n_persons):
            head, body, e, _E, _z = self._person(i, k)
            hx1, hy1, hx2, hy2 = head
            bx1, by1, bx2, by2 = body
            hw = (hx2 - hx1) / 2.0
            hh = (hy2 - hy1) / 2.0
            uc = (hx1 + hx2) / 2.0
            bw = (bx2 - bx1) / 2.0
            bh = float(by2 - by1)
            kp = np.zeros((17, 3), np.float32)
            kp[0] = (uc, e[1] + 0.35 * hh, 1.0)                 # nose
            kp[1] = (e[0] + 0.4 * hw, e[1], 1.0)                # l_eye (image right)
            kp[2] = (e[0] - 0.4 * hw, e[1], 1.0)                # r_eye
            kp[3] = (uc + 0.85 * hw, e[1] + 0.1 * hh, 1.0)      # l_ear
            kp[4] = (uc - 0.85 * hw, e[1] + 0.1 * hh, 1.0)      # r_ear
            fr = {5: (0.8, 0.08), 7: (0.95, 0.4), 9: (0.9, 0.75), 11: (0.5, 0.95)}
            for idx, (fx, fy) in fr.items():
                kp[idx] = (uc + fx * bw, by1 + fy * bh, 1.0)
                kp[idx + 1] = (uc - fx * bw, by1 + fy * bh, 1.0)
            box = np.array(
                [max(min(hx1, bx1), 0), max(min(hy1, by1), 0), min(max(hx2, bx2), W), min(max(hy2, by2), H)],
                dtype=np.float32,
            )
            out.append(PersonDetection(head_box=np.asarray(head, np.float32), body_box=box,
                                       keypoints=kp, score=DETECTION_SCORE))
        return out

    def center(self, i: int, sid: int) -> np.ndarray:
        """3D centre of segment ``sid`` in frame ``i``: per-axis median pixel of its
        VISIBLE mask unprojected at its depth (SceneBuilder's rule). A segment that
        is not visible falls back to its unclipped rectangle centre."""
        st = self._stats(i, sid)
        if st is not None:
            return self._unproject(st[0], st[1], st[6])
        rect, z = self._entity_rect(i, sid)
        x1, y1, x2, y2 = rect
        return self._unproject((x1 + x2 - 1) / 2.0, (y1 + y2 - 1) / 2.0, z)

    def aabb(self, i: int, sid: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Axis-aligned box (min, max) of the visible mask of ``sid`` (metres), or None."""
        st = self._stats(i, sid)
        if st is None:
            return None
        z = st[6]
        a = self._unproject(st[2], st[4], z)
        b = self._unproject(st[3], st[5], z)
        return np.minimum(a, b), np.maximum(a, b)

    def _entity_rect(self, i: int, sid: int) -> tuple[_Rect, float]:
        if 1 <= sid <= self.n_objects:
            return self._obj[sid - 1]
        k = sid - self.n_objects - 1
        if 0 <= k < self.n_persons:
            head, _body, _e, _E, z = self._person(i, k)
            return head, z
        if sid == self.panel_id:
            occ = self._occ(0)
            if occ is not None:
                return occ
        raise KeyError(f"unknown segment id {sid}")

    # ------------------------------------------------------------ script / truth
    def aim(self, i: int) -> dict[int, Optional[int]]:
        """Scripted segment each person aims at: {person index: segment id | None}."""
        p = self.phase(i)
        out: dict[int, Optional[int]] = {}
        for k in range(self.n_persons):
            if p == 2 and k % 2 == 0 and self.n_persons >= 2:
                nb = k + 1 if k + 1 < self.n_persons else k - 1
                out[k] = self.person_segment_id(nb)
            elif self.n_objects > 0:
                out[k] = (k + p) % self.n_objects + 1
            else:
                out[k] = None
        return out

    def truth(self, i: int) -> dict[int, Optional[int]]:
        """Ground-truth attended segment (first surface along the gaze) per person index."""
        out = self.aim(i)
        if self.occluder_active(i):
            out[0] = self.panel_id
        return out

    def _direction(self, i: int, k: int, cue: int) -> np.ndarray:
        if not 0 <= k < self.n_persons:
            raise IndexError(f"person index {k} out of range 0..{self.n_persons - 1}")
        target = self.aim(i)[k]
        if target is None:
            g = np.array([0.0, 0.0, -1.0])
        else:
            g = _normalize(self.center(i, target) - self.eye_point(i, k))
        if self.noise_deg > 0:
            rng = np.random.default_rng([self.seed, int(i), int(k), cue])
            a = math.radians(abs(rng.normal(0.0, 1.0)) * self.noise_deg)
            phi = rng.uniform(0.0, 2.0 * math.pi)
            u1, u2 = _basis(g)
            g = _normalize(math.cos(a) * g + math.sin(a) * (math.cos(phi) * u1 + math.sin(phi) * u2))
        return g

    def gaze(self, i: int, k: int) -> np.ndarray:
        """Unit gaze direction of person ``k`` (eye point -> aimed segment centre)."""
        return self._direction(i, k, 0)

    def head(self, i: int, k: int) -> np.ndarray:
        """Unit head direction of person ``k`` (same aim as the gaze, independent noise)."""
        return self._direction(i, k, 1)

    def match_person(self, i: int, head_box: Optional[np.ndarray]) -> Optional[int]:
        """World person index whose head-box centre is nearest to ``head_box``'s centre,
        accepted if the distance <= max(box width, height); else None."""
        if head_box is None or self.n_persons == 0:
            return None
        b = np.asarray(head_box, dtype=np.float64).reshape(-1)
        if b.shape != (4,) or not np.isfinite(b).all():
            return None
        c = np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])
        best, best_d = None, math.inf
        for k in range(self.n_persons):
            h = self._person(i, k)[0]
            d = float(np.hypot((h[0] + h[2]) / 2.0 - c[0], (h[1] + h[3]) / 2.0 - c[1]))
            if d < best_d:
                best, best_d = k, d
        if best_d <= max(b[2] - b[0], b[3] - b[1]):
            return best
        return None

    def source(self, start: int = 0, max_frames: Optional[int] = None) -> "SyntheticSource":
        """Frame source over frames ``start .. n_frames-1`` (at most ``max_frames``)."""
        return SyntheticSource(self, start=start, max_frames=max_frames)


_WORLDS: dict[str, SyntheticWorld] = {}
_WORLDS_LOCK = threading.Lock()


def clear_world_cache() -> None:
    """Forget the memoised worlds of :meth:`SyntheticWorld.from_config`."""
    with _WORLDS_LOCK:
        _WORLDS.clear()


class SyntheticSource:
    """Re-iterable frame source over a :class:`SyntheticWorld` (FrameSource duck type).

    Attributes: ``fps``, ``width``, ``height``, ``n_frames`` (frames yielded),
    ``fingerprint`` (``"synthetic:<world key>"``), ``intrinsics``.
    """

    def __init__(self, world: SyntheticWorld, start: int = 0, max_frames: Optional[int] = None) -> None:
        start = int(start or 0)
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}")
        self.world = world
        self.start = start
        stop = world.n_frames
        if max_frames is not None:
            stop = min(stop, start + int(max_frames))
        self.stop = max(stop, start)
        self.fps = world.fps
        self.width, self.height = world.width, world.height
        self.n_frames: Optional[int] = self.stop - self.start
        self.fingerprint = f"synthetic:{world.key}"
        self.intrinsics = world.intrinsics
        self.closed = False

    def __iter__(self) -> Iterator[Frame]:
        for i in range(self.start, self.stop):
            if self.closed:
                return
            yield self.world.frame(i)

    def __len__(self) -> int:
        return self.stop - self.start

    def __enter__(self) -> "SyntheticSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------ backends
class _SyntheticMixin:
    """Shared behaviour: world lookup, ``calls``/``loaded`` counters, emulated cost."""

    name = "synthetic"
    kind: str

    def _init_counters(self) -> None:
        self.calls = 0
        self.loaded = 0
        self.slept_ms = 0.0
        self.world: Optional[SyntheticWorld] = None

    def _load(self) -> None:
        self.world = SyntheticWorld.from_config(self.cfg)  # type: ignore[attr-defined]
        self.loaded += 1

    def _world(self) -> SyntheticWorld:
        if self.world is None:
            self.load()  # type: ignore[attr-defined]
        assert self.world is not None
        return self.world

    def _tick(self) -> None:
        self.calls += 1
        cost = (self.cfg.get("synthetic") or {}).get("cost_ms") or {}  # type: ignore[attr-defined]
        ms = float(cost.get(self.kind, 0) or 0)
        if ms > 0:
            _sleep(ms / 1000.0)
            self.slept_ms += ms

    def _directions(self, frame: Frame, persons: Sequence[Person], cue: str) -> dict[int, DirectionEstimate]:
        world = self._world()
        out: dict[int, DirectionEstimate] = {}
        for p in persons:
            k = world.match_person(frame.index, p.detection.head_box if p.detection is not None else None)
            if k is None:
                continue
            if cue == "head":
                d = world.head(frame.index, k)
                yaw = math.degrees(math.asin(float(np.clip(-d[0], -1.0, 1.0))))
                pitch = math.degrees(math.atan2(-d[1], -d[2]))
                out[p.track_id] = DirectionEstimate(
                    d, 1.0, euler_deg=np.array([yaw, pitch, 0.0]), frame_index=frame.index
                )
            else:
                out[p.track_id] = DirectionEstimate(world.gaze(frame.index, k), 1.0, frame_index=frame.index)
        return out


class SyntheticDepth(_SyntheticMixin, DepthBackend):
    """Depth from the synthetic world (exact, valid everywhere)."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._init_counters()

    def compute(self, frame: Frame) -> DepthFrame:
        self._tick()
        return self._world().depth(frame.index)


class SyntheticPersons(_SyntheticMixin, PersonBackend):
    """Person detections from the synthetic world (person-index order)."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._init_counters()

    def detect(self, frame: Frame) -> list[PersonDetection]:
        self._tick()
        return self._world().detections(frame.index)


class SyntheticHeadPose(_SyntheticMixin, HeadPoseBackend):
    """Head directions from the world, keyed by the track ids of the given persons."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._init_counters()

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        self._tick()
        return self._directions(frame, persons, "head")


class SyntheticGaze(_SyntheticMixin, GazeBackend):
    """Gaze directions from the world; ``observe`` only counts calls (``observed``)."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._init_counters()
        self.observed = 0

    def observe(self, frame: Frame, persons: Sequence[Person]) -> None:
        self.observed += 1

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        self._tick()
        return self._directions(frame, persons, "gaze")


class SyntheticSegmentation(_SyntheticMixin, SegmentationBackend):
    """Segmentation (label map + segments) from the synthetic world."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)
        self._init_counters()

    def segment(self, frame: Frame) -> Segmentation:
        self._tick()
        return self._world().segmentation(frame.index)
