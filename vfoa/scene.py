"""Metric scene objects from segmentation + depth, robust 3D eyes and own-segment matching.

Everything lives in the camera frame of the left camera (OpenCV pinhole,
+X right, +Y down, +Z forward, metres). Objects are built directly from the
NumPy arrays of a :class:`~vfoa.types.Segmentation` and a
:class:`~vfoa.types.DepthFrame`:

* only pixels on a ``scene.point_stride`` grid are unprojected (no full-frame
  backprojection, no PLY round-trip, no open3d on the default path, #32);
* depth is metric (``Z > 0``), never a normalised disparity image (#1);
* invalid depth (``valid == False``, 0, NaN, inf, negative) is removed BEFORE
  any geometry, so a bad pixel can never end up in an AABB (#11);
* object ids are persistent across segmentation refreshes through
  :class:`ObjectRegistry` (needed by temporal target smoothing).

Eyes are located with a median over the valid depths of a patch clipped to the
image and the head box; the patch grows when empty and the result is ``None``
rather than NaN near borders or holes (#14, #16). Each person's own
``person`` segment is found by box overlap so attention can exclude it (#6).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from vfoa.geometry import eye_pixel, patch_depth, unproject
from vfoa.types import DepthFrame, Person, SceneObject, Segmentation

logger = logging.getLogger(__name__)

__all__ = ["ObjectRegistry", "SceneBuilder"]

_DEFAULTS: dict[str, Any] = {
    "point_stride": 2,
    "max_points_per_object": 256,
    "min_points": 20,
    "outlier": "percentile",
    "outlier_percentiles": [2.0, 98.0],
    "ignore_labels": [],
    "person_labels": ["person"],
    "self_min_overlap": 0.3,
    "eye_patch_px": 3,
    "match_dist_m": 0.75,
    "object_max_age": 5,
    # Optional keys (not in vfoa.config DEFAULTS):
    "outlier_nb_points": 11,
    "outlier_radius_m": None,
    "outlier_radius_scale": 4.0,
    "self_aabb_margin_m": 0.1,
}
_OUTLIER_MODES = ("percentile", "radius", "none")
_NO_MATCH_COST = 1e9


# --------------------------------------------------------------------------- registry


class ObjectRegistry:
    """Persistent object ids across segmentation refreshes.

    New objects are matched to live entries with the same label by Hungarian
    assignment on the centroid distance (metres, camera frame). Thing objects
    must be closer than ``match_dist_m``; stuff objects (``is_thing`` False)
    are unique per image and match by label only. Ids start at 1, increase
    monotonically and are never reused. A live entry not matched in more than
    ``max_age`` refreshes is forgotten.
    """

    def __init__(self, match_dist_m: float = 0.75, max_age: int = 5) -> None:
        self.match_dist_m = float(match_dist_m)
        self.max_age = int(max_age)
        self.reset()

    def reset(self) -> None:
        """Forget every object; the next id is 1."""
        self._ids: list[int] = []
        self._labels: list[str] = []
        self._centroids: list[np.ndarray] = []
        self._missed: list[int] = []
        self._next_id = 1

    @property
    def next_id(self) -> int:
        """Id the next new object will get."""
        return self._next_id

    @property
    def live_ids(self) -> list[int]:
        """Ids currently remembered (matched within the last ``max_age`` refreshes)."""
        return list(self._ids)

    def assign(
        self,
        labels: Sequence[str],
        centroids: np.ndarray,
        is_thing: Optional[Sequence[bool]] = None,
        *,
        new_refresh: bool = True,
    ) -> list[int]:
        """Return one persistent id per object, in input order.

        ``centroids`` is (N, 3) metres. With ``new_refresh`` False (the same
        segmentation rebuilt with new depth) unmatched entries are not aged.
        """
        labels = [str(x) for x in labels]
        n = len(labels)
        cen = np.asarray(centroids, dtype=np.float64).reshape(n, 3) if n else np.zeros((0, 3))
        thing = np.ones(n, dtype=bool) if is_thing is None else np.asarray(is_thing, dtype=bool).reshape(n)
        m = len(self._ids)
        out: list[Optional[int]] = [None] * n
        matched_live = np.zeros(m, dtype=bool)
        if n and m:
            live_c = np.stack(self._centroids)
            dist = np.linalg.norm(cen[:, None, :] - live_c[None, :, :], axis=-1)
            same = np.asarray(labels, dtype=object)[:, None] == np.asarray(self._labels, dtype=object)[None, :]
            allowed = same & ((dist < self.match_dist_m) | ~thing[:, None])
            cost = np.where(allowed, dist, _NO_MATCH_COST)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if not allowed[r, c]:
                    continue
                out[r] = self._ids[c]
                self._centroids[c] = cen[r].copy()
                self._missed[c] = 0
                matched_live[c] = True
        if new_refresh and m:
            keep = []
            for j in range(m):
                if not matched_live[j]:
                    self._missed[j] += 1
                if self._missed[j] <= self.max_age:
                    keep.append(j)
            self._ids = [self._ids[j] for j in keep]
            self._labels = [self._labels[j] for j in keep]
            self._centroids = [self._centroids[j] for j in keep]
            self._missed = [self._missed[j] for j in keep]
        for i in range(n):
            if out[i] is None:
                oid = self._next_id
                self._next_id += 1
                out[i] = oid
                self._ids.append(oid)
                self._labels.append(labels[i])
                self._centroids.append(cen[i].copy())
                self._missed.append(0)
        return [int(x) for x in out]  # type: ignore[arg-type]


# --------------------------------------------------------------------------- helpers


def _clip_box(box: Optional[np.ndarray], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """Integer ``(r0, r1, c0, c1)`` window of a box clipped to the image, or None if empty."""
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.shape != (4,) or not np.isfinite(b).all():
        return None
    c0 = max(int(math.floor(min(b[0], b[2]))), 0)
    c1 = min(int(math.ceil(max(b[0], b[2]))), width)
    r0 = max(int(math.floor(min(b[1], b[3]))), 0)
    r1 = min(int(math.ceil(max(b[1], b[3]))), height)
    if r0 >= r1 or c0 >= c1:
        return None
    return r0, r1, c0, c1


def _radius_inliers(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, nb_points: int, radius: Optional[float], scale: float
) -> np.ndarray:
    """Boolean inlier mask equal to open3d ``remove_radius_outlier(nb_points, radius)``.

    A point is kept when more than ``nb_points`` points (itself included) lie
    within ``radius`` metres. ``radius`` None -> ``scale`` x the median
    nearest-neighbour distance. Uses scipy's cKDTree (imported lazily).
    """
    from scipy.spatial import cKDTree

    pts = np.stack([x, y, z], axis=1)
    if pts.shape[0] < 2:
        return np.zeros(pts.shape[0], dtype=bool)
    tree = cKDTree(pts)
    if radius is None:
        nn = tree.query(pts, k=2)[0][:, 1]
        radius = float(scale) * float(np.median(nn))
    if not radius > 0:
        return np.zeros(pts.shape[0], dtype=bool)
    cnt = tree.query_ball_point(pts, float(radius), return_length=True)
    return np.asarray(cnt) > int(nb_points)


# --------------------------------------------------------------------------- builder


class SceneBuilder:
    """Builds :class:`SceneObject` lists, 3D eyes and own segments for one stream.

    ``cfg`` is a Config/mapping holding a ``scene`` section (and optionally
    ``persons.kpt_thr``); keyword overrides take precedence.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, **overrides: Any) -> None:
        params = dict(_DEFAULTS)
        kpt_thr = 0.35
        if cfg is not None and hasattr(cfg, "get"):
            params.update(dict(cfg.get("scene", {}) or {}))
            kpt_thr = float((cfg.get("persons", {}) or {}).get("kpt_thr", 0.35))
        params.update(overrides)
        if params["outlier"] not in _OUTLIER_MODES:
            raise ValueError(f"scene.outlier must be one of {_OUTLIER_MODES}, got {params['outlier']!r}")
        if int(params["point_stride"]) < 1:
            raise ValueError(f"scene.point_stride must be >= 1, got {params['point_stride']!r}")
        p_lo, p_hi = (float(v) for v in params["outlier_percentiles"])
        if not (0.0 <= p_lo < p_hi <= 100.0):
            raise ValueError(f"scene.outlier_percentiles must satisfy 0 <= lo < hi <= 100, got {[p_lo, p_hi]}")
        self.params = params
        self.kpt_thr = kpt_thr
        self.stride = int(params["point_stride"])
        self.max_points = max(int(params["max_points_per_object"]), 1)
        self.min_points = max(int(params["min_points"]), 1)
        self.outlier = str(params["outlier"])
        self.percentiles = (p_lo, p_hi)
        self.ignore_labels = frozenset(str(x) for x in (params["ignore_labels"] or []))
        self.person_labels = frozenset(str(x) for x in (params["person_labels"] or []))
        self.self_min_overlap = float(params["self_min_overlap"])
        self.eye_patch_px = max(int(params["eye_patch_px"]), 0)
        self.self_aabb_margin_m = float(params["self_aabb_margin_m"])
        self.registry = ObjectRegistry(float(params["match_dist_m"]), int(params["object_max_age"]))
        self._last_seg: Optional[Segmentation] = None

    def reset(self) -> None:
        """Forget persistent object ids and the last segmentation (new stream)."""
        self.registry.reset()
        self._last_seg = None

    # ------------------------------------------------------------------ objects
    def build_objects(
        self, seg: Optional[Segmentation], depth: Optional[DepthFrame]
    ) -> list[SceneObject]:
        """Metric objects (camera frame, metres) of every kept segment, ascending segment id.

        Only valid-depth pixels on the ``point_stride`` grid are used. Segments
        with an ignored label, without SegmentInfo, or with fewer than
        ``min_points`` valid points (before or after outlier rejection) are
        dropped. Never raises on empty or fully invalid inputs.
        """
        if seg is None or depth is None:
            return []
        new_refresh = seg is not self._last_seg
        self._last_seg = seg
        s = self.stride
        D = np.asarray(depth.depth)
        V = np.asarray(depth.valid, dtype=bool)
        Hd, Wd = D.shape[:2]
        Ds = D[::s, ::s]
        Vs = V[::s, ::s]
        Hs, Ws = Ds.shape
        L = np.asarray(seg.label_map)
        if L.shape[:2] == (Hd, Wd):
            Ls = L[::s, ::s]
        else:
            Hl, Wl = L.shape[:2]
            ri = (np.arange(0, Hd, s, dtype=np.int64) * Hl) // Hd
            ci = (np.arange(0, Wd, s, dtype=np.int64) * Wl) // Wd
            Ls = L[ri[:, None], ci[None, :]]
        with np.errstate(invalid="ignore"):
            ok = (Ls > 0) & Vs & np.isfinite(Ds) & (Ds > 0)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            self.registry.assign([], np.zeros((0, 3)), [], new_refresh=new_refresh)
            return []
        ids = Ls.reshape(-1)[idx]
        z = Ds.reshape(-1)[idx].astype(np.float64)
        order = np.argsort(ids, kind="stable")
        idx = idx[order]
        ids = ids[order]
        z = z[order]
        rows, cols = np.divmod(idx, Ws)
        intr = depth.intrinsics
        fu = (np.arange(Ws, dtype=np.float64) * s - intr.cx) / intr.fx
        fv = (np.arange(Hs, dtype=np.float64) * s - intr.cy) / intr.fy
        x = fu[cols] * z
        y = fv[rows] * z

        info = {int(si.id): si for si in seg.segments}
        bounds = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [ids.size]))

        kept: list[tuple[int, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]] = []
        for a, b in zip(starts.tolist(), ends.tolist()):
            sid = int(ids[a])
            si = info.get(sid)
            if si is None or str(si.label) in self.ignore_labels or (b - a) < self.min_points:
                continue
            xs, ys, zs = x[a:b], y[a:b], z[a:b]
            if self.outlier == "percentile":
                lo, hi = np.percentile(zs, self.percentiles)
                keep = (zs >= lo) & (zs <= hi)
                xs, ys, zs = xs[keep], ys[keep], zs[keep]
            elif self.outlier == "radius":
                keep = _radius_inliers(
                    xs, ys, zs,
                    int(self.params["outlier_nb_points"]),
                    self.params["outlier_radius_m"],
                    float(self.params["outlier_radius_scale"]),
                )
                xs, ys, zs = xs[keep], ys[keep], zs[keep]
            n = int(zs.size)
            if n == 0:
                continue
            bmin = np.array([xs.min(), ys.min(), zs.min()], dtype=np.float64)
            bmax = np.array([xs.max(), ys.max(), zs.max()], dtype=np.float64)
            cen = np.array([np.median(xs), np.median(ys), np.median(zs)], dtype=np.float64)
            step = max(int(math.ceil(n / self.max_points)), 1)
            pts = np.stack([xs[::step], ys[::step], zs[::step]], axis=1).astype(np.float32)
            kept.append((sid, si, bmin, bmax, cen, pts, n))

        object_ids = self.registry.assign(
            [str(k[1].label) for k in kept],
            np.stack([k[4] for k in kept]) if kept else np.zeros((0, 3)),
            [bool(k[1].is_thing) for k in kept],
            new_refresh=new_refresh,
        )
        return [
            SceneObject(
                object_id=oid,
                segment_id=sid,
                label=str(si.label),
                is_thing=bool(si.is_thing),
                aabb_min=bmin,
                aabb_max=bmax,
                centroid=cen,
                points=pts,
                n_points=n,
            )
            for oid, (sid, si, bmin, bmax, cen, pts, n) in zip(object_ids, kept)
        ]

    # ------------------------------------------------------------------ eyes
    def locate_eyes(self, persons: Sequence[Person], depth: Optional[DepthFrame]) -> None:
        """Set ``eyes_px`` (pixels) and ``eyes_3d`` (camera frame, metres) of each person in place.

        ``eyes_px`` follows :func:`vfoa.geometry.eye_pixel`. The eye depth is the
        median of the valid depths in a patch around the eye pixel, restricted
        to the head box clipped to the image; the patch grows up to three
        times. ``eyes_3d`` is None (never NaN) when no valid depth is found.
        """
        D = V = None
        if depth is not None:
            D = np.asarray(depth.depth)
            V = np.asarray(depth.valid, dtype=bool)
        base = max(self.eye_patch_px, 1)
        patches = (self.eye_patch_px, base * 2, base * 4, base * 8)
        for p in persons:
            p.eyes_px = None
            p.eyes_3d = None
            det = p.detection
            if det is None:
                continue
            px = eye_pixel(det.head_box, det.keypoints, self.kpt_thr)
            if px is None:
                continue
            p.eyes_px = px
            if D is None:
                continue
            h, w = D.shape[:2]
            r0, c0 = 0, 0
            cd, cv = D, V
            if det.head_box is not None and np.isfinite(det.head_box).all():
                win = _clip_box(det.head_box, w, h)
                if win is None:
                    continue
                r0, r1, c0, c1 = win
                cd, cv = D[r0:r1, c0:c1], V[r0:r1, c0:c1]
            z = None
            for patch in patches:
                z = patch_depth(cd, cv, float(px[0]) - c0, float(px[1]) - r0, patch)
                if z is not None:
                    break
            if z is None or not math.isfinite(z) or z <= 0:
                continue
            e = unproject(px.astype(np.float64), np.float64(z), depth.intrinsics)
            if np.isfinite(e).all():
                p.eyes_3d = e

    # ------------------------------------------------------------------ own segment
    def match_own_segments(
        self,
        persons: Sequence[Person],
        seg: Optional[Segmentation],
        objects: Sequence[SceneObject],
    ) -> None:
        """Set ``own_object_id`` of each person in place (the object attention must skip).

        Candidates are objects with a label in ``scene.person_labels``. The head
        box, then the body box, is clipped to the label map (frame resolution);
        the candidate covering the largest fraction ``>= self_min_overlap`` of
        the box wins. Fallback: the nearest candidate whose AABB, grown by
        ``self_aabb_margin_m``, contains ``eyes_3d``. Otherwise None.
        """
        cands = [o for o in objects if o.label in self.person_labels]
        for p in persons:
            p.own_object_id = None
        if not cands:
            return
        L = None if seg is None else np.asarray(seg.label_map)
        seg_ids = np.array([o.segment_id for o in cands], dtype=np.int64)
        for p in persons:
            det = p.detection
            found: Optional[int] = None
            if L is not None and det is not None:
                h, w = L.shape[:2]
                for box in (det.head_box, det.body_box):
                    win = _clip_box(box, w, h)
                    if win is None:
                        continue
                    r0, r1, c0, c1 = win
                    crop = L[r0:r1, c0:c1]
                    frac = np.array([np.count_nonzero(crop == sid) for sid in seg_ids], dtype=np.float64)
                    frac /= crop.size
                    j = int(np.argmax(frac))
                    if frac[j] >= self.self_min_overlap:
                        found = cands[j].object_id
                        break
            if found is None and p.eyes_3d is not None:
                e = np.asarray(p.eyes_3d, dtype=np.float64)
                if np.isfinite(e).all():
                    m = self.self_aabb_margin_m
                    best = None
                    for o in cands:
                        if np.all(e >= o.aabb_min - m) and np.all(e <= o.aabb_max + m):
                            d = float(np.linalg.norm(e - o.centroid))
                            if best is None or d < best[0]:
                                best = (d, o.object_id)
                    if best is not None:
                        found = best[1]
            p.own_object_id = found
