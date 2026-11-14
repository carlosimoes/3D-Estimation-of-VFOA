"""Attention scoring: which scene object each person is looking at.

Everything is expressed in the camera frame (OpenCV pinhole of the left camera,
+X right, +Y down, +Z forward, metres). Gaze and head directions are unit
vectors pointing from the eyes towards what is looked at.

Two scoring methods are available (``attention.method``):

* ``angular`` (default): for each cue direction ``d`` and object, ``theta`` is
  the robust minimum (5th percentile) of the angle between ``d`` and the
  vectors from the eyes to the object's surface sample points (or centroid);
  the cue score is ``exp(-theta^2 / (2 sigma^2))``, optionally times a
  distance falloff.
* ``rays`` (thesis method, bug-fixed): cones of rays around each cue are
  slab-tested against the object AABBs; every hitting ray contributes
  ``1.5 - |entry - centre| / half_diagonal`` and the sum is normalised by the
  maximum achievable value, so each cue score lies in [0, 1].

Per-cue scores are normalised to [0, 1] FIRST and then fused with weights that
sum to one (``fixed``, ``agreement`` - the thesis 45-degree rule - or
``confidence``). Occluded pairs (``depth`` map marching, ``aabb`` blockers, or
nearest-hit-only in rays mode) keep their per-cue scores but get
``score_total = 0``. A person's own object is never scored.

The module is pure NumPy (plus :mod:`vfoa.geometry`), never prints and does
not emit NumPy floating-point warnings.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from vfoa.geometry import angle_between, cone_rays, ray_aabb, segment_aabb
from vfoa.types import (
    AttentionScore,
    DepthFrame,
    DirectionEstimate,
    Person,
    SceneObject,
    Segmentation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AGREE_WEIGHTS",
    "DISAGREE_WEIGHTS",
    "AttentionParams",
    "agreement_weights",
    "fusion_weights",
    "box_ratio",
    "ray_box_values",
    "rays_score",
    "best_targets",
    "score_frame",
]

#: Thesis weights (gaze, head) when gaze and head agree (angle < agreement_deg).
AGREE_WEIGHTS: tuple[float, float] = (1.9, 1.1)
#: Thesis weights (gaze, head) when gaze and head disagree.
DISAGREE_WEIGHTS: tuple[float, float] = (1.55, 1.45)

_METHODS = ("angular", "rays")
_FUSIONS = ("fixed", "agreement", "confidence")
_TARGETS = ("surface", "centroid")
_OCCLUSIONS = ("depth", "aabb", "none")
_PERCENTILE = 5.0  # robust minimum angle
_EPS = 1e-12
_CAND_MARGIN = 2  # extra candidates kept by the partial sort (round-off safety)


# --------------------------------------------------------------------------- params


@dataclass(frozen=True)
class AttentionParams:
    """Parameters of the ``attention`` config section (angles in degrees, distances in metres).

    ``rays.*`` keys are flattened to ``rays_n``, ``half_angles_gaze_deg`` and
    ``half_angles_head_deg``.
    """

    method: str = "angular"
    sigma_gaze_deg: float = 10.0
    sigma_head_deg: float = 20.0
    fusion: str = "confidence"
    w_gaze: float = 0.6
    w_head: float = 0.4
    agreement_deg: float = 45.0
    target: str = "surface"
    occlusion: str = "depth"
    occlusion_margin_m: float = 0.15
    occlusion_samples: int = 24
    min_target_distance_m: float = 0.15
    distance_full_m: Optional[float] = None
    distance_decay_m: float = 2.0
    min_score: float = 0.1
    rays_n: int = 20
    half_angles_gaze_deg: tuple[float, ...] = (11.3, 21.8)
    half_angles_head_deg: tuple[float, ...] = (12.5, 24.0)

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"attention.method must be one of {_METHODS}, got {self.method!r}")
        if self.fusion not in _FUSIONS:
            raise ValueError(f"attention.fusion must be one of {_FUSIONS}, got {self.fusion!r}")
        if self.target not in _TARGETS:
            raise ValueError(f"attention.target must be one of {_TARGETS}, got {self.target!r}")
        if self.occlusion not in _OCCLUSIONS:
            raise ValueError(
                f"attention.occlusion must be one of {_OCCLUSIONS}, got {self.occlusion!r}"
            )
        if not (self.sigma_gaze_deg > 0 and self.sigma_head_deg > 0):
            raise ValueError("attention.sigma_gaze_deg and sigma_head_deg must be > 0")
        if int(self.rays_n) < 1:
            raise ValueError(f"attention.rays.n must be >= 1, got {self.rays_n}")
        if int(self.occlusion_samples) < 1:
            raise ValueError("attention.occlusion_samples must be >= 1")
        if not self.distance_decay_m > 0:
            raise ValueError("attention.distance_decay_m must be > 0")
        if not (len(self.half_angles_gaze_deg) and len(self.half_angles_head_deg)):
            raise ValueError("attention.rays half-angle lists must not be empty")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "AttentionParams":
        """Build from a full config (uses ``cfg["attention"]``) or the section itself."""
        sec: Mapping[str, Any] = cfg["attention"] if "attention" in cfg else cfg
        d = cls()
        rays = sec.get("rays", None) or {}
        full = sec.get("distance_full_m", d.distance_full_m)
        return cls(
            method=str(sec.get("method", d.method)),
            sigma_gaze_deg=float(sec.get("sigma_gaze_deg", d.sigma_gaze_deg)),
            sigma_head_deg=float(sec.get("sigma_head_deg", d.sigma_head_deg)),
            fusion=str(sec.get("fusion", d.fusion)),
            w_gaze=float(sec.get("w_gaze", d.w_gaze)),
            w_head=float(sec.get("w_head", d.w_head)),
            agreement_deg=float(sec.get("agreement_deg", d.agreement_deg)),
            target=str(sec.get("target", d.target)),
            occlusion=str(sec.get("occlusion", d.occlusion)),
            occlusion_margin_m=float(sec.get("occlusion_margin_m", d.occlusion_margin_m)),
            occlusion_samples=int(sec.get("occlusion_samples", d.occlusion_samples)),
            min_target_distance_m=float(sec.get("min_target_distance_m", d.min_target_distance_m)),
            distance_full_m=None if full is None else float(full),
            distance_decay_m=float(sec.get("distance_decay_m", d.distance_decay_m)),
            min_score=float(sec.get("min_score", d.min_score)),
            rays_n=int(rays.get("n", d.rays_n)),
            half_angles_gaze_deg=tuple(
                float(a) for a in rays.get("half_angles_gaze_deg", d.half_angles_gaze_deg)
            ),
            half_angles_head_deg=tuple(
                float(a) for a in rays.get("half_angles_head_deg", d.half_angles_head_deg)
            ),
        )


# --------------------------------------------------------------------------- helpers


def _unit_or_none(v: Any) -> Optional[np.ndarray]:
    """Unit float64 (3,) copy of ``v``, or None when missing, non-finite or ~zero."""
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    n = float(np.linalg.norm(a))
    if n <= 1e-9:
        return None
    return a / n


def _cue_dir(est: Optional[DirectionEstimate]) -> Optional[np.ndarray]:
    return None if est is None else _unit_or_none(est.direction)


def agreement_weights(
    g: np.ndarray, h: np.ndarray, agreement_deg: float = 45.0
) -> tuple[float, float]:
    """Raw thesis weights (gaze, head): AGREE iff angle(g, h) < agreement_deg degrees."""
    ang = math.degrees(float(angle_between(np.asarray(g, float), np.asarray(h, float))))
    return AGREE_WEIGHTS if ang < float(agreement_deg) else DISAGREE_WEIGHTS


def fusion_weights(
    gaze: Optional[DirectionEstimate],
    head: Optional[DirectionEstimate],
    params: AttentionParams,
) -> tuple[float, float]:
    """Fusion weights ``(w_gaze, w_head)`` summing to 1 over the available cues.

    A cue is available when its direction is finite and non-zero. Both missing
    -> (0, 0); one missing -> all weight on the other.
    """
    g = _cue_dir(gaze)
    h = _cue_dir(head)
    if g is None and h is None:
        return 0.0, 0.0
    if g is None:
        return 0.0, 1.0
    if h is None:
        return 1.0, 0.0

    def _fixed() -> tuple[float, float]:
        a, b = max(float(params.w_gaze), 0.0), max(float(params.w_head), 0.0)
        if not (math.isfinite(a) and math.isfinite(b)) or a + b <= _EPS:
            return 0.5, 0.5
        return a / (a + b), b / (a + b)

    if params.fusion == "agreement":
        a, b = agreement_weights(g, h, params.agreement_deg)
        return a / 3.0, b / 3.0
    if params.fusion == "confidence":

        def _c(x: float) -> float:
            x = float(x)
            return min(max(x, 0.0), 1.0) if math.isfinite(x) else 0.0

        a = max(float(params.w_gaze), 0.0) * _c(gaze.confidence)  # type: ignore[union-attr]
        b = max(float(params.w_head), 0.0) * _c(head.confidence)  # type: ignore[union-attr]
        if not (math.isfinite(a) and math.isfinite(b)) or a + b <= _EPS:
            return _fixed()
        return a / (a + b), b / (a + b)
    return _fixed()


def _boxes(bmin: np.ndarray, bmax: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.atleast_2d(np.asarray(bmin, dtype=np.float64))
    hi = np.atleast_2d(np.asarray(bmax, dtype=np.float64))
    return np.minimum(lo, hi), np.maximum(lo, hi)


def box_ratio(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """``(B,)`` ratio (min half-extent) / (half diagonal) of AABBs (thesis ``calc_ratio``).

    Translation invariant; 0 for degenerate boxes.
    """
    lo, hi = _boxes(bmin, bmax)
    ext = np.abs(hi - lo)
    hd = np.linalg.norm(ext, axis=-1) / 2.0
    return (ext.min(axis=-1) / 2.0) / np.maximum(hd, _EPS)


def ray_box_values(
    origins: np.ndarray, rays: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thesis per-ray values against AABBs.

    ``origins`` ``(R, 3)`` or ``(3,)``, ``rays`` ``(R, 3)`` (any length), boxes
    ``(B, 3)`` or ``(3,)``. Returns ``(values, hit, t_enter)`` each ``(R, B)``:
    ``value = 1.5 - |entry - centre| / half_diagonal`` on hits (entry point =
    first point of the ray inside the box, ``t >= 0``), else 0.
    """
    lo, hi = _boxes(bmin, bmax)
    o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    d = np.atleast_2d(np.asarray(rays, dtype=np.float64))
    o, d = np.broadcast_arrays(o, d)
    hit, t_enter, _ = ray_aabb(o, d, lo, hi)
    o_ok = np.where(np.isfinite(o), o, 0.0)
    d_ok = np.where(np.isfinite(d), d, 0.0)
    entry = o_ok[:, None, :] + t_enter[..., None] * d_ok[:, None, :]  # (R, B, 3)
    centre = np.where(np.isfinite(lo + hi), (lo + hi) / 2.0, 0.0)
    hd = np.linalg.norm(np.where(np.isfinite(hi - lo), hi - lo, 0.0), axis=-1) / 2.0
    dist = np.linalg.norm(entry - centre[None], axis=-1)
    values = np.where(hit, 1.5 - dist / np.maximum(hd, _EPS)[None, :], 0.0)
    return values, hit, t_enter


def rays_score(value_sum: Any, n_rays: Any, ratio: Any) -> Any:
    """Normalised ray-cone score ``clip(value_sum / (n_rays * (1.5 - ratio)), 0, 1)``; 0 if no rays."""
    v = np.asarray(value_sum, dtype=np.float64)
    n = np.asarray(n_rays, dtype=np.float64)
    r = np.asarray(ratio, dtype=np.float64)
    den = n * (1.5 - r)
    good = (n > 0) & (den > _EPS) & np.isfinite(v) & np.isfinite(den)
    out = np.where(good, v / np.where(good, den, 1.0), 0.0)
    out = np.clip(out, 0.0, 1.0)
    return float(out) if out.ndim == 0 else out


def best_targets(
    scores: Iterable[AttentionScore], min_score: float
) -> dict[int, Optional[int]]:
    """Per person, the object with the highest ``score_total`` (first on ties), or None below ``min_score``."""
    best: dict[int, tuple[float, Optional[int]]] = {}
    for s in scores:
        cur = best.get(s.person_id)
        val = float(s.score_total)
        if cur is None or val > cur[0]:
            best[s.person_id] = (val, s.object_id)
    return {
        pid: (oid if math.isfinite(val) and val >= float(min_score) else None)
        for pid, (val, oid) in best.items()
    }


# --------------------------------------------------------------------------- scoring


def _angular(eyes, D, objs, params):
    """Angular per-cue scores ``S (2,P,B)`` and primary-cue target points ``T (P,B,3)``."""
    P = eyes.shape[0]
    B = len(objs)
    cent = np.stack([np.asarray(o.centroid, np.float64).reshape(3) for o in objs])
    plist = []
    for o in objs:
        pts = None
        if params.target == "surface" and o.points is not None:
            pts = np.asarray(o.points, dtype=np.float64).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if pts.shape[0] == 0:
                pts = None
        plist.append(pts if pts is not None else cent[len(plist)][None, :])
    K = max(p.shape[0] for p in plist)
    pts = np.zeros((B, K, 3), dtype=np.float64)
    pmask = np.zeros((B, K), dtype=bool)
    for b, p in enumerate(plist):
        pts[b, : p.shape[0]] = p
        pmask[b, : p.shape[0]] = True

    v = pts[None, :, :, :] - eyes[:, None, None, :]  # (P,B,K,3)
    dist = np.linalg.norm(v, axis=-1)
    ok = pmask[None] & (dist >= params.min_target_distance_m) & (dist > _EPS)
    dot = np.einsum("cpd,pbkd->cpbk", D, v)
    key = np.where(ok[None], -dot / np.maximum(dist, _EPS)[None], np.inf)  # -cos(angle)

    n = ok.sum(axis=-1)  # (P,B)
    has_pts = n > 0
    nm1 = np.maximum(n - 1, 0)
    pos = (_PERCENTILE / 100.0) * nm1
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, nm1)
    frac = pos - lo
    m = min(int(hi.max()) + 1 + _CAND_MARGIN, K)
    if m < K:
        idx = np.argpartition(key, m - 1, axis=-1)[..., :m]
    else:
        idx = np.broadcast_to(np.arange(K), key.shape).copy()
    vb = np.broadcast_to(v[None], (2,) + v.shape)
    vc = np.take_along_axis(vb, idx[..., None], axis=3)  # (2,P,B,m,3)
    okc = np.take_along_axis(np.broadcast_to(ok[None], key.shape), idx, axis=-1)
    ang = np.where(okc, angle_between(D[:, :, None, None, :], vc), np.inf)
    order = np.argsort(ang, axis=-1, kind="stable")
    ang_s = np.take_along_axis(ang, order, axis=-1)
    a_lo = np.take_along_axis(ang_s, np.broadcast_to(lo[None, ..., None], (2, P, B, 1)), -1)[..., 0]
    a_hi = np.take_along_axis(ang_s, np.broadcast_to(hi[None, ..., None], (2, P, B, 1)), -1)[..., 0]
    a_lo = np.where(has_pts[None], a_lo, 0.0)
    a_hi = np.where(has_pts[None], a_hi, 0.0)
    theta = a_lo + frac[None] * (a_hi - a_lo)

    sig = np.radians(np.array([params.sigma_gaze_deg, params.sigma_head_deg]))[:, None, None]
    S = np.where(has_pts[None], np.exp(-(theta**2) / (2.0 * sig**2)), 0.0)

    if params.distance_full_m is not None:
        Dist = np.linalg.norm(cent[None] - eyes[:, None], axis=-1)  # (P,B)
        full = float(params.distance_full_m)
        fall = np.exp(-np.maximum(Dist - full, 0.0) / float(params.distance_decay_m))
        S = S * np.where(Dist <= full, 1.0, fall)[None]

    # Target point of the primary cue: the minimum-angle valid point.
    best_k = np.take_along_axis(idx, order[..., :1], axis=-1)[..., 0]  # (2,P,B)
    return S, best_k, pts, has_pts, cent


def _rays(eyes, D, has, bmin, bmax, is_thing, pair_ok, params):
    """Rays-method per-cue scores ``S (2,P,B)`` and the occlusion flags ``(P,B)``."""
    P = eyes.shape[0]
    B = bmin.shape[0]
    ratio = box_ratio(bmin, bmax)  # (B,)
    S = np.zeros((2, P, B))
    raw_any = np.zeros((P, B), dtype=bool)
    cred_any = np.zeros((P, B), dtype=bool)
    for c, angles in enumerate((params.half_angles_gaze_deg, params.half_angles_head_deg)):
        rays = np.concatenate(
            [cone_rays(D[c], a, params.rays_n) for a in angles], axis=1
        )  # (P,Rc,3)
        Rc = rays.shape[1]
        origins = np.repeat(eyes, Rc, axis=0)
        values, hit, t_enter = ray_box_values(origins, rays.reshape(-1, 3), bmin, bmax)
        values = values.reshape(P, Rc, B)
        hit = hit.reshape(P, Rc, B)
        t_enter = t_enter.reshape(P, Rc, B)
        if params.occlusion != "none":
            blocker = hit & is_thing[None, None, :] & pair_ok[:, None, :] & (t_enter > 0)
            t_block = np.where(blocker, t_enter, np.inf).min(axis=-1)  # (P,Rc)
            credit = hit & (t_enter <= t_block[..., None] + 1e-9)
        else:
            credit = hit
        vsum = np.where(credit, values, 0.0).sum(axis=1)  # (P,B)
        S[c] = rays_score(vsum, Rc, ratio[None, :])
        hc = has[c][:, None]
        raw_any |= hc & hit.any(axis=1)
        cred_any |= hc & credit.any(axis=1)
    occluded = raw_any & ~cred_any if params.occlusion != "none" else np.zeros((P, B), bool)
    return S, occluded


def _depth_occlusion(eyes, T, persons, objs, own_idx, depth, segmentation, params):
    """``(P,B)`` flags: the depth map is nearer than the eye->target segment.

    Samples whose pixel shows the target's segment, the person's own segment
    or lies inside the person's ``detection.head_box`` never block. The
    person's ``detection.body_box`` is excluded only as a fallback when the
    own segment is unknown (no segmentation, no own object, or own object
    ``segment_id <= 0``): when the own segment is labelled, a real occluder
    overlapping the full-person box must still count.
    """
    dmap = np.asarray(depth.depth, dtype=np.float64)
    valid = np.asarray(depth.valid, dtype=bool)
    H, W = dmap.shape
    S_n = int(params.occlusion_samples)
    f = np.linspace(0.1, 0.9, S_n)
    s = eyes[:, None, None, :] + f[None, None, :, None] * (T - eyes[:, None, :])[:, :, None, :]
    uv = depth.intrinsics.project(s)  # (P,B,S,2), NaN behind the camera
    fin = np.isfinite(uv).all(axis=-1)
    uvs = np.where(fin[..., None], uv, -1.0)
    px = np.floor(uvs[..., 0] + 0.5)
    py = np.floor(uvs[..., 1] + 0.5)
    inside = fin & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    xi = np.where(inside, px, 0).astype(np.int64)
    yi = np.where(inside, py, 0).astype(np.int64)
    z_map = dmap[yi, xi]
    usable = inside & valid[yi, xi] & np.isfinite(z_map) & (z_map > 0)
    blocked = usable & (s[..., 2] - np.where(usable, z_map, 0.0) > params.occlusion_margin_m)

    excluded = np.zeros_like(blocked)
    own_known = np.zeros(len(persons), dtype=bool)
    if segmentation is not None:
        lm = np.asarray(segmentation.label_map)
        Hl, Wl = lm.shape[:2]
        if (Hl, Wl) != (H, W):
            lx = np.clip((xi * Wl) // max(W, 1), 0, Wl - 1)
            ly = np.clip((yi * Hl) // max(H, 1), 0, Hl - 1)
        else:
            lx, ly = xi, yi
        lab = lm[ly, lx].astype(np.int64)
        tseg = np.array([int(o.segment_id) for o in objs], dtype=np.int64)  # (B,)
        oseg = np.array(
            [int(objs[i].segment_id) if i >= 0 else -1 for i in own_idx], dtype=np.int64
        )  # (P,)
        excluded |= (lab == tseg[None, :, None]) & (tseg[None, :, None] > 0)
        excluded |= (lab == oseg[:, None, None]) & (oseg[:, None, None] > 0)
        own_known = oseg > 0
    for p, person in enumerate(persons):
        det = person.detection
        if det is None:
            continue
        boxes = [det.head_box]
        if not own_known[p]:
            boxes.append(det.body_box)
        for box in boxes:
            if box is None:
                continue
            bx = np.asarray(box, dtype=np.float64).reshape(4)
            if not np.isfinite(bx).all():
                continue
            excluded[p] |= (
                (px[p] >= bx[0]) & (px[p] <= bx[2]) & (py[p] >= bx[1]) & (py[p] <= bx[3])
            )
    blocked &= ~excluded
    if S_n < 4:
        return blocked.any(axis=-1)
    return (blocked[..., 1:] & blocked[..., :-1]).any(axis=-1)


def _aabb_occlusion(eyes, T, bmin, bmax, is_thing, own_idx):
    """``(P,B)`` flags: another thing-object's AABB intersects the eye->target segment."""
    P, B = T.shape[:2]
    p0 = np.repeat(eyes, B, axis=0)
    p1 = T.reshape(-1, 3)
    inter = segment_aabb(p0, p1, bmin, bmax).reshape(P, B, B)  # [person, target, blocker]
    eye_in = ((eyes[:, None, :] >= bmin[None]) & (eyes[:, None, :] <= bmax[None])).all(-1)  # (P,K)
    t_in = ((T[:, :, None, :] >= bmin[None, None]) & (T[:, :, None, :] <= bmax[None, None])).all(-1)
    k = np.arange(B)
    cand = (
        inter
        & is_thing[None, None, :]
        & (k[None, :] != k[:, None])[None]
        & (k[None, None, :] != own_idx[:, None, None])
        & ~eye_in[:, None, :]
        & ~t_in
    )
    return cand.any(axis=-1)


def score_frame(
    persons: Sequence[Person],
    objects: Sequence[SceneObject],
    params: AttentionParams,
    depth: Optional[DepthFrame] = None,
    segmentation: Optional[Segmentation] = None,
) -> list[AttentionScore]:
    """Score every (person, object) pair except a person's own object.

    Persons need finite ``eyes_3d`` and at least one usable cue (gaze or head
    direction); a missing cue gets ``None`` and its weight goes to the other.
    Output order: persons, then objects, in input order. ``depth`` (and
    optionally ``segmentation``) is used by ``occlusion: depth``; without a
    depth map that mode is a no-op. Pass the segmentation the objects were
    built from: the target's and the person's own segment never occlude, nor
    does the head box; the body box is ignored only when the own segment is
    unknown (fallback), so real occluders in front of the person still count.
    """
    keep_p: list[Person] = []
    eyes_l, dirs_l, has_l = [], [], []
    for person in persons:
        if person.eyes_3d is None:
            continue
        e = np.asarray(person.eyes_3d, dtype=np.float64).reshape(-1)
        if e.shape != (3,) or not np.isfinite(e).all():
            continue
        g = _cue_dir(person.gaze)
        h = _cue_dir(person.head)
        if g is None and h is None:
            continue
        keep_p.append(person)
        eyes_l.append(e)
        dirs_l.append([g if g is not None else np.array([0.0, 0.0, 1.0]),
                       h if h is not None else np.array([0.0, 0.0, 1.0])])
        has_l.append([g is not None, h is not None])
    objs: list[SceneObject] = []
    for o in objects:
        vals = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (o.aabb_min, o.aabb_max, o.centroid)]
        if all(v.shape == (3,) and np.isfinite(v).all() for v in vals):
            objs.append(o)
    if not keep_p or not objs:
        return []

    P, B = len(keep_p), len(objs)
    eyes = np.stack(eyes_l)
    D = np.transpose(np.asarray(dirs_l, dtype=np.float64), (1, 0, 2))  # (2,P,3)
    has = np.asarray(has_l, dtype=bool).T  # (2,P)
    bmin, bmax = _boxes(
        np.stack([np.asarray(o.aabb_min, np.float64).reshape(3) for o in objs]),
        np.stack([np.asarray(o.aabb_max, np.float64).reshape(3) for o in objs]),
    )
    is_thing = np.array([bool(o.is_thing) for o in objs])
    id_to_idx = {}
    for i, o in enumerate(objs):
        id_to_idx.setdefault(o.object_id, i)
    own_idx = np.array(
        [id_to_idx.get(p.own_object_id, -1) if p.own_object_id is not None else -1 for p in keep_p],
        dtype=np.int64,
    )
    pair_ok = np.arange(B)[None, :] != own_idx[:, None]

    with np.errstate(all="ignore"):
        if params.method == "angular":
            S, best_k, pts, has_pts, cent = _angular(eyes, D, objs, params)
            occluded = np.zeros((P, B), dtype=bool)
            if params.occlusion != "none":
                prim = np.where(has[0], 0, 1)  # (P,)
                kk = best_k[prim, np.arange(P)]  # (P,B)
                T = pts[np.arange(B)[None, :], kk]  # (P,B,3)
                T = np.where(has_pts[..., None], T, cent[None])
                if params.occlusion == "depth":
                    if depth is not None:
                        occluded = _depth_occlusion(
                            eyes, T, keep_p, objs, own_idx, depth, segmentation, params
                        )
                else:
                    occluded = _aabb_occlusion(eyes, T, bmin, bmax, is_thing, own_idx)
        else:
            S, occluded = _rays(eyes, D, has, bmin, bmax, is_thing, pair_ok, params)
    S = np.clip(np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    out: list[AttentionScore] = []
    for p, person in enumerate(keep_p):
        wg, wh = fusion_weights(
            person.gaze if has[0, p] else None, person.head if has[1, p] else None, params
        )
        for b, o in enumerate(objs):
            if not pair_ok[p, b]:
                continue
            g = float(S[0, p, b]) if has[0, p] else None
            h = float(S[1, p, b]) if has[1, p] else None
            total = (wg * g if g is not None else 0.0) + (wh * h if h is not None else 0.0)
            occ = bool(occluded[p, b])
            if occ:
                total = 0.0
            out.append(
                AttentionScore(
                    person_id=int(person.track_id),
                    object_id=int(o.object_id),
                    label=str(o.label),
                    score_gaze=g,
                    score_head=h,
                    score_total=float(min(max(total, 0.0), 1.0)),
                    occluded=occ,
                )
            )
    return out
