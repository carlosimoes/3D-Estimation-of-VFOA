"""Geometry core of the VFOA pipeline (pure NumPy, vectorised, NaN-free).

Conventions (architecture section 2):

* The only 3D frame is the OpenCV pinhole frame of the left camera: origin at
  the optical centre, +X right, +Y down, +Z forward, units METRES.
  ``u = fx * X / Z + cx`` (column), ``v = fy * Y / Z + cy`` (row).
* Depth is ``Z > 0`` in metres; invalid depth is 0 / non-finite / masked out.
* Disparity is positive pixels, ``d = fx * B / Z`` with the baseline ``B`` in metres.
* Directions are float64 unit vectors pointing from the eyes towards what is
  looked at; a person looking at the camera has direction ``(0, 0, -1)``.

NaN is only used as the invalid marker of :func:`backproject` and
:func:`project_points`; every other function returns finite values (or
``None``) and never emits NumPy floating-point warnings.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from vfoa.types import Intrinsics

__all__ = [
    "normalize",
    "angle_between",
    "unproject",
    "backproject",
    "project_points",
    "disparity_to_depth",
    "depth_to_disparity",
    "patch_depth",
    "eye_pixel",
    "euler_to_forward",
    "ray_aligned",
    "gaze360_to_camera",
    "orthonormal_basis",
    "cone_rays",
    "ray_aabb",
    "segment_aabb",
    "aabb_from_points",
]

_EPS = 1e-12
_PAR_EPS = 1e-12


# --------------------------------------------------------------------------- vectors


def normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Return ``v`` scaled to unit length along ``axis`` (float64).

    Vectors with norm below ``eps`` are divided by ``eps`` (so a zero vector
    stays zero instead of becoming NaN).
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in radians between vectors ``a`` and ``b`` (``(..., 3)``, broadcasting).

    Uses ``atan2(|a x b|, a . b)``, accurate for parallel, antiparallel and nearly
    parallel vectors. Inputs need not be unit length. Returns a float for
    1-D inputs, otherwise an array of the broadcast leading shape.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    cross = np.cross(a, b)
    out = np.arctan2(np.linalg.norm(cross, axis=-1), np.sum(a * b, axis=-1))
    if np.ndim(out) == 0:
        return float(out)
    return out


# --------------------------------------------------------------------------- camera


def unproject(uv: np.ndarray, z: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Pixels ``uv (..., 2)`` (column, row) at depth ``z (...)`` metres -> camera points ``(..., 3)``.

    ``X = (u - cx) * Z / fx``, ``Y = (v - cy) * Z / fy``. float64.
    """
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if uv.shape[-1] != 2:
        raise ValueError(f"uv must have shape (..., 2), got {uv.shape}")
    x = (uv[..., 0] - intr.cx) * z / intr.fx
    y = (uv[..., 1] - intr.cy) * z / intr.fy
    x, y, z = np.broadcast_arrays(x, y, z)
    return np.stack([x, y, z], axis=-1)


def backproject(
    depth: np.ndarray, intr: Intrinsics, valid: Optional[np.ndarray] = None
) -> np.ndarray:
    """Back-project a full depth map (HxW metres) to camera points (H, W, 3) float32.

    Pixels with non-finite or non-positive depth, or ``valid == False``, are NaN.
    The pixel grid comes from ``depth.shape`` (column ``u``, row ``v``). Use only
    for offline/visualisation work: the real-time hot path unprojects only the
    pixels it needs with :func:`unproject`.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got shape {depth.shape}")
    h, w = depth.shape
    d = depth.astype(np.float32, copy=False)
    ok = np.isfinite(d) & (d > 0)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    z = np.where(ok, d, np.float32(np.nan)).astype(np.float32, copy=False)
    fu = ((np.arange(w, dtype=np.float64) - intr.cx) / intr.fx).astype(np.float32)
    fv = ((np.arange(h, dtype=np.float64) - intr.cy) / intr.fy).astype(np.float32)
    out = np.empty((h, w, 3), dtype=np.float32)
    np.multiply(z, fu[None, :], out=out[..., 0])
    np.multiply(z, fv[:, None], out=out[..., 1])
    out[..., 2] = z
    return out


def project_points(points: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Project camera points ``(..., 3)`` metres to pixels ``(..., 2)`` (NaN where Z <= 0)."""
    return intr.project(points)


def depth_to_disparity(depth: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """Depth (metres) -> disparity ``fx * B / Z`` float32 pixels; 0 where depth is invalid."""
    if not (fx > 0 and baseline > 0):
        raise ValueError(f"fx and baseline must be positive, got fx={fx!r}, baseline={baseline!r}")
    z = np.asarray(depth, dtype=np.float64)
    ok = np.isfinite(z) & (z > 0)
    d = np.where(ok, float(fx) * float(baseline) / np.where(ok, z, 1.0), 0.0)
    return d.astype(np.float32)


def disparity_to_depth(
    disp: np.ndarray, fx: float, baseline: float, min_disp: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    """Disparity (positive pixels) -> ``(depth float32 metres, valid bool)``.

    ``Z = fx * B / d``; disparities that are non-finite or below ``min_disp``
    (including ``d <= 0``) are invalid and get depth 0.
    """
    if not (fx > 0 and baseline > 0):
        raise ValueError(f"fx and baseline must be positive, got fx={fx!r}, baseline={baseline!r}")
    d = np.asarray(disp, dtype=np.float64)
    valid = np.isfinite(d) & (d >= float(min_disp)) & (d > 0)
    z = np.where(valid, float(fx) * float(baseline) / np.where(valid, d, 1.0), 0.0)
    return z.astype(np.float32), valid


def patch_depth(
    depth: np.ndarray,
    valid: Optional[np.ndarray],
    u: float,
    v: float,
    patch: int = 3,
) -> Optional[float]:
    """Median of the valid depths (metres) in a ``(2*patch+1)^2`` window around pixel ``(u, v)``.

    ``u`` is the column, ``v`` the row. The window is clipped to the image, so
    pixels near (or partly beyond) the border work. NaN/inf/<=0 depths and
    ``valid == False`` pixels are ignored. Returns ``None`` when the centre is
    non-finite or no valid depth is in the window.
    """
    if u is None or v is None:
        return None
    u = float(u)
    v = float(v)
    if not (math.isfinite(u) and math.isfinite(v)):
        return None
    depth = np.asarray(depth)
    h, w = depth.shape[:2]
    patch = max(int(patch), 0)
    cu = int(math.floor(u + 0.5))
    cv = int(math.floor(v + 0.5))
    r0, r1 = max(0, cv - patch), min(h, cv + patch + 1)
    c0, c1 = max(0, cu - patch), min(w, cu + patch + 1)
    if r0 >= r1 or c0 >= c1:
        return None
    win = np.asarray(depth[r0:r1, c0:c1], dtype=np.float64)
    keep = np.isfinite(win) & (win > 0)
    if valid is not None:
        keep &= np.asarray(valid[r0:r1, c0:c1], dtype=bool)
    vals = win[keep]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def eye_pixel(
    head_box: Optional[np.ndarray],
    keypoints: Optional[np.ndarray] = None,
    kpt_thr: float = 0.35,
) -> Optional[np.ndarray]:
    """Eye pixel ``(u, v)`` float32 of a person.

    Midpoint of the two COCO eye keypoints (rows 1 and 2) when both are finite
    with confidence ``>= kpt_thr``; otherwise the thesis/Gaze360 head-box rule
    ``((x1 + x2) / 2, 0.65 * y1 + 0.35 * y2)``. Returns ``None`` when neither is
    available (no box and no confident eyes).
    """
    if keypoints is not None:
        kp = np.asarray(keypoints, dtype=np.float64)
        if kp.ndim == 2 and kp.shape[0] >= 3 and kp.shape[1] >= 3:
            eyes = kp[1:3]
            if np.isfinite(eyes).all() and (eyes[:, 2] >= kpt_thr).all():
                return eyes[:, :2].mean(axis=0).astype(np.float32)
    if head_box is not None:
        b = np.asarray(head_box, dtype=np.float64).reshape(-1)
        if b.shape == (4,) and np.isfinite(b).all():
            x1, y1, x2, y2 = b
            return np.array([(x1 + x2) / 2.0, 0.65 * y1 + 0.35 * y2], dtype=np.float32)
    return None


# --------------------------------------------------------------------------- directions


def euler_to_forward(yaw_deg, pitch_deg, roll_deg=0.0) -> np.ndarray:
    """FSA-Net head pose (degrees) -> unit forward vector in the camera frame.

    ``forward = (-sin(yaw), -cos(yaw) sin(pitch), -cos(yaw) cos(pitch))``:
    yaw = pitch = 0 looks at the camera ``(0, 0, -1)``, +pitch looks up (-Y),
    +yaw turns towards the image left (-X). Roll does not change the forward
    direction. Broadcasts: scalars -> (3,), arrays -> (..., 3).
    """
    yaw = np.radians(np.asarray(yaw_deg, dtype=np.float64))
    pitch = np.radians(np.asarray(pitch_deg, dtype=np.float64))
    yaw, pitch = np.broadcast_arrays(yaw, pitch)
    cy = np.cos(yaw)
    out = np.stack([-np.sin(yaw), -cy * np.sin(pitch), -cy * np.cos(pitch)], axis=-1)
    return normalize(out) + 0.0


def ray_aligned(direction, eye_ray: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotate a direction given in the eye-ray frame into the camera frame.

    The ray frame has ``z`` along the camera -> eye ray, ``x`` towards the image
    right and ``y`` down: ``ez = unit(eye_ray)``,
    ``rx = unit(cross([0, 1, 0], ez))`` (``(1, 0, 0)`` when the ray is parallel
    to Y), ``ry = cross(ez, rx)`` and ``out = d.x rx + d.y ry + d.z ez``. A ray
    along the optical axis gives the identity, so ``(0, 0, -1)`` in the ray frame
    always looks back along the ray at the camera. Models that work on a crop
    around the face (Gaze360, L2CS, FSA-Net) predict directions relative to
    that ray, not to the optical axis.

    ``direction`` is ``(3,)`` or ``(..., 3)``; ``eye_ray`` is ``None``, ``(3,)``
    or ``(..., 3)`` (any length). The output shape is
    ``broadcast_shapes(direction.shape[:-1], eye_ray.shape[:-1]) + (3,)``, unit
    float64. ``None`` and rows whose ray is zero or non-finite return the
    normalised input direction unchanged. Raises ValueError when the last
    dimension of either argument is not 3.
    """
    d = np.asarray(direction, dtype=np.float64)
    if d.ndim == 0 or d.shape[-1] != 3:
        raise ValueError(f"direction must have shape (..., 3), got {d.shape}")
    if eye_ray is None:
        return normalize(d) + 0.0
    e = np.asarray(eye_ray, dtype=np.float64)
    if e.ndim == 0 or e.shape[-1] != 3:
        raise ValueError(f"eye_ray must have shape (..., 3), got {e.shape}")
    shape = np.broadcast_shapes(d.shape[:-1], e.shape[:-1])
    d = np.broadcast_to(d, shape + (3,))
    e = np.broadcast_to(e, shape + (3,))

    finite = np.isfinite(e).all(axis=-1)
    e0 = np.where(finite[..., None], e, 0.0)
    n = np.linalg.norm(e0, axis=-1)
    ok = finite & (n > _EPS)
    ez = np.where(ok[..., None], e0 / np.where(ok, n, 1.0)[..., None],
                  np.array([0.0, 0.0, 1.0]))
    c = np.cross(np.array([0.0, 1.0, 0.0]), ez)
    cn = np.linalg.norm(c, axis=-1)
    cok = cn >= 1e-9
    rx = np.where(cok[..., None], c / np.where(cok, cn, 1.0)[..., None],
                  np.array([1.0, 0.0, 0.0]))
    ry = np.cross(ez, rx)
    out = d[..., 0:1] * rx + d[..., 1:2] * ry + d[..., 2:3] * ez
    return normalize(np.where(ok[..., None], out, d)) + 0.0


def gaze360_to_camera(yaw_rad, pitch_rad, eye_ray: Optional[np.ndarray] = None) -> np.ndarray:
    """Gaze360-convention angles (yaw ``t``, pitch ``p``, radians) -> unit gaze in the camera frame.

    Gaze360 dataset eye frame (erkil1452/gaze360, dataset/README.md): right
    handed, +x points to the image LEFT, +y up, +z away from the camera, so
    ``[-1, 0, 0]`` looks to the image right and ``[0, 0, -1]`` straight into the
    camera. The labels are ``t = atan2(g.x, -g.z)`` and ``p = asin(g.y)``
    (code/data_loader.py), i.e. ``g = (cos p sin t, sin p, -cos p cos t)``. In
    the OpenCV camera frame (+X right, +Y down, +Z forward) this is
    ``(-g.x, -g.y, g.z) = (-cos p sin t, -sin p, -cos p cos t)``: a proper
    rotation, ``diag(-1, -1, 1)``. ``t = p = 0`` looks at the camera, +t turns
    towards the image left, +p looks up. The same labels train L2CS/MobileGaze.

    The models predict gaze relative to the camera -> eye ray, so when
    ``eye_ray`` (camera -> eye vector in the camera frame, any length) is given
    the vector is rotated with :func:`ray_aligned`; ``t = p = 0`` then looks back
    along the ray at the camera.

    ``eye_ray`` may be ``None``, a ``(3,)`` vector or a ``(..., 3)`` array (one
    ray per person). The output shape is
    ``broadcast_shapes(t.shape, p.shape, eye_ray.shape[:-1]) + (3,)``. Rows whose
    ray is zero or non-finite are returned uncorrected. Raises ValueError when
    the last dimension of ``eye_ray`` is not 3.
    """
    t = np.asarray(yaw_rad, dtype=np.float64)
    p = np.asarray(pitch_rad, dtype=np.float64)
    t, p = np.broadcast_arrays(t, p)
    d = np.stack([-np.cos(p) * np.sin(t), -np.sin(p), -np.cos(p) * np.cos(t)], axis=-1)
    return ray_aligned(d, eye_ray)


def orthonormal_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors ``(u, v)`` orthogonal to ``d`` with ``cross(u, v) = unit(d)``.

    Valid for ANY direction (the helper axis is the one least aligned with
    ``d``). Accepts ``(3,)`` or ``(..., 3)``. Raises ValueError for zero or
    non-finite directions.
    """
    d = np.asarray(d, dtype=np.float64)
    if d.shape[-1] != 3:
        raise ValueError(f"direction must have shape (..., 3), got {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("direction must be finite")
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    if (n < _EPS).any():
        raise ValueError("direction must be non-zero")
    dn = d / n
    helper = np.eye(3)[np.argmin(np.abs(dn), axis=-1)]
    u = normalize(np.cross(helper, dn))
    v = np.cross(dn, u)
    return u, v


def cone_rays(direction: np.ndarray, half_angle_deg: float, n: int = 20) -> np.ndarray:
    """``n`` unit rays on a cone of half-angle ``half_angle_deg`` around ``direction``.

    Rays are evenly spaced in azimuth (``phi = 2 pi k / n``, no duplicated
    endpoint), each exactly ``half_angle_deg`` from the axis. ``direction``
    ``(3,)`` -> ``(n, 3)``; ``(..., 3)`` -> ``(..., n, 3)``.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    d = np.asarray(direction, dtype=np.float64)
    u, v = orthonormal_basis(d)
    dn = normalize(d)
    a = math.radians(float(half_angle_deg))
    phi = 2.0 * np.pi * np.arange(n) / n
    cphi = np.cos(phi)[:, None]
    sphi = np.sin(phi)[:, None]
    ring = cphi * u[..., None, :] + sphi * v[..., None, :]
    rays = math.cos(a) * dn[..., None, :] + math.sin(a) * ring
    return normalize(rays) + 0.0


# --------------------------------------------------------------------------- ray / box


def _prep(origins, dirs, bmin, bmax):
    o = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    d = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    o, d = np.broadcast_arrays(o, d)
    lo_in = np.atleast_2d(np.asarray(bmin, dtype=np.float64))
    hi_in = np.atleast_2d(np.asarray(bmax, dtype=np.float64))
    lo = np.minimum(lo_in, hi_in)
    hi = np.maximum(lo_in, hi_in)
    if o.shape[-1] != 3 or lo.shape[-1] != 3 or lo.ndim != 2 or o.ndim != 2:
        raise ValueError("origins/dirs must be (R, 3) or (3,), boxes (B, 3) or (3,)")
    ray_ok = np.isfinite(o).all(-1) & np.isfinite(d).all(-1)
    box_ok = np.isfinite(lo).all(-1) & np.isfinite(hi).all(-1)
    o = np.where(ray_ok[:, None], o, 0.0)
    d = np.where(ray_ok[:, None], d, 0.0)
    lo = np.where(box_ok[:, None], lo, 0.0)
    hi = np.where(box_ok[:, None], hi, 0.0)
    ok = ray_ok[:, None] & box_ok[None, :]
    return o, d, lo, hi, ok


def _slabs(o, d, lo, hi):
    """Per (ray, box) parametric interval ``(t_near, t_far)`` of the infinite line."""
    par = np.abs(d) < _PAR_EPS  # (R, 3)
    inv = 1.0 / np.where(par, 1.0, d)
    t1 = (lo[None, :, :] - o[:, None, :]) * inv[:, None, :]  # (R, B, 3)
    t2 = (hi[None, :, :] - o[:, None, :]) * inv[:, None, :]
    near = np.minimum(t1, t2)
    far = np.maximum(t1, t2)
    inside = (o[:, None, :] >= lo[None, :, :]) & (o[:, None, :] <= hi[None, :, :])
    parb = np.broadcast_to(par[:, None, :], near.shape)
    near = np.where(parb, np.where(inside, -np.inf, np.inf), near)
    far = np.where(parb, np.where(inside, np.inf, -np.inf), far)
    return near.max(axis=-1), far.min(axis=-1)


def ray_aabb(
    origins: np.ndarray, dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised slab test of rays against axis-aligned boxes (camera frame, metres).

    ``origins`` ``(R, 3)`` or ``(3,)`` (broadcast against ``dirs``), ``dirs``
    ``(R, 3)`` (need not be unit; ``t`` is in units of ``|dir|``), boxes
    ``bmin, bmax`` ``(B, 3)``. Returns ``(hit, t_enter, t_exit)``, each ``(R, B)``.
    A ray hits iff ``t_exit >= max(t_enter, 0)``, so boxes behind the origin do
    not count; ``t_enter`` is clipped to ``>= 0`` (origin inside -> 0). Zero
    direction components are handled without inf/NaN; an all-zero direction is
    a point (hit iff inside). Returned ``t`` are always finite: 0 on misses, and
    ``t_exit`` is 0 when unbounded (zero direction). Non-finite rays/boxes never hit.
    """
    o, d, lo, hi, ok = _prep(origins, dirs, bmin, bmax)
    t_near, t_far = _slabs(o, d, lo, hi)
    t_enter = np.maximum(t_near, 0.0)
    hit = ok & (t_far >= t_enter)
    t_enter_out = np.where(hit, t_enter, 0.0)
    t_exit_out = np.where(hit & np.isfinite(t_far), t_far, 0.0)
    return hit, t_enter_out, t_exit_out


def segment_aabb(
    p0: np.ndarray, p1: np.ndarray, bmin: np.ndarray, bmax: np.ndarray
) -> np.ndarray:
    """Whether segments ``p0 -> p1`` ``(N, 3)`` intersect boxes ``(B, 3)``: ``(N, B)`` bool."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    o, d, lo, hi, ok = _prep(p0, p1 - p0, bmin, bmax)
    t_near, t_far = _slabs(o, d, lo, hi)
    return ok & (np.minimum(t_far, 1.0) >= np.maximum(t_near, 0.0))


def aabb_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds ``(min, max)`` float64 (3,) of the finite rows of ``points (N, 3)``."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = p[np.isfinite(p).all(axis=1)]
    if p.shape[0] == 0:
        raise ValueError("aabb_from_points needs at least one finite point")
    return p.min(axis=0), p.max(axis=0)
