"""Unit tests of the geometry core (vfoa.geometry) and the legacy Arrow_line wrapper.

Refs #1 #2 #5 #16 #30.
"""

from __future__ import annotations

import inspect
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from vfoa.geometry import (
    aabb_from_points,
    angle_between,
    backproject,
    cone_rays,
    depth_to_disparity,
    disparity_to_depth,
    euler_to_forward,
    eye_pixel,
    gaze360_to_camera,
    normalize,
    orthonormal_basis,
    patch_depth,
    project_points,
    ray_aabb,
    segment_aabb,
    unproject,
)
from vfoa.types import Intrinsics

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

AXES = [
    np.array(a, dtype=np.float64)
    for a in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])
]


# --------------------------------------------------------------------------- camera / depth


def test_backproject_matches_sintel_ground_truth(sintel_sample):
    _, depth, intr = sintel_sample(1)
    pts = backproject(depth, intr)
    assert pts.dtype == np.float32
    assert pts.shape == (436, 1024, 3)
    assert np.array_equal(pts[..., 2], depth)

    uv = project_points(pts, intr)
    uu, vv = np.meshgrid(np.arange(1024), np.arange(436))
    assert np.abs(uv[..., 0] - uu).max() < 1e-3
    assert np.abs(uv[..., 1] - vv).max() < 1e-3

    disp = depth_to_disparity(depth, intr.fx, intr.baseline)
    ref = cv2.reprojectImageTo3D(disp, intr.Q())
    rel = np.linalg.norm(ref - pts, axis=-1) / np.linalg.norm(pts, axis=-1)
    assert rel.max() < 1e-4
    assert ref[..., 2].min() > 0
    assert pts[0, 0, 0] < 0 and pts[0, 0, 1] < 0


def test_backproject_invalid_is_nan():
    intr = Intrinsics(100.0, 100.0, 2.0, 1.0, 4, 3)
    depth = np.ones((3, 4), dtype=np.float32)
    depth[0, 0] = 0
    depth[0, 1] = -1
    depth[0, 2] = np.nan
    depth[0, 3] = np.inf
    valid = np.ones((3, 4), dtype=bool)
    valid[1, 0] = False
    pts = backproject(depth, intr, valid=valid)
    bad = np.zeros((3, 4), dtype=bool)
    bad[0, :] = True
    bad[1, 0] = True
    assert np.isnan(pts[bad]).all()
    assert np.isfinite(pts[~bad]).all()


def test_unproject_project_roundtrip(rng):
    intr = Intrinsics(500.0, 450.0, 320.0, 240.0, 640, 480)
    uv = rng.uniform(0, 640, size=(50, 2))
    z = rng.uniform(0.5, 10, size=50)
    p = unproject(uv, z, intr)
    assert p.shape == (50, 3)
    np.testing.assert_allclose(p[:, 2], z)
    np.testing.assert_allclose(project_points(p, intr), uv, atol=1e-9)
    # scalar
    q = unproject(np.array([320.0, 240.0]), 2.0, intr)
    np.testing.assert_allclose(q, [0, 0, 2])


def test_disparity_depth_roundtrip(rng):
    z = rng.uniform(0.2, 50.0, size=1000).astype(np.float32)
    d = depth_to_disparity(z, 896.0, 0.1)
    assert d.dtype == np.float32
    z2, valid = disparity_to_depth(d, 896.0, 0.1)
    assert valid.all()
    np.testing.assert_allclose(z2, z, rtol=1e-5)
    assert depth_to_disparity(np.array([2.0]), 896.0, 0.1)[0] == pytest.approx(44.8)

    disp = np.array([-1.0, 0.0, 1e-4, np.nan, np.inf, 2.0])
    zz, vv = disparity_to_depth(disp, 896.0, 0.1)
    assert vv.tolist() == [False] * 5 + [True]
    assert (zz[:5] == 0).all()
    assert zz[5] == pytest.approx(896 * 0.1 / 2.0)
    assert np.isfinite(zz).all()
    dd = depth_to_disparity(np.array([0.0, -1.0, np.nan, np.inf]), 896.0, 0.1)
    assert (dd == 0).all()
    with pytest.raises(ValueError):
        disparity_to_depth(disp, 0.0, 0.1)


# --------------------------------------------------------------------------- directions


def test_euler_to_forward_axes():
    np.testing.assert_allclose(euler_to_forward(0, 0, 0), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(euler_to_forward(90, 0), [-1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(euler_to_forward(0, 90), [0, -1, 0], atol=1e-12)
    assert euler_to_forward(0, 0).shape == (3,)


def test_euler_unit_roll_invariant(rng):
    yaw = rng.uniform(-180, 180, 200)
    pitch = rng.uniform(-90, 90, 200)
    roll = rng.uniform(-180, 180, 200)
    f = euler_to_forward(yaw, pitch, roll)
    assert f.shape == (200, 3)
    np.testing.assert_allclose(np.linalg.norm(f, axis=-1), 1.0, atol=1e-12)
    np.testing.assert_allclose(f, euler_to_forward(yaw, pitch, 0.0), atol=1e-15)


def test_euler_matches_draw_axis(rng):
    for yaw, pitch in rng.uniform(-80, 80, size=(20, 2)):
        f = euler_to_forward(yaw, pitch)
        y, p = math.radians(yaw), math.radians(pitch)
        # thesis draw_axis: z-axis image endpoint (sin(-yaw), -cos(yaw) sin(pitch))
        assert f[0] == pytest.approx(math.sin(-y), abs=1e-12)
        assert f[1] == pytest.approx(-math.cos(y) * math.sin(p), abs=1e-12)


def test_gaze360_axes():
    np.testing.assert_allclose(gaze360_to_camera(0, 0), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(gaze360_to_camera(math.pi / 2, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(gaze360_to_camera(0, math.pi / 2), [0, -1, 0], atol=1e-12)
    g = gaze360_to_camera(np.linspace(-3, 3, 7), np.linspace(-1, 1, 7))
    assert g.shape == (7, 3)
    np.testing.assert_allclose(np.linalg.norm(g, axis=-1), 1.0, atol=1e-12)


def test_gaze360_eye_ray(rng):
    ray = np.array([0.5, -0.3, 2.0])
    np.testing.assert_allclose(gaze360_to_camera(0, 0, ray), -normalize(ray), atol=1e-12)
    for t, p in rng.uniform(-1.5, 1.5, size=(10, 2)):
        np.testing.assert_allclose(
            gaze360_to_camera(t, p, [0, 0, 5]), gaze360_to_camera(t, p), atol=1e-12
        )
    a1, a2 = (0.3, -0.2), (-0.5, 0.4)
    ang0 = angle_between(gaze360_to_camera(*a1), gaze360_to_camera(*a2))
    ang1 = angle_between(gaze360_to_camera(*a1, ray), gaze360_to_camera(*a2, ray))
    assert ang1 == pytest.approx(ang0, abs=1e-9)
    g = gaze360_to_camera(0.3, 0.2, [0, 2, 0])
    assert np.isfinite(g).all()
    assert np.linalg.norm(g) == pytest.approx(1.0, abs=1e-12)
    # None / zero / non-finite eye rays mean no correction
    for bad in (None, [0, 0, 0], [np.nan, 0, 1]):
        np.testing.assert_allclose(gaze360_to_camera(0.3, 0.2, bad), gaze360_to_camera(0.3, 0.2))
    gs = gaze360_to_camera(rng.uniform(-3, 3, 50), rng.uniform(-1.5, 1.5, 50), ray)
    np.testing.assert_allclose(np.linalg.norm(gs, axis=-1), 1.0, atol=1e-12)


# --------------------------------------------------------------------------- cones


def test_orthonormal_basis_any_direction(rng):
    dirs = AXES + list(rng.normal(size=(100, 3)) * rng.uniform(0.01, 100, size=(100, 1)))
    for d in dirs:
        u, v = orthonormal_basis(d)
        dn = d / np.linalg.norm(d)
        assert abs(np.dot(u, dn)) < 1e-12 and abs(np.dot(v, dn)) < 1e-12
        assert abs(np.dot(u, v)) < 1e-12
        assert np.linalg.norm(u) == pytest.approx(1, abs=1e-12)
        assert np.linalg.norm(v) == pytest.approx(1, abs=1e-12)
        np.testing.assert_allclose(np.cross(u, v), dn, atol=1e-12)
    for bad in ([0, 0, 0], [np.nan, 1, 0]):
        with pytest.raises(ValueError):
            orthonormal_basis(np.array(bad, dtype=float))


def test_cone_rays_half_angle_unique(rng):
    axes = AXES + [normalize(x) for x in rng.normal(size=(10, 3))]
    for d in axes:
        for a in (0.5, 11.3, 21.8, 24.0, 89.0):
            rays = cone_rays(d, a, 20)
            assert rays.shape == (20, 3)
            assert np.isfinite(rays).all()
            np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-12)
            ang = angle_between(rays, d)
            assert np.abs(ang - math.radians(a)).max() < 1e-9
            assert len(np.unique(np.round(rays, 9), axis=0)) == 20
            np.testing.assert_allclose(rays.mean(axis=0), math.cos(math.radians(a)) * d, atol=1e-12)
            np.testing.assert_allclose(cone_rays(300 * d, a, 20), rays, atol=1e-12)
    batch = cone_rays(np.stack(AXES), 10.0, 8)
    assert batch.shape == (6, 8, 3)
    with pytest.raises(ValueError):
        cone_rays([0, 0, 1], 10, 0)


# --------------------------------------------------------------------------- ray / box


def test_ray_aabb_basic():
    bmin = np.array([[-1, -1, 2], [-1, -1, -3], [-1, -1, -1]], dtype=float)
    bmax = np.array([[1, 1, 3], [1, 1, -2], [1, 1, 1]], dtype=float)
    hit, te, tx = ray_aabb(np.zeros(3), np.array([[0, 0, 1.0]]), bmin, bmax)
    assert hit.shape == te.shape == tx.shape == (1, 3)
    assert hit.tolist() == [[True, False, True]]
    assert te[0, 0] == pytest.approx(2) and tx[0, 0] == pytest.approx(3)
    assert te[0, 2] == 0 and tx[0, 2] == pytest.approx(1)
    # zero x/y components: inside the slab hits, outside misses
    origins = np.array([[0.5, 0.5, 0.0], [1.5, 0.0, 0.0]])
    dirs = np.array([[0, 0, 1.0], [0, 0, 1.0]])
    hit, te, tx = ray_aabb(origins, dirs, bmin[:1], bmax[:1])
    assert hit[:, 0].tolist() == [True, False]
    for arr in (te, tx):
        assert np.isfinite(arr).all()
    assert te[1, 0] == 0 and tx[1, 0] == 0
    # all-zero direction: a point
    hit, te, tx = ray_aabb(np.array([[0, 0, 0.0], [5, 5, 5]]), np.zeros((2, 3)), bmin, bmax)
    assert hit.tolist() == [[False, False, True], [False, False, False]]
    assert np.isfinite(te).all() and np.isfinite(tx).all()


def _scalar_ray_box(o, d, lo, hi):
    t0, t1 = -math.inf, math.inf
    for k in range(3):
        if d[k] == 0.0:
            if o[k] < lo[k] or o[k] > hi[k]:
                return False, 0.0, 0.0
            continue
        a = (lo[k] - o[k]) / d[k]
        b = (hi[k] - o[k]) / d[k]
        if a > b:
            a, b = b, a
        t0 = max(t0, a)
        t1 = min(t1, b)
    te = max(t0, 0.0)
    if t1 >= te:
        return True, te, (t1 if math.isfinite(t1) else 0.0)
    return False, 0.0, 0.0


def test_ray_aabb_matches_scalar_reference(rng):
    R, B = 50, 20
    centers = rng.uniform(-2, 2, size=(B, 3))
    half = rng.uniform(0.5, 2.0, size=(B, 3))
    bmin, bmax = centers - half, centers + half
    origins = rng.uniform(-4, 4, size=(R, 3))
    origins[:10] = centers[:10]
    dirs = rng.normal(size=(R, 3))
    zero = rng.uniform(size=(R, 3)) < 0.3
    zero[zero.all(axis=1), 0] = False
    dirs[zero] = 0.0
    assert (np.abs(dirs).sum(axis=1) > 0).all()
    hit, te, tx = ray_aabb(origins, dirs, bmin, bmax)
    ref_hit = np.zeros((R, B), dtype=bool)
    ref_te = np.zeros((R, B))
    ref_tx = np.zeros((R, B))
    for i in range(R):
        for j in range(B):
            ref_hit[i, j], ref_te[i, j], ref_tx[i, j] = _scalar_ray_box(
                origins[i].tolist(), dirs[i].tolist(), bmin[j].tolist(), bmax[j].tolist()
            )
    assert np.array_equal(hit, ref_hit)
    np.testing.assert_allclose(te, ref_te, atol=1e-9)
    np.testing.assert_allclose(tx, ref_tx, atol=1e-9)
    assert 100 < hit.sum() < 900

    # independent property check
    ii, jj = np.nonzero(hit)
    for t in (te, tx):
        p = origins[ii] + t[ii, jj, None] * dirs[ii]
        assert (p >= bmin[jj] - 1e-9).all() and (p <= bmax[jj] + 1e-9).all()
    ts = np.linspace(0, 20, 400)
    ii, jj = np.nonzero(~hit)
    p = origins[ii, None, :] + ts[None, :, None] * dirs[ii, None, :]
    inside = ((p >= bmin[jj, None]) & (p <= bmax[jj, None])).all(-1)
    assert not inside.any()


def test_segment_aabb_agrees_with_ray_aabb(rng):
    bmin = rng.uniform(-3, 0, size=(15, 3))
    bmax = bmin + rng.uniform(0.3, 2, size=(15, 3))
    p0 = rng.uniform(-4, 4, size=(60, 3))
    p1 = rng.uniform(-4, 4, size=(60, 3))
    p1[::7, 1] = p0[::7, 1]  # some zero components
    seg = segment_aabb(p0, p1, bmin, bmax)
    hit, te, _ = ray_aabb(p0, p1 - p0, bmin, bmax)
    assert seg.shape == (60, 15)
    assert np.array_equal(seg, hit & (te <= 1.0))
    one = (np.array([[-1.0, -1.0, -1.0]]), np.array([[1.0, 1.0, 1.0]]))
    assert not segment_aabb([[0, 0, -5.0]], [[0, 0, -2.0]], *one)[0, 0]
    assert segment_aabb([[0, 0, -5.0]], [[0, 0, -0.5]], *one)[0, 0]
    assert segment_aabb([[0.1, 0.1, 0.1]], [[0.1, 0.1, 0.1]], *one)[0, 0]


def test_ray_aabb_non_finite_inputs_never_hit():
    bmin = np.array([[-1, -1, 1.0], [np.nan, -1, 1]])
    bmax = np.array([[1, 1, 2.0], [1, 1, 2]])
    origins = np.array([[0, 0, 0.0], [np.nan, 0, 0], [0, 0, 0]])
    dirs = np.array([[0, 0, 1.0], [0, 0, 1], [0, np.inf, 1]])
    hit, te, tx = ray_aabb(origins, dirs, bmin, bmax)
    assert hit.tolist() == [[True, False], [False, False], [False, False]]
    assert np.isfinite(te).all() and np.isfinite(tx).all()
    assert not segment_aabb(origins, origins + dirs, bmin, bmax)[1:].any()


# --------------------------------------------------------------------------- pixels


def test_patch_depth_borders():
    h, w = 20, 30
    depth = np.full((h, w), 2.0, dtype=np.float32)
    for u, v in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        val = patch_depth(depth, None, u, v, 3)
        assert isinstance(val, float) and val == 2.0
    assert patch_depth(depth, None, -2.0, -2.0, 3) == 2.0
    assert patch_depth(depth, None, w + 1.0, h + 1.0, 3) == 2.0
    assert patch_depth(depth, None, -50.0, 5.0, 3) is None
    assert patch_depth(depth, None, np.nan, 5.0, 3) is None

    bad = np.zeros((h, w), dtype=np.float32)
    bad[0, 0] = np.nan
    bad[0, 1] = np.inf
    bad[1, 0] = -1
    bad[1, 1] = 5.0
    valid = np.ones((h, w), dtype=bool)
    valid[1, 1] = False
    assert patch_depth(bad, valid, 0, 0, 3) is None
    assert patch_depth(bad, None, 0, 0, 3) == 5.0

    mixed = np.zeros((h, w), dtype=np.float32)
    mixed[10, 10] = 1.0
    mixed[10, 11] = 3.0
    mixed[11, 10] = 2.0
    mixed[9, 9] = np.nan
    assert patch_depth(mixed, None, 10.4, 10.4, 1) == 2.0


def test_eye_pixel_rules():
    box = np.array([10, 20, 50, 80], dtype=np.float32)
    e = eye_pixel(box)
    assert e.dtype == np.float32
    np.testing.assert_allclose(e, [30, 41], atol=1e-5)
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[1] = (20, 30, 0.9)
    kp[2] = (40, 34, 0.8)
    np.testing.assert_allclose(eye_pixel(box, kp, 0.35), [30, 32])
    kp[2, 2] = 0.1
    np.testing.assert_allclose(eye_pixel(box, kp, 0.35), [30, 41], atol=1e-5)
    assert eye_pixel(None, kp, 0.35) is None
    assert eye_pixel(None) is None
    kp[2, 2] = 0.9
    np.testing.assert_allclose(eye_pixel(None, kp), [30, 32])


# --------------------------------------------------------------------------- misc


def test_angle_between_accuracy_and_broadcast(rng):
    a = np.array([1.0, 2.0, 3.0])
    assert angle_between(a, 2 * a) == pytest.approx(0, abs=1e-9)
    assert angle_between(a, -a) == pytest.approx(math.pi, abs=1e-9)
    b = np.array([math.cos(1e-6), math.sin(1e-6), 0.0])
    assert angle_between([1, 0, 0], b) == pytest.approx(1e-6, abs=1e-9)
    out = angle_between(rng.normal(size=(5, 1, 3)), rng.normal(size=(4, 3)))
    assert out.shape == (5, 4)
    assert ((out >= 0) & (out <= math.pi)).all()


def test_aabb_from_points():
    pts = np.array([[0, 1, 2], [np.nan, 0, 0], [-1, 5, 3], [np.inf, 0, 0]], dtype=np.float32)
    lo, hi = aabb_from_points(pts)
    assert lo.dtype == np.float64
    np.testing.assert_allclose(lo, [-1, 1, 2])
    np.testing.assert_allclose(hi, [0, 5, 3])
    with pytest.raises(ValueError):
        aabb_from_points(np.full((2, 3), np.nan))


def test_normalize_zero():
    out = normalize(np.array([[0, 0, 0], [3, 0, 4.0]]))
    np.testing.assert_allclose(out, [[0, 0, 0], [0.6, 0, 0.8]])


# --------------------------------------------------------------------------- legacy Arrow_line


def test_arrow_line_imports_without_open3d(repo_root):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    code = (
        "import sys, inspect, Arrow_line\n"
        "assert 'open3d' not in sys.modules, 'open3d imported'\n"
        "assert 'print(' not in inspect.getsource(Arrow_line.vect2cone)\n"
        "print('ok')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root / "thesis_final"), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout


def test_vect2cone_wrapper():
    import Arrow_line

    X0 = np.array([0.1, 0.2, 0.3])
    X1 = X0 + np.array([0.0, 0.0, -4.0])
    v = Arrow_line.vect2cone(X0, X1, 1.0, 20)
    assert v.shape == (20, 3)
    np.testing.assert_allclose(angle_between(v, X1 - X0), math.atan2(1.0, 4.0), atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), math.hypot(4.0, 1.0), atol=1e-9)
    assert len(np.unique(np.round(v, 9), axis=0)) == 20
    vx = Arrow_line.vect2cone([0, 0, 0], [5, 0, 0], 2, 12)
    assert np.isfinite(vx).all()
    with pytest.raises(ValueError):
        Arrow_line.vect2cone([1, 1, 1], [1, 1, 1], 1, 5)


def test_align_matrix_guards(rng):
    import Arrow_line

    z = np.array([0.0, 0.0, 1.0])
    vecs = list(rng.normal(size=(50, 3))) + [z, -z, np.array([1e-9, 0, -1.0]), np.array([1e-13, 0, -1])]
    for v in vecs:
        R = Arrow_line.caculate_align_mat(v)
        np.testing.assert_allclose(R @ z, v / np.linalg.norm(v), atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(Arrow_line.caculate_align_mat(np.zeros(3)), np.eye(3))
    Rz, Ry = Arrow_line.calculate_zy_rotation_for_arrow(np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(Rz).all() and np.isfinite(Ry).all()
    Rz, Ry = Arrow_line.calculate_zy_rotation_for_arrow(np.array([0.0, 1.0, 0.0]))
    assert np.isfinite(Rz).all() and np.isfinite(Ry).all()


def test_get_arrow_open3d():
    pytest.importorskip("open3d")
    import Arrow_line

    o = np.array([0.5, -0.2, 3.0])
    origin_copy = o.copy()
    mesh = Arrow_line.get_arrow(o, vec=[0, 0, -2.0])
    pts = np.asarray(mesh.vertices)
    assert pts[:, 2].min() == pytest.approx(o[2] - 2, abs=1e-6)
    assert pts[:, 2].max() == pytest.approx(o[2], abs=1e-6)
    assert np.array_equal(o, origin_copy)
    arrows = Arrow_line.create_cone_arrow(o, np.array([0, 0, -1.0]), np.array([]), 90, 6, "gaze")
    assert len(arrows) == 7
    assert Arrow_line.create_cone_arrow(o, np.zeros(3), None, 90, 6, "gaze") == []
