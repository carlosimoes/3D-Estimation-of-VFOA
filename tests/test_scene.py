"""Tests of vfoa.scene: metric objects, robust eyes, own segments and persistent ids."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from vfoa.geometry import aabb_from_points, backproject, eye_pixel, patch_depth
from vfoa.scene import ObjectRegistry, SceneBuilder
from vfoa.types import DepthFrame, Intrinsics, Person, PersonDetection, SegmentInfo, Segmentation

W, H = 640, 360

# (segment id, label, is_thing, (r0, r1, c0, c1) half-open pixel bounds, depth m)
RECTS = [
    (1, "cup", True, (40, 120, 60, 200), 1.2),
    (2, "tv", True, (150, 300, 300, 520), 2.0),
    (3, "wall", False, (0, 40, 400, 640), 3.5),
]


def planar_scene():
    """Fronto-parallel rectangles at known depths; label 0 and depth 6 m elsewhere."""
    intr = Intrinsics.from_hfov(W, H, 60.0, baseline=0.1)
    depth = np.full((H, W), 6.0, dtype=np.float32)
    labels = np.zeros((H, W), dtype=np.int32)
    for sid, _, _, (r0, r1, c0, c1), z in RECTS:
        depth[r0:r1, c0:c1] = z
        labels[r0:r1, c0:c1] = sid
    seg = Segmentation(labels, [SegmentInfo(sid, lab, th) for sid, lab, th, _, _ in RECTS])
    return DepthFrame(depth, np.ones((H, W), dtype=bool), intr), seg, intr


def gt_aabb(rect, intr):
    """Ground-truth AABB / centroid from the first and last pixel centres of a rectangle."""
    (r0, r1, c0, c1), z = rect
    x0, x1 = (c0 - intr.cx) * z / intr.fx, (c1 - 1 - intr.cx) * z / intr.fx
    y0, y1 = (r0 - intr.cy) * z / intr.fy, (r1 - 1 - intr.cy) * z / intr.fy
    bmin = np.array([x0, y0, z])
    bmax = np.array([x1, y1, z])
    return bmin, bmax, (bmin + bmax) / 2


def by_segment(objects):
    return {o.segment_id: o for o in objects}


def person(track_id=0, head=None, body=None, kps=None):
    return Person(track_id, PersonDetection(head_box=head, body_box=body, keypoints=kps))


# --------------------------------------------------------------------------- objects


def test_planar_scene_matches_ground_truth(make_config):
    depth, seg, intr = planar_scene()
    builder = SceneBuilder(make_config("fast"))
    objs = builder.build_objects(seg, depth)
    assert [o.segment_id for o in objs] == [1, 2, 3]
    ids = [o.object_id for o in objs]
    assert len(set(ids)) == 3 and min(ids) >= 1
    s = builder.stride
    for sid, lab, th, rect, z in RECTS:
        o = by_segment(objs)[sid]
        bmin, bmax, cen = gt_aabb((rect, z), intr)
        np.testing.assert_allclose(o.aabb_min, bmin, atol=0.01)
        np.testing.assert_allclose(o.aabb_max, bmax, atol=0.01)
        np.testing.assert_allclose(o.centroid, cen, atol=0.01)
        assert o.label == lab and o.is_thing is th
        assert o.points.dtype == np.float32 and o.points.shape[1] == 3
        assert 0 < o.points.shape[0] <= 256
        assert np.all(o.points >= o.aabb_min - 1e-6) and np.all(o.points <= o.aabb_max + 1e-6)
        r0, r1, c0, c1 = rect
        n_rows = len([r for r in range(0, H, s) if r0 <= r < r1])
        n_cols = len([c for c in range(0, W, s) if c0 <= c < c1])
        assert o.n_points == n_rows * n_cols


def test_max_points_subsample():
    depth, seg, _ = planar_scene()
    objs = SceneBuilder(max_points_per_object=50).build_objects(seg, depth)
    for o in objs:
        assert 0 < o.points.shape[0] <= 50
        assert o.n_points > 50


def test_corrupted_depth_never_affects_aabb(rng):
    depth, seg, intr = planar_scene()
    clean = by_segment(SceneBuilder().build_objects(seg, depth))
    d = depth.depth.copy()
    v = depth.valid.copy()
    r0, r1, c0, c1 = RECTS[1][3]
    rr, cc = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    pick = rng.permutation(rr.size)[: int(0.3 * rr.size)]
    parts = np.array_split(pick, 6)
    d[rr[parts[0]], cc[parts[0]]] = 0.0
    v[rr[parts[0]], cc[parts[0]]] = False
    d[rr[parts[1]], cc[parts[1]]] = np.nan
    v[rr[parts[1]], cc[parts[1]]] = False
    d[rr[parts[2]], cc[parts[2]]] = np.inf
    v[rr[parts[2]], cc[parts[2]]] = False
    d[rr[parts[3]], cc[parts[3]]] = 1e4
    v[rr[parts[3]], cc[parts[3]]] = False
    d[rr[parts[4]], cc[parts[4]]] = np.nan  # valid True but NaN
    d[rr[parts[5]], cc[parts[5]]] = -3.0  # valid True but negative
    # keep corners clean so the ground truth AABB is still reachable
    for r, c in ((r0, c0), (r0, c1 - 2), (r1 - 2, c0), (r1 - 2, c1 - 2)):
        d[r, c], v[r, c] = 2.0, True
    bad = DepthFrame(d, v, intr)
    o = by_segment(SceneBuilder().build_objects(seg, bad))[2]
    bmin, bmax, cen = gt_aabb((RECTS[1][3], RECTS[1][4]), intr)
    for got, want in ((o.aabb_min, bmin), (o.aabb_max, bmax), (o.centroid, cen)):
        np.testing.assert_allclose(got, want, atol=0.01)
    np.testing.assert_allclose(o.aabb_min, clean[2].aabb_min, atol=1e-9)
    np.testing.assert_allclose(o.aabb_max, clean[2].aabb_max, atol=1e-9)
    assert np.isfinite(o.points).all()
    assert o.n_points < clean[2].n_points


def test_min_points_and_ignore_labels():
    depth, seg, intr = planar_scene()
    labels = seg.label_map.copy()
    labels[330:333, 10:13] = 4  # 9 px -> few strided points
    seg2 = Segmentation(labels, seg.segments + [SegmentInfo(4, "spoon")])
    objs = SceneBuilder(ignore_labels=["wall"]).build_objects(seg2, depth)
    assert sorted(o.segment_id for o in objs) == [1, 2]
    objs = SceneBuilder(min_points=1, point_stride=1).build_objects(seg2, depth)
    assert 4 in by_segment(objs)


def test_unknown_segment_id_is_skipped():
    depth, seg, _ = planar_scene()
    seg2 = Segmentation(seg.label_map, seg.segments[:2])
    assert sorted(o.segment_id for o in SceneBuilder().build_objects(seg2, depth)) == [1, 2]


def test_all_invalid_depth_returns_empty():
    depth, seg, intr = planar_scene()
    assert SceneBuilder().build_objects(seg, DepthFrame(depth.depth, np.zeros_like(depth.valid), intr)) == []
    nan = np.full_like(depth.depth, np.nan)
    assert SceneBuilder().build_objects(seg, DepthFrame(nan, np.ones_like(depth.valid), intr)) == []
    zeros = np.zeros_like(depth.depth)
    assert SceneBuilder().build_objects(seg, DepthFrame(zeros, depth.valid, intr)) == []


def test_none_inputs():
    depth, seg, _ = planar_scene()
    b = SceneBuilder()
    assert b.build_objects(None, depth) == []
    assert b.build_objects(seg, None) == []
    assert b.registry.next_id == 1


def test_invalid_params():
    with pytest.raises(ValueError):
        SceneBuilder(outlier="bogus")
    with pytest.raises(ValueError):
        SceneBuilder(point_stride=0)
    with pytest.raises(ValueError):
        SceneBuilder(outlier_percentiles=[50, 10])


def test_inputs_not_mutated():
    depth, seg, _ = planar_scene()
    d0, v0, l0 = depth.depth.copy(), depth.valid.copy(), seg.label_map.copy()
    SceneBuilder().build_objects(seg, depth)
    assert np.array_equal(depth.depth, d0) and np.array_equal(depth.valid, v0)
    assert np.array_equal(seg.label_map, l0)


def test_percentile_rejects_depth_bleed(rng):
    depth, seg, intr = planar_scene()
    d = depth.depth.copy()
    r0, r1, c0, c1 = RECTS[1][3]
    mask = np.zeros((H, W), dtype=bool)
    mask[r0:r1, c0:c1] = rng.random((r1 - r0, c1 - c0)) < 0.01
    d[mask] = 8.0
    bad = DepthFrame(d, depth.valid, intr)
    o = by_segment(SceneBuilder(outlier="percentile").build_objects(seg, bad))[2]
    assert o.aabb_max[2] == pytest.approx(2.0, abs=1e-6)
    o = by_segment(SceneBuilder(outlier="none").build_objects(seg, bad))[2]
    assert o.aabb_max[2] == pytest.approx(8.0, abs=1e-6)


@pytest.mark.parametrize("stride", [1, 2, 3])
def test_matches_naive_reference(sintel_sample, stride):
    _, depth, intr = sintel_sample(1)
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    rows = np.minimum(np.arange(h) * 4 // h, 3)
    cols = np.minimum(np.arange(w) * 6 // w, 5)
    labels = (rows[:, None] * 6 + cols[None, :] + 1).astype(np.int32)
    seg = Segmentation(labels, [SegmentInfo(i, f"obj{i}") for i in range(1, 25)])
    objs = by_segment(
        SceneBuilder(outlier="none", point_stride=stride).build_objects(seg, DepthFrame(depth, valid, intr))
    )
    pts = backproject(depth, intr, valid)[::stride, ::stride].reshape(-1, 3).astype(np.float64)
    ls = labels[::stride, ::stride].reshape(-1)
    ok = np.isfinite(pts).all(axis=1)
    assert len(objs) == 24
    for sid in range(1, 25):
        sel = pts[ok & (ls == sid)]
        bmin, bmax = aabb_from_points(sel)
        o = objs[sid]
        np.testing.assert_allclose(o.aabb_min, bmin, atol=1e-5)
        np.testing.assert_allclose(o.aabb_max, bmax, atol=1e-5)
        np.testing.assert_allclose(o.centroid, np.median(sel, axis=0), atol=1e-4)
        assert o.n_points == sel.shape[0]


def test_label_map_other_resolution():
    depth, seg, intr = planar_scene()
    half = Segmentation(seg.label_map[::2, ::2].copy(), seg.segments)
    full = by_segment(SceneBuilder().build_objects(seg, depth))
    low = by_segment(SceneBuilder().build_objects(half, depth))
    assert sorted(low) == sorted(full)
    for sid in full:
        np.testing.assert_allclose(low[sid].aabb_min, full[sid].aabb_min, atol=0.02)
        np.testing.assert_allclose(low[sid].aabb_max, full[sid].aabb_max, atol=0.02)
        np.testing.assert_allclose(low[sid].centroid, full[sid].centroid, atol=0.02)


def _radius_scene(rng):
    intr = Intrinsics.from_hfov(W, H, 60.0)
    depth = np.full((H, W), 2.0, dtype=np.float32)
    labels = np.zeros((H, W), dtype=np.int32)
    labels[100:200, 100:200] = 1
    # isolated far-away specks inside the segment
    specks = [(110, 110, 4.0), (150, 190, 0.8), (190, 120, 5.5)]
    for r, c, z in specks:
        depth[r, c] = z
    seg = Segmentation(labels, [SegmentInfo(1, "box")])
    return DepthFrame(depth, np.ones((H, W), dtype=bool), intr), seg


def test_radius_outlier_removes_isolated_points(rng):
    depth, seg = _radius_scene(rng)
    o = SceneBuilder(outlier="radius", point_stride=1).build_objects(seg, depth)[0]
    assert o.aabb_min[2] == pytest.approx(2.0) and o.aabb_max[2] == pytest.approx(2.0)
    assert o.n_points == 100 * 100 - 3
    o = SceneBuilder(outlier="none", point_stride=1).build_objects(seg, depth)[0]
    assert o.aabb_max[2] == pytest.approx(5.5)


def test_radius_outlier_matches_open3d(sintel_sample):
    o3d = pytest.importorskip("open3d")
    from vfoa.scene import _radius_inliers

    _, depth, intr = sintel_sample(1)
    pts = backproject(depth, intr)[::4, ::4].reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)].astype(np.float64)
    pts = pts[::7]
    radius, nb = 0.03, 11
    mine = np.flatnonzero(_radius_inliers(pts[:, 0], pts[:, 1], pts[:, 2], nb, radius, 4.0))
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    _, ind = pcd.remove_radius_outlier(nb_points=nb, radius=radius)
    theirs = np.sort(np.asarray(ind))
    # Points exactly at the radius may differ by floating-point rounding.
    diff = np.setxor1d(mine, theirs)
    assert diff.size <= max(2, int(0.001 * pts.shape[0]))
    assert 0 < mine.size < pts.shape[0]


def test_no_open3d_no_files_subprocess(tmp_path, repo_root, sintel_paths):
    code = f"""
import json, sys
import numpy as np
from vfoa.config import load_config
from vfoa.io.sintel import depth_read, intrinsics_from_cam
from vfoa.scene import SceneBuilder
from vfoa.types import DepthFrame, SegmentInfo, Segmentation
depth = depth_read({str(sintel_paths['depth_dir'] / 'frame_0001.dpt')!r})
h, w = depth.shape
intr = intrinsics_from_cam({str(sintel_paths['cam_dir'] / 'frame_0001.cam')!r}, w, h, baseline=0.1)
labels = np.ones((h, w), np.int32)
labels[:, w // 2:] += 1
labels[h // 2:, :] += 2
seg = Segmentation(labels, [SegmentInfo(i, f"q{{i}}") for i in range(1, 5)])
valid = np.isfinite(depth) & (depth > 0)
objs = SceneBuilder(load_config(profile="fast")).build_objects(seg, DepthFrame(depth, valid, intr))
assert "open3d" not in sys.modules, "open3d imported"
print(json.dumps([float(o.centroid[2]) for o in objs]))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr
    zs = json.loads(out.stdout.strip().splitlines()[-1])
    from vfoa.io.sintel import depth_read

    depth = depth_read(sintel_paths["depth_dir"] / "frame_0001.dpt")
    assert len(zs) == 4
    for z in zs:
        assert float(depth.min()) <= z <= float(depth.max())
    assert os.listdir(tmp_path) == []


# --------------------------------------------------------------------------- eyes


def plane_depth(z=2.0):
    intr = Intrinsics.from_hfov(W, H, 60.0)
    return DepthFrame(np.full((H, W), z, np.float32), np.ones((H, W), bool), intr), intr


def test_eyes_at_borders_and_corners():
    depth, intr = plane_depth(2.0)
    s = 40
    boxes = [
        [0, 0, s, s], [W - s, 0, W, s], [0, H - s, s, H], [W - s, H - s, W, H],
        [300, 0, 300 + s, s], [300, H - s, 300 + s, H], [0, 150, s, 150 + s], [W - s, 150, W, 150 + s],
        [-20, -30, 15, 10],  # partly outside the image
    ]
    persons = [person(i, head=b) for i, b in enumerate(boxes)]
    SceneBuilder().locate_eyes(persons, depth)
    for p in persons:
        assert p.eyes_3d is not None and np.isfinite(p.eyes_3d).all()
        assert p.eyes_3d[2] == pytest.approx(2.0)
        np.testing.assert_allclose(p.eyes_px, eye_pixel(p.detection.head_box))
        np.testing.assert_allclose(intr.project(p.eyes_3d), p.eyes_px, atol=1e-3)


def test_eyes_use_keypoints():
    depth, _ = plane_depth(1.5)
    kps = np.zeros((17, 3), np.float32)
    kps[1] = [210, 105, 0.9]
    kps[2] = [230, 107, 0.9]
    p = person(head=[190, 80, 250, 150], kps=kps)
    SceneBuilder().locate_eyes([p], depth)
    np.testing.assert_allclose(p.eyes_px, [220, 106])
    assert p.eyes_3d[2] == pytest.approx(1.5)


def test_eye_patch_grows_over_hole():
    depth, intr = plane_depth(2.0)
    d, v = depth.depth.copy(), depth.valid.copy()
    head = [200, 100, 260, 170]
    px = eye_pixel(np.asarray(head, np.float32))
    u, r = int(round(float(px[0]))), int(round(float(px[1])))
    v[r - 5:r + 6, u - 5:u + 6] = False
    d[r - 5:r + 6, u - 5:u + 6] = 0
    assert patch_depth(d, v, px[0], px[1], 3) is None
    p = person(head=head)
    SceneBuilder().locate_eyes([p], DepthFrame(d, v, intr))
    assert p.eyes_3d is not None and p.eyes_3d[2] == pytest.approx(2.0)


def test_eyes_none_when_head_has_no_depth():
    depth, intr = plane_depth(5.0)
    head = [200, 100, 260, 170]
    v = depth.valid.copy()
    v[100:170, 200:260] = False
    p = person(head=head)
    SceneBuilder().locate_eyes([p], DepthFrame(depth.depth, v, intr))
    assert p.eyes_px is not None
    assert p.eyes_3d is None


def test_eyes_none_without_head_box_and_stale_reset():
    depth, _ = plane_depth(2.0)
    p = person(head=None)
    p.eyes_px = np.array([1.0, 2.0], np.float32)
    p.eyes_3d = np.array([0.0, 0.0, 1.0])
    SceneBuilder().locate_eyes([p], depth)
    assert p.eyes_px is None and p.eyes_3d is None
    q = person(head=[10, 10, 50, 50])
    SceneBuilder().locate_eyes([q], None)
    assert q.eyes_px is not None and q.eyes_3d is None


def test_all_nan_depth_gives_none_not_nan():
    _, intr = plane_depth()
    nan = DepthFrame(np.full((H, W), np.nan, np.float32), np.ones((H, W), bool), intr)
    p = person(head=[100, 100, 150, 160])
    SceneBuilder().locate_eyes([p], nan)
    assert p.eyes_3d is None
    q = person(head=[1000, 1000, 1100, 1100])  # entirely outside
    SceneBuilder().locate_eyes([q], nan)
    assert q.eyes_3d is None


# --------------------------------------------------------------------------- own segment


def person_scene():
    depth, intr = plane_depth(3.0)
    d = depth.depth.copy()
    labels = np.zeros((H, W), np.int32)
    labels[50:300, 50:150] = 1  # person A
    labels[60:320, 400:520] = 2  # person B
    labels[200:300, 250:350] = 3  # table
    d[50:300, 50:150] = 2.0
    d[60:320, 400:520] = 2.5
    seg = Segmentation(
        labels, [SegmentInfo(1, "person"), SegmentInfo(2, "person"), SegmentInfo(3, "table")]
    )
    return DepthFrame(d, depth.valid, intr), seg


def test_own_segment_two_persons():
    depth, seg = person_scene()
    b = SceneBuilder()
    objs = by_segment(b.build_objects(seg, depth))
    pb = person(7, head=[430, 60, 490, 130])
    pa = person(3, head=[70, 50, 130, 120])
    b.locate_eyes([pb, pa], depth)
    b.match_own_segments([pb, pa], seg, list(objs.values()))
    assert pa.own_object_id == objs[1].object_id
    assert pb.own_object_id == objs[2].object_id


def test_no_person_segment():
    depth, seg = person_scene()
    seg2 = Segmentation(seg.label_map, [SegmentInfo(1, "statue"), SegmentInfo(2, "tree"), SegmentInfo(3, "table")])
    b = SceneBuilder()
    objs = b.build_objects(seg2, depth)
    p = person(head=[70, 50, 130, 120])
    p.own_object_id = 99
    b.locate_eyes([p], depth)
    b.match_own_segments([p], seg2, objs)
    assert p.own_object_id is None


def test_overlap_below_threshold():
    depth, seg = person_scene()
    b = SceneBuilder(self_aabb_margin_m=-10.0)  # disable the AABB fallback
    objs = b.build_objects(seg, depth)
    p = person(head=[140, 50, 240, 150])  # only 10% inside person A
    b.match_own_segments([p], seg, objs)
    assert p.own_object_id is None


def test_body_box_fallback():
    depth, seg = person_scene()
    b = SceneBuilder(self_aabb_margin_m=-10.0)
    objs = by_segment(b.build_objects(seg, depth))
    p = person(head=[150, 0, 200, 40], body=[60, 100, 140, 290])
    b.match_own_segments([p], seg, list(objs.values()))
    assert p.own_object_id == objs[1].object_id


def test_eyes_in_aabb_fallback():
    depth, seg = person_scene()
    b = SceneBuilder()
    objs = by_segment(b.build_objects(seg, depth))
    p = person(head=[70, 50, 130, 120])
    b.locate_eyes([p], depth)
    stale = Segmentation(np.zeros_like(seg.label_map), seg.segments)
    b.match_own_segments([p], stale, list(objs.values()))
    assert p.own_object_id == objs[1].object_id
    b.match_own_segments([p], None, list(objs.values()))
    assert p.own_object_id == objs[1].object_id


# --------------------------------------------------------------------------- registry


def test_ids_stable_under_permuted_segment_ids():
    reg = ObjectRegistry(match_dist_m=0.75, max_age=5)
    labels = ["chair", "chair", "tv", "floor"]
    cen = np.array([[0.0, 0, 2], [0.4, 0, 2], [1, 0, 3], [0, 1, 2]])
    ids1 = reg.assign(labels, cen, [True, True, True, False])
    perm = [3, 1, 2, 0]
    moved = cen[perm] + 0.02
    ids2 = reg.assign([labels[i] for i in perm], moved, [False, True, True, True])
    assert ids2 == [ids1[i] for i in perm]
    assert ids1 == [1, 2, 3, 4]


def test_scene_builder_ids_stable_under_permuted_segment_ids():
    depth, seg, _ = planar_scene()
    b = SceneBuilder()
    first = {o.label: o.object_id for o in b.build_objects(seg, depth)}
    remap = np.array([0, 3, 1, 2], np.int32)
    seg2 = Segmentation(
        remap[seg.label_map],
        [SegmentInfo(int(remap[s.id]), s.label, s.is_thing) for s in seg.segments],
    )
    second = {o.label: o.object_id for o in b.build_objects(seg2, depth)}
    assert first == second


def test_new_object_gets_new_id():
    reg = ObjectRegistry()
    a = reg.assign(["cup"], [[0, 0, 1.0]])
    b = reg.assign(["cup", "book"], [[0, 0, 1.0], [1, 0, 1.0]])
    assert b[0] == a[0] and b[1] not in a
    c = reg.assign(["cup", "cup"], [[0, 0, 1.0], [0.05, 0, 1.0]])
    assert c[0] == a[0] and len(set(c)) == 2


def test_reappear_within_and_after_max_age():
    reg = ObjectRegistry(match_dist_m=0.5, max_age=2)
    (cup,) = reg.assign(["cup"], [[0, 0, 1.0]])
    reg.assign([], np.zeros((0, 3)))
    reg.assign([], np.zeros((0, 3)))
    assert reg.assign(["cup"], [[0, 0, 1.0]]) == [cup]
    for _ in range(3):
        reg.assign([], np.zeros((0, 3)))
    assert cup not in reg.live_ids
    (new,) = reg.assign(["cup"], [[0, 0, 1.0]])
    assert new != cup and new > cup
    seen = set()
    for k in range(5):
        ids = reg.assign(["x"], [[10.0 * k, 0, 1]])
        assert not (set(ids) & seen) or k == 0
        seen.update(ids)
    assert cup not in seen


def test_same_segmentation_does_not_age():
    depth, seg, _ = planar_scene()
    b = SceneBuilder(object_max_age=0)
    ids = [o.object_id for o in b.build_objects(seg, depth)]
    for _ in range(3):
        assert [o.object_id for o in b.build_objects(seg, depth)] == ids
    # an empty refresh with a NEW segmentation ages (max_age 0 -> forgotten)
    b.build_objects(Segmentation(np.zeros_like(seg.label_map), []), depth)
    assert b.registry.live_ids == []
    assert all(o.object_id not in ids for o in b.build_objects(seg, depth))


def test_thing_moved_beyond_gate_vs_stuff():
    reg = ObjectRegistry(match_dist_m=0.5)
    ids = reg.assign(["cup", "wall"], [[0, 0, 1.0], [0, 0, 5.0]], [True, False])
    ids2 = reg.assign(["cup", "wall"], [[2, 0, 1.0], [3, 0, 5.0]], [True, False])
    assert ids2[1] == ids[1]
    assert ids2[0] != ids[0]


def test_reset_restarts_ids():
    reg = ObjectRegistry()
    reg.assign(["a", "b"], [[0, 0, 1.0], [1, 0, 1.0]])
    reg.reset()
    assert reg.next_id == 1 and reg.live_ids == []
    assert reg.assign(["c"], [[0, 0, 1.0]]) == [1]
    depth, seg, _ = planar_scene()
    b = SceneBuilder()
    b.build_objects(seg, depth)
    b.reset()
    assert [o.object_id for o in b.build_objects(seg, depth)] == [1, 2, 3]
