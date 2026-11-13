"""Tests of the synthetic world and the model-free synthetic backends (#30)."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

import vfoa.backends.synthetic as synth
from vfoa.backends import create_backend
from vfoa.backends.base import (
    DepthBackend,
    GazeBackend,
    HeadPoseBackend,
    PersonBackend,
    SegmentationBackend,
)
from vfoa.backends.synthetic import SyntheticWorld
from vfoa.config import load_config
from vfoa.types import DepthFrame, Frame, Person, PersonDetection, Segmentation

N_FRAMES = 120


# ------------------------------------------------------------------ local helpers
def _world(**kw) -> SyntheticWorld:
    return SyntheticWorld(**kw)


def _rule_eye(box) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    return np.array([(x1 + x2) / 2.0, 0.65 * y1 + 0.35 * y2])


def _unproject_pixels(world: SyntheticWorld, us, vs, z) -> np.ndarray:
    intr = world.intrinsics
    us = np.asarray(us, np.float64)
    vs = np.asarray(vs, np.float64)
    z = np.asarray(z, np.float64)
    return np.stack([(us - intr.cx) * z / intr.fx, (vs - intr.cy) * z / intr.fy, z], -1)


def _angles_deg(d, v) -> np.ndarray:
    cross = np.linalg.norm(np.cross(d[None], v), axis=-1)
    return np.degrees(np.arctan2(cross, v @ d))


def _seg_aabb(p0, p1, bmin, bmax) -> bool:
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for a in range(3):
        if abs(d[a]) < 1e-12:
            if p0[a] < bmin[a] or p0[a] > bmax[a]:
                return False
        else:
            ta, tb = (bmin[a] - p0[a]) / d[a], (bmax[a] - p0[a]) / d[a]
            t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
            if t0 > t1:
                return False
    return True


def _forward(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([-math.sin(y), -math.cos(y) * math.sin(p), -math.cos(y) * math.cos(p)])


def _persons(world: SyntheticWorld, i: int, ids) -> list[Person]:
    dets = world.detections(i)
    return [Person(track_id=tid, detection=det) for tid, det in zip(ids, dets)]


# ------------------------------------------------------------------ determinism / memo
def test_two_instances_identical():
    a, b = _world(), _world()
    assert a.key == b.key
    for i in (0, 17, 45, 119):
        fa, fb = a.frame(i), b.frame(i)
        assert np.array_equal(fa.image, fb.image)
        assert np.array_equal(a.depth(i).depth, b.depth(i).depth)
        assert np.array_equal(a.segmentation(i).label_map, b.segmentation(i).label_map)
        for da, db in zip(a.detections(i), b.detections(i)):
            assert np.array_equal(da.head_box, db.head_box)
            assert np.array_equal(da.body_box, db.body_box)
            assert np.array_equal(da.keypoints, db.keypoints)
        assert a.truth(i) == b.truth(i)


def test_from_config_memoised(make_config):
    cfg = make_config()
    w1 = SyntheticWorld.from_config(cfg)
    w2 = SyntheticWorld.from_config(make_config(overrides=["attention.sigma_gaze_deg=3", "depth.stride=4"]))
    assert w1 is w2
    w3 = SyntheticWorld.from_config(make_config(overrides=["synthetic.seed=5"]))
    assert w3 is not w1
    assert not np.array_equal(w1.frame(0).image, w3.frame(0).image)
    # cost_ms does not change the world
    w4 = SyntheticWorld.from_config(make_config(overrides={"synthetic": {"cost_ms": {"depth": 1.0}}}))
    assert w4 is w1


def test_returns_copies():
    w = _world()
    d = w.depth(3)
    d.depth[:] = -1
    assert w.depth(3).depth.min() > 0
    s = w.segmentation(3)
    s.label_map[:] = 99
    assert w.segmentation(3).label_map.max() < 99


@pytest.mark.parametrize("size", [(640, 360), (320, 180)])
def test_shapes_and_dtypes(size):
    W, H = size
    w = _world(width=W, height=H)
    f = w.frame(5)
    assert isinstance(f, Frame)
    assert f.image.shape == (H, W, 3) and f.image.dtype == np.uint8
    assert f.index == 5 and f.t == pytest.approx(5 / 30.0)
    assert f.intrinsics == w.intrinsics
    d = w.depth(5)
    assert isinstance(d, DepthFrame)
    assert d.depth.shape == (H, W) and d.depth.dtype == np.float32
    assert d.valid.shape == (H, W) and d.valid.dtype == bool and d.valid.all()
    assert d.depth.min() > 0
    assert d.intrinsics.width == W and d.intrinsics.height == H and d.intrinsics.baseline == 0.1
    s = w.segmentation(5)
    assert isinstance(s, Segmentation)
    assert s.label_map.shape == (H, W) and s.label_map.dtype == np.int32
    assert s.frame_index == 5
    dets = w.detections(5)
    assert len(dets) == 2
    for det in dets:
        assert isinstance(det, PersonDetection)
        assert det.head_box.dtype == np.float32 and det.keypoints.shape == (17, 3)


# ------------------------------------------------------------------ self-consistency
def test_eye_rule_depth_and_keypoints():
    w = _world()
    for i in range(N_FRAMES):
        dep = w.depth(i).depth
        for k, det in enumerate(w.detections(i)):
            e = _rule_eye(det.head_box)
            assert np.allclose(e, w.eye_px(i, k))
            E = w.eye_point(i, k)
            u, v = int(round(e[0])), int(round(e[1]))
            assert abs(dep[v, u] - E[2]) < 1e-4
            win = dep[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
            assert abs(np.median(win) - E[2]) < 1e-4
            kp = det.keypoints
            assert kp[1, 2] > 0 and kp[2, 2] > 0
            assert np.allclose((kp[1, :2] + kp[2, :2]) / 2, e, atol=1e-4)
            # eye point is the unprojection of the eye pixel
            assert np.allclose(_unproject_pixels(w, e[0], e[1], E[2]), E)


def test_gaze_ray_hits_truth_centre():
    w = _world()
    for i in range(N_FRAMES):
        tr = w.truth(i)
        for k in range(w.n_persons):
            E = w.eye_point(i, k)
            C = w.center(i, tr[k])
            for g in (w.gaze(i, k), w.head(i, k)):
                assert np.linalg.norm(g) == pytest.approx(1.0, abs=1e-12)
                assert np.linalg.norm(np.cross(C - E, g)) < 0.05
                assert np.dot(C - E, g) > 0


def test_object_centres_and_head_boxes_on_own_segments():
    w = _world()
    intr = w.intrinsics
    for i in range(0, N_FRAMES, 3):
        seg = w.segmentation(i)
        lab = seg.label_map
        assert [s.id for s in seg.segments] == [int(x) for x in np.unique(lab) if x > 0]
        for j in range(w.n_objects):
            sid = w.object_segment_id(j)
            C = w.center(i, sid)
            u, v = intr.project(C)
            assert lab[int(round(v)), int(round(u))] == sid
        for k, det in enumerate(w.detections(i)):
            x1, y1, x2, y2 = det.head_box.astype(int)
            assert (lab[y1:y2, x1:x2] == w.person_segment_id(k)).all()
            assert w.label(w.person_segment_id(k)) == "person"
        labels = [s.label for s in seg.segments if s.label != "person"]
        assert len(labels) == len(set(labels))


def _padded_boxes(w, i):
    lab = w.segmentation(i).label_map
    dep = w.depth(i).depth
    boxes = {}
    for s in [int(x) for x in np.unique(lab) if x > 0]:
        vs, us = np.nonzero(lab == s)
        P = _unproject_pixels(w, us, vs, dep[vs, us])
        boxes[s] = (P.min(0) - [0, 0, 0.05], P.max(0) + [0, 0, 0.05])
    return boxes


def test_no_box_between_eyes_and_target():
    w = _world()
    for i in range(0, N_FRAMES, 2):
        boxes = _padded_boxes(w, i)
        tr = w.truth(i)
        for k in range(w.n_persons):
            own = w.person_segment_id(k)
            E, C = w.eye_point(i, k), w.center(i, tr[k])
            for s, (bmin, bmax) in boxes.items():
                if s in (own, tr[k]):
                    continue
                assert not _seg_aabb(E, C, bmin, bmax), (i, k, s)


def test_world_aabb_matches_mask():
    w = _world()
    boxes = _padded_boxes(w, 10)
    for s, (bmin, bmax) in boxes.items():
        amin, amax = w.aabb(10, s)
        assert np.allclose(amin, bmin + [0, 0, 0.05], atol=1e-5)
        assert np.allclose(amax, bmax - [0, 0, 0.05], atol=1e-5)


def test_occluder_blocks_documented_pair():
    w = _world(occluder=True)
    occ = w.occlusion
    assert occ is not None and occ["person"] == 0 and occ["target"] == 1
    panel = occ["occluder"]
    assert panel == w.n_objects + w.n_persons + 1 and w.label(panel) == "panel"
    intr = w.intrinsics
    n_blocked_frames = 0
    for i in range(N_FRAMES):
        tr = w.truth(i)
        if w.phase(i) != 0:
            assert not w.occluder_active(i)
            assert tr == w.aim(i)
            assert panel not in [s.id for s in w.segmentation(i).segments]
            continue
        assert w.occluder_active(i)
        assert tr[0] == panel and w.aim(i)[0] == 1
        n_blocked_frames += 1
        boxes = _padded_boxes(w, i)
        E, C = w.eye_point(i, 0), w.center(i, 1)
        assert _seg_aabb(E, C, *boxes[panel])
        # depth test: samples along eye -> target are nearer than the line
        fr = np.linspace(0.1, 0.9, 24)
        S = E[None] + fr[:, None] * (C - E)[None]
        uv = np.round(intr.project(S)).astype(int)
        dep = w.depth(i).depth
        lab = w.segmentation(i).label_map
        own = w.person_segment_id(0)
        dd, ll = dep[uv[:, 1], uv[:, 0]], lab[uv[:, 1], uv[:, 0]]
        blocked = (dd < S[:, 2] - 0.15) & (ll != own) & (ll != 1)
        assert blocked.sum() >= 6
        # the gaze still points at the aimed object, and the truth panel is unoccluded
        g = w.gaze(i, 0)
        assert np.linalg.norm(np.cross(C - E, g)) < 0.05
        # person 1 is not affected
        assert tr[1] == w.aim(i)[1]
    assert n_blocked_frames == w.phase_len


def test_script_changes_and_person_target():
    w = _world()
    truths = [w.truth(i) for i in range(N_FRAMES)]
    changes = sum(1 for a, b in zip(truths, truths[1:]) if a != b)
    assert changes >= 3
    person_ids = {w.person_segment_id(k) for k in range(w.n_persons)}
    assert any(t in person_ids for tr in truths for t in tr.values())
    # a person never targets itself
    for tr in truths:
        for k, t in tr.items():
            assert t != w.person_segment_id(k)


@pytest.mark.parametrize("kw", [{}, {"n_persons": 3, "n_objects": 5}, {"occluder": True}])
def test_independent_angular_oracle(kw):
    """Angles to each segment's unprojected pixels: truth min < 2 deg, every other
    (non-aimed) segment's robust minimum (5th percentile, the attention metric) > 7 deg."""
    w = _world(**kw)
    for i in range(0, N_FRAMES, 4):
        seg = w.segmentation(i)
        lab = seg.label_map
        dep = w.depth(i).depth
        pts = {}
        for s in [g.id for g in seg.segments]:
            vs, us = np.nonzero(lab == s)
            pts[s] = _unproject_pixels(w, us, vs, dep[vs, us])
        tr = w.truth(i)
        for k in range(w.n_persons):
            own = w.person_segment_id(k)
            E = w.eye_point(i, k)
            g = w.gaze(i, k)
            for s, P in pts.items():
                if s == own:
                    continue
                v = P - E
                v = v[np.linalg.norm(v, axis=1) > 0.15]
                a = _angles_deg(g, v)
                if s == tr[k]:
                    assert a.min() < 2.0, (i, k, s, a.min())
                elif s != w.aim(i)[k]:
                    assert np.percentile(a, 5) > 7.0, (i, k, s)


def test_noise_deterministic_and_bounded():
    clean = _world()
    a = _world(noise_deg=5.0)
    b = _world(noise_deg=5.0)
    devs = []
    for i in range(N_FRAMES):
        for k in range(2):
            ga, gb = a.gaze(i, k), b.gaze(i, k)
            assert np.array_equal(ga, gb)
            assert np.linalg.norm(ga) == pytest.approx(1.0)
            devs.append(_angles_deg(clean.gaze(i, k), ga[None])[0])
            # call-order independence
            assert np.array_equal(a.head(i, k), b.head(i, k))
    assert not np.array_equal(a.gaze(3, 0), a.head(3, 0))
    assert 2.0 < float(np.mean(devs)) < 6.5


def test_zero_persons_and_objects():
    w = _world(n_persons=0, n_objects=3)
    assert w.detections(0) == [] and w.truth(0) == {}
    w2 = _world(n_persons=2, n_objects=0)
    assert all(t is None or t > 0 for t in w2.truth(0).values())
    assert np.allclose(w2.gaze(0, 1), [0, 0, -1])


def test_invalid_counts():
    with pytest.raises(ValueError):
        _world(n_persons=4)
    with pytest.raises(ValueError):
        _world(n_objects=7)


# ------------------------------------------------------------------ backends
def test_registry_builds_all_backends():
    cfg = load_config(profile="synthetic")
    expected = {
        "depth": DepthBackend, "persons": PersonBackend, "headpose": HeadPoseBackend,
        "gaze": GazeBackend, "segmentation": SegmentationBackend,
    }
    world = SyntheticWorld.from_config(cfg)
    frame = world.frame(7)
    for kind, base in expected.items():
        b = create_backend(kind, cfg)
        assert isinstance(b, base) and b.name == "synthetic" and b.kind == kind
        assert b.calls == 0 and b.loaded == 0
        b.load()
        b.load()
        assert b.loaded == 1 and b.is_loaded and b.world is world
        if kind == "depth":
            out = b.compute(frame)
            assert np.array_equal(out.depth, world.depth(7).depth)
        elif kind == "persons":
            out = b.detect(frame)
            assert len(out) == 2 and np.array_equal(out[0].head_box, world.detections(7)[0].head_box)
        elif kind == "segmentation":
            out = b.segment(frame)
            assert np.array_equal(out.label_map, world.segmentation(7).label_map)
        else:
            persons = _persons(world, 7, [0, 1])
            if kind == "gaze":
                b.observe(frame, persons)
                assert b.observed == 1
            out = b.estimate(frame, persons)
            assert set(out) == {0, 1}
        assert b.calls == 1


@pytest.mark.parametrize("kind", ["headpose", "gaze"])
def test_direction_backends_map_track_ids(kind):
    cfg = load_config(profile="synthetic")
    world = SyntheticWorld.from_config(cfg)
    b = create_backend(kind, cfg)
    b.load()
    i = 33
    frame = world.frame(i)
    dets = world.detections(i)
    persons = [Person(track_id=7, detection=dets[1]), Person(track_id=3, detection=dets[0])]
    persons.append(Person(track_id=11, detection=PersonDetection(head_box=None)))
    far = PersonDetection(head_box=np.array([0, 300, 10, 310], np.float32))
    persons.append(Person(track_id=12, detection=far))
    out = b.estimate(frame, persons)
    assert set(out) == {7, 3}
    ref = world.head if kind == "headpose" else world.gaze
    assert np.allclose(out[7].direction, ref(i, 1))
    assert np.allclose(out[3].direction, ref(i, 0))
    for est in out.values():
        assert np.linalg.norm(est.direction) == pytest.approx(1.0)
        assert est.frame_index == i and est.confidence == 1.0
        if kind == "headpose":
            yaw, pitch, roll = est.euler_deg
            assert roll == 0.0
            assert np.allclose(_forward(yaw, pitch), est.direction, atol=1e-9)
        else:
            assert est.euler_deg is None
    assert b.estimate(frame, []) == {}


def test_cost_ms_sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(synth, "_sleep", lambda s: slept.append(s))
    cfg = load_config(profile="synthetic", overrides={
        "synthetic": {"cost_ms": {"depth": 5, "headpose": 12.5}}})
    world = SyntheticWorld.from_config(cfg)
    depth = create_backend("depth", cfg)
    head = create_backend("headpose", cfg)
    seg = create_backend("segmentation", cfg)
    frame = world.frame(0)
    depth.compute(frame)
    depth.compute(frame)
    seg.segment(frame)
    head.estimate(frame, _persons(world, 0, [0, 1]))
    assert slept == [0.005, 0.005, 0.0125]
    assert depth.slept_ms == pytest.approx(10.0) and seg.slept_ms == 0.0
    assert depth.calls == 2 and seg.calls == 1 and head.calls == 1


# ------------------------------------------------------------------ source
def test_source_iteration():
    w = _world(n_frames=10)
    src = w.source()
    assert src.fps == 30.0 and (src.width, src.height) == (640, 360)
    assert src.n_frames == 10 and src.fingerprint == f"synthetic:{w.key}"
    idx = [f.index for f in src]
    assert idx == list(range(10))
    assert [f.index for f in src] == idx  # restartable
    with w.source(start=3, max_frames=4) as s2:
        frames = list(s2)
        assert s2.n_frames == 4
    assert [f.index for f in frames] == [3, 4, 5, 6]
    assert frames[0].t == pytest.approx(3 / 30.0)
    assert np.array_equal(frames[0].image, w.frame(3).image)
    assert list(w.source(start=8, max_frames=10)) and w.source(start=8, max_frames=10).n_frames == 2
    s3 = w.source()
    s3.close()
    assert list(s3) == []


def test_import_is_light(repo_root):
    code = (
        "import sys, json; import vfoa.backends.synthetic, vfoa.cache; "
        "print(json.dumps([m for m in ('torch','open3d','matplotlib','ultralytics','PIL','onnxruntime') "
        "if m in sys.modules]))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(repo_root))
    assert json.loads(res.stdout.strip().splitlines()[-1]) == []
