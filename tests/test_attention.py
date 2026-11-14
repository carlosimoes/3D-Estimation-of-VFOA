"""Tests of vfoa.attention (angular / rays scoring, fusion, self-exclusion, occlusion)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from vfoa.attention import (
    AttentionParams,
    best_targets,
    box_ratio,
    fusion_weights,
    score_frame,
)
from vfoa.geometry import cone_rays
from vfoa.types import (
    AttentionScore,
    DepthFrame,
    DirectionEstimate,
    Intrinsics,
    Person,
    PersonDetection,
    SceneObject,
    SegmentInfo,
    Segmentation,
)


# --------------------------------------------------------------------------- helpers


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _params(**kw) -> AttentionParams:
    return dataclasses.replace(AttentionParams(), **kw)


def _person(pid, eyes, gaze=None, head=None, own=None, head_box=None, body_box=None,
            gconf=1.0, hconf=1.0) -> Person:
    return Person(
        track_id=pid,
        detection=PersonDetection(head_box=head_box, body_box=body_box),
        gaze=None if gaze is None else DirectionEstimate(np.asarray(gaze, float), confidence=gconf),
        head=None if head is None else DirectionEstimate(np.asarray(head, float), confidence=hconf),
        eyes_3d=None if eyes is None else np.asarray(eyes, dtype=np.float64),
        own_object_id=own,
    )


def _point_obj(oid, p, label=None, is_thing=True) -> SceneObject:
    p = np.asarray(p, dtype=np.float64)
    return SceneObject(
        object_id=oid, segment_id=oid, label=label or f"obj{oid}", is_thing=is_thing,
        aabb_min=p.copy(), aabb_max=p.copy(), centroid=p.copy(),
        points=p[None].astype(np.float32), n_points=1,
    )


def _box_obj(oid, centre, half, n=5, label=None, is_thing=True) -> SceneObject:
    c = np.asarray(centre, dtype=np.float64)
    h = np.broadcast_to(np.asarray(half, dtype=np.float64), (3,))
    g = np.linspace(-1.0, 1.0, n)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    pts = c + grid * h
    return SceneObject(
        object_id=oid, segment_id=oid, label=label or f"obj{oid}", is_thing=is_thing,
        aabb_min=c - h, aabb_max=c + h, centroid=c.copy(),
        points=pts.astype(np.float32), n_points=pts.shape[0],
    )


def _paint(intr: Intrinsics, objs, bg=6.0):
    """Depth map + segmentation painting each box's projected rectangle with its front-face Z."""
    depth = np.full((intr.height, intr.width), bg, dtype=np.float32)
    labels = np.zeros((intr.height, intr.width), dtype=np.int32)
    segs = []
    for o in sorted(objs, key=lambda o: -float(o.aabb_min[2])):  # far to near
        lo, hi = o.aabb_min, o.aabb_max
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        uv = intr.project(corners)
        u0, v0 = np.floor(uv.min(axis=0)).astype(int)
        u1, v1 = np.ceil(uv.max(axis=0)).astype(int)
        u0, v0 = max(u0, 0), max(v0, 0)
        u1, v1 = min(u1, intr.width - 1), min(v1, intr.height - 1)
        depth[v0:v1 + 1, u0:u1 + 1] = float(lo[2])
        labels[v0:v1 + 1, u0:u1 + 1] = o.segment_id
        segs.append(SegmentInfo(id=o.segment_id, label=o.label, is_thing=o.is_thing))
    return (DepthFrame(depth=depth, valid=np.ones_like(depth, dtype=bool), intrinsics=intr),
            Segmentation(label_map=labels, segments=segs))


def _by_pair(scores):
    return {(s.person_id, s.object_id): s for s in scores}


def _point_at_angle(theta_rad, dist=3.0):
    return dist * np.array([math.sin(theta_rad), 0.0, math.cos(theta_rad)])


# --------------------------------------------------------------------------- params


def test_params_from_config_defaults_and_profiles(make_config):
    assert AttentionParams.from_config(make_config("fast")) == AttentionParams()
    p = AttentionParams.from_config(make_config("thesis"))
    assert (p.method, p.fusion, p.occlusion) == ("rays", "agreement", "none")
    assert p.half_angles_gaze_deg == (11.3, 21.8) and p.rays_n == 20
    with pytest.raises(ValueError):
        AttentionParams.from_config({"attention": {"method": "bogus"}})
    with pytest.raises(ValueError):
        AttentionParams(sigma_gaze_deg=0.0)
    with pytest.raises(ValueError):
        AttentionParams(rays_n=0)


# --------------------------------------------------------------------------- issues


def test_issue3_weights_change_total_and_do_not_cancel():
    gaze = _unit(_point_at_angle(math.radians(5.0)))
    head = _unit(_point_at_angle(math.radians(-15.0)))
    obj = _point_obj(1, [0.0, 0.0, 3.0])
    totals = []
    for wg, wh in ((0.6, 0.4), (0.2, 0.8)):
        prm = _params(fusion="fixed", w_gaze=wg, w_head=wh, occlusion="none")
        (s,) = score_frame([_person(0, [0, 0, 0], gaze, head)], [obj], prm)
        assert abs(s.score_gaze - s.score_head) > 0.05
        assert s.score_total == pytest.approx(wg * s.score_gaze + wh * s.score_head, abs=1e-12)
        totals.append(s.score_total)
    assert abs(totals[0] - totals[1]) > 1e-3


def test_issue4_far_from_origin_scores_finite_in_unit_range():
    shift = np.array([5000.0, 5000.0, 5000.0])
    objs = [_box_obj(1, [0.2, 0, 3], 0.4), _box_obj(2, [-1, 0.3, 4], [0.5, 0.2, 0.3])]
    for method in ("angular", "rays"):
        prm = _params(method=method, occlusion="aabb")
        ref = score_frame([_person(0, [0, 0, 0], _unit([0.05, 0, 1]), _unit([-0.1, 0, 1]))],
                          objs, prm)
        moved = [dataclasses.replace(o, aabb_min=o.aabb_min + shift, aabb_max=o.aabb_max + shift,
                                     centroid=o.centroid + shift,
                                     points=(o.points.astype(np.float64) + shift))
                 for o in objs]
        out = score_frame([_person(0, shift, _unit([0.05, 0, 1]), _unit([-0.1, 0, 1]))],
                          moved, prm)
        assert len(out) == 2
        for a, b in zip(ref, out):
            for v in (b.score_gaze, b.score_head, b.score_total):
                assert math.isfinite(v) and 0.0 <= v <= 1.0
            assert b.score_total == pytest.approx(a.score_total, abs=1e-6)
    assert box_ratio(np.array([0, 0, 0.0]) + 5000, np.array([2, 4, 4.0]) + 5000)[0] == \
        pytest.approx(1 / 3)


def test_issue5_object_behind_viewer_scores_zero():
    obj = _box_obj(1, [0, 0, -3.0], 0.5)
    p = _person(0, [0, 0, 0], [0, 0, 1.0], [0, 0, 1.0])
    (r,) = score_frame([p], [obj], _params(method="rays", occlusion="none"))
    assert r.score_gaze == 0.0 and r.score_head == 0.0 and r.score_total == 0.0
    (a,) = score_frame([p], [obj], _params(method="angular", occlusion="none"))
    assert a.score_gaze < 1e-6 and a.score_head < 1e-6


def test_issue6_own_object_never_scored_other_person_is_target():
    p0 = _person(0, [0, 0, 2], [1, 0, 0.0], own=10)
    p1 = _person(1, [1, 0, 2], [-1, 0, 0.0], own=11)
    objs = [_box_obj(10, [0, 0.3, 2], [0.2, 0.5, 0.15], label="person"),
            _box_obj(11, [1, 0.3, 2], [0.2, 0.5, 0.15], label="person"),
            _box_obj(1, [0, 0, 5], 0.3)]
    for method in ("angular", "rays"):
        out = _by_pair(score_frame([p0, p1], objs, _params(method=method, occlusion="none")))
        assert (0, 10) not in out and (1, 11) not in out
        assert (0, 11) in out and (1, 10) in out and (0, 1) in out
        assert out[(0, 11)].score_total > 0.3


def test_issue14_missing_cues_and_eyes_never_raise():
    obj = _point_obj(1, [0.1, 0, 3])
    prm = _params(occlusion="none")
    (s,) = score_frame([_person(0, [0, 0, 0], None, [0, 0, 1.0])], [obj], prm)
    assert s.score_gaze is None and s.score_total == pytest.approx(s.score_head, abs=1e-15)
    (s,) = score_frame([_person(0, [0, 0, 0], [0, 0, 1.0], None)], [obj], prm)
    assert s.score_head is None and s.score_total == pytest.approx(s.score_gaze, abs=1e-15)
    assert score_frame([_person(0, None, [0, 0, 1.0], [0, 0, 1.0])], [obj], prm) == []
    assert score_frame([_person(0, [np.nan, 0, 0], [0, 0, 1.0], None)], [obj], prm) == []
    assert score_frame([_person(0, [0, 0, 0], None, None)], [obj], prm) == []
    for method in ("angular", "rays"):
        assert score_frame([_person(0, [0, 0, 0], None, None)], [obj],
                           _params(method=method)) == []


def test_zero_direction_is_a_missing_cue():
    obj = _point_obj(1, [0.0, 0, 3])
    (s,) = score_frame([_person(0, [0, 0, 0], [0, 0, 0.0], [0, 0, 1.0])], [obj], _params())
    assert s.score_gaze is None and s.score_head == pytest.approx(1.0)


def test_issue19_angular_closed_form_and_smoothness():
    prm = _params(occlusion="none", min_target_distance_m=0.0)
    sigma = math.radians(prm.sigma_gaze_deg)

    def score(theta_deg):
        obj = _point_obj(1, _point_at_angle(math.radians(theta_deg)))
        (s,) = score_frame([_person(0, [0, 0, 0], [0, 0, 1.0], None)], [obj], prm)
        return s.score_gaze

    for deg in (0.0, 3.0, 12.5, 40.0, 120.0):
        x = _point_obj(1, _point_at_angle(math.radians(deg))).points[0].astype(np.float64)
        t = math.atan2(abs(x[0]), x[2])  # exact angle of the stored (float32) point
        assert score(deg) == pytest.approx(math.exp(-t * t / (2 * sigma * sigma)), abs=1e-9)
    assert score(0.0) == 1.0
    sweep = [score(d) for d in np.linspace(0, 90, 46)]
    assert all(b <= a + 1e-15 for a, b in zip(sweep, sweep[1:]))
    for deg in (2.0, 10.0, 25.0):
        assert abs(score(deg) - score(deg + 0.01)) < 1e-3


def test_issue19_distance_falloff_only_beyond_full():
    prm = _params(occlusion="none", distance_full_m=2.0, distance_decay_m=2.0)
    p = _person(0, [0, 0, 0], [0, 0, 1.0], None)
    (near,) = score_frame([p], [_point_obj(1, [0, 0, 1.5])], prm)
    (edge,) = score_frame([p], [_point_obj(1, [0, 0, 2.0])], prm)
    (far,) = score_frame([p], [_point_obj(1, [0, 0, 4.0])], prm)
    assert near.score_gaze == pytest.approx(1.0) and edge.score_gaze == pytest.approx(1.0)
    assert far.score_gaze == pytest.approx(math.exp(-1.0), abs=1e-12)


def test_angular_robust_minimum_uses_5th_percentile():
    rng = np.random.default_rng(3)
    pts = np.c_[rng.uniform(-1, 1, (57, 2)), rng.uniform(2, 4, 57)].astype(np.float32)
    obj = SceneObject(1, 1, "o", True, pts.min(0).astype(float), pts.max(0).astype(float),
                      np.median(pts, axis=0).astype(float), pts, 57)
    d = _unit([0.3, 0.1, 1])
    (s,) = score_frame([_person(0, [0, 0, 0], d, None)], [obj], _params(occlusion="none"))
    v = pts.astype(np.float64)
    ang = np.arctan2(np.linalg.norm(np.cross(d, v), axis=1), v @ d)
    theta = np.percentile(ang, 5)
    assert s.score_gaze == pytest.approx(math.exp(-theta**2 / (2 * math.radians(10) ** 2)),
                                         abs=1e-9)


def test_rays_closed_form_single_ring():
    L, a, c, A = 5.0, 1.0, 0.5, 5.0
    prm = _params(method="rays", occlusion="none", half_angles_gaze_deg=(A,), rays_n=20)
    obj = SceneObject(1, 1, "box", True, np.array([-a, -a, L - c]), np.array([a, a, L + c]),
                      np.array([0, 0, L]), np.zeros((0, 3), np.float32), 0)
    (s,) = score_frame([_person(0, [0, 0, 0], [0, 0, 1.0], None)], [obj], prm)
    rho = (L - c) * math.tan(math.radians(A))
    hd = math.sqrt(2 * a * a + c * c)
    expect = (1.5 - math.sqrt(rho**2 + c**2) / hd) / (1.5 - min(a, c) / hd)
    assert s.score_gaze == pytest.approx(expect, abs=1e-9)
    assert s.score_total == pytest.approx(expect, abs=1e-12)


# --------------------------------------------------------------------------- occlusion


def _occlusion_scene(with_near=True):
    intr = Intrinsics.from_hfov(640, 360, 90.0)
    eyes = np.array([-1.0, 0.0, 1.5])
    own = _box_obj(10, eyes + [0, 0.3, 0], [0.2, 0.5, 0.15], label="person")
    near = _box_obj(1, [0, 0, 2.5], 0.3)
    far = _box_obj(2, [1, 0, 3.5], 0.3)
    objs = [own, far] + ([near] if with_near else [])
    d = _unit(far.centroid - eyes)
    person = _person(0, eyes, d, d, own=10)
    depth, seg = _paint(intr, objs)
    return person, objs, depth, seg


@pytest.mark.parametrize("mode", ["depth", "aabb", "rays"])
def test_issue20_far_object_occluded_by_near_one(mode):
    if mode == "rays":
        prm = _params(method="rays", occlusion="aabb", half_angles_gaze_deg=(3.0, 6.0),
                      half_angles_head_deg=(3.0, 6.0))
    else:
        prm = _params(occlusion=mode)
    person, objs, depth, seg = _occlusion_scene(True)
    out = _by_pair(score_frame([person], objs, prm, depth=depth, segmentation=seg))
    assert out[(0, 2)].occluded and out[(0, 2)].score_total == 0.0
    if mode == "rays":
        assert out[(0, 2)].score_gaze == 0.0 and out[(0, 2)].score_head == 0.0
        assert out[(0, 1)].score_total > 0.5
    person, objs, depth, seg = _occlusion_scene(False)
    out = _by_pair(score_frame([person], objs, prm, depth=depth, segmentation=seg))
    assert not out[(0, 2)].occluded and out[(0, 2)].score_total > 0.5


@pytest.mark.parametrize("method", ["angular", "rays"])
def test_issue20_occlusion_none_leaves_scores_unchanged(method):
    person, objs, depth, seg = _occlusion_scene(True)
    base = dict(method=method, half_angles_gaze_deg=(3.0, 6.0), half_angles_head_deg=(3.0, 6.0))
    a = score_frame([person], objs, _params(occlusion="none", **base), depth=depth,
                    segmentation=seg)
    assert not any(s.occluded for s in a)
    if method == "angular":
        b = score_frame([person], objs, _params(occlusion="depth", **base), depth=depth,
                        segmentation=seg)
        assert [(s.score_gaze, s.score_head) for s in a] == [(s.score_gaze, s.score_head) for s in b]
        # without a depth map, depth occlusion is a no-op (rays mode always uses nearest hits)
        c = score_frame([person], objs, _params(occlusion="depth", **base))
        d = score_frame([person], objs, _params(occlusion="none", **base))
        assert [(s.score_total, s.occluded) for s in c] == [(s.score_total, s.occluded) for s in d]


def test_issue20_not_occluded_by_own_head():
    intr = Intrinsics.from_hfov(640, 360, 90.0)
    eyes = np.array([0.0, 0.0, 2.0])
    own = _box_obj(10, [0, 0.3, 2.0], [0.25, 0.5, 0.15], label="person")
    target = _box_obj(1, [0.6, 0, 4.0], 0.3)
    depth, seg = _paint(intr, [own, target])
    d = _unit(target.centroid - eyes)
    lo, hi = own.aabb_min, own.aabb_max
    corners = intr.project(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                     for z in (lo[2], hi[2])]))
    head_box = np.r_[corners.min(0), corners.max(0)]
    prm = _params(occlusion="depth")
    for kw in ({"own": 10}, {"head_box": head_box}, {"own": 10, "head_box": head_box}):
        p = _person(0, eyes, d, d, **kw)
        out = _by_pair(score_frame([p], [own, target], prm, depth=depth, segmentation=seg))
        assert not out[(0, 1)].occluded and out[(0, 1)].score_total > 0.9
    control = _person(0, eyes, d, d)
    out = _by_pair(score_frame([control], [own, target], prm, depth=depth, segmentation=seg))
    assert out[(0, 1)].occluded and out[(0, 1)].score_total == 0.0


# --------------------------------------------------------------------------- fusion


def test_issue21_fusion_modes():
    g = _unit([0, 0, 1.0])
    h30 = _unit([math.sin(math.radians(30)), 0, math.cos(math.radians(30))])
    h60 = _unit([math.sin(math.radians(60)), 0, math.cos(math.radians(60))])
    G = DirectionEstimate(g)
    agr = _params(fusion="agreement")
    assert fusion_weights(G, DirectionEstimate(h30), agr) == pytest.approx((1.9 / 3, 1.1 / 3))
    assert fusion_weights(G, DirectionEstimate(h60), agr) == pytest.approx((1.55 / 3, 1.45 / 3))
    conf = _params(fusion="confidence")
    assert fusion_weights(DirectionEstimate(g, confidence=0.0), DirectionEstimate(h30), conf) \
        == (0.0, 1.0)
    assert fusion_weights(DirectionEstimate(g, confidence=1.0), DirectionEstimate(h30, 0.5),
                          conf) == pytest.approx((0.6 / 0.8, 0.2 / 0.8))
    fixed = _params(fusion="fixed", w_gaze=0.7, w_head=0.3)
    assert fusion_weights(G, DirectionEstimate(h60), fixed) == pytest.approx((0.7, 0.3))
    assert fusion_weights(None, DirectionEstimate(h60), fixed) == (0.0, 1.0)
    assert fusion_weights(G, None, fixed) == (1.0, 0.0)
    assert fusion_weights(None, None, fixed) == (0.0, 0.0)
    both0 = fusion_weights(DirectionEstimate(g, 0.0), DirectionEstimate(h30, 0.0), conf)
    assert both0 == pytest.approx((0.6, 0.4))


def test_agreement_fusion_total_in_score_frame():
    obj = _point_obj(1, [0, 0, 3.0])
    g = _unit(_point_at_angle(math.radians(5)))
    h = _unit(_point_at_angle(math.radians(-25)))
    (s,) = score_frame([_person(0, [0, 0, 0], g, h)], [obj], _params(fusion="agreement",
                                                                     occlusion="none"))
    assert s.score_total == pytest.approx((1.9 * s.score_gaze + 1.1 * s.score_head) / 3, abs=1e-12)


# --------------------------------------------------------------------------- naive reference


def _slab(o, d, lo, hi):
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if o[i] < lo[i] or o[i] > hi[i]:
                return False, 0.0
            continue
        t1, t2 = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
        tmin, tmax = max(tmin, min(t1, t2)), min(tmax, max(t1, t2))
    t_enter = max(tmin, 0.0)
    return (tmax >= t_enter), t_enter


def _naive(persons, objs, prm):
    out = []
    sig = {0: math.radians(prm.sigma_gaze_deg), 1: math.radians(prm.sigma_head_deg)}
    rings = {0: prm.half_angles_gaze_deg, 1: prm.half_angles_head_deg}
    for p in persons:
        if p.eyes_3d is None:
            continue
        e = np.asarray(p.eyes_3d, float)
        cues = {0: p.gaze, 1: p.head}
        dirs = {c: (None if est is None else _unit(est.direction)) for c, est in cues.items()}
        if dirs[0] is None and dirs[1] is None:
            continue
        wg, wh = fusion_weights(p.gaze, p.head, prm)
        for b, o in enumerate(objs):
            if o.object_id == p.own_object_id:
                continue
            sc, raw, cred = {}, False, False
            for c, d in dirs.items():
                if d is None:
                    sc[c] = None
                    continue
                if prm.method == "angular":
                    pts = o.points.astype(np.float64)
                    if pts.shape[0] == 0:
                        pts = o.centroid[None]
                    angs = []
                    for x in pts:
                        v = x - e
                        if np.linalg.norm(v) < prm.min_target_distance_m:
                            continue
                        angs.append(math.atan2(np.linalg.norm(np.cross(d, v)), float(np.dot(d, v))))
                    if not angs:
                        sc[c] = 0.0
                        continue
                    th = np.percentile(np.array(angs), 5)
                    s = math.exp(-th * th / (2 * sig[c] ** 2))
                    if prm.distance_full_m is not None:
                        dist = np.linalg.norm(o.centroid - e)
                        if dist > prm.distance_full_m:
                            s *= math.exp(-(dist - prm.distance_full_m) / prm.distance_decay_m)
                    sc[c] = s
                else:
                    rays = np.concatenate([cone_rays(d, a, prm.rays_n) for a in rings[c]])
                    total = 0.0
                    for r in rays:
                        hits = [_slab(e, r, q.aabb_min, q.aabb_max) for q in objs]
                        hit, te = hits[b]
                        if not hit:
                            continue
                        raw = True
                        if prm.occlusion != "none":
                            tb = min([t for k, (h, t) in enumerate(hits)
                                      if h and objs[k].is_thing and t > 0
                                      and objs[k].object_id != p.own_object_id] + [np.inf])
                            if te > tb + 1e-9:
                                continue
                        cred = True
                        entry = e + te * r
                        cen = (o.aabb_min + o.aabb_max) / 2
                        hd = np.linalg.norm(o.aabb_max - o.aabb_min) / 2
                        total += 1.5 - np.linalg.norm(entry - cen) / hd
                    ext = o.aabb_max - o.aabb_min
                    ratio = (ext.min() / 2) / (np.linalg.norm(ext) / 2)
                    sc[c] = min(max(total / (len(rays) * (1.5 - ratio)), 0.0), 1.0)
            occ = prm.method == "rays" and prm.occlusion != "none" and raw and not cred
            tot = 0.0 if occ else (wg * (sc[0] or 0.0) + wh * (sc[1] or 0.0))
            out.append((p.track_id, o.object_id, sc[0], sc[1], tot, occ))
    return out


def _random_scene(seed=0):
    rng = np.random.default_rng(seed)
    objs = []
    counts = [0, 1, 2, 3, 7, 20, 40, 64, 100, 5, 13, 30]
    for j, k in enumerate(counts):
        c = np.r_[rng.uniform(-2, 2), rng.uniform(-1, 1), rng.uniform(3, 6)]
        h = rng.uniform(0.1, 0.6, 3)
        pts = (c + rng.uniform(-1, 1, (k, 3)) * h).astype(np.float32)
        objs.append(SceneObject(100 + j, 100 + j, f"o{j}", bool(j % 3), c - h, c + h, c, pts, k))
    persons = []
    for i in range(5):
        e = np.r_[rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), rng.uniform(0.5, 1.5)]
        tgt = objs[int(rng.integers(len(objs)))].centroid
        g = _unit(tgt - e + rng.normal(0, 0.2, 3))
        h = _unit(tgt - e + rng.normal(0, 0.4, 3))
        persons.append(_person(i, e, None if i == 1 else g, None if i == 2 else h,
                               own=100 + i if i == 3 else None,
                               gconf=float(rng.uniform(0.2, 1)), hconf=float(rng.uniform(0.2, 1))))
    return persons, objs


@pytest.mark.parametrize("kw", [
    dict(method="angular", occlusion="none", fusion="fixed"),
    dict(method="angular", occlusion="none", fusion="confidence", distance_full_m=4.0),
    dict(method="angular", occlusion="none", fusion="agreement", target="centroid"),
    dict(method="rays", occlusion="none", fusion="agreement"),
    dict(method="rays", occlusion="aabb", fusion="confidence"),
])
def test_vectorised_matches_naive_reference(kw):
    persons, objs = _random_scene(0)
    prm = _params(**kw)
    if prm.target == "centroid":
        objs = [dataclasses.replace(o, points=np.zeros((0, 3), np.float32)) for o in objs]
    got = score_frame(persons, objs, prm)
    ref = _naive(persons, objs, prm)
    assert len(got) == len(ref) == 5 * 12 - 1
    nonzero = 0
    for s, (pid, oid, g, h, tot, occ) in zip(got, ref):
        assert (s.person_id, s.object_id) == (pid, oid)
        for a, b in ((s.score_gaze, g), (s.score_head, h)):
            assert (a is None) == (b is None)
            if a is not None:
                assert a == pytest.approx(b, abs=1e-9)
        assert s.score_total == pytest.approx(tot, abs=1e-9)
        assert s.occluded == occ
        nonzero += s.score_total > 1e-3
    assert nonzero >= 3


# --------------------------------------------------------------------------- misc


def test_best_targets():
    sc = [AttentionScore(0, 1, "a", 0.5, None, 0.05), AttentionScore(0, 2, "b", 0.5, None, 0.08),
          AttentionScore(1, 1, "a", 0.5, None, 0.4), AttentionScore(1, 2, "b", 0.5, None, 0.7),
          AttentionScore(1, 3, "c", 0.5, None, 0.7)]
    assert best_targets(sc, 0.1) == {0: None, 1: 2}
    assert best_targets(sc, 0.05) == {0: 2, 1: 2}
    assert best_targets([], 0.1) == {}


@pytest.mark.parametrize("method", ["angular", "rays"])
@pytest.mark.parametrize("occlusion", ["none", "depth", "aabb"])
def test_empty_inputs_and_degenerate_box_emit_no_warnings(method, occlusion):
    prm = _params(method=method, occlusion=occlusion)
    intr = Intrinsics.from_hfov(64, 36, 90.0)
    depth = DepthFrame(np.zeros((36, 64), np.float32), np.zeros((36, 64), bool), intr)
    p = _person(0, [0, 0, 0], [0, 0, 1.0], [0.1, 0, 1.0])
    with np.errstate(all="raise"):
        assert score_frame([], [_point_obj(1, [0, 0, 2])], prm) == []
        assert score_frame([p], [], prm) == []
        out = score_frame([p], [_point_obj(1, [0, 0, 2.0]), _point_obj(2, [0, 0, 0.0])], prm,
                          depth=depth)
        bad = dataclasses.replace(_point_obj(3, [0, 0, 1.0]), centroid=np.array([np.nan] * 3))
        assert len(score_frame([p], [bad], prm)) == 0
    for s in out:
        assert 0.0 <= s.score_total <= 1.0 and math.isfinite(s.score_gaze)
