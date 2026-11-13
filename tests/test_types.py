from __future__ import annotations

import dataclasses
import warnings

import cv2
import numpy as np
import pytest

from vfoa.types import (
    COCO_KEYPOINTS,
    AttentionScore,
    DepthFrame,
    DirectionEstimate,
    Frame,
    FrameResult,
    Intrinsics,
    Person,
    PersonDetection,
    SceneObject,
    SegmentInfo,
    Segmentation,
)


def _pixel_grid(h: int, w: int) -> np.ndarray:
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return np.stack([u, v], axis=-1)


def test_q_reprojection_matches_ground_truth(sintel_sample):
    _, depth, intr = sintel_sample(1)
    assert intr.baseline == pytest.approx(0.1)
    disp = (intr.fx * intr.baseline / depth).astype(np.float32)
    pts = cv2.reprojectImageTo3D(disp, intr.Q())
    assert pts.shape == depth.shape + (3,)
    assert np.all(pts[..., 2] > 0)
    assert np.max(np.abs(pts[..., 2] - depth)) < 1e-4
    uv = intr.project(pts)
    assert np.max(np.abs(uv - _pixel_grid(*depth.shape))) < 1e-3


def test_q_anisotropic_focal():
    intr = Intrinsics(fx=500.0, fy=400.0, cx=63.5, cy=47.5, width=128, height=96, baseline=0.2)
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 5.0, size=(96, 128)).astype(np.float32)
    disp = (intr.fx * intr.baseline / depth).astype(np.float32)
    pts = cv2.reprojectImageTo3D(disp, intr.Q())
    assert np.max(np.abs(pts[..., 2] - depth)) < 1e-4
    assert np.max(np.abs(intr.project(pts) - _pixel_grid(96, 128))) < 1e-3


def test_q_requires_baseline():
    with pytest.raises(ValueError):
        Intrinsics(100, 100, 50, 50, 100, 100).Q()
    with pytest.raises(ValueError):
        Intrinsics(100, 100, 50, 50, 100, 100, baseline=0.0).Q()


def test_K_property():
    intr = Intrinsics(fx=10, fy=20, cx=3, cy=4, width=8, height=6)
    np.testing.assert_array_equal(intr.K, [[10, 0, 3], [0, 20, 4], [0, 0, 1]])
    assert intr.K.dtype == np.float64
    assert isinstance(intr.fx, float) and isinstance(intr.width, int)


def test_scaled_matches_from_hfov():
    full = Intrinsics.from_hfov(640, 360, 60.0, baseline=0.1)
    half = full.scaled(0.5)
    ref = Intrinsics.from_hfov(320, 180, 60.0, baseline=0.1)
    for name in ("fx", "fy", "cx", "cy"):
        assert getattr(half, name) == pytest.approx(getattr(ref, name))
    assert (half.width, half.height, half.baseline) == (320, 180, 0.1)

    aniso = full.scaled(0.5, 0.25)
    assert aniso.fx == pytest.approx(full.fx * 0.5)
    assert aniso.fy == pytest.approx(full.fy * 0.25)
    assert (aniso.width, aniso.height) == (320, 90)

    point = np.array([0.3, -0.2, 2.0])
    u, v = full.project(point)
    u2, v2 = aniso.project(point)
    assert u2 == pytest.approx((u + 0.5) * 0.5 - 0.5)
    assert v2 == pytest.approx((v + 0.5) * 0.25 - 0.5)


def test_from_hfov_values():
    intr = Intrinsics.from_hfov(640, 360, 90.0)
    assert intr.fx == pytest.approx(320.0)
    assert intr.fy == intr.fx
    assert (intr.cx, intr.cy) == (319.5, 179.5)
    assert intr.baseline is None


def test_from_K_roundtrip():
    intr = Intrinsics(fx=896, fy=896, cx=511.5, cy=217.5, width=1024, height=436, baseline=0.1)
    again = Intrinsics.from_K(intr.K, intr.width, intr.height, baseline=intr.baseline)
    assert again == intr
    with pytest.raises(ValueError):
        Intrinsics.from_K(np.eye(4), 10, 10)


def test_project_behind_camera_nan_no_warning():
    intr = Intrinsics.from_hfov(100, 100)
    pts = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, -2.0], [np.nan, 0.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        uv = intr.project(pts)
    assert uv.shape == (4, 2)
    np.testing.assert_allclose(uv[0], [intr.cx, intr.cy])
    assert np.isnan(uv[1:]).all()
    assert intr.project(np.zeros((2, 5, 3)) + [0, 0, 1]).shape == (2, 5, 2)


EXPECTED_FIELDS = {
    Intrinsics: ["fx", "fy", "cx", "cy", "width", "height", "baseline"],
    Frame: ["index", "t", "image", "right", "intrinsics"],
    DepthFrame: ["depth", "valid", "intrinsics", "disparity"],
    PersonDetection: ["head_box", "body_box", "keypoints", "score"],
    DirectionEstimate: ["direction", "confidence", "euler_deg", "frame_index"],
    Person: ["track_id", "detection", "gaze", "head", "eyes_px", "eyes_3d", "own_object_id"],
    SegmentInfo: ["id", "label", "is_thing", "score"],
    Segmentation: ["label_map", "segments", "frame_index"],
    SceneObject: [
        "object_id", "segment_id", "label", "is_thing", "aabb_min", "aabb_max",
        "centroid", "points", "n_points",
    ],
    AttentionScore: [
        "person_id", "object_id", "label", "score_gaze", "score_head", "score_total", "occluded",
    ],
    FrameResult: ["index", "t", "persons", "objects", "scores", "targets", "timings_ms", "ran"],
}


@pytest.mark.parametrize("cls", list(EXPECTED_FIELDS), ids=lambda c: c.__name__)
def test_field_names_contract(cls):
    assert [f.name for f in dataclasses.fields(cls)] == EXPECTED_FIELDS[cls]


def test_coco_keypoints():
    assert len(COCO_KEYPOINTS) == 17
    assert COCO_KEYPOINTS[:3] == ("nose", "l_eye", "r_eye")
    assert COCO_KEYPOINTS[-1] == "r_ankle"


def test_detection_coercion_and_shape_errors():
    det = PersonDetection(head_box=[1, 2, 3, 4], keypoints=np.zeros((17, 3)), score=np.float32(0.5))
    assert det.head_box.dtype == np.float32 and det.head_box.shape == (4,)
    assert det.body_box is None
    assert det.keypoints.dtype == np.float32
    assert isinstance(det.score, float)
    assert PersonDetection(head_box=None).head_box is None
    with pytest.raises(ValueError):
        PersonDetection(head_box=[1, 2, 3])
    with pytest.raises(ValueError):
        PersonDetection(head_box=None, keypoints=np.zeros((18, 3)))

    est = DirectionEstimate(direction=[0, 0, -2])
    assert est.direction.dtype == np.float64
    np.testing.assert_array_equal(est.direction, [0, 0, -2])  # not renormalised
    with pytest.raises(ValueError):
        DirectionEstimate(direction=[1, 0])


def test_default_containers_independent():
    a, b = FrameResult(index=0, t=0.0), FrameResult(index=1, t=0.1)
    a.persons.append("x")
    a.targets[0] = 1
    a.timings_ms["depth"] = 1.0
    assert b.persons == [] and b.targets == {} and b.timings_ms == {} and b.ran == {}


def test_array_dataclass_eq_safe():
    img = np.zeros((2, 2, 3), np.uint8)
    f1, f2 = Frame(0, 0.0, img), Frame(0, 0.0, img)
    assert f1 == f1 and f1 != f2  # identity comparison, never raises
    seg = Segmentation(np.zeros((2, 2), np.int32), [SegmentInfo(1, "cup")])
    assert seg == seg
    assert SegmentInfo(1, "cup") == SegmentInfo(1, "cup")
    assert AttentionScore(0, 1, "cup", None, 0.5, 0.5) == AttentionScore(0, 1, "cup", None, 0.5, 0.5)
