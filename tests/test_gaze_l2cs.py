"""Tests of the fast L2CS-style gaze backend (vfoa.backends.fast.l2cs_gaze).

CPU only and no network: the ONNX session is faked (it returns logits chosen
by the test) and weight downloads are refused (autouse fixture in conftest).
Only ``test_real_mobileone_s0_sintel`` (marked ``network``) downloads the real
MobileOne-S0 model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from vfoa import geometry
from vfoa.backends import create_backend, create_backend_with_fallback
from vfoa.backends.base import BackendUnavailable
from vfoa.backends.fast import l2cs_gaze as L
from vfoa.backends.fast.l2cs_gaze import _download as REAL_DOWNLOAD
from vfoa.backends.fast.l2cs_gaze import (
    BinnedGaze360Backend,
    angles_to_direction,
    crop_square,
    decode_bins,
    face_crop_box,
    face_visible,
    head_ray_rotation,
    preprocess,
    resolve_l2cs_weights,
)
from vfoa.geometry import unproject
from vfoa.types import Frame, Intrinsics, Person, PersonDetection

S = 448


# ---------------------------------------------------------------------- helpers
def onehot(bin_index: int, n: int = 1) -> np.ndarray:
    z = np.full((n, 90), -1000.0, np.float32)
    z[:, bin_index] = 0.0
    return z


def bin_of(deg: float) -> int:
    return int(round((deg + 180.0) / 4.0))


class FakeSession:
    """Minimal onnxruntime.InferenceSession stand-in.

    ``outputs_fn(x) -> (logits_a, logits_b)`` with ``x`` the (n, 3, S, S) input.
    ``batch_dim`` is 1 (static, like the released ONNX files) or a string
    (dynamic). Every fed array is recorded in ``self.fed``.
    """

    def __init__(self, outputs_fn, batch_dim=1, size=S, names=("yaw", "pitch"), n_outputs=2):
        self.outputs_fn = outputs_fn
        self.batch_dim = batch_dim
        self.size = size
        self.names = names
        self.n_outputs = n_outputs
        self.fed: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input", shape=[self.batch_dim, 3, self.size, self.size])]

    def get_outputs(self):
        return [SimpleNamespace(name=n) for n in self.names[: self.n_outputs]]

    def run(self, output_names, feed):
        x = feed["input"]
        assert isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
        if self.batch_dim == 1:
            assert x.shape[0] == 1, "static batch-1 model fed a batch"
        self.fed.append(x.copy())
        a, b = self.outputs_fn(x)
        return [np.asarray(a, np.float32), np.asarray(b, np.float32)]


def const_outputs(deg_a: float = 0.0, deg_b: float = 0.0):
    return lambda x: (onehot(bin_of(deg_a), len(x)), onehot(bin_of(deg_b), len(x)))


def content_outputs(x):
    """Logits that depend on the crop content (to compare batching modes)."""
    n = len(x)
    a = np.full((n, 90), -1000.0)
    b = np.full((n, 90), -1000.0)
    for i in range(n):
        m = float(x[i].mean())
        a[i, int(abs(m) * 37) % 90] = 0.0
        b[i, int(abs(m) * 53 + 7) % 90] = 0.0
        a[i, (int(abs(m) * 37) + 1) % 90] = -1.0
    return a, b


def make_cfg(make_config, tmp_path, **l2cs):
    return make_config("fast", {
        "gaze": {"l2cs": l2cs},
        "models": {"weights_dir": str(tmp_path / "weights"), "allow_download": False},
    })


def make_person(tid, cx, cy, eye_dist=20.0, nose=0.9, l_eye=0.9, r_eye=0.9, ears=0.9, keypoints=True):
    kp = np.zeros((17, 3), np.float32)
    kp[0] = (cx, cy + 0.3 * eye_dist, nose)
    kp[1] = (cx + eye_dist / 2, cy, l_eye)
    kp[2] = (cx - eye_dist / 2, cy, r_eye)
    kp[3] = (cx + eye_dist, cy + 0.1 * eye_dist, ears)
    kp[4] = (cx - eye_dist, cy + 0.1 * eye_dist, ears)
    half = 1.5 * eye_dist
    head = np.array([cx - half, cy - half, cx + half, cy + half], np.float32)
    det = PersonDetection(head_box=head, keypoints=kp if keypoints else None)
    return Person(tid, det)


def make_frame(w=320, h=240, index=0, intrinsics=None, image=None):
    img = image if image is not None else np.full((h, w, 3), 128, np.uint8)
    return Frame(index=index, t=index / 30.0, image=img, intrinsics=intrinsics)


def backend(make_config, tmp_path, session, **l2cs):
    return BinnedGaze360Backend(make_cfg(make_config, tmp_path, **l2cs), session=session)


# ---------------------------------------------------------------------- decode and axes
def test_decode_known_bins():
    deg, conf = decode_bins(np.concatenate([onehot(45), onehot(0), onehot(89)]))
    np.testing.assert_allclose(deg, [0.0, -180.0, 176.0], atol=1e-9)
    np.testing.assert_allclose(conf, 1.0, atol=1e-9)
    mix = np.full((1, 90), -1000.0)
    mix[0, 50] = mix[0, 51] = 0.0
    deg, conf = decode_bins(mix)
    assert deg[0] == pytest.approx(22.0)
    assert 0.8 < conf[0] < 1.0
    # softmax is applied (logits, not probabilities): shifting all logits changes nothing
    z = np.random.default_rng(0).normal(size=(4, 90))
    np.testing.assert_allclose(decode_bins(z)[0], decode_bins(z + 100.0)[0])


def test_axes_at_camera():
    np.testing.assert_allclose(angles_to_direction(0.0, 0.0)[0], [0.0, 0.0, -1.0], atol=1e-12)


def test_positive_a_is_image_left():
    d = angles_to_direction(math.radians(30), 0.0)[0]
    assert d[0] < 0 and abs(d[1]) < 1e-12 and d[2] < 0


def test_positive_b_is_image_up():
    d = angles_to_direction(0.0, math.radians(30))[0]
    assert d[1] < 0 and abs(d[0]) < 1e-12 and d[2] < 0


def test_angles_to_direction_matches_geometry(rng):
    a = rng.uniform(-np.pi, np.pi, 200)
    b = rng.uniform(-np.pi / 2, np.pi / 2, 200)
    d = angles_to_direction(a, b)
    assert d.shape == (200, 3)
    np.testing.assert_allclose(d, geometry.gaze360_to_camera(a, b), atol=1e-12)
    closed = np.stack([-np.cos(b) * np.sin(a), -np.sin(b), -np.cos(b) * np.cos(a)], axis=1)
    np.testing.assert_allclose(d, closed, atol=1e-12)


def test_unit_norm(rng):
    d = angles_to_direction(rng.uniform(-np.pi, np.pi, 500), rng.uniform(-np.pi / 2, np.pi / 2, 500))
    assert d.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)


def test_output_order_positional(make_config, tmp_path):
    sess = FakeSession(const_outputs(20.0, 0.0), names=("pitch", "yaw"))
    be = backend(make_config, tmp_path, sess, perspective_correction=False)
    est = be.estimate(make_frame(), [make_person(1, 160, 120)])[1]
    assert est.euler_deg[0] == pytest.approx(20.0) and est.euler_deg[1] == pytest.approx(0.0)
    assert est.direction[0] < 0 and abs(est.direction[1]) < 1e-9  # first output = a (horizontal)
    assert est.frame_index == 0 and est.direction.dtype == np.float64


def test_backend_signs_end_to_end(make_config, tmp_path):
    for (a, b), check in [((0, 0), lambda d: np.allclose(d, [0, 0, -1], atol=1e-9)),
                          ((40, 0), lambda d: d[0] < 0), ((-40, 0), lambda d: d[0] > 0),
                          ((0, 40), lambda d: d[1] < 0), ((0, -40), lambda d: d[1] > 0)]:
        be = backend(make_config, tmp_path, FakeSession(const_outputs(a, b)), perspective_correction=False)
        d = be.estimate(make_frame(), [make_person(1, 160, 120)])[1].direction
        assert check(d), (a, b, d)
        assert np.linalg.norm(d) == pytest.approx(1.0)


# ---------------------------------------------------------------------- gate
def test_gate(make_config, tmp_path):
    sess = FakeSession(const_outputs())
    be = backend(make_config, tmp_path, sess)
    persons = [
        make_person(1, 60, 60, keypoints=False),            # no keypoints
        make_person(2, 160, 60, nose=0.2),                  # nose not confident
        make_person(3, 260, 60, l_eye=0.1, r_eye=0.3),      # both eyes not confident
        make_person(4, 60, 180, l_eye=0.1, r_eye=0.9),      # nose + one eye: OK
        make_person(5, 160, 180),                           # all visible: OK
    ]
    out = be.estimate(make_frame(), persons)
    assert sorted(out) == [4, 5]
    assert len(sess.fed) == 2
    assert be.stats["gated_out"] == 3 and be.stats["estimated"] == 2
    assert be.estimate(make_frame(), []) == {}


def test_face_visible_and_box():
    kp = make_person(1, 100, 100, nose=0.8, l_eye=0.6, r_eye=0.2).detection.keypoints
    assert face_visible(kp, 0.5) == pytest.approx(0.7)
    assert face_visible(None, 0.5) == 0.0
    box = face_crop_box(kp, None, 0.5, margin=1.0)
    assert box is not None and box[2] - box[0] == pytest.approx(box[3] - box[1])
    full = make_person(1, 100, 100, eye_dist=20)
    box = face_crop_box(full.detection.keypoints, full.detection.head_box, 0.5, 1.1)
    side = box[2] - box[0]
    assert side == pytest.approx(1.1 * 46.0)  # 2.3 * eye distance > keypoint extent 40
    assert 0.5 * 60 <= side <= 60  # within the head box limits
    # fewer than 2 confident points -> the head box made square
    box = face_crop_box(np.zeros((17, 3), np.float32), np.array([10, 20, 30, 60]), 0.5)
    np.testing.assert_allclose(box, [0, 20, 40, 60])
    assert face_crop_box(None, None, 0.5) is None


# ---------------------------------------------------------------------- crop and preprocessing
def test_crop_and_preprocess(make_config, tmp_path):
    img = np.zeros((240, 320, 3), np.uint8)
    img[:, :160] = (255, 0, 0)  # BGR blue
    img[:, 160:] = (0, 0, 255)  # BGR red
    sess = FakeSession(const_outputs())
    be = backend(make_config, tmp_path, sess)
    be.estimate(make_frame(image=img), [make_person(1, 60, 120), make_person(2, 260, 120)])
    blue, red = sess.fed[0][0], sess.fed[1][0]
    assert blue.shape == red.shape == (3, S, S)
    r_on, r_off = (1 - 0.485) / 0.229, (0 - 0.485) / 0.229
    b_on, b_off = (1 - 0.406) / 0.225, (0 - 0.406) / 0.225
    g_off = (0 - 0.456) / 0.224
    np.testing.assert_allclose(red[0], r_on, atol=1e-5)    # channel 0 is R
    np.testing.assert_allclose(red[1], g_off, atol=1e-5)
    np.testing.assert_allclose(red[2], b_off, atol=1e-5)
    np.testing.assert_allclose(blue[0], r_off, atol=1e-5)
    np.testing.assert_allclose(blue[2], b_on, atol=1e-5)

    x = preprocess([np.zeros((S, S, 3), np.uint8)] * 2)
    assert x.shape == (2, 3, S, S) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]

    # dynamic-size model with input_size=224
    sess224 = FakeSession(const_outputs(), size="H")
    be224 = backend(make_config, tmp_path, sess224, input_size=224)
    be224.estimate(make_frame(image=img), [make_person(1, 60, 120)])
    assert sess224.fed[0].shape == (1, 3, 224, 224)

    # a fixed-size model overrides input_size
    sess_fixed = FakeSession(const_outputs())
    be_fixed = backend(make_config, tmp_path, sess_fixed, input_size=224)
    be_fixed.estimate(make_frame(image=img), [make_person(1, 60, 120)])
    assert sess_fixed.fed[0].shape == (1, 3, S, S)


def test_crop_outside_frame():
    img = np.zeros((100, 100, 3), np.uint8)
    img[:50] = (0, 255, 0)
    out = crop_square(img, np.array([-30.0, -30.0, 30.0, 30.0]), 64)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    assert (out == (0, 255, 0)).all()  # replicated border, no black padding
    out = crop_square(img, np.array([500.0, 500.0, 520.0, 520.0]), 32)  # entirely outside
    assert out.shape == (32, 32, 3)
    # a mirrored crop would swap left/right: check orientation is kept
    img2 = np.zeros((100, 100, 3), np.uint8)
    img2[:, 50:] = 255
    out = crop_square(img2, np.array([0.0, 0.0, 100.0, 100.0]), 100)
    assert out[:, :40].max() == 0 and out[:, 60:].min() == 255


def test_crop_partly_outside_through_backend(make_config, tmp_path):
    sess = FakeSession(const_outputs())
    be = backend(make_config, tmp_path, sess)
    out = be.estimate(make_frame(), [make_person(1, 2, 2, eye_dist=30), make_person(2, 318, 238, eye_dist=30)])
    assert sorted(out) == [1, 2]


# ---------------------------------------------------------------------- batching
def _three_persons():
    return [make_person(1, 60, 60), make_person(2, 160, 120, eye_dist=30), make_person(3, 260, 180, eye_dist=12)]


def _textured_frame():
    img = (np.random.default_rng(1).integers(0, 255, (240, 320, 3))).astype(np.uint8)
    return make_frame(image=img)


def test_batching_static(make_config, tmp_path):
    sess = FakeSession(content_outputs, batch_dim=1)
    out = backend(make_config, tmp_path, sess).estimate(_textured_frame(), _three_persons())
    assert len(sess.fed) == 3 and all(x.shape == (1, 3, S, S) for x in sess.fed)
    assert sorted(out) == [1, 2, 3]


def test_batching_dynamic(make_config, tmp_path):
    frame, persons = _textured_frame(), _three_persons()
    s_static = FakeSession(content_outputs, batch_dim=1)
    s_dyn = FakeSession(content_outputs, batch_dim="N")
    out_s = backend(make_config, tmp_path, s_static).estimate(frame, persons)
    out_d = backend(make_config, tmp_path, s_dyn).estimate(frame, persons)
    assert len(s_dyn.fed) == 1 and s_dyn.fed[0].shape == (3, 3, S, S)
    np.testing.assert_array_equal(np.concatenate(s_static.fed), s_dyn.fed[0])
    for tid in (1, 2, 3):
        np.testing.assert_allclose(out_s[tid].direction, out_d[tid].direction)
        assert out_s[tid].confidence == pytest.approx(out_d[tid].confidence)
    dirs = {tuple(np.round(out_d[t].direction, 6)) for t in (1, 2, 3)}
    assert len(dirs) > 1  # the per-person crops really differ


def test_dynamic_batch_respects_batch_size(make_config, tmp_path):
    cfg = make_config("fast", {"gaze": {"batch_size": 2},
                               "models": {"weights_dir": str(tmp_path), "allow_download": False}})
    sess = FakeSession(content_outputs, batch_dim="N")
    BinnedGaze360Backend(cfg, session=sess).estimate(_textured_frame(), _three_persons())
    assert [len(x) for x in sess.fed] == [2, 1]


# ---------------------------------------------------------------------- confidence
def test_confidence_range(make_config, tmp_path, rng):
    uniform = lambda x: (np.zeros((len(x), 90)), np.zeros((len(x), 90)))  # noqa: E731
    out = backend(make_config, tmp_path, FakeSession(uniform)).estimate(make_frame(), [make_person(1, 160, 120)])
    assert out[1].confidence == pytest.approx(0.0, abs=1e-9)
    out = backend(make_config, tmp_path, FakeSession(const_outputs(10, -10))).estimate(
        make_frame(), [make_person(1, 160, 120, nose=0.9, l_eye=0.9, r_eye=0.4)])
    assert 0.5 < out[1].confidence <= 1.0
    assert out[1].confidence == pytest.approx(0.9)
    rand = lambda x: (rng.normal(0, 5, (len(x), 90)), rng.normal(0, 5, (len(x), 90)))  # noqa: E731
    be = backend(make_config, tmp_path, FakeSession(rand))
    for _ in range(20):
        for e in be.estimate(make_frame(), _three_persons()).values():
            assert 0.0 <= e.confidence <= 1.0


# ---------------------------------------------------------------------- perspective correction
def test_head_ray_rotation():
    K = Intrinsics.from_hfov(320, 240, 90.0).K
    np.testing.assert_allclose(head_ray_rotation(K[0, 2], K[1, 2], K), np.eye(3), atol=1e-12)
    R = head_ray_rotation(300.0, 30.0, K)
    r = np.linalg.solve(K, [300.0, 30.0, 1.0])
    r /= np.linalg.norm(r)
    np.testing.assert_allclose(R @ [0, 0, 1], r, atol=1e-12)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # looking at the camera from an off-axis position: d = -ray
    np.testing.assert_allclose(R @ [0, 0, -1], -r, atol=1e-12)


def test_perspective_correction(make_config, tmp_path):
    intr = Intrinsics.from_hfov(320, 240, 90.0)
    frame = make_frame(intrinsics=intr)
    off = make_person(1, 290, 40)
    centred = make_person(2, intr.cx, intr.cy - 0.3 * 20 / 2)  # crop centre close to the principal point

    plain = backend(make_config, tmp_path, FakeSession(const_outputs()),
                    perspective_correction=False).estimate(frame, [off, centred])
    corr = backend(make_config, tmp_path, FakeSession(const_outputs())).estimate(frame, [off, centred])
    np.testing.assert_allclose(plain[1].direction, [0, 0, -1], atol=1e-9)
    assert np.degrees(np.arccos(np.clip(corr[1].direction @ plain[1].direction, -1, 1))) > 20
    assert corr[1].direction[0] < 0 and corr[1].direction[1] > 0  # points back towards the camera
    assert np.linalg.norm(corr[1].direction) == pytest.approx(1.0)
    angle_c = np.degrees(np.arccos(np.clip(corr[2].direction @ plain[2].direction, -1, 1)))
    assert angle_c < 2.0
    # without frame intrinsics the configured camera is used (no error)
    out = backend(make_config, tmp_path, FakeSession(const_outputs()),
                  perspective_correction=True).estimate(make_frame(), [off])
    assert 1 in out


def test_default_correction_uses_eye_ray(make_config, tmp_path):
    intr = Intrinsics.from_hfov(320, 240, 90.0)
    frame = make_frame(intrinsics=intr)
    person = make_person(1, 290, 40)
    box = face_crop_box(person.detection.keypoints, person.detection.head_box, 0.5, 1.1)
    centre = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])

    def unit(v):
        v = np.asarray(v, dtype=np.float64)
        return v / np.linalg.norm(v)

    d = backend(make_config, tmp_path, FakeSession(const_outputs())).estimate(frame, [person])[1].direction
    np.testing.assert_allclose(d, -unit(unproject(centre, 1.0, intr)), atol=1e-9)
    person.eyes_px = np.array([280.0, 50.0], np.float32)
    d = backend(make_config, tmp_path, FakeSession(const_outputs())).estimate(frame, [person])[1].direction
    np.testing.assert_allclose(d, -unit(unproject(np.array([280.0, 50.0]), 1.0, intr)), atol=1e-9)
    d = backend(make_config, tmp_path, FakeSession(const_outputs()),
                perspective_correction=False).estimate(frame, [person])[1].direction
    np.testing.assert_allclose(d, [0, 0, -1], atol=1e-9)


def _certainty(z):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp(z - z.max())
    p /= p.sum()
    ent = -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0))
    return 1.0 - ent / math.log(90)


def _logits_with_certainty(c):
    """``s * onehot(45)`` with ``1 - H(softmax) / log(90) == c`` (bisection on ``s``)."""
    lo, hi = 0.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        z = np.zeros(90)
        z[45] = mid
        if _certainty(z) < c:
            lo = mid
        else:
            hi = mid
    z = np.zeros(90)
    z[45] = 0.5 * (lo + hi)
    return z


def test_confidence_ref(make_config, tmp_path):
    vis = face_visible(make_person(1, 160, 120).detection.keypoints, 0.5)
    assert vis == pytest.approx(0.9)

    def conf_for(c, **l2cs):
        z = _logits_with_certainty(c)
        assert _certainty(z) == pytest.approx(c, abs=1e-9)
        fn = lambda x: (np.tile(z, (len(x), 1)), np.tile(z, (len(x), 1)))  # noqa: E731
        be = backend(make_config, tmp_path, FakeSession(fn), **l2cs)
        return be.estimate(make_frame(), [make_person(1, 160, 120)])[1].confidence

    assert conf_for(0.25) == pytest.approx(vis, abs=1e-4)
    assert conf_for(0.125) == pytest.approx(0.5 * vis, abs=1e-4)
    assert conf_for(0.5) == pytest.approx(vis, abs=1e-4)
    assert conf_for(0.25, confidence_ref=None) == pytest.approx(0.25 * vis, abs=1e-4)
    assert conf_for(0.25, confidence_ref=0) == pytest.approx(0.25 * vis, abs=1e-4)
    assert conf_for(0.125, confidence_ref=0.5) == pytest.approx(0.25 * vis, abs=1e-4)
    uniform = lambda x: (np.zeros((len(x), 90)), np.zeros((len(x), 90)))  # noqa: E731
    out = backend(make_config, tmp_path, FakeSession(uniform)).estimate(make_frame(), [make_person(1, 160, 120)])
    assert out[1].confidence == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------- availability
def test_unavailable_no_weights(make_config, tmp_path):
    cfg = make_cfg(make_config, tmp_path)
    with pytest.raises(BackendUnavailable, match=r"gaze\.l2cs"):
        resolve_l2cs_weights(cfg)
    with pytest.raises(BackendUnavailable, match=r"gaze\.l2cs"):
        BinnedGaze360Backend(cfg).load()


def test_unavailable_download_fails(make_config, tmp_path):
    cfg = make_config("fast", {"models": {"weights_dir": str(tmp_path), "allow_download": True}})
    with pytest.raises(BackendUnavailable) as info:
        resolve_l2cs_weights(cfg)
    msg = str(info.value)
    assert "mobileone_s0_gaze.onnx" in msg and "github.com/yakhyo" in msg and "gaze.l2cs" in msg
    assert "VFOA_WEIGHTS_DIR" in msg
    assert not list(tmp_path.iterdir())  # nothing half-written


def test_unavailable_no_onnxruntime(make_config, tmp_path, monkeypatch):
    def no_ort():
        raise ImportError("No module named 'onnxruntime'")

    monkeypatch.setattr(L, "_import_ort", no_ort)
    with pytest.raises(BackendUnavailable, match="onnxruntime"):
        BinnedGaze360Backend(make_cfg(make_config, tmp_path)).load()


def test_wrong_number_of_outputs(make_config, tmp_path):
    with pytest.raises(BackendUnavailable, match="2 outputs"):
        backend(make_config, tmp_path, FakeSession(const_outputs(), n_outputs=1)).load()


def test_weights_resolution(make_config, tmp_path, monkeypatch):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    explicit = tmp_path / "my.onnx"
    explicit.write_bytes(b"x")
    assert resolve_l2cs_weights(make_cfg(make_config, tmp_path, path=str(explicit))) == explicit
    (wdir / "resnet18_gaze.onnx").write_bytes(b"x")
    assert resolve_l2cs_weights(make_cfg(make_config, tmp_path, model="resnet18")) == wdir / "resnet18_gaze.onnx"
    monkeypatch.setenv("VFOA_WEIGHTS_DIR", str(tmp_path / "env"))
    with pytest.raises(BackendUnavailable):
        resolve_l2cs_weights(make_cfg(make_config, tmp_path, model="resnet18"))
    monkeypatch.delenv("VFOA_WEIGHTS_DIR")

    calls = []

    def fake_download(url, target, sha256=None):
        calls.append((url, target, sha256))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"onnx")
        return target

    monkeypatch.setattr(L, "_download", fake_download)
    cfg = make_config("fast", {"models": {"weights_dir": str(wdir), "allow_download": True}})
    assert resolve_l2cs_weights(cfg) == wdir / "mobileone_s0_gaze.onnx"
    assert calls[0][0].endswith("/weights/mobileone_s0_gaze.onnx")
    assert calls[0][2] == L.WEIGHTS["mobileone_s0"][2]
    cfg_off = make_config("fast", {"gaze": {"l2cs": {"download": False, "model": "resnet34"}},
                                   "models": {"weights_dir": str(wdir), "allow_download": True}})
    with pytest.raises(BackendUnavailable, match="disabled"):
        resolve_l2cs_weights(cfg_off)
    cfg_url = make_config("fast", {"gaze": {"l2cs": {"model": "resnet34", "url": "http://example.invalid/x.onnx"}},
                                   "models": {"weights_dir": str(wdir), "allow_download": True}})
    resolve_l2cs_weights(cfg_url)
    assert calls[-1][0] == "http://example.invalid/x.onnx" and calls[-1][2] is None


def test_download_helper_file_url(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"weights-bytes")
    sha = hashlib.sha256(b"weights-bytes").hexdigest()
    target = tmp_path / "out" / "w.onnx"
    assert REAL_DOWNLOAD(src.as_uri(), target, sha) == target
    assert target.read_bytes() == b"weights-bytes"
    bad = tmp_path / "out" / "bad.onnx"
    with pytest.raises(ValueError, match="sha256"):
        REAL_DOWNLOAD(src.as_uri(), bad, "0" * 64)
    assert not bad.exists() and not (tmp_path / "out" / "bad.onnx.part").exists()


def test_load_with_fake_onnxruntime(make_config, tmp_path, monkeypatch):
    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "mobileone_s0_gaze.onnx").write_bytes(b"x")
    made = {}

    class FakeOrt:
        class SessionOptions:
            intra_op_num_threads = 0

        @staticmethod
        def get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            made.update(path=path, providers=providers, threads=sess_options.intra_op_num_threads)
            return FakeSession(const_outputs())

    monkeypatch.setattr(L, "_import_ort", lambda: FakeOrt)
    cfg = make_config("fast", {"models": {"weights_dir": str(wdir), "allow_download": False, "device": "cpu"},
                               "runtime": {"num_threads": 3}})
    be = BinnedGaze360Backend(cfg)
    be.load()
    assert made == {"path": str(wdir / "mobileone_s0_gaze.onnx"), "providers": ["CPUExecutionProvider"],
                    "threads": 3}
    assert 1 in be.estimate(make_frame(), [make_person(1, 160, 120)])
    cfg["models"]["device"] = "auto"
    BinnedGaze360Backend(cfg).load()
    assert made["providers"][0] == "CUDAExecutionProvider"


# ---------------------------------------------------------------------- fallback
def test_create_backend_with_fallback(make_config, tmp_path, monkeypatch):
    cfg = make_cfg(make_config, tmp_path)
    b, used = create_backend_with_fallback("gaze", cfg)
    assert type(b).__name__ == "HeadProxyGaze" and used == "head_proxy" and b.is_loaded
    # no fallback configured -> unchanged behaviour (not loaded, no error yet)
    cfg2 = make_config("fast", {"gaze": {"fallback": None},
                                "models": {"weights_dir": str(tmp_path), "allow_download": False}})
    b, used = create_backend_with_fallback("gaze", cfg2)
    assert isinstance(b, BinnedGaze360Backend) and used is None and not b.is_loaded
    with pytest.raises(BackendUnavailable):
        b.load()
    # other profiles: no fallback
    for profile in ("accurate", "thesis", "synthetic"):
        assert make_config(profile).gaze.fallback is None
    # available primary -> kept (and loaded)
    monkeypatch.setattr(L, "_import_ort", lambda: (_ for _ in ()).throw(AssertionError("not used")))
    monkeypatch.setattr(BinnedGaze360Backend, "_load", lambda self: None)
    b, used = create_backend_with_fallback("gaze", cfg)
    assert isinstance(b, BinnedGaze360Backend) and used is None and b.is_loaded
    assert isinstance(create_backend("gaze", cfg), BinnedGaze360Backend)


def _pipeline_cfg(make_config, tmp_path, frames=6):
    return make_config("fast", {
        "input": {"source": "synthetic", "path": "synthetic", "max_frames": frames},
        "models": {"weights_dir": str(tmp_path / "weights"), "allow_download": False},
        "smoothing": {"direction": "none", "target": "none"},
    })


def _synthetic_backends(cfg):
    from vfoa.backends.synthetic import SyntheticDepth, SyntheticHeadPose, SyntheticPersons, SyntheticSegmentation

    return {"depth": SyntheticDepth(cfg), "persons": SyntheticPersons(cfg),
            "headpose": SyntheticHeadPose(cfg), "segmentation": SyntheticSegmentation(cfg)}


def test_pipeline_falls_back_to_head_proxy(make_config, tmp_path, caplog):
    from vfoa.pipeline import Pipeline, open_input

    cfg = _pipeline_cfg(make_config, tmp_path)
    pipe = Pipeline(cfg, backends=_synthetic_backends(cfg))
    results = []

    class Sink:
        def write(self, frame, result):
            results.append(result)

        def close(self):
            pass

    with caplog.at_level("WARNING"):
        summary = pipe.run(open_input(cfg), [Sink()])
    assert type(pipe.backends["gaze"]).__name__ == "HeadProxyGaze"
    assert pipe.cfg.gaze.backend == "head_proxy" and pipe.cfg.gaze.fallback_from == "l2cs"
    assert cfg.gaze.backend == "l2cs"  # the caller's config is not mutated
    assert any("falling back" in r.getMessage() for r in caplog.records)
    assert len(results) == 6
    assert any(p.gaze is not None for r in results for p in r.persons)
    assert summary["counters"].get("gaze_fallback") == 1, summary["counters"]
    # reopening retries the primary backend (from the original config)
    pipe.open(30.0)
    assert pipe.cfg.gaze.backend == "head_proxy" and pipe.cfg.gaze.fallback_from == "l2cs"
    pipe.close()


def test_pipeline_uses_l2cs_when_available(make_config, tmp_path, monkeypatch):
    from vfoa.pipeline import Pipeline, open_input

    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "mobileone_s0_gaze.onnx").write_bytes(b"x")
    sessions = []

    class FakeOrt:
        SessionOptions = SimpleNamespace

        @staticmethod
        def get_available_providers():
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(path, sess_options=None, providers=None):
            sessions.append(FakeSession(const_outputs(30.0, 0.0)))
            return sessions[-1]

    monkeypatch.setattr(L, "_import_ort", lambda: FakeOrt)
    cfg = _pipeline_cfg(make_config, tmp_path)
    pipe = Pipeline(cfg, backends=_synthetic_backends(cfg))
    results = []

    class Sink:
        def write(self, frame, result):
            results.append(result)

        def close(self):
            pass

    summary = pipe.run(open_input(cfg), [Sink()])
    assert isinstance(pipe.backends["gaze"], BinnedGaze360Backend)
    assert pipe.cfg is cfg and "gaze_fallback" not in summary["counters"]
    assert "load:gaze" in summary["setup_ms"]
    gazes = [p.gaze for r in results for p in r.persons if p.gaze is not None]
    assert gazes and all(g.direction[0] < 0 for g in gazes)  # +30 deg = image-left
    assert len(sessions) == 1 and sessions[0].fed


# ---------------------------------------------------------------------- import hygiene
def test_lazy_import(repo_root):
    code = textwrap.dedent(
        """
        import json, sys
        import vfoa.backends.fast.l2cs_gaze as m
        from vfoa.backends import create_backend
        from vfoa.config import load_config
        b = create_backend("gaze", load_config(profile="fast"))
        print(json.dumps({"ort": "onnxruntime" in sys.modules, "torch": "torch" in sys.modules,
                          "name": type(b).__name__, "loaded": b.is_loaded}))
        """
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo_root), env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data == {"ort": False, "torch": False, "name": "BinnedGaze360Backend", "loaded": False}


# ---------------------------------------------------------------------- real model (network)
@pytest.mark.network
def test_real_mobileone_s0_sintel(make_config, tmp_path, sintel_sample):
    pytest.importorskip("onnxruntime")
    image, _depth, intr = sintel_sample(1)
    cfg = make_config("fast", {"models": {"weights_dir": str(tmp_path), "allow_download": True, "device": "cpu"}})
    be = BinnedGaze360Backend(cfg)
    be.load()
    assert (tmp_path / "mobileone_s0_gaze.onnx").is_file()
    frame = Frame(index=0, t=0.0, image=image, intrinsics=intr)
    persons = []
    try:
        import ultralytics  # noqa: F401
    except Exception:
        pass
    else:
        from vfoa.backends.fast.yolo_pose import YoloPosePersons

        dets = YoloPosePersons(cfg).detect(frame)
        persons = [Person(i + 1, d) for i, d in enumerate(dets)]
    h, w = image.shape[:2]
    persons.append(make_person(999, w / 2, h / 3, eye_dist=max(w / 20, 10)))  # always one gated-in face
    out = be.estimate(frame, persons)
    assert 999 in out
    for e in out.values():
        assert np.isfinite(e.direction).all()
        assert np.linalg.norm(e.direction) == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= e.confidence <= 1.0
