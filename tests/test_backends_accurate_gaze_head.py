"""Tests of the accurate gaze (Gaze360) and head-pose (FSA-Net) backends with fake models.

Refs #2 #7 #8 #9 #17 #32.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vfoa.backends.accurate.fsanet import FSANetHeadPose, crop_head, preprocess_head
from vfoa.backends.accurate.gaze360 import MEAN, STD, Gaze360Gaze, strip_module_prefix
from vfoa.backends.base import BackendUnavailable
from vfoa.geometry import euler_to_forward, eye_pixel, gaze360_to_camera, ray_aligned, unproject
from vfoa.types import Frame, Intrinsics, Person, PersonDetection

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- helpers


def _frame(index: int, value=0, fps: float = 30.0, size=(120, 160)) -> Frame:
    h, w = size
    if isinstance(value, np.ndarray):
        image = value
        h, w = image.shape[:2]
    else:
        image = np.full((h, w, 3), value, dtype=np.uint8)
    return Frame(index=index, t=index / fps, image=image, intrinsics=Intrinsics.from_hfov(w, h))


def _two_value_frame(index: int, fps: float = 30.0) -> Frame:
    """Left half = index, right half = index + 100 (decodable per track)."""
    img = np.empty((120, 160, 3), dtype=np.uint8)
    img[:, :80] = index
    img[:, 80:] = index + 100
    return _frame(index, img, fps=fps)


def _person(tid: int, box) -> Person:
    return Person(track_id=tid, detection=PersonDetection(head_box=box))


LEFT = [10, 10, 50, 50]
RIGHT = [100, 10, 140, 50]


class FakeSession:
    def __init__(self, ypr):
        self.ypr = np.asarray(ypr, dtype=np.float32).reshape(1, 3)
        self.feeds: list[dict] = []

    def run(self, names, feeds):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return [self.ypr.copy()]


class FakeGazeModel:
    """Records its input; returns per-person (yaw, pitch) and uncertainties as torch tensors."""

    def __init__(self, yaw=0.0, pitch=0.0, unc=0.5):
        self.yaw, self.pitch, self.unc = yaw, pitch, unc
        self.inputs: list[np.ndarray] = []

    def __call__(self, x):
        import torch

        self.inputs.append(x.numpy().copy())
        n = x.shape[0]
        unc = np.broadcast_to(np.asarray(self.unc, dtype=np.float32), (n,))
        ang = np.stack([np.full(n, self.yaw), np.full(n, self.pitch)], axis=1).astype(np.float32)
        var = np.stack([unc, unc], axis=1)
        return torch.from_numpy(ang), torch.from_numpy(np.ascontiguousarray(var))


def _decode(x: np.ndarray) -> np.ndarray:
    """Frame value per slot from the normalised R channel of (P, 7, 3, H, W)."""
    return np.rint((x[..., 0, 0, 0] * STD[0] + MEAN[0]) * 255).astype(int)


def _fsa_cfg(make_config, tmp_path: Path, extra=None):
    m1, m2 = tmp_path / "m1.onnx", tmp_path / "m2.onnx"
    m1.write_bytes(b"")
    m2.write_bytes(b"")
    over = {"headpose": {"models": [str(m1), str(m2)]}}
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(over.get(key), dict):
            over[key] = {**over[key], **value}
        else:
            over[key] = value
    return make_config("accurate", over)


def _gaze(make_config, gaze=None, runtime=None, model=None):
    over: dict = {"gaze": {"window_step": 2, "eye_frame_correction": False, **(gaze or {})}}
    if runtime:
        over["runtime"] = runtime
    model = model or FakeGazeModel()
    calls = []

    def factory():
        calls.append(1)
        return model

    return Gaze360Gaze(make_config("accurate", over), model_factory=factory), model, calls


# --------------------------------------------------------------------------- import / load


def test_import_hygiene():
    code = textwrap.dedent(
        """
        import sys
        import GazePredictor1, HeadPredictor, transforms, functional
        import vfoa.backends.accurate.gaze360, vfoa.backends.accurate.fsanet
        bad = [m for m in ("torch", "torchvision", "onnxruntime") if m in sys.modules]
        assert not bad, bad
        print("ok")
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO / "thesis_final", env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout
    assert not (REPO / "thesis_final" / "GazePredictor.py").exists()


def test_load_unavailable_names_config_key(make_config, monkeypatch, tmp_path):
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    cfg = make_config("accurate")
    with pytest.raises(BackendUnavailable, match=r"headpose\.models"):
        FSANetHeadPose(cfg).load()
    with pytest.raises(BackendUnavailable, match=r"gaze\.weights"):
        Gaze360Gaze(cfg).load()
    weights = tmp_path / "gaze360" / "gaze360_model.pth.tar"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    cfg2 = make_config("accurate", {"models": {"third_party_root": str(tmp_path)}})
    with pytest.raises(BackendUnavailable, match=r"models\.third_party_root"):
        Gaze360Gaze(cfg2).load()
    # a missing FSA-Net file under a configured root also names the key
    with pytest.raises(BackendUnavailable, match=r"headpose\.models"):
        FSANetHeadPose(cfg2).load()


# --------------------------------------------------------------------------- FSA-Net


def test_fsanet_normalisation(make_config, tmp_path):
    sess = FakeSession((0, 0, 0))
    be = FSANetHeadPose(_fsa_cfg(make_config, tmp_path), session_factory=lambda p: sess)
    for value, expected in ((255, (255 - 127.5) / 128), (0, -127.5 / 128)):
        sess.feeds.clear()
        be.estimate(_frame(0, value), [_person(0, LEFT)])
        blob = sess.feeds[0]["input"]
        assert blob.dtype == np.float32 and blob.shape == (1, 3, 64, 64)
        np.testing.assert_array_equal(blob, np.float32(expected))
    assert expected == pytest.approx(-0.99609375)


def test_fsanet_sessions_once_and_direction(make_config, tmp_path):
    made = []
    sessions = [FakeSession((10, 20, 30)), FakeSession((30, -10, 0))]

    def factory(path):
        made.append(path)
        return sessions[len(made) - 1]

    be = FSANetHeadPose(_fsa_cfg(make_config, tmp_path), session_factory=factory)
    for i in range(5):
        out = be.estimate(_frame(i, 128), [_person(7, LEFT)])
    assert len(made) == 2
    est = out[7]
    frame = _frame(4, 128)
    ray = unproject(eye_pixel(np.asarray(LEFT, dtype=np.float32)), 1.0, frame.intrinsics)
    np.testing.assert_allclose(est.direction, ray_aligned(euler_to_forward(20, 5, 15), ray), atol=1e-12)
    np.testing.assert_allclose(est.euler_deg, (20, 5, 15))
    assert np.linalg.norm(est.direction) == pytest.approx(1.0)
    assert est.frame_index == 4


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_fsanet_eye_frame_correction(make_config, tmp_path):
    frame = _frame(0, 128)
    box_ray = unproject(eye_pixel(np.asarray(RIGHT, dtype=np.float32)), 1.0, frame.intrinsics)
    eyes = np.array([125.0, 20.0], np.float32)
    eyes_ray = unproject(eyes.astype(np.float64), 1.0, frame.intrinsics)
    no_intr = Frame(index=0, t=0.0, image=frame.image, intrinsics=None)
    for ypr in ((0, 0, 0), (30, 10, 5)):
        sess = FakeSession(ypr)
        fwd = euler_to_forward(*ypr)
        on = FSANetHeadPose(_fsa_cfg(make_config, tmp_path), session_factory=lambda p: sess)
        off = FSANetHeadPose(_fsa_cfg(make_config, tmp_path, {"headpose": {"eye_frame_correction": False}}),
                             session_factory=lambda p: sess)
        cases = [
            (on, frame, None, ray_aligned(fwd, box_ray)),
            (on, frame, eyes, ray_aligned(fwd, eyes_ray)),
            (off, frame, None, fwd),
            (on, no_intr, None, fwd),
        ]
        for be, fr, eyes_px, expected in cases:
            person = _person(3, RIGHT)
            person.eyes_px = eyes_px
            est = be.estimate(fr, [person])[3]
            np.testing.assert_allclose(est.direction, expected, atol=1e-6)
            assert np.linalg.norm(est.direction) == pytest.approx(1.0)
            np.testing.assert_allclose(est.euler_deg, ypr, atol=1e-5)
        if ypr == (0, 0, 0):
            # looking at the camera from off-axis: back along the eye ray
            np.testing.assert_allclose(cases[0][3], -_unit(box_ray), atol=1e-12)
            np.testing.assert_allclose(cases[1][3], -_unit(eyes_ray), atol=1e-12)
            np.testing.assert_allclose(cases[2][3], [0, 0, -1], atol=1e-12)


def test_fsanet_default_factory_disables_spinning(make_config, tmp_path, monkeypatch):
    created = []

    class Options:
        def __init__(self):
            self.entries = []
            self.intra_op_num_threads = 0

        def add_session_config_entry(self, key, value):
            self.entries.append((key, value))

    class PlainOptions:
        intra_op_num_threads = 0

    def inference_session(path, sess_options=None, providers=None):
        created.append((path, sess_options, providers))
        return FakeSession((0, 0, 0))

    fake = SimpleNamespace(SessionOptions=Options, InferenceSession=inference_session,
                           get_available_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    cfg = _fsa_cfg(make_config, tmp_path, {"models": {"device": "cpu"}, "runtime": {"num_threads": 3}})
    FSANetHeadPose(cfg).load()
    assert len(created) == 2
    for _path, options, providers in created:
        assert ("session.intra_op.allow_spinning", "0") in options.entries
        assert options.intra_op_num_threads == 3
        assert providers == ["CPUExecutionProvider"]
    created.clear()
    fake.SessionOptions = PlainOptions
    be = FSANetHeadPose(cfg)
    be.load()
    assert len(created) == 2 and len(be._sessions) == 2


def test_fsanet_box_clipping(make_config, tmp_path):
    sess = FakeSession((0, 0, 0))
    be = FSANetHeadPose(_fsa_cfg(make_config, tmp_path), session_factory=lambda p: sess)
    yy, xx = np.mgrid[0:120, 0:160]
    img = np.stack([xx, yy, (xx + yy) // 2], axis=-1).astype(np.uint8)
    frame = _frame(3, img)
    out = be.estimate(frame, [_person(1, [-20, -10, 30, 40])])
    assert 1 in out
    np.testing.assert_array_equal(sess.feeds[0]["input"], preprocess_head(img[0:40, 0:30]))
    sess.feeds.clear()
    persons = [
        _person(2, [50, 50, 50, 80]),        # zero width
        _person(3, [200, 200, 260, 260]),    # fully outside
        _person(4, None),                    # no head box
    ]
    assert be.estimate(frame, persons) == {}
    assert sess.feeds == []
    assert crop_head(img, [np.nan, 0, 10, 10]) is None


# --------------------------------------------------------------------------- Gaze360 windows


def test_gaze_window_causal(make_config):
    be, model, _ = _gaze(make_config)
    persons = [_person(0, LEFT), _person(1, RIGHT)]
    for i in range(21):
        be.observe(_two_value_frame(i), persons)
    be.estimate(_two_value_frame(20), persons)
    x = model.inputs[-1]
    vals = _decode(x)
    assert vals[0].tolist() == [14, 16, 18, 20, 20, 20, 20]
    assert vals[1].tolist() == [114, 116, 118, 120, 120, 120, 120]
    assert (vals[1] >= 100).all() and (vals[0] < 100).all()

    be2, model2, _ = _gaze(make_config)
    for i in range(3):
        be2.observe(_two_value_frame(i), persons)
    be2.estimate(_two_value_frame(2), persons)
    vals2 = _decode(model2.inputs[-1])
    assert vals2[0].tolist() == [0, 0, 0, 2, 2, 2, 2]
    assert vals2[1].tolist() == [100, 100, 100, 102, 102, 102, 102]


def test_gaze_window_lookahead(make_config):
    be, model, _ = _gaze(make_config, runtime={"lookahead_frames": 6})
    persons = [_person(0, LEFT)]
    for i in range(27):
        be.observe(_two_value_frame(i), persons)
    be.estimate(_two_value_frame(20), persons)
    assert _decode(model.inputs[-1])[0].tolist() == [14, 16, 18, 20, 22, 24, 26]


def test_gaze_window_static(make_config):
    be, model, _ = _gaze(make_config, gaze={"window": "static"})
    persons = [_person(0, LEFT)]
    for i in range(21):
        be.observe(_two_value_frame(i), persons)
    be.estimate(_two_value_frame(20), persons)
    assert _decode(model.inputs[-1])[0].tolist() == [20] * 7


def test_gaze_default_step_from_fps(make_config):
    be, model, _ = _gaze(make_config, gaze={"window_step": None})
    persons = [_person(0, LEFT)]
    for i in range(21):
        be.observe(_two_value_frame(i, fps=30.0), persons)
    be.estimate(_two_value_frame(20, fps=30.0), persons)
    assert be.fps == pytest.approx(30.0)
    assert _decode(model.inputs[-1])[0].tolist() == [11, 14, 17, 20, 20, 20, 20]


def test_gaze_channel_order_and_shape(make_config):
    be, model, _ = _gaze(make_config)
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[..., 0] = 255  # pure blue in BGR
    persons = [_person(0, LEFT), _person(1, RIGHT)]
    be.estimate(_frame(0, img), persons)
    x = model.inputs[-1]
    assert x.shape == (2, 7, 3, 224, 224) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, :, 2], (1 - MEAN[2]) / STD[2], rtol=1e-5)
    np.testing.assert_allclose(x[:, :, 0], -MEAN[0] / STD[0], rtol=1e-5)
    np.testing.assert_allclose(x[:, :, 1], -MEAN[1] / STD[1], rtol=1e-5)


def test_gaze_conversion(make_config):
    yaw, pitch = 0.3, -0.2
    box = [10, 10, 40, 40]
    be, _, _ = _gaze(make_config, model=FakeGazeModel(yaw, pitch, 0.1))
    frame = _frame(0, 100)
    out = be.estimate(frame, [_person(0, box)])
    np.testing.assert_allclose(out[0].direction, gaze360_to_camera(yaw, pitch), atol=1e-6)

    be2, _, _ = _gaze(make_config, gaze={"eye_frame_correction": True},
                      model=FakeGazeModel(yaw, pitch, 0.1))
    out2 = be2.estimate(frame, [_person(0, box)])
    ray = unproject(eye_pixel(np.asarray(box, dtype=np.float32)), 1.0, frame.intrinsics)
    np.testing.assert_allclose(out2[0].direction, gaze360_to_camera(yaw, pitch, ray), atol=1e-6)
    assert not np.allclose(out2[0].direction, out[0].direction, atol=1e-3)

    # +yaw turns towards the image left (Gaze360 eye frame: +x image-left)
    be3, _, _ = _gaze(make_config, model=FakeGazeModel(0.3, 0.0, 0.1))
    d = be3.estimate(frame, [_person(0, box)])[0].direction
    assert d[0] < 0
    np.testing.assert_allclose(d, [-np.sin(0.3), 0.0, -np.cos(0.3)], atol=1e-6)


def test_gaze_uncertainty_and_confidence(make_config):
    model = FakeGazeModel(0.1, 0.1, unc=[0.6, 3.1])
    be, _, _ = _gaze(make_config, gaze={"uncertainty_thr": 3.0}, model=model)
    out = be.estimate(_frame(0, 50), [_person(0, LEFT), _person(1, RIGHT)])
    assert list(out) == [0]
    assert out[0].confidence == pytest.approx(0.8)
    assert out[0].frame_index == 0


def test_gaze_single_call_and_factory_once(make_config):
    be, model, calls = _gaze(make_config)
    persons = [_person(0, [0, 0, 30, 30]), _person(1, [40, 0, 70, 30]), _person(2, [80, 0, 110, 30])]
    for i in range(5):
        f = _frame(i, i)
        be.observe(f, persons)
        out = be.estimate(f, persons)
        assert len(model.inputs) == i + 1
        assert model.inputs[-1].shape[0] == 3
        assert set(out) == {0, 1, 2}
    assert len(calls) == 1


def test_gaze_observe_does_not_load(make_config):
    be, _, calls = _gaze(make_config)
    be.observe(_frame(0, 1), [_person(0, LEFT)])
    assert calls == [] and not be.is_loaded


def test_gaze_buffers_bounded_and_pruned(make_config):
    be, _, _ = _gaze(make_config)
    s, lookahead = 2, 0
    for i in range(50):
        persons = [_person(0, LEFT)] + ([_person(1, RIGHT)] if i < 30 else [])
        be.observe(_two_value_frame(i), persons)
        for buf in be._buffers.values():
            assert len(buf) <= 3 * s + lookahead + 1
    assert set(be._buffers) == {0}
    be.reset()
    assert be._buffers == {}


def test_strip_module_prefix():
    state = {"module.a.weight": 1, "b.bias": 2, "module.module.c": 3}
    assert strip_module_prefix(state) == {"a.weight": 1, "b.bias": 2, "module.c": 3}


def test_default_model_loads_stub_repo(make_config, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    code = tmp_path / "gaze360" / "code"
    code.mkdir(parents=True)
    (code / "model.py").write_text(textwrap.dedent(
        """
        import torch.nn as nn


        class _Base(nn.Linear):
            pass


        def resnet18(pretrained=True, **kw):
            base = _Base(4, 3)
            base.pretrained_flag = pretrained
            return base


        class GazeLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = resnet18(pretrained=True)
                self.head = nn.Linear(3, 2)
        """
    ))
    ref = {"base.weight": torch.arange(12.0).view(3, 4), "base.bias": torch.ones(3),
           "head.weight": torch.full((2, 3), 0.5), "head.bias": torch.zeros(2)}
    ckpt = {"state_dict": {f"module.{k}": v for k, v in ref.items()}, "epoch": np.float64(3.0)}
    weights = tmp_path / "w.pth.tar"
    torch.save(ckpt, weights)
    cfg = make_config("accurate", {"models": {"third_party_root": str(tmp_path), "device": "cpu"},
                                   "gaze": {"weights": str(weights)}})
    be = Gaze360Gaze(cfg)
    be.load()
    model = be._model
    assert model.base.pretrained_flag is False
    assert not model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, ref[k])


# --------------------------------------------------------------------------- legacy


def test_legacy_aliases(tmp_path):
    from GazePredictor1 import GazePredictor
    from HeadPredictor import HeadPredictor

    assert issubclass(GazePredictor, Gaze360Gaze)
    assert issubclass(HeadPredictor, FSANetHeadPose)
    g = GazePredictor("my/gaze.pth", 4, uncertaintyThreshold=2.0)
    assert g.cfg.gaze.weights == "my/gaze.pth"
    assert g.cfg.gaze.batch_size == 4
    assert g.cfg.gaze.uncertainty_thr == 2.0
    assert g.cfg.gaze.backend == "gaze360"
    h = HeadPredictor("a.onnx", "b.onnx", 8)
    assert list(h.cfg.headpose.models) == ["a.onnx", "b.onnx"]
    assert h.batchSize == 8
    h2 = HeadPredictor(cfg=g.cfg)
    assert h2.cfg is g.cfg


def test_transforms_py312():
    torch = pytest.importorskip("torch")
    import functional
    import transforms

    img = (np.arange(50 * 80 * 3) % 256).astype(np.uint8).reshape(50, 80, 3)
    assert transforms.Resize((224, 224))(img).shape == (224, 224, 3)
    small = functional.resize(img, 32)
    assert small.shape == (32, 51, 3)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    t = transforms.Normalize(mean, std)(transforms.ToTensor()(img))
    expected = (img.transpose(2, 0, 1) / 255.0 - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    assert isinstance(t, torch.Tensor) and tuple(t.shape) == (3, 50, 80)
    np.testing.assert_allclose(t.numpy(), expected, rtol=1e-5, atol=1e-6)
