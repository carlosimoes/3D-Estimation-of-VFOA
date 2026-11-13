"""Tests of the optional fast backends (YOLO pose/seg, keypoint head pose, head-proxy gaze).

All models are faked with SimpleNamespace results; nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from vfoa.backends.base import BackendUnavailable
from vfoa.backends.fast import _ultra
from vfoa.backends.fast.head_proxy_gaze import HeadProxyGaze
from vfoa.backends.fast.keypoints_headpose import HEAD_MODEL, KeypointHeadPose, solve_head_pose
from vfoa.backends.fast.yolo_pose import YoloPosePersons, head_box_from_keypoints
from vfoa.backends.fast.yolo_seg import YoloSegSegmentation, _unletterbox
from vfoa.config import load_config
from vfoa.types import DirectionEstimate, Frame, Intrinsics, Person, PersonDetection

FAST_MODULES = [
    "vfoa.backends.fast._ultra",
    "vfoa.backends.fast.yolo_pose",
    "vfoa.backends.fast.keypoints_headpose",
    "vfoa.backends.fast.yolo_seg",
    "vfoa.backends.fast.head_proxy_gaze",
]


# --------------------------------------------------------------------------- fakes
class FakeTensor:
    """Minimal torch-like tensor: detach().cpu().numpy()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class FakeModel:
    def __init__(self, result, names=None):
        self.result = result
        self.names = names
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append((source, kwargs))
        return [self.result]

    def track(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("model.track must never be called")


def make_frame(w=640, h=480, index=3, intrinsics=None):
    img = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return Frame(index=index, t=index / 30.0, image=img, intrinsics=intrinsics)


def face_keypoints(cx=300.0, cy=100.0, conf=0.9):
    kp = np.zeros((17, 3), np.float32)
    kp[0] = (cx, cy, conf)  # nose
    kp[1] = (cx + 8, cy - 6, conf)  # l_eye (image right when facing camera? any)
    kp[2] = (cx - 8, cy - 6, conf)
    kp[3] = (cx + 18, cy - 2, conf)
    kp[4] = (cx - 18, cy - 2, conf)
    kp[5] = (cx + 40, cy + 50, conf)
    kp[6] = (cx - 40, cy + 50, conf)
    return kp


# --------------------------------------------------------------------------- import hygiene
def test_import_hygiene_and_registry(repo_root):
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        for name in {FAST_MODULES!r}:
            importlib.import_module(name)
        from vfoa.backends import create_backend
        from vfoa.config import load_config
        cfg = load_config(profile="fast", overrides=["gaze.backend=head_proxy"])
        out = {{}}
        for kind in ("persons", "headpose", "segmentation", "gaze"):
            be = create_backend(kind, cfg)
            out[kind] = [type(be).__name__, be.is_loaded]
        print(json.dumps({{"mods": sorted(sys.modules), "backends": out}}))
        """
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    roots = {m.split(".")[0] for m in data["mods"]}
    assert not roots & {"ultralytics", "torch", "torchvision"}, sorted(roots & {"ultralytics", "torch"})
    assert data["backends"] == {
        "persons": ["YoloPosePersons", False],
        "headpose": ["KeypointHeadPose", False],
        "segmentation": ["YoloSegSegmentation", False],
        "gaze": ["HeadProxyGaze", False],
    }


# --------------------------------------------------------------------------- YOLO pose
def pose_result(xyxy, conf, kps, tensor=False):
    wrap = FakeTensor if tensor else np.asarray
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=wrap(np.asarray(xyxy, np.float32)), conf=wrap(np.asarray(conf, np.float32))),
        keypoints=SimpleNamespace(data=wrap(np.asarray(kps, np.float32))),
    )


@pytest.mark.parametrize("tensor", [False, True])
def test_pose_detections(tensor):
    cfg = load_config(profile="fast")
    frame = make_frame()
    before = frame.image.copy()
    kp1 = face_keypoints()
    kp2 = face_keypoints(500, 200)
    res = pose_result([[250, 60, 360, 470], [450, 150, 560, 470]], [0.9, 0.2], [kp1, kp2], tensor)
    model = FakeModel(res)
    be = YoloPosePersons(cfg, model=model)
    dets = be.detect(frame)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, PersonDetection)
    np.testing.assert_allclose(d.body_box, [250, 60, 360, 470])
    assert d.keypoints.shape == (17, 3) and d.keypoints.dtype == np.float32
    assert d.score == pytest.approx(0.9)
    hb = d.head_box
    assert hb is not None and hb[2] > hb[0] and hb[3] > hb[1]
    assert hb[0] >= 0 and hb[1] >= 0 and hb[2] <= 639 and hb[3] <= 479
    face = kp1[:5, :2]
    assert (face[:, 0] >= hb[0]).all() and (face[:, 0] <= hb[2]).all()
    assert (face[:, 1] >= hb[1]).all() and (face[:, 1] <= hb[3]).all()
    source, kwargs = model.calls[0]
    assert source is frame.image
    np.testing.assert_array_equal(frame.image, before)
    assert kwargs["conf"] == pytest.approx(cfg.persons.score_thr)
    assert kwargs["imgsz"] == cfg.persons.imgsz
    assert kwargs["verbose"] is False
    assert "device" not in kwargs  # injected model: no device resolution
    assert be.is_loaded


def test_pose_clips_and_sanitises_keypoints():
    cfg = load_config(profile="fast")
    frame = make_frame(w=320, h=240)
    kp = face_keypoints(20, 15)
    kp[10] = (np.nan, 5, 0.9)
    res = pose_result([[-10, -5, 100, 300]], [0.8], [kp])
    dets = YoloPosePersons(cfg, model=FakeModel(res)).detect(frame)
    assert len(dets) == 1
    np.testing.assert_allclose(dets[0].body_box, [0, 0, 100, 239])
    assert dets[0].keypoints[10, 2] == 0 and np.isfinite(dets[0].keypoints).all()
    hb = dets[0].head_box
    assert hb[0] >= 0 and hb[1] >= 0


def test_pose_head_box_fallbacks():
    size = (640, 480)
    body = np.array([100, 50, 200, 450], np.float32)
    kp = face_keypoints()
    kp[:5, 2] = 0.0
    s_fb = min(0.45 * 100, 0.3 * 400)
    hb = head_box_from_keypoints(kp, body, size)
    np.testing.assert_allclose(hb, [150 - s_fb / 2, 50, 150 + s_fb / 2, 50 + s_fb], atol=1e-4)
    hb2 = head_box_from_keypoints(None, body, size)
    np.testing.assert_allclose(hb2, hb)
    assert head_box_from_keypoints(None, np.array([10, 10, 10, 50]), size) is None
    assert head_box_from_keypoints(kp, None, size) is None
    # profile view: nose + one eye + one ear
    kp = np.zeros((17, 3), np.float32)
    kp[0] = (300, 100, 0.9)
    kp[1] = (292, 94, 0.8)
    kp[3] = (275, 98, 0.7)
    hb = head_box_from_keypoints(kp, body, size)
    pts = kp[[0, 1, 3], :2]
    assert (pts[:, 0] >= hb[0]).all() and (pts[:, 0] <= hb[2]).all()
    assert (pts[:, 1] >= hb[1]).all() and (pts[:, 1] <= hb[3]).all()
    # single keypoint: size from the body box
    kp1 = np.zeros((17, 3), np.float32)
    kp1[0] = (300, 100, 0.9)
    hb1 = head_box_from_keypoints(kp1, body, size)
    assert hb1[2] - hb1[0] == pytest.approx(s_fb, rel=1e-4)
    # empty real-style result
    cfg = load_config(profile="fast")
    empty = pose_result(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3)))
    assert YoloPosePersons(cfg, model=FakeModel(empty)).detect(make_frame()) == []
    assert YoloPosePersons(cfg, model=FakeModel(SimpleNamespace(boxes=None))).detect(make_frame()) == []
    # missing keypoints -> body fallback head box
    res = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=np.array([[100, 50, 200, 450]], np.float32), conf=np.array([0.9])),
        keypoints=None,
    )
    dets = YoloPosePersons(cfg, model=FakeModel(res)).detect(make_frame())
    assert dets[0].keypoints is None
    np.testing.assert_allclose(dets[0].head_box, hb2)


# --------------------------------------------------------------------------- head pose (PnP)
def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _forward(yaw, pitch):
    y, p = np.radians(yaw), np.radians(pitch)
    return np.array([-np.sin(y), -np.cos(y) * np.sin(p), -np.cos(y) * np.cos(p)])


def _angle_deg(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


FX = 900.0
INTR = Intrinsics(fx=FX, fy=FX, cx=511.5, cy=217.5, width=1024, height=436)


def synth_keypoints(yaw, pitch, roll=0.0, t=(0.1, -0.05, 1.2), rng=None, noise=0.0, conf=(1, 1, 1, 1, 1)):
    R = _rx(-np.radians(pitch)) @ _ry(np.radians(yaw)) @ _rz(np.radians(roll))
    pc = (R @ HEAD_MODEL.T).T + np.asarray(t)
    uv = np.stack([FX * pc[:, 0] / pc[:, 2] + INTR.cx, FX * pc[:, 1] / pc[:, 2] + INTR.cy], axis=1)
    if noise:
        uv = uv + rng.normal(0.0, noise, uv.shape)
    kp = np.zeros((17, 3), np.float32)
    kp[:5, :2] = uv
    kp[:5, 2] = conf
    return kp


def test_headpose_pnp_grid():
    rng = np.random.default_rng(0)
    for yaw in (-40, -20, 0, 20, 40):
        for pitch in (-20, 0, 20):
            kp = synth_keypoints(yaw, pitch, 10.0, rng=rng, noise=1.0)
            sol = solve_head_pose(kp, INTR.K)
            assert sol is not None, (yaw, pitch)
            f, conf, euler = sol
            assert _angle_deg(f, _forward(yaw, pitch)) < 5.0, (yaw, pitch)
            assert np.linalg.norm(f) == pytest.approx(1.0, abs=1e-9)
            assert f.dtype == np.float64
            np.testing.assert_allclose(euler, [yaw, pitch, 10.0], atol=5.0)
            assert conf == pytest.approx(1.0)


def test_headpose_backend_frontal():
    cfg = load_config(profile="fast")
    kp = synth_keypoints(0, 0, t=(0.0, 0.0, 1.2))
    person = Person(track_id=7, detection=PersonDetection(head_box=None, keypoints=kp))
    frame = Frame(index=5, t=5 / 30, image=np.zeros((436, 1024, 3), np.uint8), intrinsics=INTR)
    out = KeypointHeadPose(cfg).estimate(frame, [person])
    est = out[7]
    np.testing.assert_allclose(est.direction, [0, 0, -1], atol=1e-6)
    assert est.frame_index == 5 and 0 < est.confidence <= 1
    np.testing.assert_allclose(est.euler_deg, [0, 0, 0], atol=1e-4)


def test_headpose_insufficient_keypoints():
    cfg = load_config(profile="fast")
    base = synth_keypoints(10, 5)
    cases = {}
    kp = base.copy(); kp[2:5, 2] = 0; cases[0] = kp  # 2 strong
    kp = base.copy(); kp[3:5, 2] = 0; cases[1] = kp  # 3 strong, rest 0
    kp = base.copy(); kp[0, 2] = 0; cases[2] = kp  # no nose
    kp = np.zeros((17, 3), np.float32); kp[:5] = (100, 100, 0.9); cases[3] = kp  # coincident
    kp = base.copy(); kp[0, 1] += 60; cases[4] = kp  # inconsistent nose
    persons = [Person(i, PersonDetection(head_box=None, keypoints=k)) for i, k in cases.items()]
    persons.append(Person(5, PersonDetection(head_box=[0, 0, 10, 10])))  # no keypoints
    frame = Frame(index=0, t=0.0, image=np.zeros((436, 1024, 3), np.uint8), intrinsics=None)
    assert KeypointHeadPose(cfg).estimate(frame, persons) == {}
    # intrinsics None still works for a valid person
    good = Person(9, PersonDetection(head_box=None, keypoints=base))
    assert set(KeypointHeadPose(cfg).estimate(frame, [good])) == {9}


def test_headpose_relaxed_fallback():
    confs = (0.9, 0.9, 0.2, 0.9, 0.0)
    kp = synth_keypoints(50, 0, conf=confs)
    sol = solve_head_pose(kp, INTR.K, kpt_thr=0.35)
    assert sol is not None
    f, conf, _ = sol
    assert _angle_deg(f, _forward(50, 0)) < 5.0
    pts = kp[:4, :2].astype(np.float64)
    span = np.linalg.norm(pts[:, None] - pts[None], axis=-1).max()
    expected = 0.5 * np.mean(confs[:4]) * np.clip(span / 40.0, 0.25, 1.0)
    assert conf == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------- YOLO seg
def seg_result(masks, conf, cls, names=None, tensor=False):
    wrap = FakeTensor if tensor else np.asarray
    return SimpleNamespace(
        masks=SimpleNamespace(data=wrap(np.asarray(masks, np.float32))),
        boxes=SimpleNamespace(conf=wrap(np.asarray(conf, np.float32)), cls=wrap(np.asarray(cls, np.float32))),
        names=names,
    )


@pytest.mark.parametrize("tensor", [False, True])
def test_seg_label_map(tensor):
    cfg = load_config(profile="fast")
    H, W = 60, 80
    m = np.zeros((4, H, W), np.float32)
    m[0, 10:30, 10:30] = 1  # score 0.5, cls 0 (person)
    m[1, 20:40, 20:40] = 1  # score 0.9, cls 56 (chair) wins overlap
    m[2, 50:55, 70:75] = 1  # score 0.1 below threshold
    m[3, 22:28, 22:28] = 1  # score 0.4, fully covered by higher-score mask 1
    res = seg_result(m, [0.5, 0.9, 0.1, 0.4], [0, 56, 2, 3], names={0: "person", 56: "chair", 2: "car", 3: "moto"},
                     tensor=tensor)
    frame = make_frame(w=W, h=H, index=11)
    model = FakeModel(res)
    seg = YoloSegSegmentation(cfg, model=model).segment(frame)
    lab = seg.label_map
    assert lab.shape == (H, W) and lab.dtype == np.int32
    ids = [s.id for s in seg.segments]
    assert ids == [1, 2]
    assert sorted(set(np.unique(lab)) - {0}) == ids
    assert [s.label for s in seg.segments] == ["chair", "person"]
    assert all(s.is_thing for s in seg.segments)
    assert seg.segments[0].score == pytest.approx(0.9)
    assert lab[25, 25] == 1  # overlap -> higher score
    assert lab[12, 12] == 2
    assert lab[52, 72] == 0
    assert seg.frame_index == 11
    source, kwargs = model.calls[0]
    assert source is frame.image
    assert kwargs["conf"] == pytest.approx(cfg.segmentation.score_thr)


def test_seg_names_from_model():
    cfg = load_config(profile="fast")
    m = np.zeros((1, 20, 20), np.float32)
    m[0, 5:10, 5:10] = 1
    res = seg_result(m, [0.9], [1], names=None)
    seg = YoloSegSegmentation(cfg, model=FakeModel(res, names=["person", "bicycle"])).segment(make_frame(20, 20))
    assert seg.segments[0].label == "bicycle"


def test_seg_unletterbox():
    lab = np.zeros((384, 640), np.int32)
    lab[112:212, 50:150] = 3
    out = _unletterbox(lab, (720, 1280))
    assert out.shape == (720, 1280) and out.dtype == np.int32
    assert out[210, 110] == 3 and out[390, 290] == 3 and out[300, 200] == 3
    assert out[190, 200] == 0 and out[410, 200] == 0 and out[300, 90] == 0 and out[300, 310] == 0
    same = np.ones((5, 5), np.int32)
    assert _unletterbox(same, (5, 5)) is same

    # end to end: letterboxed masks map to frame size
    cfg = load_config(profile="fast")
    m = np.zeros((1, 384, 640), np.float32)
    m[0, 112:212, 50:150] = 1
    seg = YoloSegSegmentation(cfg, model=FakeModel(seg_result(m, [0.8], [0], {0: "person"}))).segment(
        make_frame(1280, 720)
    )
    assert seg.label_map.shape == (720, 1280)
    assert seg.label_map[300, 200] == 1 and seg.label_map[100, 200] == 0


def test_seg_empty():
    cfg = load_config(profile="fast")
    frame = make_frame(w=50, h=40)
    for res in (
        SimpleNamespace(masks=None, boxes=SimpleNamespace(conf=np.zeros(0), cls=np.zeros(0)), names={}),
        seg_result(np.zeros((0, 40, 50)), [], []),
        seg_result(np.ones((1, 40, 50)), [0.01], [0]),
    ):
        seg = YoloSegSegmentation(cfg, model=FakeModel(res)).segment(frame)
        assert seg.label_map.shape == (40, 50) and seg.label_map.dtype == np.int32
        assert not seg.label_map.any() and seg.segments == []


# --------------------------------------------------------------------------- head proxy gaze
def test_head_proxy_gaze():
    cfg = load_config(profile="fast", overrides=["gaze.backend=head_proxy", "gaze.proxy_confidence=0.4"])
    d = np.array([0.6, 0.0, -0.8])
    p1 = Person(1, PersonDetection(head_box=[0, 0, 5, 5]), head=DirectionEstimate(d, confidence=0.75))
    p2 = Person(2, PersonDetection(head_box=[0, 0, 5, 5]))
    frame = make_frame(index=4)
    out = HeadProxyGaze(cfg).estimate(frame, [p1, p2])
    assert set(out) == {1}
    np.testing.assert_allclose(out[1].direction, d)
    assert out[1].direction is not p1.head.direction
    assert out[1].confidence == pytest.approx(0.3)
    assert out[1].frame_index == 4


# --------------------------------------------------------------------------- ultralytics loading
def test_ultralytics_missing(monkeypatch):
    monkeypatch.delenv("YOLO_AUTOINSTALL", raising=False)
    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            monkeypatch.delitem(sys.modules, name)
    seen = []

    class Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "ultralytics" or name.startswith("ultralytics."):
                seen.append(os.environ.get("YOLO_AUTOINSTALL"))
                raise ImportError("blocked for test")
            return None

    monkeypatch.setattr(sys, "meta_path", [Blocker()] + sys.meta_path)
    cfg = load_config(profile="fast")
    with pytest.raises(BackendUnavailable) as err:
        YoloPosePersons(cfg).load()
    msg = str(err.value)
    assert "ultralytics" in msg and "requirements-fast.txt" in msg and "AGPL" in msg
    assert seen and seen[0] == "false"
    with pytest.raises(BackendUnavailable):
        YoloSegSegmentation(cfg).load()


@pytest.fixture
def fake_ultralytics(monkeypatch):
    built = []

    class YOLO:
        fail = False

        def __init__(self, path):
            if YOLO.fail:
                raise RuntimeError("corrupt weights")
            built.append(path)
            self.path = path
            self.names = {0: "person"}

        def predict(self, source, **kwargs):
            return []

        def track(self, *a, **k):  # pragma: no cover
            raise AssertionError("track must never be called")

    mod = types.ModuleType("ultralytics")
    mod.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)
    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    return YOLO, built


def test_weights_resolution(fake_ultralytics, tmp_path, monkeypatch):
    YOLO, built = fake_ultralytics
    wdir = tmp_path / "w"
    cfg = load_config(profile="fast", overrides={"models": {"weights_dir": str(wdir), "allow_download": False,
                                                            "device": "cpu"}})
    with pytest.raises(BackendUnavailable) as err:
        YoloPosePersons(cfg).load()
    assert "yolo11n-pose.pt" in str(err.value) and "allow_download" in str(err.value)
    assert built == []

    cfg_dl = load_config(profile="fast", overrides={"models": {"weights_dir": str(wdir), "allow_download": True,
                                                               "device": "cpu"}})
    be = YoloPosePersons(cfg_dl)
    be.load()
    be.load()
    assert built == [str(wdir / "yolo11n-pose.pt")]
    assert wdir.is_dir()

    existing = tmp_path / "custom-seg.pt"
    existing.write_bytes(b"x")
    cfg_ex = load_config(profile="fast", overrides={"segmentation": {"model": str(existing)},
                                                    "models": {"allow_download": False, "device": "cpu"}})
    YoloSegSegmentation(cfg_ex).load()
    assert built[-1] == str(existing)

    YOLO.fail = True
    try:
        with pytest.raises(BackendUnavailable) as err:
            YoloPosePersons(cfg_dl).load()
        assert "corrupt weights" in str(err.value)
    finally:
        YOLO.fail = False

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "yolo11n-pose.pt").write_bytes(b"x")
    monkeypatch.setenv("VFOA_WEIGHTS_DIR", str(env_dir))
    YoloPosePersons(cfg).load()  # allow_download false, but the env dir has the file
    assert built[-1] == str(env_dir / "yolo11n-pose.pt")


def test_to_numpy_and_predict():
    np.testing.assert_array_equal(_ultra.to_numpy(FakeTensor([1, 2])), [1, 2])
    arr = np.ones(3)
    assert _ultra.to_numpy(arr) is arr
    model = FakeModel("r")
    assert _ultra.predict(model, arr, device=None, imgsz=32) == "r"
    assert "device" not in model.calls[0][1]
    _ultra.predict(model, arr, device="cpu")
    assert model.calls[1][1]["device"] == "cpu"


# --------------------------------------------------------------------------- real model (opt-in)
@pytest.mark.network
def test_real_yolo_pose_museum(repo_root, tmp_path):
    pytest.importorskip("ultralytics")
    wdir = os.environ.get("VFOA_WEIGHTS_DIR") or str(tmp_path)
    cfg = load_config(profile="fast", overrides={"models": {"weights_dir": wdir, "device": "cpu"}})
    cap = cv2.VideoCapture(str(repo_root / "Final_videos" / "museum.mp4"))
    ok, img = cap.read()
    cap.release()
    assert ok
    be = YoloPosePersons(cfg)
    try:
        be.load()
    except BackendUnavailable as exc:
        pytest.skip(str(exc))
    dets = be.detect(Frame(index=0, t=0.0, image=img))
    assert any(d.head_box is not None for d in dets)
