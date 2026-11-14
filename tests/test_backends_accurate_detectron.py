"""Tests of the accurate detectron2/OpenPose backends with fake models (Refs #9, #13, #18, #32)."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vfoa.backends import create_backend
from vfoa.backends.accurate import openpose_densepose as opd
from vfoa.backends.accurate import panoptic_fpn as pfpn
from vfoa.backends.accurate.openpose_densepose import (
    OPENPOSE_TO_COCO17,
    OpenPoseDensePosePersons,
    OpenPoseRuntime,
    densepose_part_labels,
    head_box_from_parts,
    match_skeletons_to_head_boxes,
    openpose_humans_to_coco17,
    openpose_preprocess,
)
from vfoa.backends.accurate.panoptic_fpn import (
    Detectron2Runtime,
    PanopticFPNSegmentation,
    panoptic_to_segmentation,
    run_detectron2,
    to_model_format,
)
from vfoa.backends.base import BackendUnavailable
from vfoa.types import Frame

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- helpers


def _frame(h: int = 100, w: int = 120, index: int = 0) -> Frame:
    rng = np.random.default_rng(index)
    img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    img[..., 0] = 10  # make channels distinguishable
    img[..., 2] = 200
    return Frame(index=index, t=index / 30.0, image=img)


class RecordingModel:
    """Fake detectron2 model: records its inputs, returns ``make_output(inputs)``."""

    def __init__(self, make_output):
        self.make_output = make_output
        self.calls: list = []

    def __call__(self, batch):
        self.calls.append(batch)
        return [self.make_output(item) for item in batch]


def _part(x: float, y: float, score: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, score=score)


def _densepose_output(boxes, head_rects, scores=None):
    """Fake DensePose instances: per box an IUV logit map with head parts in head_rects.

    head_rects[j] = (c0, c1, r0, r1) inclusive in box-local pixels, or None.
    """
    import torch

    n = len(boxes)
    if n == 0:
        tensor = torch.zeros((0, 4))
        dp = SimpleNamespace(S=torch.zeros((0, 2, 4, 4)), I=torch.zeros((0, 25, 4, 4)))
        return {"instances": SimpleNamespace(pred_boxes=SimpleNamespace(tensor=tensor), scores=torch.zeros(0), pred_densepose=dp)}
    # All boxes share the same size so the logits stack; interpolation to the same size is identity.
    w = int(round(boxes[0][2] - boxes[0][0]))
    h = int(round(boxes[0][3] - boxes[0][1]))
    S = torch.zeros((n, 2, h, w))
    S[:, 1] = 1.0
    I = torch.zeros((n, 25, h, w))  # noqa: E741
    I[:, 1] = 1.0
    for j, rect in enumerate(head_rects):
        if rect is None:
            continue
        c0, c1, r0, r1 = rect
        I[j, 23, r0 : r1 + 1, c0 : c1 + 1] = 5.0
    dp = SimpleNamespace(S=S, I=I)
    inst = SimpleNamespace(
        pred_boxes=SimpleNamespace(tensor=torch.tensor(boxes, dtype=torch.float32)),
        scores=torch.tensor(scores if scores is not None else [0.9] * n, dtype=torch.float32),
        pred_densepose=dp,
    )
    return {"instances": inst}


def _fake_openpose(humans_fn, calls=None):
    import torch

    def model(x):
        if calls is not None:
            calls.append(x)
        n, _, h, w = x.shape
        paf = torch.zeros((n, 38, h // 8, w // 8))
        heat = torch.zeros((n, 19, h // 8, w // 8))
        return (paf, heat), None

    seen = []

    def paf_to_pose(heat, paf):
        seen.append((heat.shape, paf.shape))
        return humans_fn()

    rt = OpenPoseRuntime(model=model, paf_to_pose=paf_to_pose)
    rt.seen = seen  # type: ignore[attr-defined]
    return rt


# --------------------------------------------------------------------------- imports / registry


def _clean_import(cwd: Path, module: str, attr: str) -> None:
    code = (
        f"import sys; import {module} as m; assert hasattr(m, {attr!r}); "
        "bad = [k for k in ('torch', 'detectron2', 'densepose', 'lib', 'ultralytics') if k in sys.modules]; "
        "assert not bad, bad"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "VFOA_THIRD_PARTY"}
    env["PYTHONPATH"] = str(REPO)
    res = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr


@pytest.mark.parametrize(
    "cwd,module,attr",
    [
        (REPO, "vfoa.backends.accurate.openpose_densepose", "OpenPoseDensePosePersons"),
        (REPO, "vfoa.backends.accurate.panoptic_fpn", "PanopticFPNSegmentation"),
        (REPO / "thesis_final", "OpenPosePredictor", "OpenPosePredictor"),
        (REPO / "thesis_final", "DensePosePredictor", "DensePosePredictor"),
        (REPO / "thesis_final", "PanopticPrediction", "PanopticPredictor"),
    ],
)
def test_imports_clean(cwd, module, attr):
    _clean_import(cwd, module, attr)


def test_create_and_unavailable(make_config, monkeypatch):
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    # Force the missing-package condition so the test does not depend on
    # whether detectron2 / OpenPose happen to be importable on this machine.
    for name in ("detectron2", "detectron2.checkpoint", "lib", "lib.config.default"):
        monkeypatch.setitem(sys.modules, name, None)
    cfg = make_config("accurate")
    persons = create_backend("persons", cfg)
    seg = create_backend("segmentation", cfg)
    assert isinstance(persons, OpenPoseDensePosePersons)
    assert isinstance(seg, PanopticFPNSegmentation)
    assert not persons.is_loaded and not seg.is_loaded

    for backend in (persons, seg):
        with pytest.raises(BackendUnavailable) as err:
            backend.load()
        msg = str(err.value)
        assert "detectron2" in msg and "models.third_party_root" in msg and "VFOA_THIRD_PARTY" in msg
        assert not backend.is_loaded

    rt = Detectron2Runtime(model=RecordingModel(lambda item: {}))
    p2 = OpenPoseDensePosePersons(cfg, densepose_factory=lambda c: rt)
    with pytest.raises(BackendUnavailable) as err:
        p2.load()
    msg = str(err.value)
    assert "Pose_Estimation" in msg and "models.third_party_root" in msg and "VFOA_THIRD_PARTY" in msg
    assert not p2.is_loaded


# --------------------------------------------------------------------------- colour order (#9)


def test_to_model_format():
    img = _frame().image
    assert to_model_format(img, "BGR") is img
    np.testing.assert_array_equal(to_model_format(img, "rgb"), img[:, :, ::-1])
    with pytest.raises(ValueError):
        to_model_format(img, "YUV")


@pytest.mark.parametrize("fmt", ["BGR", "RGB"])
@pytest.mark.parametrize("which", ["persons", "segmentation"])
def test_channel_order(fmt, which, make_config):
    frame = _frame()
    original = frame.image.copy()
    expected = original if fmt == "BGR" else original[:, :, ::-1]
    cfg = make_config("accurate", {"persons": {"match_keypoints": False}})
    h, w = original.shape[:2]

    for resize in (None, lambda im: im[::2, ::2]):
        if which == "persons":
            model = RecordingModel(lambda item: _densepose_output([], []))
            rt = Detectron2Runtime(model=model, input_format=fmt, resize=resize)
            backend = OpenPoseDensePosePersons(cfg, densepose_factory=lambda c: rt)
            backend.detect(frame)
        else:
            model = RecordingModel(
                lambda item: {"panoptic_seg": (np.zeros((item["height"], item["width"]), np.int32), [])}
            )
            rt = Detectron2Runtime(model=model, input_format=fmt, resize=resize)
            backend = PanopticFPNSegmentation(cfg, runtime_factory=lambda c: rt)
            backend.segment(frame)
        (item,) = model.calls[0]
        got = item["image"].numpy().transpose(1, 2, 0)
        assert item["image"].dtype.is_floating_point
        assert (item["height"], item["width"]) == (h, w)
        want = expected if resize is None else expected[::2, ::2]
        np.testing.assert_array_equal(got, want.astype(np.float32))
        np.testing.assert_array_equal(frame.image, original)


# --------------------------------------------------------------------------- load once (#32)


def test_models_built_once(make_config):
    cfg = make_config("accurate")
    counts = {"dp": 0, "op": 0, "seg": 0}
    dp_model = RecordingModel(lambda item: _densepose_output([[10, 10, 50, 70]], [(5, 20, 2, 15)]))
    op_calls: list = []

    def dp_factory(c):
        counts["dp"] += 1
        return Detectron2Runtime(model=dp_model)

    def op_factory(c):
        counts["op"] += 1
        return _fake_openpose(lambda: [], op_calls)

    backend = OpenPoseDensePosePersons(cfg, densepose_factory=dp_factory, openpose_factory=op_factory)
    backend.load()
    backend.load()
    for i in range(5):
        backend.detect(_frame(index=i))
    assert counts == {"dp": 1, "op": 1, "seg": 0}
    assert len(dp_model.calls) == 5 and len(op_calls) == 5

    seg_model = RecordingModel(lambda item: {"panoptic_seg": (np.zeros((item["height"], item["width"])), [])})

    def seg_factory(c):
        counts["seg"] += 1
        return Detectron2Runtime(model=seg_model)

    seg = PanopticFPNSegmentation(cfg, runtime_factory=seg_factory)
    seg.load()
    seg.load()
    for i in range(5):
        seg.segment(_frame(index=i))
    assert counts["seg"] == 1 and len(seg_model.calls) == 5

    counts.update(dp=0, op=0)
    cfg2 = make_config("accurate", {"persons": {"match_keypoints": False}})
    b2 = OpenPoseDensePosePersons(cfg2, densepose_factory=dp_factory, openpose_factory=op_factory)
    for i in range(5):
        b2.detect(_frame(index=i))
    assert counts["dp"] == 1 and counts["op"] == 0


# --------------------------------------------------------------------------- fake detectron2 build (#13)


class _Node:
    """Auto-attribute config node (stands in for a detectron2 CfgNode)."""

    def __init__(self):
        object.__setattr__(self, "_frozen", False)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        child = _Node()
        object.__setattr__(self, name, child)
        return child

    def __setattr__(self, name, value):
        if self._frozen:
            raise RuntimeError("frozen")
        object.__setattr__(self, name, value)

    def merge_from_file(self, path):
        self.merged = path
        self.INPUT.FORMAT = "RGB"
        self.INPUT.MIN_SIZE_TEST = 800
        self.INPUT.MAX_SIZE_TEST = 1333
        self.DATASETS.TEST = ["coco_2017_val_panoptic"]

    def freeze(self):
        object.__setattr__(self, "_frozen", True)


def _install_fake_detectron2(monkeypatch, record):
    class FakeModel:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True
            return self

        def cuda(self, *a, **k):
            raise AssertionError(".cuda() must not be called")

        def __call__(self, batch):
            return [{}]

    def get_cfg():
        node = _Node()
        record["cfg"] = node
        return node

    def build_model(d2):
        record["build_model"] = record.get("build_model", 0) + 1
        record["device_at_build"] = d2.MODEL.DEVICE
        record["model"] = FakeModel()
        return record["model"]

    class DetectionCheckpointer:
        def __init__(self, model):
            self.model = model

        def load(self, path):
            record["checkpoint"] = path

    class ResizeShortestEdge:
        def __init__(self, short, maxsize):
            record["resize"] = (short, maxsize)

        def get_transform(self, img):
            return SimpleNamespace(apply_image=lambda im: im)

    MetadataCatalog = SimpleNamespace(
        get=lambda name: SimpleNamespace(thing_classes=["person"], stuff_classes=["things", "wall"])
    )
    mods = {
        "detectron2": types.ModuleType("detectron2"),
        "detectron2.config": types.ModuleType("detectron2.config"),
        "detectron2.modeling": types.ModuleType("detectron2.modeling"),
        "detectron2.checkpoint": types.ModuleType("detectron2.checkpoint"),
        "detectron2.data": types.ModuleType("detectron2.data"),
        "detectron2.data.transforms": types.ModuleType("detectron2.data.transforms"),
    }
    mods["detectron2.config"].get_cfg = get_cfg
    mods["detectron2.modeling"].build_model = build_model
    mods["detectron2.checkpoint"].DetectionCheckpointer = DetectionCheckpointer
    mods["detectron2.data"].MetadataCatalog = MetadataCatalog
    mods["detectron2.data.transforms"].ResizeShortestEdge = ResizeShortestEdge
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_build_runtime_with_fake_detectron2(monkeypatch, tmp_path, make_config):
    import torch

    def boom(*a, **k):
        raise AssertionError("device_count must not be used")

    monkeypatch.setattr(torch.cuda, "device_count", boom)
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    cfg_file = tmp_path / "panoptic.yaml"
    weights = tmp_path / "model.pkl"
    cfg_file.write_text("x: 1\n")
    weights.write_bytes(b"0")
    record: dict = {}
    _install_fake_detectron2(monkeypatch, record)
    cfg = make_config(
        "accurate",
        {
            "models": {"device": "cpu"},
            "segmentation": {"panoptic_cfg": str(cfg_file), "panoptic_weights": str(weights), "panoptic_thr": 0.42},
        },
    )
    seg = PanopticFPNSegmentation(cfg)
    seg.load()
    rt = seg.runtime
    assert record["build_model"] == 1
    assert record["device_at_build"] == "cpu" and rt.device == "cpu"
    d2 = record["cfg"]
    assert d2.MODEL.WEIGHTS == str(weights) and record["checkpoint"] == str(weights)
    assert d2.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH == pytest.approx(0.42)
    assert record["model"].evaluated
    assert rt.input_format == "RGB"
    assert record["resize"] == ([800, 800], 1333)
    assert tuple(rt.stuff_classes) == ("things", "wall")
    seg.load()
    assert record["build_model"] == 1

    # auto device resolves through torch.cuda.is_available, never device_count.
    cfg_auto = make_config(
        "accurate",
        {"segmentation": {"panoptic_cfg": str(cfg_file), "panoptic_weights": str(weights)}},
    )
    rt2 = pfpn.build_panoptic_runtime(cfg_auto)
    assert rt2.device in ("cpu", "cuda")

    # Missing weights -> BackendUnavailable naming the key.
    cfg_bad = make_config(
        "accurate",
        {"segmentation": {"panoptic_cfg": str(cfg_file), "panoptic_weights": str(tmp_path / "nope.pkl")}},
    )
    with pytest.raises(BackendUnavailable, match="segmentation.panoptic_weights"):
        pfpn.build_panoptic_runtime(cfg_bad)

    # DensePose runtime: densepose package missing -> BackendUnavailable mentioning it.
    monkeypatch.setitem(sys.modules, "densepose", None)
    cfg_dp = make_config(
        "accurate", {"persons": {"densepose_cfg": str(cfg_file), "densepose_weights": str(weights)}}
    )
    with pytest.raises(BackendUnavailable, match="DensePose"):
        opd.build_densepose_runtime(cfg_dp)

    dp_mod = types.ModuleType("densepose")
    dp_cfg_mod = types.ModuleType("densepose.config")
    dp_cfg_mod.add_densepose_config = lambda node: setattr(node, "densepose_added", True)
    monkeypatch.setitem(sys.modules, "densepose", dp_mod)
    monkeypatch.setitem(sys.modules, "densepose.config", dp_cfg_mod)
    opd.build_densepose_runtime(cfg_dp)
    d2 = record["cfg"]
    assert d2.densepose_added is True
    assert d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST == pytest.approx(0.85)


def test_no_device_count_in_sources():
    for path in (
        REPO / "vfoa/backends/accurate/openpose_densepose.py",
        REPO / "vfoa/backends/accurate/panoptic_fpn.py",
        REPO / "thesis_final/OpenPosePredictor.py",
        REPO / "thesis_final/DensePosePredictor.py",
        REPO / "thesis_final/PanopticPrediction.py",
    ):
        text = path.read_text()
        for bad in ("device_count", "DataParallel", ".cuda(", "/home/carlos", "float16", "set_start_method"):
            assert bad not in text, (path.name, bad)


# --------------------------------------------------------------------------- DensePose post-processing


def test_head_box():
    labels = np.zeros((30, 40), dtype=np.int64)
    labels[4:14, 8:18] = 23
    labels[6, 10] = 24
    labels[20:25, 5:30] = 2  # body part, ignored
    box = head_box_from_parts(labels, (100, 50), (1000, 1000))
    np.testing.assert_allclose(box, [106.65, 52.65, 118.35, 64.35], atol=1e-4)
    assert box.dtype == np.float32

    # Clipping at the image border.
    box = head_box_from_parts(labels, (-10, -5), (5, 1000))
    np.testing.assert_allclose(box, [0.0, 0.0, 5.0, 9.35], atol=1e-4)

    assert head_box_from_parts(np.zeros((30, 40), np.int64), (0, 0), (100, 100)) is None
    single = np.zeros((30, 40), dtype=np.int64)
    single[4:14, 8] = 24
    assert head_box_from_parts(single, (0, 0), (100, 100)) is None
    # Entirely outside the image after clipping -> None.
    assert head_box_from_parts(labels, (500, 0), (100, 100)) is None


def test_part_labels():
    import torch

    S = torch.zeros((2, 2, 6, 8))
    S[:, 1, :, :4] = 1.0  # foreground on the left half only
    I = torch.zeros((2, 25, 6, 8))  # noqa: E741
    I[:, 23] = 1.0
    dp = SimpleNamespace(S=S, I=I)
    lab = densepose_part_labels(dp, 1, 6, 8)
    assert lab.shape == (6, 8)
    assert (lab[:, :4] == 23).all() and (lab[:, 4:] == 0).all()
    assert densepose_part_labels(dp, 0, 0, 8) is None
    dp2 = SimpleNamespace(coarse_segm=S, fine_segm=I)
    np.testing.assert_array_equal(densepose_part_labels(dp2, 0, 6, 8), lab)


# --------------------------------------------------------------------------- OpenPose pre/post-processing


def test_openpose_preprocess():
    img = np.zeros((100, 130, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue
    chw, scale, (net_h, net_w) = openpose_preprocess(img, 368, 8)
    assert scale == pytest.approx(3.68)
    assert chw.dtype == np.float32 and chw.shape == (3, net_h, net_w)
    assert net_h % 8 == 0 and net_w % 8 == 0
    assert net_h == 368 and net_w == 480  # 130 * 3.68 = 478.4 -> 478 -> 480
    assert chw[0, 0, 0] == pytest.approx(255 / 256 - 0.5)  # BGR kept: channel 0 is blue
    assert chw[2, 0, 0] == pytest.approx(-0.5)
    assert np.all(chw[:, :, 478:] == -0.5)


def test_humans_to_coco17():
    net = (480, 368)
    scale = 2.0
    parts = {
        0: _part(0.1, 0.2, 0.8),  # nose
        14: _part(0.12, 0.18, 0.7),  # r_eye -> COCO 2
        17: _part(0.15, 0.19, 2.0),  # l_ear -> COCO 3 (conf clipped to 1)
        9: _part(0.3, 0.8, 0.9),  # r_knee -> COCO 14 (lower body)
        1: _part(0.2, 0.3, 0.9),  # neck (dropped)
    }
    human = SimpleNamespace(body_parts=parts)
    (kp,) = openpose_humans_to_coco17([human], net, scale=scale, upper_body_only=False)
    assert kp.dtype == np.float32 and kp.shape == (17, 3)
    np.testing.assert_allclose(kp[0], [0.1 * 480 / 2, 0.2 * 368 / 2, 0.8], rtol=1e-6)
    np.testing.assert_allclose(kp[2], [0.12 * 480 / 2, 0.18 * 368 / 2, 0.7], rtol=1e-6)
    assert kp[3, 2] == pytest.approx(1.0)
    np.testing.assert_allclose(kp[14], [0.3 * 240, 0.8 * 184, 0.9], rtol=1e-6)
    assert kp[1, 2] == 0 and kp[5, 2] == 0  # missing parts
    assert OPENPOSE_TO_COCO17[14] == 9 and OPENPOSE_TO_COCO17[2] == 14

    (kp2,) = openpose_humans_to_coco17([human], net, scale=scale, upper_body_only=True)
    assert np.all(kp2[13:17, 2] == 0)
    assert kp2[0, 2] > 0

    # Outside the image -> dropped; human with nothing left -> skipped.
    lower_only = SimpleNamespace(body_parts={10: _part(0.5, 0.5)})
    assert openpose_humans_to_coco17([lower_only], net, scale, upper_body_only=True) == []
    (kp3,) = openpose_humans_to_coco17([human], net, scale, upper_body_only=False, image_size=(25, 1000))
    assert kp3[0, 2] > 0 and kp3[2, 2] == 0 and kp3[14, 2] == 0 and kp3[3, 2] == 0


# --------------------------------------------------------------------------- matching (#18)


def _skeleton(head_pts):
    kp = np.zeros((17, 3), np.float32)
    for idx, (x, y) in head_pts.items():
        kp[idx] = (x, y, 0.9)
    return kp


def test_matching():
    boxes = [np.array([0, 0, 20, 20], np.float32), np.array([50, 0, 70, 20], np.float32)]
    headless = np.zeros((17, 3), np.float32)
    headless[5] = (10, 10, 0.9)  # a shoulder inside box 0, but no head keypoints
    sk_b1 = _skeleton({0: (60, 10), 1: (58, 8)})
    m = match_skeletons_to_head_boxes([headless, sk_b1], boxes)
    assert m == {1: 1}

    # Two skeletons competing for one box: one-to-one, the better (more inside) wins.
    good = _skeleton({0: (10, 10), 1: (8, 8), 2: (12, 8)})
    half = _skeleton({0: (19, 10), 1: (30, 8)})
    m = match_skeletons_to_head_boxes([half, good], [boxes[0]])
    assert m == {0: 1}
    m = match_skeletons_to_head_boxes([good, good.copy()], [boxes[0]])
    assert len(m) == 1 and set(m.values()) <= {0, 1}

    # None boxes and no skeletons.
    assert match_skeletons_to_head_boxes([good], [None]) == {}
    assert match_skeletons_to_head_boxes([], boxes) == {}
    # Skeleton outside every box matches nothing.
    assert match_skeletons_to_head_boxes([_skeleton({0: (200, 200)})], boxes) == {}


def test_detect_end_to_end(make_config):
    import torch  # noqa: F401

    H, W = 100, 120
    frame = _frame(H, W)
    boxes = [[20, 10, 60, 70], [70, 10, 110, 70], [0, 0, 40, 60]]
    rects = [(10, 29, 5, 19), (10, 29, 5, 19), None]  # third instance: no head
    dp_model = RecordingModel(lambda item: _densepose_output(boxes, rects, [0.95, 0.9, 0.8]))
    _, scale, (net_h, net_w) = openpose_preprocess(frame.image)

    def norm(x, y):
        return _part(x * scale / net_w, y * scale / net_h)

    def humans():
        return [
            SimpleNamespace(body_parts={0: norm(40, 20), 15: norm(42, 18), 9: norm(40, 90)}),  # head of instance 0
            SimpleNamespace(body_parts={2: norm(90, 40)}),  # no head keypoints -> unmatched
        ]

    cfg = make_config("accurate")
    op = _fake_openpose(humans)
    backend = OpenPoseDensePosePersons(cfg, densepose_factory=lambda c: Detectron2Runtime(model=dp_model),
                                       openpose_factory=lambda c: op)
    dets = backend.detect(frame)
    assert len(dets) == 2
    expected0 = [20 + 10 - 2.85, 10 + 5 - 2.1, 20 + 29 + 2.85, 10 + 19 + 2.1]
    np.testing.assert_allclose(dets[0].head_box, expected0, atol=1e-4)
    np.testing.assert_allclose(dets[0].body_box, [20, 10, 60, 70])
    assert dets[0].score == pytest.approx(0.95)
    kp = dets[0].keypoints
    assert kp is not None and kp.shape == (17, 3)
    np.testing.assert_allclose(kp[0, :2], [40, 20], atol=1e-3)
    np.testing.assert_allclose(kp[1, :2], [42, 18], atol=1e-3)
    assert kp[14, 2] == 0  # lower body removed by default
    assert dets[1].keypoints is None
    # heatmap (19 ch) and PAF (38 ch) passed in the right order, HWC.
    heat_shape, paf_shape = op.seen[0]
    assert heat_shape[-1] == 19 and paf_shape[-1] == 38

    # keep_headless keeps the third instance with head_box None.
    cfg2 = make_config("accurate", {"persons": {"keep_headless": True}})
    b2 = OpenPoseDensePosePersons(cfg2, densepose_factory=lambda c: Detectron2Runtime(model=dp_model),
                                  openpose_factory=lambda c: _fake_openpose(humans))
    dets2 = b2.detect(frame)
    assert len(dets2) == 3 and dets2[2].head_box is None and dets2[2].keypoints is None

    # No head box anywhere -> OpenPose is not run.
    calls: list = []
    no_head = RecordingModel(lambda item: _densepose_output([[20, 10, 60, 70]], [None]))
    b3 = OpenPoseDensePosePersons(cfg, densepose_factory=lambda c: Detectron2Runtime(model=no_head),
                                  openpose_factory=lambda c: _fake_openpose(humans, calls))
    assert b3.detect(frame) == []
    assert calls == []
    b3.close()
    assert not b3.is_loaded


# --------------------------------------------------------------------------- panoptic


def test_panoptic_to_segmentation():
    import torch

    pan = np.zeros((6, 8), np.int64)
    pan[0:2, 0:3] = 1  # thing: person
    pan[3:6, :] = 2  # stuff: wall
    pan[2, 5] = 7  # id not in segments_info -> 0
    pan[0, 7] = 4  # thing with out-of-range category
    info = [
        {"id": 1, "isthing": True, "category_id": 0, "score": 0.9},
        {"id": 2, "isthing": False, "category_id": 1, "area": 24},
        {"id": 3, "isthing": True, "category_id": 0},  # not present in the map -> dropped
        {"id": 4, "isthing": True, "category_id": 99},
    ]
    seg = panoptic_to_segmentation(torch.as_tensor(pan), info, ["person"], ["things", "wall"], frame_index=5)
    assert seg.label_map.dtype == np.int32 and seg.label_map.shape == (6, 8)
    assert seg.frame_index == 5
    assert seg.label_map[2, 5] == 0
    assert {s.id for s in seg.segments} == {1, 2, 4}
    by_id = {s.id: s for s in seg.segments}
    assert by_id[1].label == "person" and by_id[1].is_thing and by_id[1].score == pytest.approx(0.9)
    assert by_id[2].label == "wall" and not by_id[2].is_thing and by_id[2].score == 1.0
    assert by_id[4].label == "thing_99"
    assert set(np.unique(seg.label_map).tolist()) == {0, 1, 2, 4}

    empty = panoptic_to_segmentation(np.zeros((4, 5)), [], [], [])
    assert empty.segments == [] and empty.label_map.shape == (4, 5) and not empty.label_map.any()

    stuff_oob = panoptic_to_segmentation(pan, [{"id": 2, "isthing": False, "category_id": 5}], [], [])
    assert stuff_oob.segments[0].label == "stuff_5"


def test_panoptic_backend(make_config):
    H, W = 40, 60
    small = np.zeros((H // 2, W // 2), np.int32)
    small[:, : W // 4] = 3

    def out(item):
        return {"panoptic_seg": (small, [{"id": 3, "isthing": False, "category_id": 0}])}

    rt = Detectron2Runtime(model=RecordingModel(out), stuff_classes=["floor"])
    seg = PanopticFPNSegmentation(make_config("accurate"), runtime_factory=lambda c: rt)
    res = seg.segment(_frame(H, W, index=17))
    assert res.frame_index == 17
    assert res.label_map.shape == (H, W) and res.label_map.dtype == np.int32
    assert (res.label_map[:, : W // 2] == 3).all() and (res.label_map[:, W // 2 :] == 0).all()
    assert res.segments[0].label == "floor"
    seg.close()
    assert not seg.is_loaded and seg.runtime is None


def test_run_detectron2_keeps_original_size():
    img = _frame(30, 50).image
    model = RecordingModel(lambda item: {"ok": True})
    out = run_detectron2(Detectron2Runtime(model=model, input_format="RGB", resize=lambda im: im[::3, ::3]), img)
    assert out == {"ok": True}
    (item,) = model.calls[0]
    assert (item["height"], item["width"]) == (30, 50)
    assert tuple(item["image"].shape) == (3, 10, 17)


# --------------------------------------------------------------------------- legacy wrappers


def test_legacy_wrappers(make_config):
    from DensePosePredictor import DensePosePredictor
    from OpenPosePredictor import OpenPosePredictor
    from PanopticPrediction import PanopticPredictor

    frame = _frame()
    boxes = [[20, 10, 60, 70]]
    dp_model = RecordingModel(lambda item: _densepose_output(boxes, [(10, 29, 5, 19)]))
    op_calls: list = []
    dp = DensePosePredictor(
        "cfg.yaml", "w.pkl", 0.7, 2,
        densepose_factory=lambda c: Detectron2Runtime(model=dp_model),
        openpose_factory=lambda c: pytest.fail("OpenPose must not be built"),
    )
    assert dp.cfg.persons.densepose_cfg == "cfg.yaml" and dp.cfg.persons.densepose_thr == pytest.approx(0.7)
    (heads,) = dp([frame.image])
    assert len(heads) == 1 and heads[0] is not None
    labels = np.zeros((30, 40), np.int64)
    labels[4:14, 8:18] = 23
    np.testing.assert_allclose(DensePosePredictor.extractHeadBoxFromOutput([100, 50, 140, 80], labels),
                               [106.65, 52.65, 118.35, 64.35], atol=1e-4)

    args = SimpleNamespace(cfg="exp.yaml", opM="pose.pth", opB=4)
    op = OpenPosePredictor(args, openpose_factory=lambda c: _fake_openpose(lambda: [], op_calls),
                           densepose_factory=lambda c: pytest.fail("DensePose must not be built"))
    assert op.cfg.persons.openpose_weights == "pose.pth"
    op.initModelForLoop()
    assert op([frame.image, frame.image]) == [[], []]
    assert op.predictIteration([frame.image]) == [[]]
    assert len(op_calls) == 3
    assert OpenPosePredictor.bottomBodyPartsIndex == [9, 10, 12, 13, 18]

    rt = Detectron2Runtime(model=RecordingModel(
        lambda item: {"panoptic_seg": (np.ones((item["height"], item["width"]), np.int32),
                                       [{"id": 1, "isthing": True, "category_id": 0}])}),
        thing_classes=["person"])
    pp = PanopticPredictor("pan.yaml", "pan.pkl", 0.3, 1, runtime_factory=lambda c: rt)
    assert pp.cfg.segmentation.panoptic_thr == pytest.approx(0.3)
    segs, runtime = pp([frame.image, frame.image])
    assert runtime is rt and [s.frame_index for s in segs] == [0, 1]
    assert segs[0].segments[0].label == "person"
