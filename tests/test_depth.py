"""Tests of the depth backends (stereo block matching, Sintel ground truth, constant)."""

from __future__ import annotations

import logging
import types

import cv2
import numpy as np
import pytest

from vfoa.backends import create_backend
from vfoa.backends.base import BackendUnavailable
from vfoa.backends.depth_simple import ConstantDepth, SintelGTDepth
from vfoa.backends.depth_stereo import StereoBMDepth, synthesize_right_view
from vfoa.io.sintel import depth_read
from vfoa.types import Frame, Intrinsics

_RIGHT_CACHE: dict = {}


@pytest.fixture
def stereo_sample(sintel_sample):
    """``stereo_sample(index) -> (left, right, gt_depth, intrinsics)`` with a synthesised right view."""

    def _get(index: int = 1):
        left, gt, intr = sintel_sample(index)
        if index not in _RIGHT_CACHE:
            _RIGHT_CACHE[index] = synthesize_right_view(left, gt, intr)
        return left, _RIGHT_CACHE[index].copy(), gt, intr

    return _get


def _frame(left, right, intr, index=0):
    return Frame(index=index, t=0.0, image=left, right=right, intrinsics=intr)


def _stereo_cfg(make_config, **depth):
    return make_config("fast", {"depth": {"backend": "stereo_bm", **depth}})


# ------------------------------------------------------------ synthesis
def test_synthesize_constant_depth_is_pure_shift(rng):
    h, w = 20, 64
    left = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    intr = Intrinsics(100.0, 100.0, (w - 1) / 2, (h - 1) / 2, w, h, baseline=0.1)
    depth = np.full((h, w), 100.0 * 0.1 / 8.0, dtype=np.float32)  # disparity exactly 8
    right = synthesize_right_view(left, depth, intr)
    assert right.shape == left.shape and right.dtype == np.uint8
    np.testing.assert_array_equal(right[:, : w - 8], left[:, 8:])
    # the right border is a hole, filled from the nearest filled pixel on the left
    np.testing.assert_array_equal(right[:, w - 8:], np.repeat(left[:, w - 1: w], 8, axis=1))


def test_synthesize_zbuffer_and_hole_fill():
    h, w = 30, 80
    left = np.zeros((h, w, 3), np.uint8)
    left[:] = (10, 20, 30)  # background colour
    left[10:20, 30:40] = (200, 100, 50)  # near square
    intr = Intrinsics(100.0, 100.0, (w - 1) / 2, (h - 1) / 2, w, h, baseline=0.1)
    depth = np.full((h, w), 10.0 / 4.0, np.float32)  # far plane: d = 4
    depth[10:20, 30:40] = 10.0 / 12.0  # near square: d = 12
    right = synthesize_right_view(left, depth, intr)
    # the near square moved by 12 and wins over the background it lands on
    np.testing.assert_array_equal(right[10:20, 18:28], np.broadcast_to((200, 100, 50), (10, 10, 3)))
    # disocclusion to its right (right cols 28..35 receive no source pixel) takes the background
    np.testing.assert_array_equal(right[10:20, 28:36], np.broadcast_to((10, 20, 30), (10, 8, 3)))


def test_synthesize_needs_baseline():
    img = np.zeros((4, 4, 3), np.uint8)
    with pytest.raises(ValueError):
        synthesize_right_view(img, np.ones((4, 4), np.float32), Intrinsics(1, 1, 1, 1, 4, 4))


# ------------------------------------------------------------ stereo BM
@pytest.mark.parametrize("index", [1, 25])
def test_stereo_bm_matches_ground_truth(index, stereo_sample, make_config):
    left, right, gt, intr = stereo_sample(index)
    backend = StereoBMDepth(_stereo_cfg(make_config, scale=1.0))
    df = backend.compute(_frame(left, right, intr, index - 1))
    v = df.valid
    assert df.depth.shape == gt.shape and df.depth.dtype == np.float32
    assert v.mean() > 0.30
    rel = np.abs(df.depth[v] - gt[v]) / gt[v]
    assert np.median(rel) < 0.02
    assert np.percentile(rel, 90) < 0.10
    assert (df.depth[v] > 0).all()
    assert (df.depth[~v] == 0).all() and (df.disparity[~v] == 0).all()
    fb = intr.fx * intr.baseline
    np.testing.assert_allclose(df.depth[v] * df.disparity[v], fb, rtol=1e-3)
    d = df.disparity
    assert d.dtype == np.float32
    assert d.max() < 160
    assert np.allclose(d * 16, np.round(d * 16))
    assert (d[v] != np.round(d[v])).any()  # sub-pixel, not a 0-255 image
    assert np.median(np.abs(d[v] - fb / gt[v])) < 1.0
    assert df.intrinsics.fx == pytest.approx(896.0)


def test_stereo_bm_half_scale(stereo_sample, make_config):
    left, right, gt, intr = stereo_sample(1)
    df = StereoBMDepth(_stereo_cfg(make_config, scale=0.5)).compute(_frame(left, right, intr))
    assert df.depth.shape == left.shape[:2] and df.disparity.shape == left.shape[:2]
    v = df.valid
    assert v.mean() > 0.30
    assert np.median(np.abs(df.depth[v] - gt[v]) / gt[v]) < 0.05
    np.testing.assert_allclose(df.depth[v] * df.disparity[v], intr.fx * intr.baseline, rtol=1e-3)


class _SpyMatcher:
    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def compute(self, a, b):
        self._log["compute"] += 1
        return self._inner.compute(a, b)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_matcher_built_once_one_compute_per_frame(monkeypatch, stereo_sample, make_config):
    log = {"create": 0, "compute": 0}
    real = cv2.StereoBM_create

    def factory(*args, **kwargs):
        log["create"] += 1
        return _SpyMatcher(real(*args, **kwargs), log)

    monkeypatch.setattr(cv2, "StereoBM_create", factory)
    monkeypatch.delattr(cv2, "ximgproc", raising=False)
    left, right, _, intr = stereo_sample(1)
    small_l, small_r = left[:, :512].copy(), right[:, :512].copy()
    backend = StereoBMDepth(_stereo_cfg(make_config))
    for i in range(5):
        backend.compute(_frame(small_l, small_r, intr, i))
    backend.load()
    assert log == {"create": 1, "compute": 5}


def test_wls_auto_logs_once_without_ximgproc(monkeypatch, caplog, make_config):
    monkeypatch.delattr(cv2, "ximgproc", raising=False)
    img = np.random.default_rng(0).integers(0, 255, (60, 120, 3), dtype=np.uint8)
    intr = Intrinsics(100, 100, 60, 30, 120, 60, baseline=0.1)
    backend = StereoBMDepth(_stereo_cfg(make_config, wls="auto"))
    with caplog.at_level(logging.INFO, logger="vfoa.backends.depth_stereo"):
        backend.load()
        for i in range(3):
            backend.compute(_frame(img, img, intr, i))
    records = [r for r in caplog.records if "ximgproc" in r.getMessage()]
    assert len(records) == 1


@pytest.mark.parametrize("value", [True, "true"])
def test_wls_true_raises_backend_unavailable(monkeypatch, make_config, value):
    monkeypatch.delattr(cv2, "ximgproc", raising=False)
    backend = StereoBMDepth(_stereo_cfg(make_config, wls=value))
    with pytest.raises(BackendUnavailable, match="ximgproc"):
        backend.load()


def test_wls_with_fake_ximgproc(monkeypatch, make_config):
    log = {"right_create": 0, "filter_create": 0, "left": 0, "right": 0, "filter": 0}
    params = {}

    class FakeRight:
        def compute(self, a, b):
            log["right"] += 1
            return np.zeros(a.shape, np.int16)

    class FakeFilter:
        def setLambda(self, v):
            params["lambda"] = v

        def setSigmaColor(self, v):
            params["sigma"] = v

        def filter(self, disp_left, img_left, out, disp_right):
            log["filter"] += 1
            return np.full(disp_left.shape, 16 * 10, np.int16)  # 10 px everywhere

    def create_right(left):
        log["right_create"] += 1
        return FakeRight()

    def create_filter(left):
        log["filter_create"] += 1
        return FakeFilter()

    fake = types.SimpleNamespace(createRightMatcher=create_right, createDisparityWLSFilter=create_filter)
    monkeypatch.setattr(cv2, "ximgproc", fake, raising=False)
    real = cv2.StereoBM_create

    def factory(*args, **kwargs):
        m = real(*args, **kwargs)
        spy = _SpyMatcher(m, {"compute": 0})
        orig = spy.compute

        def compute(a, b):
            log["left"] += 1
            return orig(a, b)

        spy.compute = compute
        return spy

    monkeypatch.setattr(cv2, "StereoBM_create", factory)
    img = np.random.default_rng(1).integers(0, 255, (60, 120, 3), dtype=np.uint8)
    intr = Intrinsics(100, 100, 60, 30, 120, 60, baseline=0.1)
    backend = StereoBMDepth(_stereo_cfg(make_config, wls="auto", wls_lambda=1234.0, wls_sigma=0.7))
    for i in range(3):
        df = backend.compute(_frame(img, img, intr, i))
    assert params == {"lambda": 1234.0, "sigma": 0.7}
    assert log == {"right_create": 1, "filter_create": 1, "left": 3, "right": 3, "filter": 3}
    assert np.allclose(df.disparity, 10.0)
    np.testing.assert_allclose(df.depth, 100 * 0.1 / 10.0, rtol=1e-6)


def test_missing_right_or_baseline_raises(make_config, monkeypatch):
    monkeypatch.delattr(cv2, "ximgproc", raising=False)
    img = np.zeros((40, 80, 3), np.uint8)
    backend = StereoBMDepth(_stereo_cfg(make_config))
    intr = Intrinsics(100, 100, 40, 20, 80, 40, baseline=0.1)
    with pytest.raises(ValueError, match="right"):
        backend.compute(_frame(img, None, intr))
    no_baseline = StereoBMDepth(make_config("fast", {"depth": {"backend": "stereo_bm"},
                                                      "camera": {"baseline_m": None}}))
    with pytest.raises(ValueError, match="baseline"):
        no_baseline.compute(_frame(img, img, Intrinsics(100, 100, 40, 20, 80, 40)))


def test_invalid_scale_rejected(make_config):
    with pytest.raises(ValueError):
        StereoBMDepth(_stereo_cfg(make_config, scale=1.5)).load()


# ------------------------------------------------------------ Sintel GT / constant
@pytest.mark.parametrize("index", [0, 10, 49])
def test_sintel_gt_exact(index, sintel_paths, make_config):
    cfg = make_config("fast", {
        "depth": {"backend": "sintel_gt"},
        "input": {"depth_dir": str(sintel_paths["depth_dir"]), "camera_dir": str(sintel_paths["cam_dir"])},
    })
    gt = depth_read(sintel_paths["depth_dir"] / f"frame_{index + 1:04d}.dpt")
    img = np.zeros(gt.shape + (3,), np.uint8)
    df = SintelGTDepth(cfg).compute(Frame(index=index, t=0.0, image=img))
    expected_valid = np.isfinite(gt) & (gt >= 0.05) & (gt <= 100.0)
    np.testing.assert_array_equal(df.valid, expected_valid)
    np.testing.assert_array_equal(df.depth[df.valid], gt[df.valid])
    assert df.depth.dtype == np.float32
    assert df.intrinsics.fx == pytest.approx(896.0)
    assert (df.intrinsics.width, df.intrinsics.height) == (gt.shape[1], gt.shape[0])
    if index == 49:
        assert (~df.valid).any()
        assert (df.depth[~df.valid] == 0).all()


def test_sintel_gt_missing_dir_and_missing_frame(tmp_path, sintel_paths, make_config, caplog):
    bad = SintelGTDepth(make_config("fast", {"input": {"depth_dir": str(tmp_path / "nope")}}))
    with pytest.raises(BackendUnavailable, match="input.depth_dir"):
        bad.load()
    with pytest.raises(BackendUnavailable):
        SintelGTDepth(make_config("fast", {"input": {"depth_dir": None}})).load()
    backend = SintelGTDepth(make_config("fast", {"input": {"depth_dir": str(sintel_paths["depth_dir"])}}))
    img = np.zeros((20, 30, 3), np.uint8)
    with caplog.at_level(logging.WARNING, logger="vfoa.backends.depth_simple"):
        for i in (500, 501):
            df = backend.compute(Frame(index=i, t=0.0, image=img))
            assert df.depth.shape == (20, 30) and not df.valid.any()
    assert len([r for r in caplog.records if r.levelno >= logging.WARNING]) == 1


def test_sintel_gt_resizes_to_frame(sintel_paths, make_config):
    backend = SintelGTDepth(make_config("fast", {"input": {"depth_dir": str(sintel_paths["depth_dir"])}}))
    df = backend.compute(Frame(index=0, t=0.0, image=np.zeros((218, 512, 3), np.uint8)))
    assert df.depth.shape == (218, 512) and df.valid.mean() > 0.9


def test_constant_depth(make_config):
    backend = ConstantDepth(make_config("fast", {"depth": {"backend": "constant", "constant_depth_m": 3.0}}))
    df = backend.compute(Frame(index=0, t=0.0, image=np.zeros((48, 64, 3), np.uint8)))
    assert df.depth.shape == (48, 64) and df.depth.dtype == np.float32
    assert (df.depth == 3.0).all() and df.valid.all() and df.disparity is None
    assert (df.intrinsics.width, df.intrinsics.height) == (64, 48)
    with pytest.raises(ValueError):
        ConstantDepth(make_config("fast", {"depth": {"constant_depth_m": 0}})).load()


def test_frame_intrinsics_rules(make_config, sintel_paths):
    from vfoa.backends.depth_simple import frame_intrinsics

    img = np.zeros((436, 1024, 3), np.uint8)
    cfg = make_config("fast", {"input": {"camera_dir": str(sintel_paths["cam_dir"])}})
    intr = frame_intrinsics(cfg, Frame(index=0, t=0.0, image=img))
    assert intr.fx == pytest.approx(896.0) and intr.baseline == pytest.approx(0.1)
    cfg = make_config("fast", {"camera": {"fx": 500.0}})
    intr = frame_intrinsics(cfg, Frame(index=0, t=0.0, image=img))
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (500.0, 500.0, 511.5, 217.5)
    small = Frame(index=0, t=0.0, image=np.zeros((218, 512, 3), np.uint8),
                  intrinsics=Intrinsics(896, 896, 511.5, 217.5, 1024, 436))
    intr = frame_intrinsics(make_config("fast"), small)
    assert (intr.width, intr.height) == (512, 218) and intr.fx == pytest.approx(448.0)
    assert intr.baseline == pytest.approx(0.1)
    intr = frame_intrinsics(make_config("fast"), Frame(index=0, t=0.0, image=img))
    assert intr.fx == pytest.approx(512.0 / np.tan(np.radians(30.0)))


def test_registry_builds_depth_backends(make_config, tmp_path):
    expected = {"stereo_bm": StereoBMDepth, "sintel_gt": SintelGTDepth, "constant": ConstantDepth}
    for name, cls in expected.items():
        cfg = make_config("fast", {"depth": {"backend": name}, "input": {"depth_dir": str(tmp_path / "missing")}})
        backend = create_backend("depth", cfg)
        assert type(backend) is cls
        assert not backend.is_loaded
