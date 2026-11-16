from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from vfoa.smoothing import DirectionSmoother, KalmanFilter3D, TargetSmoother

REPO = Path(__file__).resolve().parent.parent


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def ang(a, b):
    return math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b)))


def cfg(**smoothing):
    base = {"direction": "kalman", "ema_alpha": 0.5, "kalman_q": 1.0, "kalman_r": 0.05,
            "max_hold_s": 0.5, "target": "hmm", "hmm_stay_prob": 0.9, "hysteresis_frames": 3}
    base.update(smoothing)
    return {"smoothing": base, "attention": {"min_score": 0.1}}


# ---------------------------------------------------------------- KalmanFilter3D
@pytest.mark.parametrize("rel", ["vfoa/smoothing.py", "thesis_final/KalmanFilter.py",
                                 "thesis_final/BoxTracker.py", "thesis_final/Data.py"])
def test_no_syntax_warning(rel):
    src = (REPO / rel).read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(src, rel, "exec")


def test_adaptive_r_arrays_are_distinct():
    kf = KalmanFilter3D(dt=1 / 30)
    differed = False
    for i in range(30):
        z = np.array([1.0, 2.0, 3.0]) + np.array([3.0, -2.0, 1.0]) * max(10 - i, 0) / 30.0 * i
        kf.Kalman(z)
        if i >= 1:
            assert kf.R_atual is not kf.Ratual
            assert not np.shares_memory(kf.R_atual, kf.Ratual)
            np.testing.assert_allclose(np.diag(kf.R), kf.Ratual)
            if not np.allclose(kf.R_atual, kf.Ratual):
                differed = True
    assert differed


def _cv_track(rng, v, sigma, n=300, dt=1 / 30):
    t = np.arange(n)[:, None] * dt
    truth = np.array([1.0, 2.0, 3.0]) + np.asarray(v) * t
    return truth, truth + rng.normal(0.0, sigma, truth.shape)


def _rmse(a, b):
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def test_kalman_reduces_rmse_tuned():
    rng = np.random.default_rng(0)
    sigma = 0.05
    truth, meas = _cv_track(rng, (0.5, -0.2, 0.3), sigma)
    kf = KalmanFilter3D(dt=1 / 30, q=1.0, r=sigma ** 2, adaptive_r=False)
    out = np.array([kf.Kalman(z) for z in meas])
    assert _rmse(out, truth) < _rmse(meas, truth)


def test_kalman_reduces_rmse_legacy_defaults():
    rng = np.random.default_rng(1)
    truth, meas = _cv_track(rng, (2.0, -1.0, 1.5), 3.0)
    kf = KalmanFilter3D(dt=1 / 30)
    out = np.array([kf.Kalman(z) for z in meas])
    assert _rmse(out, truth) < _rmse(meas, truth)


def test_kalman_requires_F_or_dt():
    with pytest.raises(ValueError):
        KalmanFilter3D()


# ------------------------------------------------------------- DirectionSmoother
D0 = unit([0.2, -0.1, -1.0])
D1 = unit([-0.9, 0.1, -0.4])


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_direction_noise_reduction(mode):
    rng = np.random.default_rng(0)
    sm = DirectionSmoother(cfg(direction=mode), fps=30)
    sig = math.radians(3.0)
    raw_err, out_err = [], []
    for i in range(401):
        m = unit(D0 + rng.normal(0.0, sig, 3))
        out = sm.update((0, "gaze"), m, i / 30)
        assert out is not None
        assert abs(np.linalg.norm(out) - 1.0) < 1e-9
        if i >= 50:
            raw_err.append(ang(m, D0) ** 2)
            out_err.append(ang(out, D0) ** 2)
    assert np.mean(raw_err) / np.mean(out_err) >= 2.0


@pytest.mark.parametrize("mode,settle", [("kalman", 3), ("ema", 8)])
def test_direction_step_converges(mode, settle):
    sm = DirectionSmoother(cfg(direction=mode), fps=30)
    for i in range(30):
        sm.update("k", D0, i / 30)
    for j in range(30):
        out = sm.update("k", D1, (30 + j) / 30)
        if j >= settle:
            assert math.degrees(ang(out, D1)) < 5.0, (mode, j)


def test_kalman_ignores_single_outlier():
    sm = DirectionSmoother(cfg(direction="kalman"), fps=30)
    for i in range(20):
        sm.update("k", D0, i / 30)
    out = sm.update("k", D1, 20 / 30)
    assert math.degrees(ang(out, D0)) < 5.0
    for i in range(21, 30):
        out = sm.update("k", D0, i / 30)
        assert math.degrees(ang(out, D0)) < 5.0


@pytest.mark.parametrize("mode", ["kalman", "ema"])
def test_hold_then_none(mode):
    sm = DirectionSmoother(cfg(direction=mode, max_hold_s=0.35), fps=10)
    last = None
    for i in range(10):
        last = sm.update("k", D0, i / 10)
    for j in range(1, 4):
        out = sm.update("k", None, (9 + j) / 10)
        assert out is not None
        assert abs(np.linalg.norm(out) - 1) < 1e-9
        assert math.degrees(ang(out, last)) < 5.0
    assert sm.update("k", None, 13 / 10) is None
    assert "k" not in sm
    out = sm.update("k", D1, 14 / 10)
    np.testing.assert_allclose(out, D1)


def test_mode_none_passthrough():
    sm = DirectionSmoother(cfg(direction="none"), fps=30)
    v = np.array([0.0, 0.0, -2.0])
    out = sm.update("k", v, 0.0)
    np.testing.assert_array_equal(out, v)
    assert sm.update("k", None, 0.1) is None


def test_invalid_measurements_are_missing():
    sm = DirectionSmoother(cfg(), fps=30)
    assert sm.update("k", [np.nan, 0, 1], 0.0) is None
    assert sm.update("k", [0, 0, 0], 0.0) is None
    assert sm.update("k", D0, 0.0, confidence=0.0) is None


def test_keys_independent_and_drop():
    sm = DirectionSmoother(cfg(direction="kalman"), fps=30)
    for i in range(10):
        a = sm.update((0, "gaze"), D0, i / 30)
        b = sm.update((1, "head"), D1, i / 30)
    assert math.degrees(ang(a, D0)) < 1e-6 + 1.0
    assert math.degrees(ang(b, D1)) < 1.0
    sm.drop((0, "gaze"))
    assert (0, "gaze") not in sm and (1, "head") in sm
    assert sm.update((0, "gaze"), None, 11 / 30) is None
    np.testing.assert_allclose(sm.update((0, "gaze"), D1, 12 / 30), D1)
    sm.reset()
    assert (1, "head") not in sm


# --------------------------------------------------------------- TargetSmoother
A = {1: 0.8, 2: 0.2}
B = {1: 0.2, 2: 0.8}


@pytest.mark.parametrize("mode", ["hmm", "hysteresis"])
def test_target_blip_is_ignored(mode):
    ts = TargetSmoother(cfg(target=mode))
    out = [ts.update(0, s, i / 30) for i, s in enumerate([A, A, A, B, A, A])]
    assert out == [1] * 6


def test_target_sustained_switch_hysteresis():
    ts = TargetSmoother(cfg(target="hysteresis", hysteresis_frames=3))
    for i in range(5):
        assert ts.update(0, A, i) == 1
    out = [ts.update(0, B, 5 + j) for j in range(5)]
    assert out == [1, 1, 2, 2, 2]


def test_target_sustained_switch_hmm():
    ts = TargetSmoother(cfg(target="hmm"))
    for i in range(10):
        assert ts.update(0, A, i) == 1
    out = [ts.update(0, B, 10 + j) for j in range(6)]
    first = out.index(2)
    assert first < 4
    assert all(o == 2 for o in out[first:])


@pytest.mark.parametrize("mode", ["hmm", "hysteresis", "none"])
def test_target_below_min_score_is_none(mode):
    ts = TargetSmoother(cfg(target=mode))
    for i in range(5):
        assert ts.update(0, {1: 0.05, 2: 0.02}, i) is None


@pytest.mark.parametrize("mode", ["hmm", "hysteresis", "none"])
def test_target_object_disappears(mode):
    ts = TargetSmoother(cfg(target=mode))
    for i in range(5):
        ts.update(0, {1: 0.9, 2: 0.3}, i)
    for i in range(5, 10):
        assert ts.update(0, {2: 0.3}, i) != 1
    assert ts.update(0, {}, 11) is None
    ts.drop(0)
    ts.drop(42)


def test_target_mode_none_is_argmax():
    ts = TargetSmoother(cfg(target="none"))
    seq = [A, B, A, {3: 0.5, 1: 0.5}, B]
    assert [ts.update(7, s, i) for i, s in enumerate(seq)] == [1, 2, 1, 1, 2]
