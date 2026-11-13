"""Tests of the legacy depth modules in thesis_final (Depth_Estimator, sintel_data, MiDaS helpers)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import cv2
import numpy as np
import pytest
import yaml

from vfoa.backends.depth_stereo import synthesize_right_view


def _write_stereo_pngs(tmp_path, sintel_sample, n=3):
    dl, dr = tmp_path / "left", tmp_path / "right"
    dl.mkdir()
    dr.mkdir()
    samples = []
    for k in range(1, n + 1):
        left, gt, intr = sintel_sample(k)
        right = synthesize_right_view(left, gt, intr)
        cv2.imwrite(str(dl / f"frame_{k:04d}.png"), left)
        cv2.imwrite(str(dr / f"frame_{k:04d}.png"), right)
        samples.append((left, gt, intr))
    return dl, dr, samples


def _read_video(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
    finally:
        cap.release()
    return frames


def test_depth_estimator_sintel(tmp_path, sintel_sample, sintel_paths):
    from Depth_Estimator import Depth_Estimator

    dl, dr, samples = _write_stereo_pngs(tmp_path, sintel_sample)
    est = Depth_Estimator(str(dl), str(dr), str(sintel_paths["cam_dir"]), "mpi_sintel")
    out = tmp_path / "rect.mp4"
    Qs, disps = est(str(out), 0, 3)
    assert len(Qs) == 3 and len(disps) == 3
    for Q, disp, (left, gt, _) in zip(Qs, disps, samples):
        assert Q.shape == (4, 4)
        assert Q[3, 2] == pytest.approx(10.0) and Q[2, 3] == pytest.approx(896.0)
        assert disp.dtype == np.float32 and disp.shape == gt.shape
        pts = cv2.reprojectImageTo3D(disp, Q)
        v = disp > 0
        assert (pts[..., 2][v] > 0).all()
        rel = np.abs(pts[..., 2][v] - gt[v]) / gt[v]
        assert np.median(rel) < 0.05

    frames = _read_video(out)
    assert len(frames) == 3
    for img, disp, (left, _, _) in zip(frames, disps, samples):
        assert img.shape == left.shape
        diff = np.abs(img.astype(np.int16) - left.astype(np.int16)).mean(axis=2)
        assert diff.mean() < 10
        grey = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        region = (disp == 0) & (grey > 30)  # pixels the legacy code blacked out
        assert region.sum() > 1000
        assert diff[region].mean() < 10

    Qs2, disps2 = est(str(tmp_path / "one.mp4"), 1, 2)
    assert len(disps2) == 1
    np.testing.assert_array_equal(disps2[0], disps[1])
    np.testing.assert_array_equal(Qs2[0], Qs[1])


def test_depth_estimator_missing_images(tmp_path, sintel_paths):
    from Depth_Estimator import Depth_Estimator

    est = Depth_Estimator(str(tmp_path), str(tmp_path), str(sintel_paths["cam_dir"]), "mpi_sintel")
    with pytest.raises(FileNotFoundError):
        est(str(tmp_path / "x.mp4"), 0, 3)


def _cam_yaml(path, width, height, f):
    data = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": {"rows": 3, "cols": 3, "data": [f, 0, (width - 1) / 2, 0, f, (height - 1) / 2, 0, 0, 1]},
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0, 0, 0, 0, 0]},
        "rectification_matrix": {"rows": 3, "cols": 3, "data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
        "projection_matrix": {"rows": 3, "cols": 4,
                              "data": [f, 0, (width - 1) / 2, 0, 0, f, (height - 1) / 2, 0, 0, 0, 1, 0]},
    }
    path.write_text(yaml.safe_dump(data))


def test_depth_estimator_robot_identity_calibration(tmp_path, tiny_video):
    from Depth_Estimator import ROBOT_BASELINE_M, Depth_Estimator

    cam_dir = tmp_path / "calib"
    cam_dir.mkdir()
    _cam_yaml(cam_dir / "left.yaml", 320, 240, 300.0)
    _cam_yaml(cam_dir / "right.yaml", 320, 240, 300.0)
    vl = tiny_video(n_frames=5, size=(320, 240))
    vr = tiny_video(n_frames=5, size=(320, 240))
    est = Depth_Estimator(str(vl), str(vr), str(cam_dir / "left.yaml"), "robot_video")
    out = tmp_path / "robot.mp4"
    Qs, disps = est(str(out), 0, 3)
    assert isinstance(Qs, list) and len(Qs) == 3 and len(disps) == 3
    for Q, d in zip(Qs, disps):
        assert Q[3, 2] == pytest.approx(1.0 / ROBOT_BASELINE_M)
        assert Q[2, 3] > 0
        assert d.shape == (180, 240) and d.dtype == np.float32
    assert len(_read_video(out)) == 3


def test_legacy_calculate_disp_and_main_depth(tmp_path, sintel_sample, sintel_paths):
    from Depth_Estimator import Depth_Estimator

    dl, dr, samples = _write_stereo_pngs(tmp_path, sintel_sample, n=1)
    est = Depth_Estimator(str(dl), str(dr), str(sintel_paths["cam_dir"]), "mpi_sintel")
    clean, disp = est.main_depth(str(dl / "frame_0001.png"), str(dr / "frame_0001.png"))
    np.testing.assert_array_equal(clean, samples[0][0])
    d2, invalid = est.calculate_disp_sintel(clean, cv2.imread(str(dr / "frame_0001.png")))
    np.testing.assert_array_equal(d2, disp)
    np.testing.assert_array_equal(invalid, disp <= 0)


def test_sintel_data_roundtrips(tmp_path, sintel_paths, rng):
    import sintel_data as sio

    depth = rng.uniform(0.1, 10.0, size=(7, 9)).astype(np.float32)
    sio.depth_write(str(tmp_path / "d.dpt"), depth)
    np.testing.assert_array_equal(sio.depth_read(str(tmp_path / "d.dpt")), depth)

    K = np.array([[896.0, 0, 511.5], [0, 896.0, 217.5], [0, 0, 1]])
    Rt = rng.normal(size=(3, 4))
    sio.cam_write(str(tmp_path / "c.cam"), K, Rt)
    K2, Rt2 = sio.cam_read(str(tmp_path / "c.cam"))
    np.testing.assert_array_equal(K2, K)
    np.testing.assert_array_equal(Rt2, Rt)

    u = rng.normal(size=(5, 6)).astype(np.float32)
    v = rng.normal(size=(5, 6)).astype(np.float32)
    sio.flow_write(str(tmp_path / "a.flo"), u, v)
    u2, v2 = sio.flow_read(str(tmp_path / "a.flo"))
    np.testing.assert_array_equal(u2, u)
    np.testing.assert_array_equal(v2, v)
    sio.flow_write(str(tmp_path / "b.flo"), np.stack([u, v], axis=2))
    u3, v3 = sio.flow_read(str(tmp_path / "b.flo"))
    np.testing.assert_array_equal(u3, u)
    np.testing.assert_array_equal(v3, v)
    assert (tmp_path / "a.flo").read_bytes()[:4] == b"PIEH"

    disp = rng.uniform(0, 1000, size=(6, 8))
    sio.disparity_write(str(tmp_path / "disp.png"), disp, bitdepth=32)
    np.testing.assert_allclose(sio.disparity_read(str(tmp_path / "disp.png")), disp, atol=1 / 64)

    seg = rng.integers(0, 2 ** 20, size=(6, 8)).astype(np.int32)
    sio.segmentation_write(str(tmp_path / "seg.png"), seg)
    np.testing.assert_array_equal(sio.segmentation_read(str(tmp_path / "seg.png")), seg)

    gt = sio.depth_read(str(sintel_paths["depth_dir"] / "frame_0001.dpt"))
    assert gt.shape == (436, 1024)
    K, _ = sio.cam_read(str(sintel_paths["cam_dir"] / "frame_0001.cam"))
    assert K[0, 0] == pytest.approx(896.0)


def test_prediction_depth_scale_shift(rng):
    torch = pytest.importorskip("torch")
    import prediction_depth as pdp

    pred = torch.from_numpy(rng.uniform(0.5, 2.0, size=(1, 16, 20)))
    target = 2.5 * pred + 0.3
    mask = torch.ones_like(pred)
    scale, shift = pdp.compute_scale_and_shift(pred, target, mask)
    assert float(scale[0]) == pytest.approx(2.5, rel=1e-9)
    assert float(shift[0]) == pytest.approx(0.3, rel=1e-9)
    loss = pdp.ScaleAndShiftInvariantLoss()
    assert isinstance(loss, torch.nn.Module)
    assert float(loss(pred, target, mask)) == pytest.approx(0.0, abs=1e-12)
    assert pdp.MSELoss is pdp.MSELoss and pdp.GradientLoss.__module__ == "prediction_depth"
    with pytest.raises(AttributeError):
        pdp.NoSuchLoss  # noqa: B018


def test_midas_missing_raises_backend_unavailable(monkeypatch, tmp_path):
    import DepthPredictor_Mono as dpm
    from vfoa.backends.base import BackendUnavailable

    monkeypatch.setenv("VFOA_MIDAS_ROOT", str(tmp_path))
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))  # _midas() appends the root
    for name in [m for m in sys.modules if m == "midas" or m.startswith("midas.")]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(BackendUnavailable, match="VFOA_MIDAS_ROOT"):
        dpm._midas()


def test_legacy_modules_import_cleanly(repo_root):
    code = textwrap.dedent(
        """
        import importlib, sys
        heavy = ("matplotlib", "open3d", "PIL", "torch", "torchvision", "midas")
        for name in ["Depth_Estimator", "sintel_data", "DepthPredictor_Mono", "Reconstruction_3D",
                     "Reconstruction_Mono_3D", "prediction_depth",
                     "vfoa.backends.depth_stereo", "vfoa.backends.depth_simple"]:
            importlib.import_module(name)
            bad = [m for m in sys.modules if m.split(".")[0] in heavy]
            assert not bad, (name, sorted(bad)[:5])
        print("OK")
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("VFOA_")}
    env["PYTHONPATH"] = os.pathsep.join([str(repo_root / "thesis_final"), str(repo_root)])
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(repo_root), env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
