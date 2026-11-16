from __future__ import annotations

import cv2
import numpy as np
import pytest

from vfoa.io.sintel import cam_read, cam_write, depth_read, depth_write, intrinsics_from_cam


def test_repo_cam_file(sintel_paths):
    K, Rt = cam_read(sintel_paths["cam_dir"] / "frame_0001.cam")
    assert K.shape == (3, 3) and Rt.shape == (3, 4)
    assert K.dtype == np.float64
    np.testing.assert_allclose(K, [[896, 0, 511.5], [0, 896, 217.5], [0, 0, 1]])
    R = Rt[:, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_repo_depth_file(sintel_paths):
    depth = depth_read(sintel_paths["depth_dir"] / "frame_0001.dpt")
    assert depth.shape == (436, 1024)
    assert depth.dtype == np.float32 and depth.dtype.isnative
    assert 0.2 < depth.min() < 0.35
    assert 2.0 < depth.max() < 2.5
    assert 1.0 < float(np.median(depth)) < 1.5


def test_depth_roundtrip_non_square(tmp_path, rng):
    depth = rng.uniform(0.1, 10.0, size=(7, 13)).astype(np.float32)
    path = tmp_path / "d.dpt"
    depth_write(path, depth)
    assert path.stat().st_size == 12 + 4 * 7 * 13
    back = depth_read(path)
    assert back.shape == (7, 13)
    np.testing.assert_array_equal(back, depth)
    with pytest.raises(ValueError):
        depth_write(tmp_path / "x.dpt", np.zeros((2, 2, 2)))


def test_cam_roundtrip(tmp_path, rng):
    K = np.array([[500.0, 0, 320], [0, 510, 240], [0, 0, 1]])
    Rt = rng.normal(size=(3, 4))
    path = tmp_path / "c.cam"
    cam_write(path, K, Rt)
    assert path.stat().st_size == 4 + 8 * 21
    K2, Rt2 = cam_read(path)
    np.testing.assert_array_equal(K2, K)
    np.testing.assert_array_equal(Rt2, Rt)


def test_bad_files_raise(tmp_path):
    bad_tag = tmp_path / "bad.dpt"
    bad_tag.write_bytes(b"XXXX" + np.array([2, 2], "<i4").tobytes() + bytes(16))
    with pytest.raises(ValueError):
        depth_read(bad_tag)
    with pytest.raises(ValueError):
        cam_read(bad_tag)
    truncated = tmp_path / "trunc.dpt"
    depth_write(truncated, np.ones((4, 5), np.float32))
    truncated.write_bytes(truncated.read_bytes()[:-3])
    with pytest.raises(ValueError):
        depth_read(truncated)
    bad_size = tmp_path / "size.dpt"
    bad_size.write_bytes(b"PIEH" + np.array([0, 5], "<i4").tobytes())
    with pytest.raises(ValueError):
        depth_read(bad_size)
    short_cam = tmp_path / "short.cam"
    short_cam.write_bytes(b"PIEH" + bytes(50))
    with pytest.raises(ValueError):
        cam_read(short_cam)


def test_intrinsics_from_cam(sintel_paths):
    intr = intrinsics_from_cam(sintel_paths["cam_dir"] / "frame_0001.cam", 1024, 436)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (896.0, 896.0, 511.5, 217.5)
    assert (intr.width, intr.height, intr.baseline) == (1024, 436, 0.1)
    assert intrinsics_from_cam(sintel_paths["cam_dir"] / "frame_0001.cam", 1024, 436,
                               baseline=None).baseline is None


def test_sintel_sample_fixture(sintel_sample):
    image, depth, intr = sintel_sample(1)
    assert image.shape == (436, 1024, 3) and image.dtype == np.uint8
    assert depth.shape == (436, 1024) and depth.dtype == np.float32
    assert 1.0 < float(np.median(depth)) < 1.5
    assert intr.fx == 896.0 and intr.cx == 511.5 and intr.baseline == 0.1
    image[:] = 0  # returns copies
    again, _, _ = sintel_sample(1)
    assert again.any()
    img3, depth3, _ = sintel_sample(3)
    assert img3.shape == (436, 1024, 3) and depth3.shape == (436, 1024)


def test_tiny_video_fixture(tiny_video):
    path = tiny_video(n_frames=9, size=(96, 64), fps=12.0)
    cap = cv2.VideoCapture(str(path))
    try:
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            frames.append(img)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    assert len(frames) == 9
    assert frames[0].shape == (64, 96, 3)
    assert fps == pytest.approx(12.0)
    assert tiny_video() != path
