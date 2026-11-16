"""Tests of vfoa.io.sources (streaming frame sources)."""

from __future__ import annotations

import inspect
import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from vfoa.io import sources as S
from vfoa.io.sintel import cam_write
from vfoa.types import Frame, Intrinsics


def _cfg(make_config, **inp):
    camera = inp.pop("camera", None)
    overrides = {"input": inp}
    if camera:
        overrides["camera"] = camera
    return make_config("fast", overrides)


def _write_pngs(directory: Path, names_values: list[tuple[str, int]], size=(32, 24)) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    w, h = size
    for name, value in names_values:
        img = np.full((h, w, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(directory / name), img)


def test_sintel_metadata_and_frames(make_config, sintel_paths, sintel_sample):
    src = S.VideoSource(_cfg(make_config, path=str(sintel_paths["video"])))
    assert src.fps == 30.0
    assert (src.width, src.height) == (1024, 436)
    assert src.n_frames == 50
    first_equal = None
    indices = []
    for f in src:
        assert isinstance(f, Frame)
        indices.append(f.index)
        assert type(f.t) is float
        assert f.t == f.index / 30
        assert f.image.dtype == np.uint8 and f.image.shape == (436, 1024, 3)
        assert f.intrinsics is not None
        if f.index == 0:
            first_equal = np.array_equal(f.image, sintel_sample(1)[0])
    assert indices == list(range(50))
    assert first_equal


def test_start_and_max_frames(make_config, sintel_paths, sintel_sample):
    src = S.VideoSource(_cfg(make_config, path=str(sintel_paths["video"]), start=10, max_frames=5))
    assert src.n_frames == 5
    frames = list(src)
    assert [f.index for f in frames] == [10, 11, 12, 13, 14]
    assert frames[0].t == 10 / 30
    assert np.array_equal(frames[0].image, sintel_sample(11)[0])


def test_is_lazy(make_config, tiny_video, monkeypatch):
    path = tiny_video(n_frames=12)
    counts = {"read": 0, "grab": 0}
    real = cv2.VideoCapture

    class Counting:
        def __init__(self, *a, **k):
            self._cap = real(*a, **k)

        def read(self):
            counts["read"] += 1
            return self._cap.read()

        def grab(self):
            counts["grab"] += 1
            return self._cap.grab()

        def __getattr__(self, name):
            return getattr(self._cap, name)

    monkeypatch.setattr(S.cv2, "VideoCapture", Counting)
    src = S.VideoSource(_cfg(make_config, path=str(path)))
    it = iter(src)
    assert inspect.isgenerator(it)
    assert counts["read"] == 0
    next(it)
    assert counts["read"] == 1
    it.close()

    counts.update(read=0, grab=0)
    src = S.VideoSource(_cfg(make_config, path=str(path), start=4))
    f = next(iter(src))
    assert f.index == 4
    assert counts["grab"] == 4 and counts["read"] == 1


def test_fps_override(make_config, tiny_video):
    path = tiny_video(n_frames=4, fps=10.0)
    assert S.VideoSource(_cfg(make_config, path=str(path))).fps == 10.0
    src = S.VideoSource(_cfg(make_config, path=str(path), fps=25))
    assert src.fps == 25.0
    assert [f.t for f in src] == [i / 25.0 for i in range(4)]


def test_reiterable(make_config, tiny_video):
    src = S.VideoSource(_cfg(make_config, path=str(tiny_video(n_frames=5))))
    a = [f.index for f in src]
    b = [f.index for f in src]
    assert a == b == list(range(5))
    assert src.n_frames == 5


def test_missing_file_raises(make_config, tmp_path):
    with pytest.raises(FileNotFoundError):
        S.VideoSource(_cfg(make_config, path=str(tmp_path / "nope.mp4")))
    with pytest.raises(FileNotFoundError):
        S.ImageDirSource(_cfg(make_config, path=str(tmp_path / "nodir")))


def test_camera_dir_mapping(make_config, tiny_video, tmp_path, sintel_paths, caplog):
    path = tiny_video(n_frames=6)
    cam_dir = tmp_path / "cams"
    cam_dir.mkdir()
    Rt = np.hstack([np.eye(3), np.zeros((3, 1))])
    for num in (1, 2, 3, 4):  # frames 4 and 5 (files 5, 6) missing
        K = np.array([[100.0 + num, 0, 80], [0, 100.0 + num, 60], [0, 0, 1]])
        cam_write(cam_dir / f"frame_{num:04d}.cam", K, Rt)
    src = S.VideoSource(_cfg(make_config, path=str(path), camera_dir=str(cam_dir)))
    with caplog.at_level(logging.WARNING, logger="vfoa.io.sources"):
        frames = list(src)
    fx = [f.intrinsics.fx for f in frames]
    assert fx[:4] == [101.0, 102.0, 103.0, 104.0]
    assert fx[4:] == [104.0, 104.0]
    assert sum("missing" in r.getMessage() for r in caplog.records) == 1
    assert frames[0].intrinsics.baseline == 0.1

    with pytest.raises(FileNotFoundError):
        S.VideoSource(_cfg(make_config, path=str(path), camera_dir=str(tmp_path / "none")))

    src = S.VideoSource(
        _cfg(make_config, path=str(sintel_paths["video"]), camera_dir=str(sintel_paths["cam_dir"]), max_frames=2)
    )
    f = next(iter(src))
    assert f.intrinsics.fx == 896.0 and f.intrinsics.baseline == 0.1
    assert (f.intrinsics.width, f.intrinsics.height) == (1024, 436)
    src = S.VideoSource(
        _cfg(
            make_config, path=str(sintel_paths["video"]), camera_dir=str(sintel_paths["cam_dir"]),
            max_frames=1, camera={"baseline_m": 0.25},
        )
    )
    assert next(iter(src)).intrinsics.baseline == 0.25


def test_camera_config(make_config, tiny_video):
    path = tiny_video(n_frames=2, size=(160, 120))
    src = S.VideoSource(_cfg(make_config, path=str(path), camera={"fx": 200.0}))
    intr = next(iter(src)).intrinsics
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (200.0, 200.0, 79.5, 59.5)
    src = S.VideoSource(
        _cfg(make_config, path=str(path), camera={"fx": 200.0, "fy": 210.0, "cx": 70.0, "cy": 50.0})
    )
    intr = next(iter(src)).intrinsics
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (200.0, 210.0, 70.0, 50.0)


def test_hfov_fallback(make_config, tiny_video):
    path = tiny_video(n_frames=2, size=(160, 120))
    src = S.VideoSource(_cfg(make_config, path=str(path), camera={"hfov_deg": 90.0}))
    intr = next(iter(src)).intrinsics
    assert intr.fx == pytest.approx(80.0) and intr.fy == pytest.approx(80.0)
    assert isinstance(intr, Intrinsics)


def test_image_dir_pairs_right_view(make_config, tmp_path, caplog):
    left, right = tmp_path / "left", tmp_path / "right"
    order = ["img_1.png", "img_2.png", "img_10.png"]
    _write_pngs(left, [(n, 10 * (k + 1)) for k, n in enumerate(order)])
    _write_pngs(right, [(n, 10 * (k + 1) + 1) for k, n in enumerate(order)])
    src = S.ImageDirSource(_cfg(make_config, path=str(left), right_path=str(right), fps=10))
    assert src.n_frames == 3 and (src.width, src.height) == (32, 24)
    frames = list(src)
    assert [int(f.image[0, 0, 0]) for f in frames] == [10, 20, 30]
    assert [int(f.right[0, 0, 0]) for f in frames] == [11, 21, 31]
    assert [f.t for f in frames] == [0.0, 0.1, 0.2]

    # Mismatched counts: 5 left vs 3 right -> 3 frames + warning.
    l5, r3 = tmp_path / "l5", tmp_path / "r3"
    _write_pngs(l5, [(f"f{k}.png", k) for k in range(5)])
    _write_pngs(r3, [(f"f{k}.png", 100 + k) for k in range(3)])
    with caplog.at_level(logging.WARNING, logger="vfoa.io.sources"):
        src = S.ImageDirSource(_cfg(make_config, path=str(l5), right_path=str(r3)))
        frames = list(src)
    assert src.n_frames == 3 and len(frames) == 3
    assert any("shorter" in r.getMessage() for r in caplog.records)

    src = S.ImageDirSource(_cfg(make_config, path=str(l5), start=1, max_frames=3))
    frames = list(src)
    assert [f.index for f in frames] == [1, 2, 3]
    assert [int(f.image[0, 0, 0]) for f in frames] == [1, 2, 3]
    assert src.n_frames == 3


def test_video_with_right_video(make_config, tiny_video):
    left = tiny_video(n_frames=12)
    right = tiny_video(n_frames=8)
    src = S.VideoSource(_cfg(make_config, path=str(left), right_path=str(right)))
    assert src.n_frames == 8
    frames = list(src)
    assert len(frames) == 8
    assert all(f.right is not None and f.right.shape == f.image.shape for f in frames)


def test_open_source_auto(make_config, tiny_video, tmp_path, monkeypatch):
    video = tiny_video(n_frames=3)
    assert isinstance(S.open_source(_cfg(make_config, path=str(video))), S.VideoSource)
    d = tmp_path / "imgs"
    _write_pngs(d, [("a.png", 1)])
    assert isinstance(S.open_source(_cfg(make_config, path=str(d))), S.ImageDirSource)
    assert isinstance(S.open_source(_cfg(make_config, path=str(d), source="images")), S.ImageDirSource)

    class FakeCap:
        opened = True

        def __init__(self, index):
            assert index == 0
            self.n = 0

        def isOpened(self):
            return FakeCap.opened

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 15.0, cv2.CAP_PROP_FRAME_WIDTH: 8, cv2.CAP_PROP_FRAME_HEIGHT: 6}.get(prop, 0)

        def read(self):
            self.n += 1
            return True, np.zeros((6, 8, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(S.cv2, "VideoCapture", FakeCap)
    cam = S.open_source(_cfg(make_config, path="0", max_frames=3))
    assert isinstance(cam, S.CameraSource)
    assert cam.fps == 15.0 and cam.n_frames == 3
    frames = list(cam)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[0].intrinsics is not None
    cam.close()
    FakeCap.opened = False
    with pytest.raises(OSError):
        S.open_source(_cfg(make_config, path="0"))

    with pytest.raises(ValueError):
        S.open_source(_cfg(make_config, path=str(video), source="webcam"))
    with pytest.raises(ValueError):
        S.open_source(_cfg(make_config, path=None))


def test_synthetic_is_lazy(make_config, monkeypatch):
    seen = {}
    sentinel = object()

    class World:
        @classmethod
        def from_config(cls, cfg):
            seen["cfg"] = cfg
            return cls()

        def source(self):
            return sentinel

    stand_in = types.ModuleType("vfoa.backends.synthetic")
    stand_in.SyntheticWorld = World
    monkeypatch.setitem(sys.modules, "vfoa.backends.synthetic", stand_in)
    cfg = _cfg(make_config, path="synthetic")
    assert S.open_source(cfg) is sentinel
    assert seen["cfg"] is cfg
    cfg2 = _cfg(make_config, source="synthetic", path=None)
    assert S.open_source(cfg2) is sentinel


def test_fingerprint(make_config, tiny_video):
    a = tiny_video(n_frames=4)
    b = tiny_video(n_frames=4)
    fp = S.VideoSource(_cfg(make_config, path=str(a))).fingerprint
    assert fp == S.VideoSource(_cfg(make_config, path=str(a))).fingerprint
    assert fp != S.VideoSource(_cfg(make_config, path=str(b))).fingerprint
    assert fp != S.VideoSource(_cfg(make_config, path=str(a), start=1)).fingerprint
    assert fp != S.VideoSource(_cfg(make_config, path=str(a), max_frames=2)).fingerprint
