"""Shared pytest fixtures and markers for the vfoa test-suite.

Fixtures: repo_root, sintel_paths, sintel_sample(index=1), tiny_video(...),
make_config(profile="synthetic", overrides=None), rng.
Markers: ``slow`` and ``network`` (network tests run only with VFOA_RUN_NETWORK=1).
Autouse: ``_no_weight_downloads`` makes the l2cs weight download fail (except in
``network`` tests).
Models are always faked; tests never download and never need a GPU.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: slow tests")
    config.addinivalue_line("markers", "network: tests that need network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list) -> None:
    if os.environ.get("VFOA_RUN_NETWORK") == "1":
        return
    skip = pytest.mark.skip(reason="network test (set VFOA_RUN_NETWORK=1 to run)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_weight_downloads(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fast profile loads the l2cs gaze backend: never let it download weights in tests
    (tests marked ``network`` are exempt)."""
    if "network" in request.keywords:
        return
    from vfoa.backends.fast import l2cs_gaze

    def _refuse(url, target, *a, **k):
        raise OSError(f"network disabled in tests (would download {url})")

    monkeypatch.setattr(l2cs_gaze, "_download", _refuse)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def sintel_paths(repo_root: Path) -> dict:
    """Paths of the Sintel bandage_2 sample: video, cam_dir, depth_dir."""
    return {
        "video": repo_root / "Final_videos" / "mpi_sintel.mp4",
        "cam_dir": repo_root / "Final_videos" / "stereo" / "camdata_left" / "bandage_2",
        "depth_dir": repo_root / "Final_videos" / "stereo" / "training" / "bandage_2_depth",
    }


@pytest.fixture(scope="session")
def _sintel_cache() -> dict:
    return {}


@pytest.fixture
def sintel_sample(sintel_paths: dict, _sintel_cache: dict):
    """Factory ``sintel_sample(index=1) -> (image_bgr, depth_m, Intrinsics)``.

    ``index`` is the 1-based Sintel frame number (video frame ``index - 1``).
    Intrinsics carry the Sintel baseline 0.1 m. Returns copies.
    """
    from vfoa.io.sintel import depth_read, intrinsics_from_cam

    def _sample(index: int = 1):
        if index not in _sintel_cache:
            cap = cv2.VideoCapture(str(sintel_paths["video"]))
            try:
                image = None
                for _ in range(index):
                    ok, image = cap.read()
                    if not ok:
                        raise RuntimeError(f"cannot read frame {index - 1} of {sintel_paths['video']}")
            finally:
                cap.release()
            depth = depth_read(sintel_paths["depth_dir"] / f"frame_{index:04d}.dpt")
            h, w = depth.shape
            intr = intrinsics_from_cam(
                sintel_paths["cam_dir"] / f"frame_{index:04d}.cam", w, h, baseline=0.1
            )
            _sintel_cache[index] = (image, depth, intr)
        image, depth, intr = _sintel_cache[index]
        return image.copy(), depth.copy(), intr

    return _sample


@pytest.fixture
def tiny_video(tmp_path: Path):
    """Factory ``tiny_video(n_frames=12, size=(160, 120), fps=10.0) -> Path``.

    Writes an mp4v video (``size`` = (width, height)) of a moving rectangle.
    """
    counter = itertools.count()

    def _make(n_frames: int = 12, size: tuple = (160, 120), fps: float = 10.0) -> Path:
        width, height = int(size[0]), int(size[1])
        path = tmp_path / f"tiny_{next(counter)}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
        try:
            for i in range(n_frames):
                img = np.full((height, width, 3), 40, dtype=np.uint8)
                x = (i * 7) % max(width - 20, 1)
                cv2.rectangle(img, (x, height // 3), (x + 20, height // 3 + 20), (0, 200, 255), -1)
                writer.write(img)
        finally:
            writer.release()
        return path

    return _make


@pytest.fixture
def make_config():
    """Factory ``make_config(profile="synthetic", overrides=None) -> Config``."""
    from vfoa.config import load_config

    def _make(profile: str = "synthetic", overrides=None):
        return load_config(profile=profile, overrides=overrides)

    return _make


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic NumPy generator (seed 0)."""
    return np.random.default_rng(0)
