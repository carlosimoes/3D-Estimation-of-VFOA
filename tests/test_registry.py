from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types

import pytest

import vfoa.backends as backends
from vfoa.backends import REGISTRY, available, create_backend, register, resolve
from vfoa.backends.accurate import _thirdparty
from vfoa.backends.base import (
    Backend,
    BackendUnavailable,
    DepthBackend,
    GazeBackend,
    resolve_device,
    weights_dir,
)
from vfoa.config import load_config

HEAVY = ["torch", "open3d", "matplotlib", "ultralytics", "onnxruntime", "PIL"]
LIGHT_MODULES = [
    "vfoa", "vfoa.types", "vfoa.config", "vfoa.timing", "vfoa.backends", "vfoa.backends.base",
    "vfoa.io.sintel", "vfoa.backends.accurate._thirdparty",
]

EXPECTED = {
    "depth": {
        "stereo_bm": "vfoa.backends.depth_stereo:StereoBMDepth",
        "sintel_gt": "vfoa.backends.depth_simple:SintelGTDepth",
        "constant": "vfoa.backends.depth_simple:ConstantDepth",
        "synthetic": "vfoa.backends.synthetic:SyntheticDepth",
    },
    "persons": {
        "openpose_densepose": "vfoa.backends.accurate.openpose_densepose:OpenPoseDensePosePersons",
        "yolo_pose": "vfoa.backends.fast.yolo_pose:YoloPosePersons",
        "synthetic": "vfoa.backends.synthetic:SyntheticPersons",
    },
    "headpose": {
        "fsanet": "vfoa.backends.accurate.fsanet:FSANetHeadPose",
        "keypoints": "vfoa.backends.fast.keypoints_headpose:KeypointHeadPose",
        "synthetic": "vfoa.backends.synthetic:SyntheticHeadPose",
    },
    "gaze": {
        "gaze360": "vfoa.backends.accurate.gaze360:Gaze360Gaze",
        "head_proxy": "vfoa.backends.fast.head_proxy_gaze:HeadProxyGaze",
        "synthetic": "vfoa.backends.synthetic:SyntheticGaze",
    },
    "segmentation": {
        "panoptic_fpn": "vfoa.backends.accurate.panoptic_fpn:PanopticFPNSegmentation",
        "yolo_seg": "vfoa.backends.fast.yolo_seg:YoloSegSegmentation",
        "synthetic": "vfoa.backends.synthetic:SyntheticSegmentation",
    },
}


@pytest.fixture(scope="module")
def import_probe(repo_root):
    code = textwrap.dedent(
        f"""
        import importlib, json, sys
        for name in {LIGHT_MODULES!r}:
            importlib.import_module(name)
        print(json.dumps(sorted(sys.modules)))
        """
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo_root), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


def test_light_imports_have_no_heavy_deps(import_probe):
    loaded = {m.split(".")[0] for m in import_probe}
    assert not (loaded & set(HEAVY)), sorted(loaded & set(HEAVY))


def test_importing_registry_imports_no_backend(import_probe):
    extra = {
        m for m in import_probe
        if m.startswith("vfoa.backends.")
        and m not in ("vfoa.backends.base", "vfoa.backends.accurate", "vfoa.backends.accurate._thirdparty")
    }
    assert not extra, sorted(extra)


def test_bootstrap_from_thesis_final(repo_root):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", "import _bootstrap, vfoa; print(vfoa.__file__)"],
        cwd=str(repo_root / "thesis_final"), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "vfoa" in proc.stdout


def test_preregistered_names():
    for kind, names in EXPECTED.items():
        for name, target in names.items():
            assert REGISTRY[kind][name] == target
            assert isinstance(REGISTRY[kind][name], str)
    assert set(REGISTRY) == set(EXPECTED)
    assert available("gaze") == ["gaze360", "head_proxy", "synthetic"]


class FakeDepth(DepthBackend):
    name = "fake"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.load_calls = 0

    def _load(self):
        self.load_calls += 1

    def compute(self, frame):  # pragma: no cover - not called
        raise NotImplementedError


def test_create_with_registered_class(monkeypatch):
    monkeypatch.setitem(REGISTRY, "depth", dict(REGISTRY["depth"]))
    register("depth", "fake", FakeDepth)
    cfg = load_config(overrides=["depth.backend=fake"])
    be = create_backend("depth", cfg)
    assert isinstance(be, FakeDepth)
    assert be.cfg is cfg
    assert be.section is cfg.depth
    assert not be.is_loaded and be.load_calls == 0
    be.load()
    be.load()
    assert be.is_loaded and be.load_calls == 1
    be.reset()
    be.close()


def test_create_with_string_target(monkeypatch, tmp_path):
    mod_name = "vfoa_fake_backend_mod"
    (tmp_path / f"{mod_name}.py").write_text(
        "from vfoa.backends.base import GazeBackend\n"
        "class FakeGaze(GazeBackend):\n"
        "    name = 'fake'\n"
        "    def estimate(self, frame, persons):\n"
        "        return {}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    monkeypatch.setitem(REGISTRY, "gaze", dict(REGISTRY["gaze"]))
    register("gaze", "fake", f"{mod_name}:FakeGaze")
    assert mod_name not in sys.modules
    be = create_backend("gaze", load_config(overrides=["gaze.backend=fake"]))
    assert mod_name in sys.modules
    assert isinstance(be, GazeBackend)
    assert be.observe(None, []) is None
    assert be.estimate(None, []) == {}
    assert resolve("gaze", "fake") is type(be)


def test_none_and_unknown():
    cfg = load_config()  # gaze.backend == none
    assert create_backend("gaze", cfg) is None
    with pytest.raises(ValueError) as err:
        create_backend("depth", load_config(overrides=["depth.backend=nope"]))
    msg = str(err.value)
    for name in ("stereo_bm", "sintel_gt", "constant", "synthetic", "none"):
        assert name in msg
    with pytest.raises(ValueError):
        create_backend("nokind", cfg)


def test_register_validation():
    with pytest.raises(ValueError):
        register("nokind", "x", FakeDepth)
    with pytest.raises(ValueError):
        register("depth", "x", "no_colon_here")
    with pytest.raises(ValueError):
        register("depth", "none", FakeDepth)
    with pytest.raises(TypeError):
        register("depth", "x", 42)
    assert "x" not in REGISTRY["depth"]


def test_abstract_interfaces():
    with pytest.raises(TypeError):
        DepthBackend(load_config())
    base = Backend(load_config())
    base.load()
    assert base.is_loaded
    assert issubclass(BackendUnavailable, RuntimeError)
    assert backends.BackendUnavailable is BackendUnavailable


def test_thirdparty_paths(monkeypatch, tmp_path):
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    cfg = load_config()
    assert _thirdparty.third_party_root(cfg) is None
    with pytest.raises(BackendUnavailable) as err:
        _thirdparty.resolve_path(cfg, "gaze360/gaze360_model.pth.tar")
    assert "models.third_party_root" in str(err.value) and "VFOA_THIRD_PARTY" in str(err.value)
    assert "gaze360_model.pth.tar" in str(err.value)
    absolute = tmp_path / "w.pth"
    assert _thirdparty.resolve_path(cfg, absolute) == absolute

    env_root = tmp_path / "env_root"
    (env_root / "gaze360").mkdir(parents=True)
    monkeypatch.setenv("VFOA_THIRD_PARTY", str(env_root))
    assert _thirdparty.third_party_root(cfg) == env_root
    assert _thirdparty.resolve_path(cfg, "a/b.pt") == env_root / "a" / "b.pt"

    cfg_root = tmp_path / "cfg_root"
    cfg2 = load_config(overrides={"models": {"third_party_root": str(cfg_root)}})
    assert _thirdparty.third_party_root(cfg2) == cfg_root  # config wins over env

    monkeypatch.setattr(sys, "path", list(sys.path))
    added = _thirdparty.add_sys_paths(cfg, "gaze360")
    assert added == [env_root / "gaze360"]
    _thirdparty.add_sys_paths(cfg, "gaze360")
    assert sys.path.count(str(env_root / "gaze360")) == 1
    with pytest.raises(BackendUnavailable):
        _thirdparty.add_sys_paths(cfg, "missing_dir")
    with pytest.raises(BackendUnavailable):
        _thirdparty.add_sys_paths(cfg2, "gaze360")  # root does not exist


def test_weights_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    assert str(weights_dir(load_config())) == "weights"
    cfg = load_config(overrides=["models.weights_dir=/opt/w"])
    assert str(weights_dir(cfg)) == "/opt/w"
    monkeypatch.setenv("VFOA_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir(cfg) == tmp_path


def test_resolve_device(monkeypatch):
    assert resolve_device(load_config(overrides=["models.device=cuda:1"])) == "cuda:1"
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert resolve_device(load_config()) == "cuda"
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)
    assert resolve_device(load_config()) == "cpu"
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert resolve_device(load_config()) == "cpu"
