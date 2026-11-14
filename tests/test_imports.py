"""Import hygiene: every module of vfoa/ and thesis_final/ imports in a fresh interpreter.

Each module is imported in its own subprocess (cwd = repo root, PYTHONPATH =
repo root + thesis_final, no VFOA_THIRD_PARTY) with a bogus ``sys.argv`` so
that a module parsing arguments at import time fails. Heavy optional
dependencies (torch, open3d, ultralytics, ...) must not be loaded by the
import itself. No source file may contain a hard-coded ``/home/carlos`` path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

HEAVY = (
    "torch", "torchvision", "onnxruntime", "open3d", "matplotlib", "PIL", "ultralytics",
    "ffmpeg", "detectron2", "lucid", "OpenGL",
)

CORE = (
    "vfoa.types", "vfoa.config", "vfoa.timing", "vfoa.geometry", "vfoa.attention", "vfoa.tracking",
    "vfoa.smoothing", "vfoa.scene", "vfoa.cache", "vfoa.pipeline", "vfoa.cli", "vfoa.io",
    "vfoa.io.sintel", "vfoa.io.sources", "vfoa.io.outputs", "vfoa.backends", "vfoa.backends.base",
)
OWNED = ("FinalTest", "VideoProcessor", "Reconstruction_scene", "GazeFrameRenderer")


def _discover() -> list[str]:
    mods: list[str] = []
    for path in sorted((ROOT / "vfoa").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).with_suffix("")
        parts = list(rel.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        mods.append(".".join(parts))
    for path in sorted((ROOT / "thesis_final").glob("*.py")):
        mods.append(path.stem)
    return mods


MODULES = _discover()

_CODE = (
    "import sys, json\n"
    "sys.argv = ['prog', '--bogus-flag']\n"
    "import importlib\n"
    "importlib.import_module({mod!r})\n"
    "heavy = {heavy!r}\n"
    "print('HEAVY=' + json.dumps(sorted(h for h in heavy if h in sys.modules)))\n"
)


def _run(mod: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("VFOA_THIRD_PARTY", None)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "thesis_final")])
    env.setdefault("OMP_NUM_THREADS", "1")
    return subprocess.run(
        [sys.executable, "-c", _CODE.format(mod=mod, heavy=HEAVY)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )


@pytest.fixture(scope="session")
def import_results() -> dict[str, subprocess.CompletedProcess]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(MODULES, pool.map(_run, MODULES)))


def _heavy(proc: subprocess.CompletedProcess) -> list[str]:
    for line in proc.stdout.splitlines():
        if line.startswith("HEAVY="):
            return json.loads(line[len("HEAVY="):])
    raise AssertionError(f"no HEAVY line in output:\n{proc.stdout}\n{proc.stderr}")


@pytest.mark.parametrize("mod", MODULES)
def test_module_imports(mod: str, import_results: dict) -> None:
    proc = import_results[mod]
    assert proc.returncode == 0, f"import {mod} failed:\n{proc.stderr[-3000:]}"


@pytest.mark.parametrize("mod", MODULES)
def test_no_heavy_import(mod: str, import_results: dict) -> None:
    proc = import_results[mod]
    if proc.returncode != 0:
        pytest.fail(f"import {mod} failed:\n{proc.stderr[-2000:]}")
    assert _heavy(proc) == [], f"{mod} imports heavy optional dependencies at import time"


def test_discovery_covers_core() -> None:
    missing = [m for m in CORE + OWNED if m not in MODULES]
    assert not missing, missing
    assert "vfoa" in MODULES
    assert not any("__pycache__" in m for m in MODULES)


def test_no_hardcoded_home_paths() -> None:
    needle = b"/home/carlos"
    offenders = []
    for base in (ROOT / "vfoa", ROOT / "thesis_final"):
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts:
                continue
            if needle in path.read_bytes():
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders
