"""Tests of benchmarks/bench_pipeline.py, the config reconciliation, the docs and the CI workflow.

No test asserts a duration: the machine is shared. Structure, run counts and
schedules are checked instead.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import pytest
import yaml

from vfoa.config import DEFAULTS, PROFILES, load_config

REPO = Path(__file__).resolve().parents[1]
STAGES = ("depth", "persons", "tracking", "headpose", "gaze", "segmentation", "scene", "smoothing",
          "attention", "total")
META_KEYS = {"profile", "input", "frames", "warmup", "resolution", "config_hash", "python", "numpy", "opencv",
             "torch", "ultralytics", "cpu_count", "date", "backends", "fallbacks", "env"}
STAT_KEYS = {"runs", "mean_ms", "p50_ms", "p95_ms", "max_ms", "per_frame_ms", "total_ms", "share"}


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_pipeline_under_test", REPO / "benchmarks" / "bench_pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(bench, tmp_path, capsys, *argv, name="r.json"):
    out = tmp_path / name
    code = bench.main([*argv, "--out", str(out)])
    text = capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
    return code, text, data


# --------------------------------------------------------------------------- benchmark harness
def test_import_is_light():
    import subprocess

    code = (
        "import sys, importlib.util;"
        f"spec = importlib.util.spec_from_file_location('b', {str(REPO / 'benchmarks' / 'bench_pipeline.py')!r});"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
        "bad = [n for n in ('torch', 'open3d', 'ultralytics', 'onnxruntime', 'matplotlib') if n in sys.modules];"
        "print(','.join(bad))"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert res.stdout.strip() == ""


def test_synthetic_structure(bench, tmp_path, capsys):
    code, text, data = _run(bench, tmp_path, capsys, "--profile", "synthetic", "--frames", "12", "--warmup", "3")
    assert code == 0
    for name in STAGES:
        assert re.search(rf"^{name}\s", text, re.M), name
    assert data["status"] == "ok" and data["reason"] is None
    assert META_KEYS <= set(data["meta"])
    meta = data["meta"]
    assert meta["profile"] == "synthetic" and meta["resolution"] == [640, 360]
    assert meta["frames"] == 12 and meta["warmup"] == 3
    assert re.fullmatch(r"[0-9a-f]{16}", meta["config_hash"])
    assert set(meta["backends"].values()) == {"synthetic"} and meta["fallbacks"] == {}
    assert set(meta["env"]) == {"OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS"}
    assert all(v is None or isinstance(v, str) for v in meta["env"].values())
    timing = data["timing"]
    assert timing["frames"] == 12 and math.isfinite(timing["fps"]) and timing["fps"] > 0
    stages = timing["stages"]
    assert set(STAGES) <= set(stages)
    assert stages["persons"]["runs"] == 12
    assert stages["segmentation"]["runs"] == 2  # stride 5 on frames 3..14 (warm-up excluded)
    for st in stages.values():
        assert STAT_KEYS <= set(st)
        assert all(math.isfinite(st[k]) for k in STAT_KEYS)
    assert stages["read"]["runs"] == 12
    assert data["run"]["frames_total"] == 15
    assert data["micro"] == {}


def test_overrides_change_schedule_and_hash(bench, tmp_path, capsys):
    _, _, base = _run(bench, tmp_path, capsys, "--frames", "12", "--warmup", "3", name="a.json")
    code, _, data = _run(bench, tmp_path, capsys, "--frames", "12", "--warmup", "3",
                         "--set", "segmentation.stride=4", "--set", "gaze.stride=3", name="b.json")
    assert code == 0 and data["status"] == "ok"
    stages = data["timing"]["stages"]
    assert stages["segmentation"]["runs"] == 3  # frames 4, 8, 12
    assert stages["gaze"]["runs"] == 4  # frames 3, 6, 9, 12
    assert stages["persons"]["runs"] == 12
    assert data["meta"]["config_hash"] != base["meta"]["config_hash"]
    assert data["meta"]["overrides"] == ["segmentation.stride=4", "gaze.stride=3"]


def test_real_video_with_synthetic_backends(bench, tmp_path, capsys, tiny_video):
    video = tiny_video(n_frames=12, size=(160, 120))
    code, text, data = _run(bench, tmp_path, capsys, "--profile", "synthetic", "--input", str(video),
                            "--frames", "8", "--warmup", "2")
    assert code == 0 and data["status"] == "ok"
    assert data["meta"]["resolution"] == [160, 120]
    assert data["timing"]["stages"]["read"]["runs"] == 8
    assert data["timing"]["frames"] == 8
    assert re.search(r"^read\s", text, re.M)
    # more frames requested than the video has: still ok, measured on what exists
    code, _, data = _run(bench, tmp_path, capsys, "--profile", "synthetic", "--input", str(video),
                         "--frames", "100", "--warmup", "2", name="long.json")
    assert code == 0 and data["status"] == "ok"
    assert data["timing"]["frames"] == 10 and data["run"]["frames_total"] == 12


def test_fast_profile_skips_without_ultralytics(bench, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    code, text, data = _run(
        bench, tmp_path, capsys, "--profile", "fast", "--input", "synthetic", "--frames", "3", "--warmup", "1",
        "--set", "gaze.backend=none", "--set", "models.allow_download=false",
        "--set", f"models.weights_dir={tmp_path / 'noweights'}",
    )
    assert code == 0
    assert "SKIPPED:" in text
    assert data["status"] == "skipped"
    assert "ultralytics" in data["reason"]
    assert data["timing"] == {} and data["meta"]["profile"] == "fast"


def test_l2cs_without_weights_falls_back_to_head_proxy(bench, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("VFOA_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    code, text, data = _run(
        bench, tmp_path, capsys, "--profile", "synthetic", "--frames", "6", "--warmup", "2",
        "--set", "gaze.backend=l2cs", "--set", "gaze.fallback=head_proxy",
        "--set", "models.allow_download=false", "--set", f"models.weights_dir={tmp_path / 'noweights'}",
    )
    assert code == 0 and data["status"] == "ok"
    assert data["meta"]["fallbacks"] == {"gaze": {"from": "l2cs", "to": "head_proxy"}}
    assert data["meta"]["env"]["OMP_WAIT_POLICY"] == "PASSIVE"
    assert re.search(r"^NOTE:", text, re.M)
    assert data["timing"]["stages"]["gaze"]["runs"] == 6


def test_stereo_synth_exercises_stereo_bm(bench, tmp_path, capsys, sintel_paths):
    args = ["--profile", "synthetic", "--input", str(sintel_paths["video"]),
            "--camera-dir", str(sintel_paths["cam_dir"]), "--depth-dir", str(sintel_paths["depth_dir"]),
            "--set", "depth.backend=stereo_bm", "--set", "depth.scale=0.5", "--frames", "4", "--warmup", "1"]
    code, _, data = _run(bench, tmp_path, capsys, *args, "--stereo-synth")
    assert code == 0 and data["status"] == "ok"
    assert data["meta"]["backends"]["depth"] == "stereo_bm"
    assert data["meta"]["resolution"] == [1024, 436] and data["meta"]["stereo_synth"] is True
    assert data["timing"]["stages"]["depth"]["runs"] == 4
    # --stereo-synth without ground-truth depth is a usage error
    no_depth = [a for a in args if a not in ("--depth-dir", str(sintel_paths["depth_dir"]))]
    code, _, _ = _run(bench, tmp_path, capsys, *no_depth, "--stereo-synth", name="err.json")
    assert code == 2


def test_input_error_exits_2(bench, tmp_path, capsys):
    code, _, data = _run(bench, tmp_path, capsys, "--input", str(tmp_path / "missing.mp4"), "--frames", "2")
    assert code == 2 and data is None


def test_micro_benchmarks(bench):
    micro = bench.micro_benchmarks(repeats=1, small=True)
    assert set(micro) == {"ply_roundtrip", "slab_test", "stereo_bm"}
    for name, m in micro.items():
        if name == "ply_roundtrip" and m["status"] == "skipped":  # open3d not installed
            assert "open3d" in m["reason"]
            continue
        assert m["status"] == "ok", m
        assert m["agree"] is True, name
        assert m["legacy_ms"] > 0 and m["new_ms"] > 0
        assert m["speedup"] == pytest.approx(m["legacy_ms"] / m["new_ms"])


def test_micro_without_open3d(bench, monkeypatch):
    monkeypatch.setitem(sys.modules, "open3d", None)
    micro = bench.micro_benchmarks(repeats=1, small=True)
    assert micro["ply_roundtrip"]["status"] == "skipped"
    assert "open3d" in micro["ply_roundtrip"]["reason"]
    assert micro["slab_test"]["status"] == "ok" and micro["stereo_bm"]["status"] == "ok"


def test_micro_only_cli(bench, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bench, "micro_benchmarks", lambda repeats=5, small=False: {
        "slab_test": {"status": "ok", "legacy_ms": 4.0, "new_ms": 1.0, "speedup": 4.0, "agree": True}})
    code, text, data = _run(bench, tmp_path, capsys, "--frames", "0", "--micro", "--markdown")
    assert code == 0 and data["status"] == "ok" and data["timing"] == {}
    assert data["micro"]["slab_test"]["speedup"] == 4.0
    assert "speed-up 4.0x" in text and "| slab_test |" in text


def test_format_markdown(bench):
    result = {"status": "ok", "meta": {"profile": "synthetic", "input": "synthetic", "resolution": [64, 48],
                                       "frames": 2, "warmup": 1},
              "timing": {"fps": 50.0, "stages": {"total": {"runs": 2, "mean_ms": 20.0, "p50_ms": 20.0,
                                                           "p95_ms": 20.0, "max_ms": 20.0, "per_frame_ms": 20.0,
                                                           "share": 1.0}}},
              "micro": {"stereo_bm": {"status": "skipped", "reason": "no"}}}
    md = bench.format_markdown(result)
    assert "| total | 2 | 20.00 |" in md and "**50.00**" in md and "skipped: no" in md


# --------------------------------------------------------------------------- config reconciliation
#: Config sections each vfoa module reads through ``.get("key", ...)``.
FILE_SECTIONS = {
    "vfoa/cli.py": ("output",),
    "vfoa/cache.py": ("runtime", "input", "gaze", "cache", "output"),
    "vfoa/attention.py": ("attention",),
    "vfoa/pipeline.py": ("input", "timing", "cache", "runtime", "segmentation", "tracking", "headpose", "gaze",
                         "depth", "persons"),
    "vfoa/tracking.py": ("tracking",),
    "vfoa/smoothing.py": ("smoothing", "attention"),
    "vfoa/scene.py": ("scene", "persons"),
    "vfoa/backends/__init__.py": ("gaze",),
    "vfoa/backends/base.py": ("models",),
    "vfoa/backends/depth_stereo.py": ("depth",),
    "vfoa/backends/depth_simple.py": ("depth", "camera", "input"),
    "vfoa/backends/synthetic.py": ("synthetic",),
    "vfoa/backends/accurate/_thirdparty.py": ("models",),
    "vfoa/backends/accurate/fsanet.py": ("headpose", "runtime"),
    "vfoa/backends/accurate/gaze360.py": ("gaze", "input", "runtime", "persons"),
    "vfoa/backends/accurate/openpose_densepose.py": ("persons",),
    "vfoa/backends/accurate/panoptic_fpn.py": ("segmentation",),
    "vfoa/backends/fast/_ultra.py": ("models", "runtime"),
    "vfoa/backends/fast/head_proxy_gaze.py": ("gaze",),
    "vfoa/backends/fast/keypoints_headpose.py": ("headpose", "camera", "persons"),
    "vfoa/backends/fast/l2cs_gaze.py": ("gaze", "models", "runtime"),
    "vfoa/backends/fast/yolo_pose.py": ("persons",),
    "vfoa/backends/fast/yolo_seg.py": ("segmentation",),
    "vfoa/io/sources.py": ("input", "camera"),
}
#: ``.get`` keys that are not config keys (state dicts, cache metadata, env vars, summaries).
ALLOW = {
    "vfoa/backends/accurate/gaze360.py": {"state_dict"},
    "vfoa/backends/accurate/panoptic_fpn.py": {"category_id", "isthing", "score"},
    "vfoa/backends/base.py": {"VFOA_WEIGHTS_DIR"},
    "vfoa/backends/fast/_ultra.py": {"torch"},  # sys.modules.get("torch")
    "vfoa/cache.py": {"complete", "config", "disparity", "fingerprint", "format_version", "key", "n_frames", "stage"},
    "vfoa/cli.py": {"async", "latency"},
    "vfoa/pipeline.py": {"latency"},
}
#: (section, key) whose literal ``.get`` fallback deliberately differs from DEFAULTS.
FALLBACK_EXCEPTIONS = {("attention", "rays"), ("runtime", "log_every"), ("camera", "baseline_m"), ("input", "path")}
GET_KEY = re.compile(r"""\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']""")
GET_DEFAULT = re.compile(r"""\.get\(\s*["']([A-Za-z_][A-Za-z0-9_]*)["']\s*,\s*([^()\[\]{},]+?)\s*\)""")
ATTR_READ = re.compile(r"\bcfg\.([a-z_]+)\.([a-z_]+)\b")


def _keys_any_depth(d: dict, out: set | None = None) -> set:
    out = set() if out is None else out
    for k, v in d.items():
        out.add(k)
        if isinstance(v, dict):
            _keys_any_depth(v, out)
    return out


def _vfoa_files():
    return [p for p in sorted((REPO / "vfoa").rglob("*.py")) if p.relative_to(REPO).as_posix() != "vfoa/config.py"]


def test_every_get_key_is_in_defaults():
    problems = []
    for path in _vfoa_files():
        rel = path.relative_to(REPO).as_posix()
        keys = set(GET_KEY.findall(path.read_text(encoding="utf-8")))
        if not keys:
            continue
        if rel not in FILE_SECTIONS:
            problems.append(f"{rel}: not mapped in FILE_SECTIONS (keys {sorted(keys)})")
            continue
        allowed = set(DEFAULTS) | ALLOW.get(rel, set())
        for section in FILE_SECTIONS[rel]:
            allowed |= _keys_any_depth(DEFAULTS[section])
        missing = sorted(keys - allowed)
        if missing:
            problems.append(f"{rel}: {missing}")
    assert not problems, "config keys read in code but missing from DEFAULTS:\n" + "\n".join(problems)


def test_attribute_reads_are_in_defaults():
    problems = []
    for path in _vfoa_files():
        rel = path.relative_to(REPO).as_posix()
        for section, key in ATTR_READ.findall(path.read_text(encoding="utf-8")):
            if section not in DEFAULTS or not isinstance(DEFAULTS[section], dict):
                problems.append(f"{rel}: cfg.{section}")
            elif key not in DEFAULTS[section] and key not in ("get", "to_dict"):
                problems.append(f"{rel}: cfg.{section}.{key}")
    assert not problems, problems


def test_literal_fallbacks_match_defaults():
    mismatches, checked = [], 0
    for rel, sections in FILE_SECTIONS.items():
        text = (REPO / rel).read_text(encoding="utf-8")
        for key, token in GET_DEFAULT.findall(text):
            try:
                value = ast.literal_eval(token)
            except (ValueError, SyntaxError):
                continue
            owners = [s for s in sections if key in DEFAULTS[s]]
            nested = [(s, k) for s in sections for k, v in DEFAULTS[s].items() if isinstance(v, dict) and key in v]
            if len(owners) == 1 and not nested:
                checked += 1
                if (owners[0], key) not in FALLBACK_EXCEPTIONS and DEFAULTS[owners[0]][key] != value:
                    mismatches.append(f"{rel}: {owners[0]}.{key} code {value!r} != {DEFAULTS[owners[0]][key]!r}")
            elif len(owners) > 1:
                checked += 1
                if value not in [DEFAULTS[s][key] for s in owners]:
                    mismatches.append(f"{rel}: {key} code {value!r} not among {owners}")
            elif len(nested) == 1 and not owners:
                checked += 1
                s, k = nested[0]
                if DEFAULTS[s][k][key] != value:
                    mismatches.append(f"{rel}: {s}.{k}.{key} code {value!r} != {DEFAULTS[s][k][key]!r}")
    assert checked > 50
    assert not mismatches, mismatches


def test_module_defaults_are_consistent():
    from vfoa import scene, tracking

    for module, section in ((tracking, "tracking"), (scene, "scene")):
        for key, value in module._DEFAULTS.items():
            assert key in DEFAULTS[section], f"{section}.{key}"
            assert DEFAULTS[section][key] == value, f"{section}.{key}"


def test_reconciled_keys_and_fast_strides():
    cfg = load_config(profile="fast")
    assert cfg.gaze.stride == 2 and PROFILES["fast"]["gaze"]["stride"] == 2
    assert cfg.persons.stride == 1 and cfg.segmentation.stride == 10
    assert "fallback_from" not in DEFAULTS["gaze"]
    for section, key in [("input", "pace"), ("depth", "speckle_window"), ("depth", "speckle_range"),
                         ("persons", "upper_body_only"), ("persons", "match_min_inside"),
                         ("persons", "keep_headless"), ("tracking", "motion"), ("tracking", "max_predict_frames"),
                         ("headpose", "relaxed_kpt_thr"), ("headpose", "fallback_confidence"),
                         ("headpose", "max_reproj_frac"), ("scene", "outlier_nb_points"),
                         ("scene", "outlier_radius_m"), ("scene", "outlier_radius_scale"),
                         ("scene", "self_aabb_margin_m"), ("smoothing", "jump_deg"), ("synthetic", "sway_m"),
                         ("gaze", "fallback"), ("gaze", "l2cs")]:
        assert key in DEFAULTS[section], f"{section}.{key}"


def test_all_config_files_load():
    files = sorted((REPO / "configs").glob("*.yaml"))
    assert len(files) >= 5
    for path in files:
        load_config(path)


# --------------------------------------------------------------------------- requirements, CI, docs
def test_ultralytics_only_in_fast_requirements():
    for path in sorted(REPO.glob("requirements*.txt")):
        lines = [ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        named = [ln for ln in lines if "ultralytics" in ln.lower()]
        if path.name == "requirements-fast.txt":
            assert named, "requirements-fast.txt must list ultralytics"
        else:
            assert not named, f"{path.name} must not depend on ultralytics: {named}"


def test_ci_workflow_runs_pytest():
    data = yaml.safe_load((REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True))  # PyYAML reads the key `on` as True
    assert "push" in triggers and "pull_request" in triggers
    steps = [s for job in data["jobs"].values() for s in job["steps"]]
    runs = " ".join(str(s.get("run", "")) for s in steps)
    assert "pytest" in runs
    assert "requirements.txt" in runs and "requirements-dev.txt" in runs


def test_readme_and_performance_doc():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for topic in ("requirements-fast.txt", "requirements-accurate.txt", "AGPL-3.0", "--profile synthetic",
                  "--profile fast", "configs/accurate.yaml", "configs/thesis.yaml", "cache.mode=read", "--realtime",
                  "segmentation.async", "bench_pipeline.py", "l2cs", "head_proxy", "+Z forward",
                  "docs/performance.md", "#1", "#35"):
        assert topic in readme, topic
    perf = (REPO / "docs" / "performance.md").read_text(encoding="utf-8")
    for topic in ("bench_pipeline.py", "--profile synthetic", "--profile fast", "| total |", "GPU", "--micro"):
        assert topic in perf, topic
