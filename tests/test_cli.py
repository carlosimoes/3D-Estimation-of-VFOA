"""Tests of the command line interface (python -m vfoa run ...)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vfoa.cli import build_parser, config_from_args, main


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sintel_args(sintel_paths: dict, out: Path, *extra: str) -> list[str]:
    return [
        "run", "--profile", "synthetic",
        "--input", str(sintel_paths["video"]),
        "--camera-dir", str(sintel_paths["cam_dir"]),
        "--depth-dir", str(sintel_paths["depth_dir"]),
        "--set", "depth.backend=sintel_gt",
        "--max-frames", "10",
        "--output-dir", str(out),
        *extra,
    ]


def test_module_entry_point_synthetic(repo_root, tmp_path):
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "vfoa", "run", "--profile", "synthetic", "--max-frames", "30",
         "--output-dir", str(out)],
        cwd=repo_root, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    rows = read_jsonl(out / "results.jsonl")
    assert [r["frame"] for r in rows] == list(range(30))
    assert all(r["persons"] for r in rows)
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert timing["frames"] == 30 and "attention" in timing["stages"]
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["profile"] == "synthetic" and cfg["input"]["max_frames"] == 30
    assert "attention" in proc.stdout and "fps" in proc.stdout


def test_sintel_ground_truth_run(sintel_paths, tmp_path):
    out = tmp_path / "sintel"
    assert main(sintel_args(sintel_paths, out)) == 0
    rows = read_jsonl(out / "results.jsonl")
    assert [r["frame"] for r in rows] == list(range(10))
    assert any(r["objects"] for r in rows)
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["input"]["source"] == "auto"
    assert cfg["depth"]["backend"] == "sintel_gt"


def test_cache_then_rescore(sintel_paths, tmp_path):
    out = tmp_path / "cached"
    assert main(sintel_args(sintel_paths, out, "--set", "cache.enabled=true")) == 0
    first = read_jsonl(out / "results.jsonl")
    assert main(sintel_args(sintel_paths, out, "--set", "cache.enabled=true", "--set", "cache.mode=read",
                            "--set", "attention.sigma_gaze_deg=5")) == 0
    second = read_jsonl(out / "results.jsonl")
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert not any(k.startswith("load:") for k in timing["setup_ms"])
    assert timing["cache"] and all(c["misses"] == 0 for c in timing["cache"].values())
    s1 = [a["score_total"] for r in first for a in r["attention"]]
    s2 = [a["score_total"] for r in second for a in r["attention"]]
    assert len(s1) == len(s2) and s1 != s2
    # the literal acceptance variant (cache not enabled on the second run) also works
    assert main(sintel_args(sintel_paths, tmp_path / "plain", "--set", "cache.mode=read",
                            "--set", "attention.sigma_gaze_deg=5")) == 0


def test_flags_map_to_config(tmp_path):
    args = build_parser().parse_args([
        "run", "--profile", "synthetic", "--input", "clip.mp4", "--right", "right.mp4",
        "--camera-dir", "cams", "--depth-dir", "depths", "--output-dir", str(tmp_path),
        "--max-frames", "7", "--realtime", "--video", "--show", "--set", "attention.sigma_gaze_deg=4",
    ])
    cfg = config_from_args(args)
    assert cfg.input.path == "clip.mp4" and cfg.input.source == "auto"
    assert cfg.input.right_path == "right.mp4"
    assert cfg.input.camera_dir == "cams" and cfg.input.depth_dir == "depths"
    assert cfg.input.max_frames == 7 and cfg.input.realtime is True
    assert cfg.output.dir == str(tmp_path) and cfg.output.video is True and cfg.output.show is True
    assert cfg.attention.sigma_gaze_deg == 4

    pinned = config_from_args(build_parser().parse_args(
        ["run", "--profile", "synthetic", "--input", "x", "--set", "input.source=video"]))
    assert pinned.input.source == "video"
    default = config_from_args(build_parser().parse_args(["run", "--profile", "synthetic"]))
    assert default.input.source == "synthetic" and default.output.video is False


def test_video_and_csv_outputs(tmp_path):
    out = tmp_path / "vid"
    assert main(["run", "--profile", "synthetic", "--max-frames", "6", "--output-dir", str(out),
                 "--video", "--set", "output.csv=true"]) == 0
    assert (out / "overlay.mp4").stat().st_size > 0
    lines = (out / "results.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("frame,t,person_id,object_id") and len(lines) > 1
    assert len(read_jsonl(out / "results.jsonl")) == 6


@pytest.mark.parametrize("argv", [
    ["run", "--profile", "no_such_profile"],
    ["run", "--profile", "fast", "--input", "does/not/exist.mp4"],
    ["run", "--profile", "synthetic", "--set", "depth.backend=none"],
])
def test_errors_exit_2(argv, tmp_path, capsys):
    assert main([*argv, "--output-dir", str(tmp_path / "err")]) == 2
    assert "vfoa: error:" in capsys.readouterr().err
