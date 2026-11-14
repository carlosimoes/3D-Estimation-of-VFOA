"""Legacy entry points on the vfoa pipeline: GazeFrameRenderer, Reconstruction, VideoProcessor, FinalTest.

Models are always the Synthetic* backends; videos come from ``tiny_video``.
Refs #12 #15 #17 #31 #32.
"""

from __future__ import annotations

import builtins
import csv
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

import FinalTest
import GazeFrameRenderer
import Reconstruction_scene
import VideoProcessor as VP
from Data import MapFramesOutputs
from vfoa.backends.base import BackendUnavailable
from vfoa.backends.synthetic import SyntheticWorld
from vfoa.geometry import depth_to_disparity, euler_to_forward
from vfoa.io.outputs import CSV_COLUMNS
from vfoa.types import Intrinsics

ROOT = Path(__file__).resolve().parent.parent
W, H = 160, 120


# ------------------------------------------------------------------ helpers
def _video_frames(path: Path) -> list[np.ndarray]:
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


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _synthetic_cfg(make_config, out_dir: Path, overrides=None):
    ov = {"synthetic": {"width": W, "height": H}, "output": {"dir": str(out_dir)}}
    for k, v in (overrides or {}).items():
        ov.setdefault(k, {}).update(v)
    return make_config("synthetic", ov)


def _env() -> dict:
    env = dict(os.environ)
    env.pop("VFOA_THIRD_PARTY", None)
    return env


class _Recorder:
    def __init__(self) -> None:
        self.frames: list = []
        self.closed = False

    def write(self, frame, result) -> None:
        self.frames.append((result.index, frame.image))

    def close(self) -> None:
        self.closed = True


# ------------------------------------------------------------------ GazeFrameRenderer
def test_draw_gaze_arrow_copy_and_direction() -> None:
    img = np.zeros((100, 120, 3), np.uint8)
    out = GazeFrameRenderer.draw_gaze_arrow(img, (50, 50), (1.0, 0.0, 0.0), 30)
    assert out is not img and not img.any()
    ys, xs = np.nonzero(out.any(axis=2))
    assert xs.size > 0
    assert xs.min() >= 50 - 5 and xs.max() >= 75
    assert np.abs(ys - 50).max() <= 8

    dot = GazeFrameRenderer.draw_gaze_arrow(img, (50, 50), (0.0, 0.0, -1.0), 30)
    ys, xs = np.nonzero(dot.any(axis=2))
    assert xs.size > 0
    assert np.hypot(xs - 50, ys - 50).max() <= 5

    for bad in ((np.nan, 0, 0), (0, 0, 0)):
        assert not GazeFrameRenderer.draw_gaze_arrow(img, (50, 50), bad, 30).any()


def test_opengl_renderer_falls_back_to_cv2() -> None:
    r = GazeFrameRenderer.OpenGLVectorRenderer(W, H, 140)
    assert r.compileShader() is False
    arrow = r.renderArrow((0.5, 0.5), (1.0, 0.0, 0.0), 0.1)
    assert arrow.shape == (H, W, 3) and arrow.dtype == np.uint8
    assert arrow.max() > 0


# ------------------------------------------------------------------ Reconstruction
IDX = (0, 31, 62, 93)  # one frame per script phase


def _recon_inputs(world: SyntheticWorld, idx=IDX, zero_pos=None):
    intr = world.intrinsics
    labels = sorted({world.label(s) for i in idx for s in np.unique(world.segmentation(i).label_map) if s > 0})
    meta = types.SimpleNamespace(thing_classes=labels, stuff_classes=["wall"])
    frames, disp, pan = [], [], []
    mfo = MapFramesOutputs()
    for j, i in enumerate(idx):
        frames.append(world.frame(i).image)
        disp.append(depth_to_disparity(world.depth(i).depth, intr.fx, intr.baseline))
        seg = world.segmentation(i)
        infos = [{"id": s.id, "category_id": labels.index(s.label), "isthing": True} for s in seg.segments]
        if zero_pos is not None:
            pos = {"first": 0, "middle": len(infos) // 2, "last": len(infos)}[zero_pos]
            infos.insert(pos, {"id": 0, "category_id": 0, "isthing": False})
        pan.append((seg.label_map, infos))
        mfo.addListEntries(j, {
            k: {"b": world.head_box(i, k), "g": world.gaze(i, k), "hp": world.head(i, k)}
            for k in range(world.n_persons)
        })
    return meta, intr.Q(), frames, disp, pan, mfo


@pytest.fixture
def world(make_config):
    return SyntheticWorld.from_config(make_config("synthetic"))


def test_reconstruction_matches_truth_and_writes_csv(world, make_config, tmp_path, monkeypatch) -> None:
    meta, Q, frames, disp, pan, mfo = _recon_inputs(world)
    originals = [f.copy() for f in frames]
    opened: list = []
    real_open = builtins.open

    def counting_open(file, *a, **kw):
        if str(file).endswith(Reconstruction_scene.CSV_NAME):
            opened.append(str(file))
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", counting_open)
    rec = Reconstruction_scene.Reconstruction(meta, Q, tmp_path, cfg=make_config("synthetic"), fps=10.0)
    out = rec(frames, disp, pan, mfo)
    monkeypatch.setattr(builtins, "open", real_open)

    assert len(opened) == 1
    assert sorted(out) == list(range(len(IDX)))
    for j, i in enumerate(IDX):
        truth = world.truth(i)
        for k in range(world.n_persons):
            att = out[j][k]
            assert att, f"no attention values for person {k} at frame {i}"
            best = max(att.values(), key=lambda v: v["total_Attention"])
            assert best["object"] == world.label(truth[k]), (i, k)
            for v in att.values():
                assert set(v) >= {"object", "attentionG", "total_Attention_G", "attentionH",
                                  "total_Attention_H", "total_Attention"}
    for f, o in zip(frames, originals):
        assert np.array_equal(f, o)
    assert not list(tmp_path.rglob("*.ply"))
    with open(tmp_path / Reconstruction_scene.CSV_NAME, encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert {int(r[0]) for r in rows[1:]} == set(range(len(IDX)))
    for r in rows[1:]:
        assert float(r[1]) == pytest.approx(int(r[0]) / 10.0)


def test_reconstruction_person_without_cues(world, make_config, tmp_path) -> None:
    meta, Q, frames, disp, pan, mfo = _recon_inputs(world, idx=(0,))
    mfo.addListEntries(0, {0: {}, 1: {"b": world.head_box(0, 1)}, 7: {"g": np.array([0.0, 0.0, 1.0])}})
    out = Reconstruction_scene.Reconstruction(meta, Q, tmp_path, cfg=make_config("synthetic"))(
        frames, disp, pan, mfo)
    assert out[0][0] == {} and out[0][1] == {} and out[0][7] == {}


def test_reconstruction_segment_zero_anywhere(world, make_config, tmp_path) -> None:
    results = []
    for pos in (None, "first", "middle", "last"):
        meta, Q, frames, disp, pan, mfo = _recon_inputs(world, idx=(0, 1), zero_pos=pos)
        d = tmp_path / str(pos)
        d.mkdir()
        rec = Reconstruction_scene.Reconstruction(meta, Q, d, cfg=make_config("synthetic"))
        results.append(rec(frames, disp, pan, mfo))
        n_segments = len(world.segmentation(0).segments)
        assert len(rec.results[0].objects) == n_segments
    assert all(r == results[0] for r in results[1:])
    seg_labels = {world.label(s.id) for s in world.segmentation(0).segments}
    for k in range(world.n_persons):
        seen = {v["object"] for v in results[0][0][k].values()}
        assert seen | {"person"} == seg_labels | {"person"}


def test_reconstruction_euler_head_pose() -> None:
    hp = np.array([20.0, -10.0, 5.0])
    persons = Reconstruction_scene.persons_from_legacy({3: {"b": [10, 10, 30, 30], "hp": hp}}, 4)
    assert persons[0].head is not None
    assert np.allclose(persons[0].head.direction, euler_to_forward(20.0, -10.0, 5.0))
    assert np.allclose(persons[0].head.euler_deg, hp)
    unit = np.array([0.0, 0.6, -0.8])
    persons = Reconstruction_scene.persons_from_legacy({3: {"hp": unit, "g": 2 * unit}}, 4)
    assert np.allclose(persons[0].head.direction, unit)
    assert np.allclose(persons[0].gaze.direction, unit)


def test_intrinsics_from_Q_round_trip_and_sign() -> None:
    intr = Intrinsics(500.0, 480.0, 320.5, 180.25, 640, 360, 0.12)
    back = Reconstruction_scene.intrinsics_from_Q(intr.Q(), 640, 360)
    for name in ("fx", "fy", "cx", "cy", "baseline"):
        assert getattr(back, name) == pytest.approx(getattr(intr, name))
    disp = np.full((360, 640), 30.0, np.float32)
    disp[0, 0] = 0.0
    a = Reconstruction_scene.depth_from_disparity(disp, intr.Q())
    Qneg = intr.Q()
    Qneg[3, 2] *= -1.0
    b = Reconstruction_scene.depth_from_disparity(disp, Qneg)
    assert np.array_equal(a.depth, b.depth) and np.array_equal(a.valid, b.valid)
    assert a.depth[5, 5] == pytest.approx(500.0 * 0.12 / 30.0)
    assert not a.valid[0, 0] and (a.depth[a.valid] > 0).all()


def test_reconstruction_visualize_flag(world, make_config, tmp_path, monkeypatch) -> None:
    calls: list = []
    monkeypatch.setattr(Reconstruction_scene, "draw_geometries", lambda g, i: calls.append((len(g), i)))
    meta, Q, frames, disp, pan, mfo = _recon_inputs(world, idx=(0, 1, 2))
    Reconstruction_scene.Reconstruction(meta, Q, tmp_path, cfg=make_config("synthetic"))(frames, disp, pan, mfo)
    assert calls == []
    pytest.importorskip("open3d")
    rec = Reconstruction_scene.Reconstruction(meta, Q, tmp_path, visualize=True, visualize_frame=1,
                                              cfg=make_config("synthetic"))
    rec(frames, disp, pan, mfo)
    assert len(calls) == 1 and calls[0][1] == 1 and calls[0][0] > 0


def test_reconstruction_does_not_import_open3d(tmp_path) -> None:
    code = (
        "import sys, types, numpy as np\n"
        "import Reconstruction_scene as R\n"
        "from Data import MapFramesOutputs\n"
        "from vfoa.config import load_config\n"
        "from vfoa.backends.synthetic import SyntheticWorld\n"
        "from vfoa.geometry import depth_to_disparity\n"
        "cfg = load_config(profile='thesis')\n"
        "w = SyntheticWorld.from_config(load_config(profile='synthetic'))\n"
        "seg = w.segmentation(0)\n"
        "meta = types.SimpleNamespace(thing_classes=[w.label(s.id) for s in seg.segments], stuff_classes=[])\n"
        "infos = [{'id': s.id, 'category_id': j, 'isthing': True} for j, s in enumerate(seg.segments)]\n"
        "m = MapFramesOutputs()\n"
        "m.addListEntries(0, {k: {'b': w.head_box(0, k), 'g': w.gaze(0, k)} for k in range(w.n_persons)})\n"
        "d = depth_to_disparity(w.depth(0).depth, w.intrinsics.fx, 0.1)\n"
        f"out = R.Reconstruction(meta, w.intrinsics.Q(), {str(tmp_path)!r})([w.frame(0).image], [d],"
        " [(seg.label_map, infos)], m)\n"
        "assert 0 in out\n"
        "print('O3D', 'open3d' in sys.modules)\n"
    )
    env = _env()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "thesis_final")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "O3D False" in proc.stdout


# ------------------------------------------------------------------ VideoProcessor
def test_video_processor_short_video(tiny_video, make_config, tmp_path) -> None:
    path = tiny_video(n_frames=7)
    originals = _video_frames(path)
    assert len(originals) == 7
    vp = VP.VideoProcessor(_synthetic_cfg(make_config, tmp_path / "out"), w=W, h=H)
    rec = _Recorder()
    vp.extraSinks.append(rec)
    summary = vp.processVideo(str(path))
    out_dir = tmp_path / "out" / path.stem
    assert vp.directory == out_dir and summary["frames"] == 7
    for name in ("gazes", "keypoints", "headpose", "overlay", "mosaic"):
        frames = _video_frames(out_dir / f"{name}.mp4")
        assert len(frames) == 7, name
        if name == "gazes":
            assert max(int(f.max()) for f in frames) > 100
        if name == "mosaic":
            assert frames[0].shape[:2] == (2 * H, 2 * W)
    recs = _read_jsonl(out_dir / "results.jsonl")
    assert [r["frame"] for r in recs] == list(range(7))
    for r in recs:
        assert r["t"] == pytest.approx(r["frame"] / 10.0)
    with open(out_dir / "results.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == CSV_COLUMNS and len(rows) > 1
    for r in rows[1:]:
        assert np.isfinite(float(r[1])) and float(r[1]) == pytest.approx(int(r[0]) / 10.0)
    for name in ("timing.json", "config.yaml", "features.desc"):
        assert (out_dir / name).is_file()
    desc = json.loads((out_dir / "features.desc").read_text())
    assert desc == {"Width": W, "Height": H, "NumberFrames": 7, "FrameRate": pytest.approx(10.0)}
    # the frames handed to the sinks were never modified
    assert rec.closed and [i for i, _ in rec.frames] == list(range(7))
    for (_, img), orig in zip(rec.frames, originals):
        assert np.array_equal(img, orig)


def test_video_processor_batches_consistent_ids(tiny_video, make_config, tmp_path) -> None:
    path = tiny_video(n_frames=40)
    runs = []
    for b in (16, 10000):
        vp = VP.VideoProcessor(_synthetic_cfg(make_config, tmp_path / f"b{b}"), w=W, h=H)
        vp.processVideo(str(path), imagesBatchLength=b, writeVideo=False)
        runs.append(_read_jsonl(tmp_path / f"b{b}" / path.stem / "results.jsonl"))
    a, b = runs
    assert len(a) == len(b) == 40
    for ra, rb in zip(a, b):
        assert [p["id"] for p in ra["persons"]] == [p["id"] for p in rb["persons"]]
        assert [p["target"] for p in ra["persons"]] == [p["target"] for p in rb["persons"]]
    prev: dict = {}
    for r in a:
        assert sorted(p["id"] for p in r["persons"]) == [0, 1]
        for p in r["persons"]:
            box = np.asarray(p["head_box"], dtype=float)
            c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            if p["id"] in prev:
                assert np.linalg.norm(c - prev[p["id"]]) < 0.1 * W
            prev[p["id"]] = c


def test_video_processor_without_gaze(tiny_video, make_config, tmp_path) -> None:
    path = tiny_video(n_frames=5)
    vp = VP.VideoProcessor(_synthetic_cfg(make_config, tmp_path / "out"), w=W, h=H)
    vp.processVideo(str(path), extractGaze=False, writeVideo=False)
    recs = _read_jsonl(tmp_path / "out" / path.stem / "results.jsonl")
    assert recs and all(p["gaze"] is None and p["head_dir"] is None for r in recs for p in r["persons"])


def test_video_processor_precomputed_disparity(tiny_video, make_config, tmp_path) -> None:
    n = 8
    path = tiny_video(n_frames=n)
    cfg = _synthetic_cfg(make_config, tmp_path / "ref")
    VP.VideoProcessor(cfg, w=W, h=H).processVideo(str(path), writeVideo=False)
    ref = _read_jsonl(tmp_path / "ref" / path.stem / "results.jsonl")

    world = SyntheticWorld.from_config(cfg)
    intr = world.intrinsics
    disp = [depth_to_disparity(world.depth(i).depth, intr.fx, intr.baseline) for i in range(n)]
    cfg2 = _synthetic_cfg(make_config, tmp_path / "pre")
    vp = VP.VideoProcessor(cfg2, intr.Q(), disp=disp, w=W, h=H)
    vp.processVideo(str(path), writeVideo=False)
    pre = _read_jsonl(tmp_path / "pre" / path.stem / "results.jsonl")
    assert vp.summary["backends"]["depth"] == "precomputed"
    assert [[p["target"] for p in r["persons"]] for r in pre] == [[p["target"] for p in r["persons"]] for r in ref]
    assert any(p["target"] is not None for r in pre for p in r["persons"])


def test_video_processor_stereo_without_right_view(tiny_video, make_config, tmp_path) -> None:
    import yaml

    path = tiny_video(n_frames=3)
    cfg = _synthetic_cfg(make_config, tmp_path / "out", {"depth": {"backend": "stereo_bm"}})
    VP.VideoProcessor(cfg, w=W, h=H).processVideo(str(path), writeVideo=False)
    saved = yaml.safe_load((tmp_path / "out" / path.stem / "config.yaml").read_text())
    assert saved["depth"]["backend"] == "constant"


def test_video_processor_thesis_profile_needs_thesis_models(tiny_video, make_config, tmp_path,
                                                            monkeypatch) -> None:
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    path = tiny_video(n_frames=3)
    cfg = make_config("thesis", {"output": {"dir": str(tmp_path / "out")}})
    with pytest.raises(BackendUnavailable) as exc:
        VP.VideoProcessor(cfg, w=W, h=H).processVideo(str(path), writeVideo=False)
    assert "third_party_root" in str(exc.value) or "VFOA_THIRD_PARTY" in str(exc.value)


def test_feature_sink_handles_missing_frame(tmp_path) -> None:
    from vfoa.types import FrameResult

    sink = VP.FeatureVideoSink(tmp_path, 10.0, (W, H))
    sink.write(None, FrameResult(index=0, t=0.0))
    sink.close()
    for name in VP.FeatureVideoSink.NAMES:
        assert len(_video_frames(tmp_path / f"{name}.mp4")) == 1


# ------------------------------------------------------------------ FinalTest
def test_finaltest_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc:
        FinalTest.main(["--help"])
    assert exc.value.code == 0


def test_finaltest_boolean_flags() -> None:
    p = FinalTest.build_parser()
    a = p.parse_args(["-w", "false", "-m"])
    assert a.w is False and a.m is True and a.g is True
    a = p.parse_args([])
    assert a.w is True and a.g is True and a.m is False
    with pytest.raises(SystemExit) as exc:
        p.parse_args(["-w", "maybe"])
    assert exc.value.code == 2


def test_finaltest_flag_mapping() -> None:
    args = FinalTest.build_parser().parse_args(
        ["-opM", "x.pth", "-dpT", "0.7", "-gT", "2.5", "-gB", "4", "-hM2", "b.onnx", "-PpCfg", "p.yaml",
         "-od", "somewhere", "-dev", "cpu", "--max-frames", "9", "--set", "attention.sigma_gaze_deg=5"]
    )
    cfg = VP.config_from_legacy_args(args)
    assert cfg.profile == "thesis"
    assert cfg.persons.backend == "openpose_densepose"
    assert cfg.persons.openpose_weights == "x.pth"
    assert cfg.persons.densepose_thr == pytest.approx(0.7)
    assert cfg.gaze.uncertainty_thr == pytest.approx(2.5) and cfg.gaze.batch_size == 4
    assert cfg.headpose.models[1] == "b.onnx" and cfg.headpose.models[0].endswith("688590.onnx")
    assert cfg.segmentation.panoptic_cfg == "p.yaml"
    assert cfg.output.dir == "somewhere" and cfg.models.device == "cpu"
    assert cfg.input.max_frames == 9 and cfg.attention.sigma_gaze_deg == pytest.approx(5.0)
    assert "/home/" not in json.dumps(cfg.to_dict())
    cfg = VP.config_from_legacy_args(FinalTest.build_parser().parse_args(["--profile", "synthetic"]))
    assert cfg.profile == "synthetic"


@pytest.mark.parametrize("cwd", [".", "thesis_final"])
def test_finaltest_script_synthetic(cwd, tmp_path) -> None:
    script = "thesis_final/FinalTest.py" if cwd == "." else "FinalTest.py"
    out = tmp_path / "od"
    proc = subprocess.run(
        [sys.executable, script, "--profile", "synthetic", "-od", str(out), "--max-frames", "6", "-w", "false"],
        cwd=str(ROOT / cwd), env=_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert len(_read_jsonl(out / "synthetic" / "results.jsonl")) == 6
    assert (out / "synthetic" / "timing.json").is_file()


def test_finaltest_missing_stereo_views(tmp_path) -> None:
    out = tmp_path / "od"
    proc = subprocess.run(
        [sys.executable, "thesis_final/FinalTest.py", "--profile", "synthetic", "-dvL", str(tmp_path / "L"),
         "-dvR", str(tmp_path / "R"), "-od", str(out), "--max-frames", "3", "-w", "false"],
        cwd=str(ROOT), env=_env(), capture_output=True, text=True, timeout=600,
    )
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, text[-3000:]
    assert "Traceback" not in text and "NameError" not in text
    assert "Stereo views not found" in text
    assert (out / "synthetic" / "results.jsonl").is_file()


def test_finaltest_missing_video(tmp_path, capsys) -> None:
    assert FinalTest.main(["-d", str(tmp_path / "nope.mp4")]) == 2
    assert "File doesnt exist" in capsys.readouterr().out


def test_finaltest_default_profile_reports_missing_thesis_models(tiny_video, tmp_path, capsys,
                                                                 monkeypatch) -> None:
    monkeypatch.delenv("VFOA_THIRD_PARTY", raising=False)
    path = tiny_video(n_frames=3)
    assert FinalTest.main(["-d", str(path), "-od", str(tmp_path / "od"), "-w", "false"]) == 2
    err = capsys.readouterr().err
    assert "FinalTest: error:" in err
    assert "third_party_root" in err or "VFOA_THIRD_PARTY" in err
