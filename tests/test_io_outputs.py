"""Tests of vfoa.io.outputs (JSONL/CSV sinks, overlays, video, mosaic, display)."""

from __future__ import annotations

import builtins
import csv
import json
import math
import subprocess
import sys

import cv2
import numpy as np
import pytest

from vfoa.io import outputs as O
from vfoa.types import (
    AttentionScore,
    DirectionEstimate,
    Frame,
    FrameResult,
    Intrinsics,
    Person,
    PersonDetection,
    SceneObject,
)

INTR = Intrinsics(fx=200.0, fy=200.0, cx=159.5, cy=119.5, width=320, height=240, baseline=0.1)


def _object(oid: int, seg: int, label: str, centre) -> SceneObject:
    c = np.asarray(centre, dtype=np.float64)
    return SceneObject(
        object_id=oid, segment_id=seg, label=label, is_thing=True,
        aabb_min=c - 0.1, aabb_max=c + 0.1, centroid=c,
        points=np.zeros((0, 3), np.float32), n_points=np.int64(30),
    )


def _result(index: int = 3, fps: float = 30.0) -> FrameResult:
    full = Person(
        track_id=0,
        detection=PersonDetection(head_box=[100, 60, 140, 110], score=0.9),
        gaze=DirectionEstimate([1.0, 0.0, 0.0], confidence=np.float32(0.8)),
        head=DirectionEstimate([0.0, 0.0, -1.0], confidence=0.7, euler_deg=[10.0, 5.0, 0.0]),
        eyes_px=np.array([120.0, 78.0], np.float32),
        eyes_3d=np.array([-0.2, -0.2, 2.0]),
        own_object_id=7,
    )
    bare = Person(track_id=1, detection=PersonDetection(head_box=None))
    objects = [_object(5, 1, "cup", [0.3, 0.0, 2.0]), _object(6, 2, "book", [-0.4, 0.1, 3.0])]
    scores = [
        AttentionScore(0, 5, "cup", np.float32(0.9), None, np.float64(0.85)),
        AttentionScore(0, 6, "book", np.float64(0.1), float("nan"), np.float32(0.05), occluded=np.bool_(True)),
    ]
    return FrameResult(
        index=index, t=index / fps, persons=[full, bare], objects=objects, scores=scores,
        targets={0: 5, 1: None}, timings_ms={"total": np.float64(12.5), "depth": 1.0},
    )


def _assert_plain(x) -> None:
    if isinstance(x, dict):
        for k, v in x.items():
            assert type(k) is str
            _assert_plain(v)
    elif isinstance(x, list):
        for v in x:
            _assert_plain(v)
    else:
        assert x is None or type(x) in (str, int, float, bool), type(x)
        if type(x) is float:
            assert math.isfinite(x)


def test_result_to_dict_keys_and_types():
    d = O.result_to_dict(_result())
    assert set(d) == {"frame", "t", "persons", "objects", "attention", "timings_ms"}
    assert set(d["persons"][0]) == {
        "id", "head_box", "eyes_3d", "gaze", "gaze_conf", "head_dir", "head_conf", "target", "target_label"
    }
    assert set(d["objects"][0]) == {"id", "segment_id", "label", "is_thing", "aabb_min", "aabb_max", "centroid"}
    assert set(d["attention"][0]) == {
        "person_id", "object_id", "label", "score_gaze", "score_head", "score_total", "occluded"
    }
    _assert_plain(d)
    p0, p1 = d["persons"]
    assert p0["target"] == 5 and p0["target_label"] == "cup"
    assert p0["gaze"] == [1.0, 0.0, 0.0] and p0["head_box"] == [100.0, 60.0, 140.0, 110.0]
    assert all(p1[k] is None for k in p1 if k != "id")
    assert d["attention"][0]["score_head"] is None
    assert d["attention"][1]["score_head"] is None  # NaN -> None
    assert d["attention"][1]["occluded"] is True
    assert d["frame"] == 3 and d["t"] == 3 / 30.0
    json.dumps(d, allow_nan=False)


def test_jsonl_one_object_per_frame(tmp_path):
    path = tmp_path / "sub" / "results.jsonl"
    with O.JsonlSink(path) as sink:
        for i in range(4):
            sink.write(None, _result(index=i, fps=25.0))
    sink.close()  # idempotent
    raw = path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 4
    for i, line in enumerate(lines):
        obj = json.loads(line)
        assert obj["frame"] == i and obj["t"] == i / 25.0
        assert obj["persons"][1]["target"] is None


def test_csv_rows(tmp_path):
    path = tmp_path / "scores.csv"
    sink = O.CsvSink(path)
    sink.write(None, _result(index=0))
    sink.write(None, _result(index=1))
    sink.close()
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    assert tuple(rows[0]) == O.CSV_COLUMNS
    assert rows[0] == ["frame", "t", "person_id", "object_id", "label", "score_gaze", "score_head",
                       "score_total", "occluded", "is_target"]
    body = rows[1:]
    assert len(body) == 4
    for r in body:
        rec = dict(zip(rows[0], r))
        assert rec["is_target"] == ("1" if (rec["person_id"], rec["object_id"]) == ("0", "5") else "0")
        assert rec["score_head"] == ""
    assert body[1][8] == "1"
    assert float(body[2][1]) == 1 / 30.0


def test_csv_opens_file_once(tmp_path, monkeypatch):
    calls = []
    real_open = builtins.open

    def counting(*a, **k):
        calls.append(a[0] if a else k.get("file"))
        return real_open(*a, **k)

    monkeypatch.setattr(builtins, "open", counting)
    sink = O.CsvSink(tmp_path / "x.csv")
    for i in range(5):
        sink.write(None, _result(index=i))
    sink.close()
    monkeypatch.setattr(builtins, "open", real_open)
    assert len(calls) == 1


def test_draw_overlay_new_image():
    img = np.full((240, 320, 3), 30, np.uint8)
    before = img.copy()
    out = O.draw_overlay(img, _result(), INTR)
    assert out is not img
    assert not np.shares_memory(out, img)
    assert np.array_equal(img, before)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_draw_overlay_missing_cues():
    img = np.zeros((240, 320, 3), np.uint8)
    res = _result()
    res.persons.append(Person(track_id=2, detection=PersonDetection(head_box=[10, 10, 30, 40])))
    res.persons.append(
        Person(track_id=3, detection=PersonDetection(head_box=None),
               gaze=DirectionEstimate([0, 0, 1.0]), eyes_3d=np.array([np.nan, 0, 1.0]))
    )
    res.persons[0].eyes_3d = np.array([np.nan, np.nan, np.nan])
    for intr in (INTR, None):
        out = O.draw_overlay(img, res, intr)
        assert out.shape == img.shape
    res.objects = []
    O.draw_overlay(img, res, INTR)
    # Non-contiguous input.
    big = np.zeros((240, 640, 3), np.uint8)
    view = big[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    out = O.draw_overlay(view, _result(), INTR)
    assert out.shape == view.shape and np.count_nonzero(big) == 0
    # Empty result -> equal copy.
    empty = FrameResult(index=0, t=0.0)
    out = O.draw_overlay(img, empty, INTR)
    assert np.array_equal(out, img) and out is not img


def test_gaze_arrow_direction():
    img = np.zeros((200, 200, 3), np.uint8)
    p = Person(
        track_id=0, detection=PersonDetection(head_box=None),
        gaze=DirectionEstimate([1.0, 0.0, 0.0]), eyes_px=np.array([100.0, 100.0]),
    )
    out = O.draw_overlay(img, FrameResult(index=0, t=0.0, persons=[p]), None)
    ys, xs = np.nonzero(out.any(axis=2))
    assert xs.size > 0
    assert xs.max() >= 150
    assert xs.min() >= 95


def test_head_axes_endpoints():
    s = 50.0
    e = O.head_axes_endpoints(0, 0, 0, (0, 0), s)
    np.testing.assert_allclose(e, [[s, 0], [0, s], [0, 0]], atol=1e-12)
    e = O.head_axes_endpoints(30, 0, 0, (0, 0), s)
    assert e[2, 0] == pytest.approx(-s * math.sin(math.radians(30)))
    e = O.head_axes_endpoints(0, 20, 0, (0, 0), s)
    assert e[2, 1] < 0
    a = O.head_axes_endpoints(15, 10, 0, (5, 7), s)
    b = O.head_axes_endpoints(15, 10, 40, (5, 7), s)
    np.testing.assert_allclose(a[2], b[2])
    img = np.zeros((100, 100, 3), np.uint8)
    out = O.draw_head_axes(img, 10, 10, 10)
    assert np.count_nonzero(img) == 0 and np.count_nonzero(out) > 0


@pytest.mark.parametrize("n", [3, 40])
def test_video_sink_roundtrip(tmp_path, n):
    path = tmp_path / f"v{n}.mp4"
    sink = O.VideoSink(path, fps=10.0, overlay=True)
    for i in range(n):
        img = np.full((120, 160, 3), (i * 5) % 255, np.uint8)
        sink.write(Frame(index=i, t=i / 10.0, image=img, intrinsics=INTR), _result(index=i))
    sink.close()
    sink.close()
    assert sink.frames_written == n
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.read()[0]:
        count += 1
    cap.release()
    assert count == n


def test_video_sink_mixed_sizes_and_odd_size(tmp_path):
    path = tmp_path / "mixed.mp4"
    sink = O.VideoSink(path, fps=0)
    sizes = [(160, 120), (161, 121), (80, 60), (320, 240), (160, 120)]
    for w, h in sizes:
        sink.write_image(np.full((h, w, 3), 100, np.uint8))
    sink.write(None, None)
    sink.close()
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == len(sizes)

    path2 = tmp_path / "odd.mp4"
    sink = O.VideoSink(path2, fps=10, size=(101, 75), overlay=False)
    assert sink.size == (100, 74)
    for _ in range(3):
        sink.write_image(np.zeros((75, 101, 3), np.uint8))
    sink.close()
    cap = cv2.VideoCapture(str(path2))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    assert (w, h) == (100, 74)

    empty = O.VideoSink(tmp_path / "empty.mp4", fps=10)
    empty.close()
    assert empty.frames_written == 0


@pytest.mark.parametrize("k", [2, 3, 4])
def test_compose_mosaic(k):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), None]
    tiles = [
        np.full((100, 200, 3), colors[0], np.uint8),
        np.full((50, 50, 3), colors[1], np.uint8),
        np.full((30, 90), 128, np.uint8),
        None,
    ][:k]
    expected = [colors[0], colors[1], (128, 128, 128), (0, 0, 0)][:k]
    W, H = 400, 300
    out = O.compose_mosaic(tiles, (2, 2), (W, H))
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    for idx, col in enumerate(expected):
        r, c = divmod(idx, 2)
        cy, cx = r * H // 2 + H // 4, c * W // 2 + W // 4
        assert tuple(int(v) for v in out[cy, cx]) == col
    with pytest.raises(ValueError):
        O.compose_mosaic(tiles * 3, (1, 2), (W, H))


def test_display_sink(monkeypatch):
    calls = {"imshow": 0}

    def broken(*a, **k):
        calls["imshow"] += 1
        raise cv2.error("no display")

    monkeypatch.setattr(O.cv2, "imshow", broken)
    monkeypatch.setattr(O.cv2, "waitKey", lambda *a: -1)
    monkeypatch.setattr(O.cv2, "destroyWindow", lambda *a: None)
    frame = Frame(index=0, t=0.0, image=np.zeros((20, 20, 3), np.uint8))
    sink = O.DisplaySink()
    sink.write(frame, FrameResult(index=0, t=0.0))
    sink.write(frame, FrameResult(index=0, t=0.0))
    assert calls["imshow"] == 1 and sink.enabled is False
    sink.close()

    shown = []
    monkeypatch.setattr(O.cv2, "imshow", lambda name, img: shown.append((name, img.shape)))
    monkeypatch.setattr(O.cv2, "waitKey", lambda *a: ord("q"))
    sink = O.DisplaySink(window="w")
    assert not sink.stop_requested
    sink.write(frame, FrameResult(index=0, t=0.0))
    assert shown == [("w", (20, 20, 3))] and sink.stop_requested
    sink.close()


def test_import_hygiene(repo_root):
    code = (
        "import sys, vfoa.io.sources, vfoa.io.outputs\n"
        "bad = [m for m in ('ffmpeg', 'matplotlib', 'torch', 'PIL', 'open3d', 'vfoa.backends.synthetic')"
        " if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
