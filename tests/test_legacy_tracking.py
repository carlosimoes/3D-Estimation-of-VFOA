from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import BoxTracker
from BoxTracker import Tracker, TrackerMatcher
from Data import MapFramesOutputs


def test_legacy_modules_import_light(repo_root):
    code = textwrap.dedent(
        """
        import json, sys
        import BoxTracker, Data, KalmanFilter
        print(json.dumps(sorted(sys.modules)))
        """
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(repo_root / "thesis_final"), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    mods = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    for heavy in ("matplotlib", "lib", "torch"):
        assert heavy not in mods


def test_tracker_ids_persist_across_batches():
    tr = Tracker()
    a = [10, 10, 50, 50]
    b = [200, 10, 240, 50]
    m1 = tr.trackInterFrames([None, None], [[a], [a, b]])
    ids1 = {k for frame in m1.values() for k in frame}
    assert sorted(m1.keys()) == [0, 1]
    assert ids1 == {0, 1}
    a2 = [11, 10, 51, 50]
    c = [400, 300, 440, 340]
    m2 = tr.trackInterFrames([None], [[a2, None, c]])
    assert sorted(m2.keys()) == [0]
    frame = m2[0]
    by_box = {id(v["b"]): k for k, v in frame.items()}
    assert by_box[id(a2)] == 0
    assert by_box[id(c)] not in ids1
    assert frame[by_box[id(c)]]["b"] is c


class _Human:
    """Hashable stand-in for an OpenPose Human (normalised part coordinates)."""

    def __init__(self, points):
        self.body_parts = {i: SimpleNamespace(x=x, y=y) for i, (x, y) in points.items()}


def _human(points):
    return _Human(points)


def test_matcher_rules():
    tm = TrackerMatcher()
    tm.setDim(100, 100)
    box = [10, 10, 50, 50]
    no_head = _human({2: (0.3, 0.3)})
    h1 = _human({0: (0.30, 0.30), 14: (0.28, 0.28)})
    h2 = _human({0: (0.32, 0.32), 15: (0.34, 0.30)})
    m = tm.matchKeypointsBoxes([no_head, h1, h2], [box])
    assert m[no_head] is None
    assert [m[h1], m[h2]].count(box) == 1
    # overlapping boxes -> one each
    box2 = [12, 12, 52, 52]
    m = tm.matchKeypointsBoxes([h1, h2], [box, box2])
    assert m[h1] is not None and m[h2] is not None and m[h1] is not m[h2]


def test_matcher_tracking_unmatched_gets_unique_id():
    tm = TrackerMatcher()
    tm.setDim(100, 100)
    box = [10, 10, 50, 50]
    h1 = _human({0: (0.30, 0.30)})
    lonely = _human({0: (0.9, 0.9)})
    res = tm.trackInterFrames([[h1, lonely]], [[box]])
    frame = res[0]
    assert len(frame) == 2
    kinds = sorted(tuple(sorted(v)) for v in frame.values())
    assert kinds == [("b", "kp"), ("kp",)]
    res2 = tm.trackInterFrames([[h1, lonely]], [[box]])
    ids2 = set(res2[0])
    assert len(ids2) == 2
    assert ids2 & set(frame) == {k for k, v in frame.items() if "b" in v}


def test_data_contains_and_append():
    m = MapFramesOutputs()
    m.addListEntries(3, {7: {"b": [0, 0, 1, 1]}})
    assert 3 in m
    assert 0 not in m
    assert m[3] == {7: {"b": [0, 0, 1, 1]}}
    base = MapFramesOutputs()
    base.addListEntries(0, {0: {"b": 1}})
    base.addListEntries(2, {1: {"b": 2}})
    other = MapFramesOutputs()
    other.addListEntries(0, {5: {"b": 3}})
    offset = base.appendMap(other)
    assert offset == 3
    assert sorted(base.keys()) == [0, 2, 3]
    assert base[2] == {1: {"b": 2}}
    assert base[3] == {5: {"b": 3}}


def test_iou_known_values():
    tr = Tracker()
    assert tr.intersectionOverUnionRatio([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert tr.intersectionOverUnionRatio([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert tr.intersectionOverUnionRatio([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert BoxTracker.HEAD_PARTS == [14, 16, 15, 17, 0]
