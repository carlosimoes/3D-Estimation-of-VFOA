from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

from vfoa.tracking import PersonTracker, box_iou, iou_matrix
from vfoa.types import PersonDetection

REPO = Path(__file__).resolve().parent.parent


def det(x1, y1, w=40, h=40, score=0.9, body=None, head=True):
    hb = [x1, y1, x1 + w, y1 + h] if head else None
    return PersonDetection(head_box=hb, body_box=body, score=score)


def ids(persons):
    return [p.track_id for p in persons]


def test_iou_matrix_values():
    a = np.array([[0, 0, 10, 10], [0, 0, 0, 10]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    np.testing.assert_allclose(m[0], [1.0, 1 / 3, 0.0])
    np.testing.assert_allclose(m[1], 0.0)  # zero-area box
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert np.all(np.isfinite(m))


def test_crossing_boxes_keep_ids():
    tr = PersonTracker()
    for i in range(50):
        a = det(10 + 4 * i, 50)
        b = det(213 - 4 * i, 50)
        dets = [a, b] if i % 2 == 0 else [b, a]
        persons = tr.update(dets, i)
        got = {id(p.detection): p.track_id for p in persons}
        assert got[id(a)] == 0, i
        assert got[id(b)] == 1, i


def test_reacquire_within_max_age_then_new_id():
    tr = PersonTracker(max_age=3, motion=False)
    assert ids(tr.update([det(10, 10)], 0)) == [0]
    for f in range(1, 4):  # 3 misses
        assert tr.update([], f) == []
    assert ids(tr.update([det(10, 10)], 4)) == [0]
    for f in range(5, 9):  # 4 misses
        tr.update([], f)
    assert ids(tr.update([det(10, 10)], 9)) == [1]


def test_overlapping_detections_never_share_id():
    tr = PersonTracker()
    tr.update([det(10, 10)], 0)
    persons = tr.update([det(11, 10), det(12, 11)], 1)
    got = ids(persons)
    assert len(set(got)) == 2
    assert 0 in got
    tr2 = PersonTracker(type="greedy_iou", iou_thr=0.5, max_age=0)
    tr2.update([det(10, 10)], 0)
    got = ids(tr2.update([det(11, 10), det(12, 11)], 1))
    assert len(set(got)) == 2


def test_new_ids_in_detection_order_from_zero():
    tr = PersonTracker()
    assert ids(tr.update([det(300, 10), det(10, 10), det(150, 10)], 0)) == [0, 1, 2]
    assert ids(tr.update([det(500, 200), det(10, 10)], 1)) == [3, 1]


def test_ids_monotonic_never_reused_over_three_batches():
    tr = PersonTracker(max_age=2)
    seen: list[int] = []
    frame = 0
    for batch in range(3):
        for k in range(20):
            dets = []
            if (k // 5) % 2 == 0:
                dets.append(det(10 + k, 10))
            if k % 7 != 3:
                dets.append(det(200, 100 + k))
            if k >= 10 and batch != 1:
                dets.append(det(400, 300))
            persons = tr.update(dets, frame)
            frame += 1
            got = ids(persons)
            assert len(got) == len(set(got))
            for tid in got:
                if tid not in seen:
                    assert all(tid > s for s in seen)
                    seen.append(tid)
    assert seen == sorted(seen)
    assert len(seen) > 3


def test_body_box_fallback():
    tr = PersonTracker()
    for f in range(10):
        body = [20 + 2 * f, 20, 120 + 2 * f, 300]
        p = tr.update([PersonDetection(head_box=None, body_box=body)], f)
        assert ids(p) == [0]
    tr = PersonTracker()
    for f in range(10):
        body = [20 + 2 * f, 20, 120 + 2 * f, 300]
        head = [50 + 2 * f, 20, 90 + 2 * f, 60] if f % 3 == 0 else None
        p = tr.update([PersonDetection(head_box=head, body_box=body)], f)
        assert ids(p) == [0], f


def test_greedy_iou_previous_update_only():
    tr = PersonTracker(type="greedy_iou", iou_thr=0.5, max_age=0)
    assert ids(tr.update([det(10, 10), det(100, 10)], 0)) == [0, 1]
    assert ids(tr.update([det(11, 10)], 1)) == [0]
    # id 1 was missing on the previous update -> new id
    assert ids(tr.update([det(12, 10), det(100, 10)], 2)) == [0, 2]
    # missing one update loses the id
    tr.update([], 3)
    assert ids(tr.update([det(12, 10)], 4)) == [3]


def test_low_score_continues_but_never_starts_a_track():
    tr = PersonTracker()
    assert tr.update([det(10, 10, score=0.3)], 0) == []
    assert ids(tr.update([det(10, 10, score=0.9)], 1)) == [0]
    assert ids(tr.update([det(11, 10, score=0.3)], 2)) == [0]
    assert tr.update([det(300, 10, score=0.3)], 3) == []
    assert tr.update([det(10, 10, score=0.05)], 4) == []


def test_empty_and_boxless_never_raise_and_reset_restarts_at_zero():
    tr = PersonTracker()
    assert tr.update([], 0) == []
    p = tr.update([PersonDetection(head_box=None), det(10, 10)], 1)
    assert ids(p) == [0, 1]
    p = tr.update([PersonDetection(head_box=[np.nan, 0, 1, 1]), det(10, 10)], 2)
    assert ids(p) == [2, 1]
    tr.reset()
    assert ids(tr.update([det(10, 10)], 0)) == [0]
    with pytest.raises(ValueError):
        PersonTracker(type="nope")


def test_same_detection_object_returned():
    tr = PersonTracker()
    d = det(10, 10)
    (p,) = tr.update([d], 0)
    assert p.detection is d


STDLIB = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}


@pytest.mark.parametrize("name", ["vfoa/tracking.py", "vfoa/smoothing.py"])
def test_module_level_imports_are_light(name):
    tree = ast.parse((REPO / name).read_text())
    allowed = STDLIB | {"numpy", "scipy"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            assert m == "vfoa.types" or m.split(".")[0] in allowed, m


def moving(f, head=True, body=True, step=20.0):
    x = 100.0 + step * f
    hb = [x + 25, 50, x + 55, 80] if head else None
    bb = [x, 50, x + 80, 250] if body else None
    return PersonDetection(head_box=hb, body_box=bb, score=0.9)


def test_default_tracks_small_fast_head_on_body_box(make_config):
    cfg = make_config("fast")
    tr = PersonTracker(cfg)
    assert tr.box_kind == "body" and PersonTracker().box_kind == "body"
    assert [tr.update([moving(f)], f)[0].track_id for f in range(15)] == [0] * 15
    tr = PersonTracker(cfg, box="head")
    got = [tr.update([moving(f)], f)[0].track_id for f in range(15)]
    assert len(set(got)) > 1
    assert PersonTracker(make_config("thesis")).box_kind == "head"


def test_head_only_detection_falls_back_to_head_box():
    tr = PersonTracker()
    got = [tr.update([moving(f, body=f not in (6, 7))], f)[0].track_id for f in range(15)]
    assert got == [0] * 15
    tr = PersonTracker()
    assert [tr.update([det(10 + 2 * f, 10)], f)[0].track_id for f in range(8)] == [0] * 8
