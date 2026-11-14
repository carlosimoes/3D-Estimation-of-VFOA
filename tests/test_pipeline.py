"""Tests of the streaming pipeline (vfoa.pipeline) against the synthetic world."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import types
from typing import Any, Optional

import numpy as np
import pytest

from vfoa.backends.synthetic import (
    SyntheticDepth,
    SyntheticGaze,
    SyntheticHeadPose,
    SyntheticPersons,
    SyntheticSegmentation,
    SyntheticWorld,
)
from vfoa.io.outputs import result_to_dict
from vfoa.pipeline import Pipeline, open_input
from vfoa.tracking import PersonTracker
from vfoa.types import DepthFrame

NONE_SMOOTHING = {"smoothing": {"direction": "none", "target": "none"}}
MODEL_STAGES = ("depth", "persons", "headpose", "gaze", "segmentation")
SUMMARY_STAGES = ("depth", "persons", "tracking", "headpose", "gaze", "segmentation",
                  "scene", "smoothing", "attention", "read", "output")


# ---------------------------------------------------------------------- helpers
class ListSink:
    def __init__(self) -> None:
        self.results: list = []
        self.closed = False

    def write(self, frame: Any, result: Any) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def merge(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
    return out


def fakes(cfg, **overrides) -> dict:
    """Fresh Synthetic* backends for every kind (overrides: kind -> class or None)."""
    classes = {
        "depth": SyntheticDepth,
        "persons": SyntheticPersons,
        "headpose": SyntheticHeadPose,
        "gaze": SyntheticGaze,
        "segmentation": SyntheticSegmentation,
    }
    classes.update(overrides)
    return {k: (None if c is None else c(cfg)) for k, c in classes.items()}


def run_cfg(cfg, backends=None):
    """Run the whole configured input; returns (results, summary, pipeline)."""
    pipe = Pipeline(cfg, backends=backends)
    sink = ListSink()
    summary = pipe.run(open_input(cfg), [sink])
    assert sink.closed
    return sink.results, summary, pipe


def seg_targets(world: SyntheticWorld, result) -> dict:
    """{world person index: target segment id or None} of one FrameResult."""
    seg_of = {o.object_id: o.segment_id for o in result.objects}
    out = {}
    for p in result.persons:
        k = world.match_person(result.index, p.detection.head_box)
        if k is None:
            continue
        tgt = result.targets.get(p.track_id)
        out[k] = None if tgt is None else seg_of.get(tgt)
    return out


def accuracy(world, results):
    ok = tot = 0
    misses = []
    for r in results:
        got = seg_targets(world, r)
        for k, t in world.truth(r.index).items():
            if t is None:
                continue
            tot += 1
            if got.get(k) == t:
                ok += 1
            else:
                misses.append((r.index, k))
    return ok, tot, misses


def comparable(results) -> list:
    out = []
    for r in results:
        d = result_to_dict(r)
        d.pop("timings_ms")
        out.append(d)
    return out


# ---------------------------------------------------------------------- end to end
def test_truth_exact_without_smoothing(make_config):
    cfg = make_config("synthetic", NONE_SMOOTHING)
    world = SyntheticWorld.from_config(cfg)
    results, summary, _ = run_cfg(cfg)
    assert [r.index for r in results] == list(range(world.n_frames))
    ok, tot, misses = accuracy(world, results)
    assert tot == 240 and ok == tot, misses[:20]
    person_target = False
    for r in results:
        for p in r.persons:
            tgt = r.targets.get(p.track_id)
            assert tgt is None or tgt != p.own_object_id
        for k, sid in seg_targets(world, r).items():
            if sid is not None and sid > world.n_objects and sid != world.person_segment_id(k):
                person_target = True
    assert person_target
    assert summary["frames"] == world.n_frames


def test_default_smoothing_accuracy(make_config):
    cfg = make_config("synthetic")
    world = SyntheticWorld.from_config(cfg)
    results, _, _ = run_cfg(cfg)
    ok, tot, misses = accuracy(world, results)
    assert ok / tot >= 0.90
    # every miss is a short transition right after a scripted switch (every 30 frames)
    assert all(i % 30 < 8 for i, _ in misses), misses
    for r in results:
        for p in r.persons:
            tgt = r.targets.get(p.track_id)
            assert tgt is None or tgt != p.own_object_id


# ---------------------------------------------------------------------- scheduling
def test_multirate_calls_ran_loads(make_config):
    strides = {"segmentation": 5, "gaze": 2, "headpose": 3, "depth": 1, "persons": 1}
    cfg = make_config("synthetic", {"input": {"max_frames": 31}, "segmentation": {"stride": 5},
                                    "gaze": {"stride": 2}, "headpose": {"stride": 3}})
    bk = fakes(cfg)
    results, summary, _ = run_cfg(cfg, bk)
    assert len(results) == 31
    expected = {s: len(range(0, 31, strides[s])) for s in strides}
    assert expected == {"segmentation": 7, "gaze": 16, "headpose": 11, "depth": 31, "persons": 31}
    for s, b in bk.items():
        assert b.calls == expected[s], s
        assert b.loaded == 1, s
    assert bk["gaze"].observed == 31
    for r in results:
        for s in strides:
            assert r.ran[s] == (r.index % strides[s] == 0), (s, r.index)
        assert r.ran["tracking"] and r.ran["scene"] and r.ran["smoothing"] and r.ran["attention"]
    for s in strides:
        assert summary["stages"][s]["runs"] == expected[s]
    # held estimates in between: every person has both cues on every frame
    for r in results:
        for p in r.persons:
            assert p.gaze is not None and p.head is not None

    cfg2 = make_config("synthetic", {"input": {"max_frames": 13},
                                     "segmentation": {"stride": 5, "phase": 2}})
    results2, _, _ = run_cfg(cfg2, fakes(cfg2))
    assert [r.index for r in results2 if r.ran["segmentation"]] == [0, 2, 7, 12]


def test_schedule_survives_frame_gaps(make_config):
    cfg = make_config("synthetic", {"segmentation": {"stride": 5}})
    world = SyntheticWorld.from_config(cfg)
    bk = fakes(cfg)
    pipe = Pipeline(cfg, backends=bk).open(world.fps, None)
    ran = []
    for i in (0, 3, 7, 12, 16, 21):
        (r,) = pipe.process(world.frame(i))
        ran.append((r.index, r.ran["segmentation"]))
    pipe.close()
    # one run per stride bucket, even though no index is a multiple of 5 after 0
    assert ran == [(0, True), (3, False), (7, True), (12, True), (16, True), (21, True)]
    assert bk["segmentation"].calls == 5


class EarlyGaze(SyntheticGaze):
    """Answers only for frames < 3."""

    def estimate(self, frame, persons):
        out = super().estimate(frame, persons)
        return out if frame.index < 3 else {}


@pytest.mark.parametrize("smoothing", ["none", "default"])
def test_hold_then_drop(make_config, smoothing):
    ov = {"input": {"max_frames": 12}, "gaze": {"max_hold_frames": 4}, "headpose": {"backend": "none"}}
    if smoothing == "none":
        ov = merge(ov, NONE_SMOOTHING)
    cfg = make_config("synthetic", ov)
    bk = fakes(cfg, gaze=EarlyGaze, headpose=None)
    results, _, _ = run_cfg(cfg, bk)
    ref = {p.track_id: p.gaze for p in results[2].persons}
    for r in results:
        assert r.ran["gaze"]
        for p in r.persons:
            if r.index <= 6:
                assert p.gaze is not None
                if r.index >= 3 and smoothing == "none":
                    assert p.gaze.frame_index == 2
                    np.testing.assert_array_equal(p.gaze.direction, ref[p.track_id].direction)
            else:
                assert p.gaze is None and p.head is None
                assert r.targets[p.track_id] is None


# ---------------------------------------------------------------------- cache
def test_cache_rescoring_without_models(make_config, tmp_path):
    base = {"input": {"max_frames": 40}, "cache": {"enabled": True, "dir": str(tmp_path / "cache")}}
    cfg1 = make_config("synthetic", base)
    res1, sum1, _ = run_cfg(cfg1, fakes(cfg1))
    assert set(sum1["cache"]) == set(MODEL_STAGES)
    assert all(c["complete"] and c["hits"] == 0 for c in sum1["cache"].values())

    cfg2 = make_config("synthetic", merge(base, {
        "cache": {"mode": "read"},
        "attention": {"sigma_gaze_deg": 5.0},
        "smoothing": {"target": "hysteresis"},
    }))
    bk2 = fakes(cfg2)
    res2, sum2, _ = run_cfg(cfg2, bk2)
    assert len(res2) == 40
    for b in bk2.values():
        assert b.loaded == 0 and b.calls == 0 and not b.is_loaded
    assert bk2["gaze"].observed == 0
    assert all(c["misses"] == 0 and c["hits"] > 0 for c in sum2["cache"].values())
    assert not any(k.startswith("load:") for k in sum2["setup_ms"])
    s1 = [s.score_total for r in res1 for s in r.scores]
    s2 = [s.score_total for r in res2 for s in r.scores]
    assert len(s1) == len(s2) and s1 != s2

    cfg3 = make_config("synthetic", merge(base, {"cache": {"mode": "read"}}))
    bk3 = fakes(cfg3)
    res3, _, _ = run_cfg(cfg3, bk3)
    assert all(b.calls == 0 for b in bk3.values())
    assert comparable(res3) == comparable(res1)


# ---------------------------------------------------------------------- lookahead
class RecordingGaze(SyntheticGaze):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_seen = -1
        self.at_estimate: dict[int, int] = {}

    def observe(self, frame, persons):
        super().observe(frame, persons)
        self.max_seen = max(self.max_seen, frame.index)

    def estimate(self, frame, persons):
        self.at_estimate[frame.index] = self.max_seen
        return super().estimate(frame, persons)


def test_lookahead_order_and_future(make_config):
    n, L = 30, 6
    cfg = make_config("synthetic", {"runtime": {"lookahead_frames": L}})
    world = SyntheticWorld.from_config(cfg)
    bk = fakes(cfg, gaze=RecordingGaze)
    pipe = Pipeline(cfg, backends=bk).open(world.fps, None)
    out = []
    for i in range(n):
        got = pipe.process(world.frame(i))
        if i < L:
            assert got == []
        else:
            assert len(got) == 1
        out += got
    rest = pipe.flush()
    assert len(rest) == L
    out += rest
    pipe.close()
    assert [r.index for r in out] == list(range(n))
    assert bk["gaze"].at_estimate == {t: min(t + L, n - 1) for t in range(n)}

    cfg0 = make_config("synthetic", {"input": {"max_frames": n}})
    res0, _, _ = run_cfg(cfg0, fakes(cfg0))
    assert [r.targets for r in out] == [r.targets for r in res0]


# ---------------------------------------------------------------------- robustness
class NoHeadBoxPersons(SyntheticPersons):
    def detect(self, frame):
        return [dataclasses.replace(d, head_box=None) for d in super().detect(frame)]


class InvalidDepth(SyntheticDepth):
    def compute(self, frame):
        d = super().compute(frame)
        return DepthFrame(np.zeros_like(d.depth), np.zeros(d.depth.shape, dtype=bool), d.intrinsics)


ROBUST = {
    "no_persons": ({"synthetic": {"n_persons": 0}}, {}),
    "no_head_box": ({}, {"persons": NoHeadBoxPersons}),
    "gaze_none": ({"gaze": {"backend": "none"}}, {"gaze": None}),
    "no_cues": ({"gaze": {"backend": "none"}, "headpose": {"backend": "none"}},
                {"gaze": None, "headpose": None}),
    "segmentation_none": ({"segmentation": {"backend": "none"}}, {"segmentation": None}),
    "invalid_depth": ({}, {"depth": InvalidDepth}),
    "no_objects": ({"synthetic": {"n_objects": 0}}, {}),
}


@pytest.mark.parametrize("case", sorted(ROBUST))
def test_robustness(make_config, case):
    ov, over = ROBUST[case]
    cfg = make_config("synthetic", merge({"input": {"max_frames": 40}}, NONE_SMOOTHING, ov))
    world = SyntheticWorld.from_config(cfg)
    results, _, _ = run_cfg(cfg, fakes(cfg, **over))
    assert len(results) == 40
    for r in results:
        json.dumps(result_to_dict(r), allow_nan=False)
    if case == "no_persons":
        assert all(r.targets == {} and r.persons == [] for r in results)
    elif case in ("no_head_box", "no_cues", "segmentation_none", "invalid_depth"):
        assert all(r.persons for r in results)
        assert all(t is None for r in results for t in r.targets.values())
    elif case == "gaze_none":
        ok, tot, misses = accuracy(world, results)
        assert ok == tot, misses
        assert all(s.score_gaze is None for r in results for s in r.scores)
        assert any(r.scores for r in results)
    if case == "segmentation_none":
        assert all(r.objects == [] and not r.ran["segmentation"] for r in results)


def test_depth_none_rejected(make_config):
    cfg = make_config("synthetic", {"depth": {"backend": "none"}})
    with pytest.raises(ValueError, match="depth.backend"):
        Pipeline(cfg).open(30.0)


def test_process_before_open_raises(make_config):
    cfg = make_config("synthetic")
    with pytest.raises(RuntimeError):
        Pipeline(cfg).process(SyntheticWorld.from_config(cfg).frame(0))


# ---------------------------------------------------------------------- timing
def test_timings_and_summary(make_config):
    cfg = make_config("synthetic", {"input": {"max_frames": 31}, "gaze": {"stride": 2},
                                    "headpose": {"stride": 3}})
    results, summary, _ = run_cfg(cfg, fakes(cfg))
    for r in results:
        for s in ("depth", "persons", "tracking", "headpose", "gaze", "segmentation"):
            assert (s in r.timings_ms) == r.ran[s], (s, r.index)
        for s in ("scene", "smoothing", "attention", "read", "total"):
            assert s in r.timings_ms
        assert r.timings_ms["total"] >= 0.0
    for s in SUMMARY_STAGES:
        assert "runs" in summary["stages"][s] and "per_frame_ms" in summary["stages"][s], s
    assert summary["stages"]["gaze"]["runs"] == 16
    assert {k for k in summary["setup_ms"] if k.startswith("load:")} == {f"load:{k}" for k in MODEL_STAGES}
    assert summary["frames"] == 31
    json.dumps(summary)


def test_timing_disabled(make_config):
    cfg = make_config("synthetic", {"input": {"max_frames": 5}, "timing": {"enabled": False}})
    results, summary, pipe = run_cfg(cfg, fakes(cfg))
    assert len(results) == 5
    assert all(r.timings_ms == {} for r in results)
    assert summary["frames"] == 5 and "stages" not in summary
    assert pipe.timer.report() == ""


# ---------------------------------------------------------------------- tracking ids
@pytest.mark.parametrize("stride", [1, 4])
def test_track_ids_match_continuous_tracker(make_config, stride):
    cfg = make_config("synthetic", {"input": {"max_frames": 100}, "synthetic": {"n_persons": 3},
                                    "persons": {"stride": stride}})
    world = SyntheticWorld.from_config(cfg)
    results, _, _ = run_cfg(cfg, fakes(cfg))
    assert len(results) == 100
    tracker = PersonTracker(cfg)
    expected = None
    for r in results:
        if r.index % stride == 0:
            expected = [(p.track_id, tuple(p.detection.head_box))
                        for p in tracker.update(world.detections(r.index), r.index)]
        got = [(p.track_id, tuple(p.detection.head_box)) for p in r.persons]
        assert got == expected, r.index
        ids = [p.track_id for p in r.persons]
        assert len(ids) == len(set(ids)) == 3
    assert {p.track_id for r in results for p in r.persons} == {0, 1, 2}


def test_reopen_resets_without_reload(make_config):
    cfg = make_config("synthetic", {"input": {"max_frames": 20}})
    bk = fakes(cfg)
    pipe = Pipeline(cfg, backends=bk)
    sink1, sink2 = ListSink(), ListSink()
    s1 = pipe.run(open_input(cfg), [sink1])
    s2 = pipe.run(open_input(cfg), [sink2])
    assert all(b.loaded == 1 for b in bk.values())
    assert comparable(sink1.results) == comparable(sink2.results)
    assert len(s1["setup_ms"]) == 5 and s2["setup_ms"] == {}
    assert s2["frames"] == 20


# ---------------------------------------------------------------------- async hook
def test_async_fallback_and_fake_runner(make_config, monkeypatch):
    base = {"input": {"max_frames": 12}}
    cfg_sync = make_config("synthetic", base)
    ref, _, _ = run_cfg(cfg_sync, fakes(cfg_sync))

    cfg = make_config("synthetic", merge(base, {"segmentation": {"async": True}}))
    monkeypatch.setitem(sys.modules, "vfoa.realtime", None)
    with pytest.warns(RuntimeWarning, match="vfoa.realtime"):
        res, _, _ = run_cfg(cfg, fakes(cfg))
    assert comparable(res) == comparable(ref)

    runners: list = []

    class LateRunner:
        """Delivers each result one poll later than a synchronous runner."""

        def __init__(self, fn, name):
            self.fn, self.name = fn, name
            self.queued: Optional[Any] = None
            self.ready: Optional[tuple] = None
            self.closed = False
            runners.append(self)

        def submit(self, frame):
            if self.queued is not None:
                return False
            self.queued = frame
            return True

        def poll(self):
            out, self.ready = self.ready, None
            if self.queued is not None:
                self.ready = (self.queued.index, self.fn(self.queued))
                self.queued = None
            return out

        def close(self):
            self.closed = True

    fake = types.ModuleType("vfoa.realtime")
    fake.AsyncStageRunner = LateRunner
    monkeypatch.setitem(sys.modules, "vfoa.realtime", fake)
    bk = fakes(cfg)
    res2, _, _ = run_cfg(cfg, bk)
    assert len(runners) == 1 and runners[0].name == "segmentation" and runners[0].closed
    assert res2[0].objects == [] and not res2[0].ran["segmentation"]
    assert res2[1].objects and res2[1].ran["segmentation"]
    # submitted on frames 0, 5, 10 (stride 5); each result arrives one frame later
    assert bk["segmentation"].calls == 3
    assert [r.index for r in res2 if r.ran["segmentation"]] == [1, 6, 11]


# ---------------------------------------------------------------------- import hygiene
def test_import_is_light(repo_root):
    code = (
        "import sys; import vfoa.pipeline, vfoa.cli, vfoa.__main__; "
        "bad = [m for m in ('torch', 'ultralytics', 'open3d', 'matplotlib') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
