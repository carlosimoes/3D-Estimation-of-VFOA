from __future__ import annotations

import json

import pytest

from vfoa.timing import CANONICAL_STAGES, StageTimer


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.now

    def advance(self, ms: float) -> None:
        self.now += ms / 1000.0


def test_canonical_stages():
    assert CANONICAL_STAGES == (
        "read", "depth", "persons", "tracking", "headpose", "gaze", "segmentation",
        "scene", "smoothing", "attention", "output", "total",
    )


def test_stage_and_add_accumulate():
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    with timer.stage("depth"):
        clock.advance(10)
    with timer.stage("depth"):
        clock.advance(5)
    timer.add("persons", 0.020)
    out = timer.frame_done()
    assert out == pytest.approx({"depth": 15.0, "persons": 20.0, "total": 35.0})
    summ = timer.summary()
    assert summ["stages"]["depth"]["runs"] == 1  # several adds in one frame = one run


def test_exception_still_recorded():
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    with pytest.raises(RuntimeError):
        with timer.stage("scene"):
            clock.advance(7)
            raise RuntimeError("boom")
    assert timer.frame_done() == pytest.approx({"scene": 7.0, "total": 7.0})


def test_summary_statistics():
    clock = FakeClock()
    timer = StageTimer(clock=clock)
    for i, ms in enumerate([10.0, 20.0, 30.0, 40.0]):
        with timer.stage("depth"):
            clock.advance(ms)
        if i % 2 == 0:
            timer.add("segmentation", 0.050)
        timer.frame_done()
    timer.add_setup("load:persons", 1.5)
    timer.count("dropped_frames")
    timer.count("dropped_frames", 2)

    s = timer.summary()
    assert s["frames"] == 4
    depth = s["stages"]["depth"]
    assert depth["runs"] == 4
    assert depth["mean_ms"] == pytest.approx(25.0)
    assert depth["p50_ms"] == pytest.approx(25.0)
    assert depth["p95_ms"] == pytest.approx(38.5)
    assert depth["max_ms"] == pytest.approx(40.0)
    assert depth["total_ms"] == pytest.approx(100.0)
    assert depth["per_frame_ms"] == pytest.approx(25.0)
    seg = s["stages"]["segmentation"]
    assert seg["runs"] == 2
    assert seg["mean_ms"] == pytest.approx(50.0)
    assert seg["per_frame_ms"] == pytest.approx(100.0 / 4)
    total = s["stages"]["total"]
    assert total["total_ms"] == pytest.approx(200.0)
    assert s["fps"] == pytest.approx(1000.0 / 50.0)
    assert sum(v["share"] for k, v in s["stages"].items() if k != "total") == pytest.approx(1.0)
    assert total["share"] == pytest.approx(1.0)
    for st in s["stages"].values():
        assert set(st) == {"runs", "mean_ms", "p50_ms", "p95_ms", "max_ms", "total_ms",
                           "per_frame_ms", "share"}
    assert s["setup_ms"] == pytest.approx({"load:persons": 1500.0})
    assert s["counters"] == {"dropped_frames": 3}
    json.dumps(s)


def test_report_canonical_order():
    timer = StageTimer(clock=FakeClock())
    for name in ("custom", "attention", "read", "depth", "depth:cache"):
        timer.add(name, 0.001)
    timer.frame_done()
    names = list(timer.summary()["stages"])
    assert names == ["read", "depth", "attention", "custom", "depth:cache", "total"]
    report = timer.report()
    rows = [line.split()[0] for line in report.splitlines()[2:2 + len(names)]]
    assert rows == names
    assert "fps" in report


def test_empty_frame_and_no_frames():
    timer = StageTimer(clock=FakeClock())
    s = timer.summary()
    assert s["frames"] == 0 and s["fps"] == 0.0 and s["stages"] == {}
    assert timer.frame_done() == {"total": 0.0}
    assert timer.summary()["frames"] == 1


def test_disabled_timer():
    def bad_clock() -> float:
        raise AssertionError("disabled timer must not call the clock")

    timer = StageTimer(enabled=False, clock=bad_clock)
    with timer.stage("depth"):
        pass
    timer.add("depth", 1.0)
    timer.add_setup("load", 1.0)
    timer.count("x")
    assert timer.frame_done() == {}
    assert timer.summary() == {}
    assert timer.report() == ""


def test_reset_and_reserved_total():
    timer = StageTimer(clock=FakeClock())
    timer.add("depth", 0.01)
    timer.frame_done()
    timer.reset()
    assert timer.summary()["frames"] == 0
    with pytest.raises(ValueError):
        timer.add("total", 1.0)
    with pytest.raises(ValueError):
        with timer.stage("total"):
            pass
