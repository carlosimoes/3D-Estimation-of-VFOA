"""Tests of the real-time helpers (vfoa.realtime) and their pipeline/CLI wiring (Refs #35).

Threads are synchronised with Events/Semaphores; WAIT is only a deadlock guard and
nothing asserts wall-clock durations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

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
from vfoa.cli import main
from vfoa.pipeline import Pipeline, open_input
from vfoa.realtime import AsyncStageRunner, LatestFrameSource
from vfoa.types import Frame

WAIT = 30.0  # deadlock guard only, never a performance assertion
GOLDEN_60 = "2c3d5cd2ba936fe02791066d492834bf8e0285da6ce0fd8b0e63e750a9cc5c9d"


def mk(i: int, fps: float = 30.0) -> Frame:
    return Frame(index=i, t=i / fps, image=np.zeros((4, 4, 3), np.uint8))


class GatedSource:
    """Fake source producing one frame per `permits` release.

    `published` is released when the reader asks for the next frame, i.e. after
    the previous frame reached the LatestFrameSource slot; allow(k) lets k frames
    through and waits until all k were published.
    """

    def __init__(self, n: int, fps: float = 30.0, fail_at: int | None = None) -> None:
        self.n, self.fps, self.width, self.height = n, fps, 4, 4
        self.n_frames, self.fingerprint = n, "gated"
        self.permits = threading.Semaphore(0)
        self.published = threading.Semaphore(0)
        self.fail_at = fail_at
        self.closed = False

    def __iter__(self):
        for i in range(self.n):
            assert self.permits.acquire(timeout=WAIT)
            if self.closed:
                return
            if i == self.fail_at:
                raise RuntimeError("camera unplugged")
            yield mk(i, self.fps)
            self.published.release()

    def allow(self, k: int) -> None:
        self.permits.release(k)
        for _ in range(k):
            assert self.published.acquire(timeout=WAIT)

    def close(self) -> None:
        self.closed = True


def threads(prefix: str) -> list:
    return [t for t in threading.enumerate() if t.name.startswith(prefix)]


class ListSink:
    def __init__(self) -> None:
        self.results: list = []

    def write(self, frame: Any, result: Any) -> None:
        self.results.append(result)

    def close(self) -> None:
        pass


def test_latest_keeps_newest_and_counts_drops():
    src = GatedSource(12)
    lfs = LatestFrameSource(src, pace=False)
    assert (lfs.fps, lfs.width, lfs.height, lfs.n_frames, lfs.fingerprint) == (30.0, 4, 4, None, "")
    it = iter(lfs)
    got = []
    src.allow(5)
    got.append(next(it).index)
    c1 = lfs.last_capture_time
    src.allow(3)
    got.append(next(it).index)
    assert lfs.last_capture_time >= c1
    src.allow(4)
    got.append(next(it).index)
    with pytest.raises(StopIteration):
        next(it)
    assert got == [4, 7, 11]
    assert lfs.dropped_frames == 9 and lfs.frames_read == 12
    lfs.close()
    assert src.closed and not threads("vfoa-latest-frame")
    lfs.close()


def test_latest_no_drop_when_consumer_keeps_up():
    src = GatedSource(8)
    with LatestFrameSource(src, pace=False) as lfs:
        got = []
        it = iter(lfs)
        for _ in range(8):
            src.permits.release()
            got.append(next(it).index)
        with pytest.raises(StopIteration):
            next(it)
    assert got == list(range(8)) and lfs.dropped_frames == 0
    assert not threads("vfoa-latest-frame")


def test_latest_pacing_follows_frame_timestamps():
    class Clock:
        t = 100.0

    sleeps: list = []

    def sleep(dt: float) -> None:
        sleeps.append(dt)
        Clock.t += dt

    class Plain:
        fps = 10.0
        width = height = 4

        def __iter__(self):
            return iter([mk(i, 10.0) for i in range(3, 8)])

    lfs = LatestFrameSource(Plain(), pace=True, clock=lambda: Clock.t, sleep=sleep)
    last = [f.index for f in lfs][-1]
    lfs.close()
    assert last == 7 and sleeps == pytest.approx([0.1] * 4)

    cam = Plain()
    cam.kind = "camera"
    assert LatestFrameSource(cam, pace=True).pace is False


def test_latest_source_error_reaches_consumer():
    src = GatedSource(5, fail_at=2)
    lfs = LatestFrameSource(src, pace=False)
    it = iter(lfs)
    src.allow(2)
    src.permits.release()
    assert next(it).index == 1
    with pytest.raises(RuntimeError, match="unplugged"):
        next(it)
    with pytest.raises(StopIteration):
        next(it)
    lfs.close()
    assert not threads("vfoa-latest-frame")


def test_latest_close_interrupts_pacing():
    class Slow:
        fps = 0.001
        width = height = 4

        def __iter__(self):
            return iter([mk(i, 0.001) for i in range(3)])

    lfs = LatestFrameSource(Slow(), pace=True)
    assert next(iter(lfs)).index == 0
    lfs.close()
    assert not threads("vfoa-latest-frame")
    with pytest.raises(StopIteration):
        next(lfs)


def test_async_runner_submit_poll_cycle():
    started, release = threading.Event(), threading.Event()

    def fn(frame):
        started.set()
        assert release.wait(WAIT)
        return f"seg-{frame.index}"

    r = AsyncStageRunner(fn, "unit")
    assert r.poll() is None and not r.busy and not threads("vfoa-async-unit")
    assert r.submit(mk(0)) is True
    assert started.wait(WAIT)
    assert r.busy and r.submit(mk(1)) is False and r.poll() is None
    release.set()
    assert r.wait(WAIT)
    assert r.poll() == (0, "seg-0") and r.poll() is None
    assert r.submit(mk(2)) and r.wait(WAIT)
    assert r.submit(mk(3)) and r.wait(WAIT)
    assert r.poll() == (3, "seg-3") and r.poll() is None
    st = r.stats()
    assert (st["submitted"], st["skipped"], st["completed"], st["failed"]) == (3, 1, 3, 0)
    r.close()
    assert not threads("vfoa-async-unit") and r.submit(mk(4)) is False
    r.close()


def test_async_runner_error_is_logged_and_reraised(caplog):
    def fn(frame):
        if frame.index == 1:
            raise ValueError("boom")
        return frame.index * 10

    r = AsyncStageRunner(fn, "err")
    with caplog.at_level(logging.ERROR, logger="vfoa.realtime"):
        assert r.submit(mk(1)) and r.wait(WAIT)
    assert any("err" in rec.getMessage() for rec in caplog.records)
    with pytest.raises(ValueError, match="boom"):
        r.poll()
    assert r.poll() is None
    assert r.submit(mk(2)) and r.wait(WAIT)
    assert r.poll() == (2, 20) and r.stats()["failed"] == 1
    r.close()
    assert not threads("vfoa-async-err")


class BlockingSeg(SyntheticSegmentation):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.entered, self.gate = threading.Event(), threading.Event()

    def segment(self, frame):
        self.entered.set()
        assert self.gate.wait(WAIT)
        return super().segment(frame)


def test_pipeline_async_segmentation_never_blocks(make_config):
    cfg = make_config("synthetic", {"segmentation": {"async": True, "stride": 1}})
    world = SyntheticWorld.from_config(cfg)
    seg = BlockingSeg(cfg)
    bk = {"depth": SyntheticDepth(cfg), "persons": SyntheticPersons(cfg), "headpose": SyntheticHeadPose(cfg),
          "gaze": SyntheticGaze(cfg), "segmentation": seg}
    pipe = Pipeline(cfg, backends=bk)
    pipe.open(world.fps)
    try:
        for i in range(12):
            (res,) = pipe.process(world.frame(i))
            assert res.index == i and res.persons and res.objects == [] and not res.ran["segmentation"]
        assert seg.entered.wait(WAIT)
        seg.gate.set()
        assert pipe._runner.wait(WAIT)
        (res,) = pipe.process(world.frame(12))
        assert res.ran["segmentation"] and res.objects
        valid = {s.id for s in world.segmentation(0).segments} | {s.id for s in world.segmentation(12).segments}
        assert {o.segment_id for o in res.objects} <= valid
    finally:
        seg.gate.set()
        summary = pipe.close()
    c = summary["counters"]
    assert c["segmentation_async_skips"] == 11 and c["segmentation_async_submits"] == 2
    assert c["segmentation_async_results"] >= 1
    assert summary["async"]["segmentation"]["submitted"] == 2
    assert not threads("vfoa-async-segmentation")


def digest(results) -> str:
    rows = []
    for r in results:
        rows.append([
            int(r.index),
            sorted([int(k), v] for k, v in r.targets.items()),
            sorted([int(s.person_id), int(s.object_id), round(float(s.score_total), 6), bool(s.occluded)]
                   for s in r.scores),
        ])
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def test_offline_results_match_pre_change_golden(make_config):
    cfg = make_config("synthetic", {"input": {"max_frames": 60}})
    sink = ListSink()
    summary = Pipeline(cfg).run(open_input(cfg), [sink])
    assert len(sink.results) == 60 and digest(sink.results) == GOLDEN_60
    assert summary["counters"] == {} and summary["async"] == {}
    assert summary["latency"]["frames"] == 60


def test_run_realtime_counters_and_latency(make_config):
    cfg = make_config("synthetic", {"input": {"max_frames": 20, "realtime": True}})
    src = open_input(cfg)
    assert isinstance(src, LatestFrameSource)
    sink = ListSink()
    summary = Pipeline(cfg).run(src, [sink])
    idx = [r.index for r in sink.results]
    assert idx == sorted(set(idx)) and idx[-1] == 19
    c = summary["counters"]
    assert c["dropped_frames"] == src.dropped_frames and c["captured_frames"] == 20
    assert len(idx) + c["dropped_frames"] == 20
    lat = summary["latency"]
    assert lat["frames"] == len(idx) and 0.0 <= lat["p50_ms"] <= lat["p95_ms"] <= lat["max_ms"]
    assert not threads("vfoa-latest-frame")


@pytest.mark.parametrize("extra", [[], ["--set", "segmentation.async=true", "--set",
                                        "synthetic.cost_ms.segmentation=50"]])
def test_cli_realtime(tmp_path, capsys, extra):
    out = tmp_path / "rt"
    assert main(["run", "--profile", "synthetic", "--realtime", "--max-frames", "60", "--output-dir", str(out),
                 *extra]) == 0
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) + timing["counters"]["dropped_frames"] == 60
    assert timing["latency"]["frames"] == len(rows)
    if extra:
        assert timing["counters"]["segmentation_async_submits"] >= 1
        assert timing["async"]["segmentation"]["submitted"] == timing["counters"]["segmentation_async_submits"]
    assert "latency capture->result" in capsys.readouterr().out


def test_cli_interrupt_writes_summary(tmp_path, monkeypatch, capsys):
    import vfoa.cli as cli

    class Bomb(ListSink):
        def write(self, frame, result):
            super().write(frame, result)
            if len(self.results) == 3:
                raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_sinks", lambda cfg, source: [Bomb()])
    out = tmp_path / "int"
    assert cli.main(["run", "--profile", "synthetic", "--realtime", "--max-frames", "30",
                     "--output-dir", str(out)]) == 130
    timing = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    assert timing["interrupted"] is True and timing["frames"] == 3
    assert "dropped_frames" in timing["counters"]
    assert not threads("vfoa-latest-frame")
    assert "latency capture->result" in capsys.readouterr().out
