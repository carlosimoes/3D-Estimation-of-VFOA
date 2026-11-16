"""Streaming VFOA pipeline: one frame at a time, models loaded once, multi-rate stages.

The offline batch flow of the thesis (read the whole video, run every model over
every frame, reload models per call) is replaced by :class:`Pipeline`:

* ``open(fps, fingerprint)`` creates the backends through the registry (unless
  injected), loads each model exactly ONCE (a backend whose stage cache is
  complete is not loaded at all) and resets the per-stream state (tracker,
  smoothers, object registry).
* ``process(frame)`` runs the INGEST part on the newest frame (persons +
  tracking when scheduled, ``gaze.observe``) and FINALIZES the frame
  ``runtime.lookahead_frames`` behind it (depth, head pose, gaze, segmentation,
  scene, smoothing, attention). With ``lookahead_frames == 0`` (the real-time
  default) exactly one :class:`FrameResult` comes out per call.
* Stage ``S`` in {depth, persons, headpose, gaze, segmentation} runs on frame
  ``i`` when ``(i - cfg[S].phase) // cfg[S].stride`` differs from the bucket of
  the last frame it ran on (the modulo rule on contiguous indices; it also
  never starves when frames are dropped) or when it has no result yet.
  Between runs the last depth/segmentation and per-track head/gaze estimates
  are held (at most ``max_hold_frames``).
* Stage outputs are optionally cached on disk (:mod:`vfoa.cache`) so that the
  attention/smoothing stages can be re-run without loading any model.
* Every stage is timed with :class:`vfoa.timing.StageTimer` under the
  canonical stage names.

Real-time mode (:mod:`vfoa.realtime`, Refs #35):

* ``segmentation.async: true`` runs segmentation on an
  :class:`~vfoa.realtime.AsyncStageRunner`. The frame loop never waits for
  it: a scheduled frame is submitted only when the worker is idle (counters
  ``segmentation_async_submits`` / ``segmentation_async_skips``), the previous
  segmentation is used meanwhile, and on the frame where a new result is
  swapped in ``ran["segmentation"]`` is True and the objects are rebuilt
  (``segmentation_async_results``). A worker error propagates from
  ``process()`` as in synchronous mode.
* ``input.realtime: true`` makes :func:`open_input` wrap the source in a
  :class:`~vfoa.realtime.LatestFrameSource`; :meth:`Pipeline.run` then adds
  the counters ``dropped_frames`` and ``captured_frames``.
* Capture-to-result latency (capture time of the frame, or its read time for
  plain sources, to the end of its finalisation) is kept in a separate timer
  and reported as ``summary["latency"]`` (frames, mean/p50/p95/max ms); it is
  not part of ``FrameResult.timings_ms``. ``summary["async"]`` holds the
  runner statistics. Offline runs (both options false) are unchanged.

Backend fallback: when ``cfg[kind].fallback`` is set (the fast profile sets
``gaze.fallback: head_proxy`` for ``gaze.backend: l2cs``), the primary
backend is loaded eagerly in ``open()``; if it is unavailable (missing
package or weights) a warning is logged, the fallback backend is used and
``pipeline.cfg`` becomes a copy with ``cfg[kind].backend = <fallback>`` and
``cfg[kind].fallback_from = <primary>`` (so cache keys describe what really
ran), and the counter ``<kind>_fallback`` is set. Exception: when the
cache of the PRIMARY configuration for that kind is complete (cache enabled,
fingerprint given, kind in ``cache.stages``, mode read or readwrite), the
primary backend is created without loading it, no fallback is applied,
``pipeline.cfg`` stays the caller's config and nothing is counted; the stage
is served from the cache. A frame missing from that cache loads the primary
lazily and, if it is unavailable, ``BackendUnavailable`` propagates (the
pipeline never switches backend in the middle of a cache).

All 3D quantities are in the camera frame of the left camera (metres, +Z
forward); times are seconds (``frame.index / fps``).
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import time
import warnings
from collections import deque
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from vfoa.attention import AttentionParams, score_frame
from vfoa.backends import KINDS, create_backend, create_backend_with_fallback
from vfoa.backends.base import Backend
from vfoa.cache import StageCache, cache_root, is_complete, stage_key, stage_meta
from vfoa.io.sources import open_source
from vfoa.scene import SceneBuilder
from vfoa.smoothing import DirectionSmoother, TargetSmoother
from vfoa.timing import StageTimer
from vfoa.tracking import PersonTracker
from vfoa.types import DepthFrame, DirectionEstimate, Frame, FrameResult, Person, Segmentation

__all__ = ["Pipeline", "open_input", "MODEL_STAGES", "CUES"]

logger = logging.getLogger(__name__)

#: Stages backed by a model (scheduled with stride/phase, cacheable).
MODEL_STAGES: tuple[str, ...] = ("depth", "persons", "headpose", "gaze", "segmentation")
#: (stage, Person attribute, smoother key suffix) of the direction cues.
CUES: tuple[tuple[str, str], ...] = (("headpose", "head"), ("gaze", "gaze"))


# ---------------------------------------------------------------------- input
def open_input(cfg: Mapping[str, Any]) -> Any:
    """Frame source described by ``cfg.input`` (see :func:`vfoa.io.sources.open_source`).

    The synthetic source is re-created with ``input.start``/``input.max_frames``
    (and both are appended to its fingerprint, so cache keys differ). With
    ``input.realtime`` the source is wrapped in
    ``vfoa.realtime.LatestFrameSource(src, pace=True)``; when that module is not
    available a warning is issued and the plain source is returned.
    """
    src = open_source(cfg)
    inp = cfg.get("input") or {}
    if hasattr(src, "world"):
        start = int(inp.get("start") or 0)
        max_frames = inp.get("max_frames")
        max_frames = None if max_frames is None else int(max_frames)
        base_fp = getattr(src, "fingerprint", "synthetic")
        src = src.world.source(start, max_frames)
        src.fingerprint = f"{base_fp}|start={start}|max={max_frames}"
    if inp.get("realtime"):
        try:
            from vfoa.realtime import LatestFrameSource  # noqa: PLC0415 (optional module)
        except ImportError:
            warnings.warn(
                "input.realtime is set but vfoa.realtime is not available; reading every frame",
                RuntimeWarning,
                stacklevel=2,
            )
            return src
        return LatestFrameSource(src, pace=bool(inp.get("pace", True)))
    return src


# ---------------------------------------------------------------------- runners
class _SyncRunner:
    """Synchronous stand-in for ``vfoa.realtime.AsyncStageRunner``.

    ``submit(frame)`` computes ``fn(frame)`` inline and always returns True;
    ``poll()`` returns (and clears) ``(frame.index, result)``.
    """

    def __init__(self, fn: Callable[[Frame], Any], name: str) -> None:
        self.fn = fn
        self.name = name
        self._result: Optional[tuple[int, Any]] = None

    def submit(self, frame: Frame) -> bool:
        self._result = (int(frame.index), self.fn(frame))
        return True

    def poll(self) -> Optional[tuple[int, Any]]:
        out, self._result = self._result, None
        return out

    def close(self) -> None:
        self._result = None


# ---------------------------------------------------------------------- pipeline
@dataclasses.dataclass
class _Pending:
    """A frame between INGEST and FINALIZE."""

    frame: Frame
    persons: list[Person]
    persons_ran: bool
    timings: dict[str, float]  # seconds measured during ingest (replayed at finalize)
    capture: Optional[float] = None  # time.perf_counter() when the frame was captured


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return cfg.get(name) or {}


def _primary_cache_complete(cfg: Mapping[str, Any], kind: str, fingerprint: Optional[str]) -> bool:
    """True when ``kind`` has a fallback and the cache of its primary config is complete.

    Read-only probe (creates and writes nothing). False when no fallback is
    configured, the cache is disabled, there is no fingerprint, ``kind`` is
    not in ``cache.stages`` or the mode is ``write``.
    """
    sec, cc = _section(cfg, kind), _section(cfg, "cache")
    fb, name = sec.get("fallback"), sec.get("backend")
    if not fb or fb == name or name in (None, "none"):
        return False
    if not (cc.get("enabled") and fingerprint and kind in (cc.get("stages") or ())):
        return False
    if str(cc.get("mode", "readwrite")) not in ("read", "readwrite"):
        return False
    return is_complete(cache_root(cfg), kind, stage_key(cfg, kind, fingerprint))


class Pipeline:
    """Frame-by-frame VFOA pipeline (see the module docstring).

    Parameters
    ----------
    cfg:
        Full :class:`vfoa.config.Config`.
    backends:
        Optional ``{kind: Backend | None}``; a present key overrides the
        registry (``None`` disables that stage). Injected backends are not
        closed by :meth:`close`.
    timer:
        Optional :class:`StageTimer` (default: enabled per ``timing.enabled``).
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        backends: Optional[Mapping[str, Optional[Backend]]] = None,
        timer: Optional[StageTimer] = None,
    ) -> None:
        self.cfg = cfg
        self._base_cfg = cfg
        self._derived_cfg: Optional[Mapping[str, Any]] = None
        self._injected: dict[str, Optional[Backend]] = dict(backends or {})
        unknown = set(self._injected) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown backend kinds {sorted(unknown)}; valid: {', '.join(KINDS)}")
        self.timer = timer if timer is not None else StageTimer(
            enabled=bool(_section(cfg, "timing").get("enabled", True))
        )
        self.backends: dict[str, Optional[Backend]] = {}
        self.caches: dict[str, StageCache] = {}
        self._owned: set[str] = set()
        self._opened = False
        self.fps = 30.0
        self.last_summary: dict = {}

    # ------------------------------------------------------------------ open
    def open(self, fps: Optional[float], fingerprint: Optional[str] = None) -> "Pipeline":
        """Create/load the backends and reset the per-stream state.

        On failure every resource created so far is released and the error
        propagates (``depth.backend: none`` raises ValueError).
        """
        if self._opened:
            self._close_resources(complete=False)
        self.caches = {}
        self._owned = set()
        self._runner = None
        self.last_summary = {}
        try:
            self._open(fps, fingerprint)
        except BaseException:
            self._close_resources(complete=False)
            raise
        self._opened = True
        return self

    def _open(self, fps: Optional[float], fingerprint: Optional[str]) -> None:
        if self.cfg is not self._derived_cfg:  # the user's config (not one rewritten by a fallback)
            self._base_cfg = self.cfg
        cfg = self.cfg = self._base_cfg
        self._derived_cfg = None
        self.fps = float(fps) if fps is not None and fps > 0 else 30.0
        self.timer.reset()
        self._latency = StageTimer(enabled=self.timer.enabled)
        self._t_open = time.perf_counter()

        # 1. backends
        backends: dict[str, Optional[Backend]] = {}
        self.backends = backends
        for kind in KINDS:
            if kind in self._injected:
                backends[kind] = self._injected[kind]
            else:
                t0 = time.perf_counter()
                if _primary_cache_complete(cfg, kind, fingerprint):
                    b, used = create_backend(kind, cfg), None  # unloaded, no fallback
                else:
                    b, used = create_backend_with_fallback(kind, cfg)
                if b is not None and b.is_loaded:
                    self.timer.add_setup(f"load:{kind}", time.perf_counter() - t0)
                if used is not None:
                    cfg = copy.deepcopy(cfg)
                    cfg[kind]["fallback_from"] = cfg[kind].get("backend")
                    cfg[kind]["backend"] = used
                    self.cfg = self._derived_cfg = cfg
                    if b is not None:
                        b.cfg = cfg
                    self.timer.count(f"{kind}_fallback")
                backends[kind] = b
                if b is not None:
                    self._owned.add(kind)
        if backends.get("depth") is None:
            raise ValueError(
                "depth.backend is 'none': the pipeline needs a depth stage; "
                "use depth.backend=constant for monocular input without depth"
            )

        # per-stream state (config errors surface before any model is loaded)
        self.tracker = PersonTracker(cfg)
        self.scene = SceneBuilder(cfg)
        self.dir_smoother = DirectionSmoother(cfg, self.fps)
        self.target_smoother = TargetSmoother(cfg)
        self.params = AttentionParams.from_config(cfg)

        # 2. caches
        self._cache_stats: dict[str, dict[str, int]] = {}
        cc = _section(cfg, "cache")
        if cc.get("enabled"):
            if not fingerprint:
                logger.warning("cache.enabled but the input has no fingerprint: caching disabled")
            else:
                root = cache_root(cfg)
                mode = str(cc.get("mode", "readwrite"))
                for stage in cc.get("stages") or ():
                    if stage not in MODEL_STAGES or backends.get(stage) is None:
                        continue
                    self.caches[stage] = StageCache(
                        root, stage, stage_key(cfg, stage, fingerprint), mode=mode,
                        chunk_size=int(cc.get("chunk_size", 64) or 64),
                        meta=stage_meta(cfg, stage, fingerprint),
                    )
                    self._cache_stats[stage] = {"hits": 0, "misses": 0}

        # 3. load models once (skip stages fully served by a complete cache)
        self._ready: set[str] = set()
        for kind, b in backends.items():
            if b is None:
                continue
            c = self.caches.get(kind)
            if c is not None and c.complete and c.mode != "write":
                logger.info("%s: served from complete cache %s; model not loaded", kind, c.dir)
                continue
            self._ensure_loaded(kind)

        # 4. per-stream buffers
        self.lookahead = max(int(_section(cfg, "runtime").get("lookahead_frames", 0) or 0), 0)
        self._pending: deque[_Pending] = deque()
        self._last: dict[str, Optional[int]] = {s: None for s in MODEL_STAGES}
        self._last_persons: list[Person] = []
        self._depth: Optional[DepthFrame] = None
        self._seg: Optional[Segmentation] = None
        self._objects: list = []
        self._obj_src: Optional[tuple[Any, Any]] = None  # (segmentation, depth) the objects were built from
        self._held: dict[str, dict[int, tuple[DirectionEstimate, int]]] = {s: {} for s, _ in CUES}
        self._last_seen: dict[int, int] = {}
        self._frames = 0
        self._seg_async = bool(_section(cfg, "segmentation").get("async")) and "segmentation" not in self.caches
        self._runner = self._make_runner("segmentation", self._segment_fn())
        if self._seg_async:
            for name in ("submits", "skips", "results"):
                self.timer.count(f"segmentation_async_{name}", 0)

    def _ensure_loaded(self, kind: str) -> Optional[Backend]:
        """Load (once) and reset backend ``kind``; load time goes to ``setup_ms``."""
        b = self.backends.get(kind)
        if b is None or kind in self._ready:
            return b
        was_loaded = b.is_loaded
        t0 = time.perf_counter()
        b.load()
        if not was_loaded:
            self.timer.add_setup(f"load:{kind}", time.perf_counter() - t0)
        b.reset()
        if getattr(b, "fps", 0) is None:
            b.fps = self.fps
        self._ready.add(kind)
        return b

    def _make_runner(self, stage: str, fn: Callable[[Frame], Any]) -> Any:
        if not _section(self.cfg, stage).get("async"):
            return _SyncRunner(fn, stage)
        if stage in self.caches:
            warnings.warn(f"{stage}.async is ignored while the {stage} cache is enabled",
                          RuntimeWarning, stacklevel=3)
            return _SyncRunner(fn, stage)
        try:
            from vfoa.realtime import AsyncStageRunner  # noqa: PLC0415 (optional module)
        except ImportError:
            warnings.warn(f"{stage}.async is set but vfoa.realtime is not available; running {stage} "
                          "synchronously", RuntimeWarning, stacklevel=3)
            return _SyncRunner(fn, stage)
        return AsyncStageRunner(fn, stage)

    def _segment_fn(self) -> Callable[[Frame], Any]:
        """Function run by the segmentation runner.

        Asynchronous mode calls the backend directly (no cache, no per-frame
        timing from the worker thread); synchronous mode goes through the
        cache and times ``segmentation`` itself.
        """
        if self._seg_async:
            backend = self.backends.get("segmentation")

            def _direct(frame: Frame) -> Any:
                assert backend is not None
                return backend.segment(frame)

            return _direct

        def _cached(frame: Frame) -> Any:
            return self._fetch("segmentation", frame.index, lambda b: b.segment(frame), self.timer.add)

        return _cached

    # ------------------------------------------------------------------ helpers
    def _due(self, stage: str, index: int) -> bool:
        last = self._last[stage]
        if last is None:
            return True
        sec = _section(self.cfg, stage)
        stride = max(int(sec.get("stride", 1) or 1), 1)
        phase = int(sec.get("phase", 0) or 0)
        return (index - phase) // stride != (last - phase) // stride

    def _fetch(self, stage: str, index: int, fn: Callable[[Backend], Any],
               record: Callable[[str, float], None]) -> Any:
        """Stage payload of frame ``index``: cache hit, else compute (lazy load) and store."""
        cache = self.caches.get(stage)
        if cache is not None:
            t0 = time.perf_counter()
            payload = cache.get(index)
            record(f"{stage}:cache", time.perf_counter() - t0)
            if payload is not None:
                self._cache_stats[stage]["hits"] += 1
                return payload
            self._cache_stats[stage]["misses"] += 1
        backend = self._ensure_loaded(stage)
        assert backend is not None
        t0 = time.perf_counter()
        payload = fn(backend)
        record(stage, time.perf_counter() - t0)
        if cache is not None:
            cache.put(index, payload)
        return payload

    # ------------------------------------------------------------------ process
    def process(self, frame: Frame, extra_timings: Optional[Mapping[str, float]] = None, *,
                capture_time: Optional[float] = None) -> list[FrameResult]:
        """Ingest ``frame`` and return the frames finalised by it (in frame order).

        ``extra_timings`` (seconds, e.g. ``{"read": dt}``) are attributed to
        this frame's timing record.
        """
        if not self._opened:
            raise RuntimeError("Pipeline.process() called before open()")
        self._ingest(frame, extra_timings, capture_time)
        out: list[FrameResult] = []
        while len(self._pending) > self.lookahead:
            out.append(self._finalize(self._pending.popleft()))
        return out

    def flush(self) -> list[FrameResult]:
        """Finalise every frame still waiting for its lookahead."""
        out: list[FrameResult] = []
        while self._pending:
            out.append(self._finalize(self._pending.popleft()))
        return out

    def _ingest(self, frame: Frame, extra_timings: Optional[Mapping[str, float]],
                capture_time: Optional[float] = None) -> None:
        timed = self.timer.enabled
        timings: dict[str, float] = {}

        def record(name: str, sec: float) -> None:
            if timed:
                timings[name] = timings.get(name, 0.0) + float(sec)

        for name, sec in (extra_timings or {}).items():
            record(name, sec)
        i = int(frame.index)
        pb = self.backends.get("persons")
        ran = False
        if pb is not None and self._due("persons", i):
            dets = self._fetch("persons", i, lambda b: b.detect(frame), record)
            t0 = time.perf_counter()
            persons = self.tracker.update(dets or [], i)
            record("tracking", time.perf_counter() - t0)
            self._last["persons"] = i
            ran = True
        else:
            persons = [Person(p.track_id, p.detection) for p in self._last_persons]
        self._last_persons = persons
        gb = self.backends.get("gaze")
        if gb is not None and gb.is_loaded:
            t0 = time.perf_counter()
            gb.observe(frame, persons)
            record("gaze:observe", time.perf_counter() - t0)
        self._pending.append(_Pending(frame, persons, ran, timings, capture_time))

    def _finalize(self, entry: _Pending) -> FrameResult:
        cfg = self.cfg
        timer = self.timer
        frame = entry.frame
        i = int(frame.index)
        t = float(frame.t)
        for name, sec in entry.timings.items():
            timer.add(name, sec)
        ran = {s: False for s in ("depth", "persons", "tracking", "headpose", "gaze", "segmentation")}
        ran["persons"] = ran["tracking"] = entry.persons_ran
        persons = [Person(p.track_id, p.detection) for p in entry.persons]
        for p in persons:
            self._last_seen[p.track_id] = i

        # depth
        if self._due("depth", i):
            self._depth = self._fetch("depth", i, lambda b: b.compute(frame), timer.add)
            self._last["depth"] = i
            ran["depth"] = True
        depth = self._depth

        # eyes first: gaze backends may use eyes_px / eyes_3d
        with timer.stage("scene"):
            self.scene.locate_eyes(persons, depth)

        # head pose, then gaze (held per track)
        fresh: dict[str, dict[int, DirectionEstimate]] = {s: {} for s, _ in CUES}
        for stage, attr in CUES:
            held = self._held[stage]
            if self.backends.get(stage) is not None and self._due(stage, i):
                est = self._fetch(stage, i, lambda b, _s=stage: b.estimate(frame, persons), timer.add) or {}
                self._last[stage] = i
                ran[stage] = True
                fresh[stage] = dict(est)
                for tid, e in est.items():
                    held[int(tid)] = (e, i)
            max_hold = int(_section(cfg, stage).get("max_hold_frames", 15) or 0)
            for tid in [k for k, (_, j) in held.items() if i - j > max_hold]:
                del held[tid]
            for p in persons:
                h = held.get(p.track_id)
                setattr(p, attr, h[0] if h is not None else None)

        # segmentation (sync or async runner)
        if self.backends.get("segmentation") is not None:
            if self._due("segmentation", i):
                t0 = time.perf_counter()
                accepted = self._runner.submit(frame)
                if accepted:
                    self._last["segmentation"] = i
                    if self._seg_async:
                        # only accepted submissions count as segmentation runs
                        timer.add("segmentation", time.perf_counter() - t0)
                if self._seg_async:
                    timer.count("segmentation_async_submits" if accepted else "segmentation_async_skips")
            got = self._runner.poll()
            if got is not None:
                self._seg = got[1]
                ran["segmentation"] = True
                if self._seg_async:
                    timer.count("segmentation_async_results")
        seg = self._seg

        # scene objects (rebuilt only when segmentation or depth changed)
        with timer.stage("scene"):
            src = self._obj_src
            if src is None or src[0] is not seg or src[1] is not depth:
                self._objects = self.scene.build_objects(seg, depth)
                self._obj_src = (seg, depth)
            objects = list(self._objects)
            self.scene.match_own_segments(persons, seg, objects)

        # smoothing of the direction cues
        with timer.stage("smoothing"):
            for p in persons:
                for stage, attr in CUES:
                    key = (p.track_id, attr)
                    h = self._held[stage].get(p.track_id)
                    if h is None:
                        self.dir_smoother.drop(key)
                        setattr(p, attr, None)
                        continue
                    f = fresh[stage].get(p.track_id)
                    d = self.dir_smoother.update(
                        key,
                        None if f is None else f.direction,
                        t,
                        h[0].confidence if f is None else f.confidence,
                    )
                    setattr(p, attr, h[0] if d is None else dataclasses.replace(h[0], direction=d))

        # attention
        with timer.stage("attention"):
            scores = score_frame(persons, objects, self.params, depth, seg)
            by_person: dict[int, dict[int, float]] = {}
            for s in scores:
                by_person.setdefault(s.person_id, {})[s.object_id] = s.score_total
            targets = {
                p.track_id: self.target_smoother.update(p.track_id, by_person.get(p.track_id, {}), t)
                for p in persons
            }

        self._prune(i)
        self._frames += 1
        ran.update(scene=True, smoothing=True, attention=True)
        timings = timer.frame_done()
        if entry.capture is not None and self._latency.enabled:
            self._latency.add("latency", max(time.perf_counter() - entry.capture, 0.0))
            self._latency.frame_done()
        log_every = int(_section(cfg, "runtime").get("log_every", 0) or 0)
        if log_every > 0 and self._frames % log_every == 0:
            logger.info("frame %d (%d processed)", i, self._frames)
        return FrameResult(i, t, persons, objects, scores, targets, timings, ran)

    def _prune(self, index: int) -> None:
        """Forget smoother/held state of tracks unseen for a long time."""
        limit = int(_section(self.cfg, "tracking").get("max_age", 30) or 0) + max(
            int(_section(self.cfg, s).get("max_hold_frames", 15) or 0) for s, _ in CUES
        )
        dead = [tid for tid, j in self._last_seen.items() if index - j > limit]
        for tid in dead:
            del self._last_seen[tid]
            for stage, attr in CUES:
                self._held[stage].pop(tid, None)
                self.dir_smoother.drop((tid, attr))
            self.target_smoother.drop(tid)

    # ------------------------------------------------------------------ close
    def _close_resources(self, complete: bool) -> None:
        runner = getattr(self, "_runner", None)
        if runner is not None:
            try:
                runner.close()
            except Exception:  # pragma: no cover - defensive
                logger.exception("closing the segmentation runner failed")
            self._runner = None
        for c in self.caches.values():
            # never demote a cache that was complete and received no new frame
            # (StageCache.complete turns False on any put() or dropped chunk)
            c.close(complete=complete or c.complete)
        for kind in sorted(self._owned):
            b = self.backends.get(kind)
            if b is not None:
                try:
                    b.close()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("closing backend %s failed", kind)
        self._opened = False

    def close(self, complete: bool = True) -> dict:
        """Release runner, caches and owned backends; return the run summary (JSON-serialisable).

        ``complete`` marks the caches as complete (only after a full run).
        """
        if not self._opened:
            return {}
        caches = {
            stage: {"key": c.key, "mode": c.mode, **self._cache_stats.get(stage, {})}
            for stage, c in self.caches.items()
        }
        # frames still waiting in the lookahead buffer were never finalised
        complete = complete and not getattr(self, "_pending", None)
        runner = getattr(self, "_runner", None)
        self._close_resources(complete)
        for stage, c in self.caches.items():
            caches[stage]["complete"] = bool(c.complete)
        summary = dict(self.timer.summary())
        summary["frames"] = self._frames
        summary["backends"] = {
            k: (None if b is None else (getattr(b, "name", "") or type(b).__name__))
            for k, b in self.backends.items()
        }
        summary["cache"] = caches
        summary["wall_s"] = time.perf_counter() - self._t_open
        lat = (self._latency.summary().get("stages") or {}).get("latency")
        summary["latency"] = {} if lat is None else {
            "frames": lat["runs"], "mean_ms": lat["mean_ms"], "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"], "max_ms": lat["max_ms"],
        }
        stats = getattr(runner, "stats", None)
        summary["async"] = {"segmentation": dict(stats())} if self._seg_async and callable(stats) else {}
        self.last_summary = summary
        return summary

    # ------------------------------------------------------------------ run
    def run(self, source: Any, sinks: Sequence[Any] = ()) -> dict:
        """Process every frame of ``source``, writing results to ``sinks`` in frame order.

        Opens with ``source.fps``/``source.fingerprint``, times ``read`` and
        ``output``, and always closes the pipeline, sinks and source. Returns
        the summary of :meth:`close`.
        """
        complete = False
        summary: dict = {}
        timer = self.timer

        def emit(results: Iterable[FrameResult], frames: Mapping[int, Frame]) -> None:
            for r in results:
                t0 = time.perf_counter()
                fr = frames.pop(r.index, None)
                for s in sinks:
                    s.write(fr, r)
                timer.add("output", time.perf_counter() - t0)

        try:
            self.open(getattr(source, "fps", None), getattr(source, "fingerprint", None) or None)
            frames: dict[int, Frame] = {}
            it = iter(source)
            while True:
                t0 = time.perf_counter()
                try:
                    frame = next(it)
                except StopIteration:
                    break
                now = time.perf_counter()
                captured = getattr(source, "last_capture_time", None)
                frames[int(frame.index)] = frame
                emit(self.process(frame, extra_timings={"read": now - t0},
                                  capture_time=captured if isinstance(captured, (int, float)) else now), frames)
            while self._pending:
                emit([self._finalize(self._pending.popleft())], frames)
            complete = not bool(_section(self.cfg, "input").get("realtime"))
        finally:
            dropped = getattr(source, "dropped_frames", None)
            if dropped is None:
                dropped = getattr(source, "dropped", None)
            if isinstance(dropped, int) and dropped >= 0:
                timer.count("dropped_frames", dropped)
            captured_n = getattr(source, "frames_read", None)
            if isinstance(captured_n, int):
                timer.count("captured_frames", captured_n)
            try:
                summary = self.close(complete=complete)
            finally:
                for s in sinks:
                    try:
                        s.close()
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("closing sink %r failed", s)
                close = getattr(source, "close", None)
                if callable(close):
                    close()
        return summary
