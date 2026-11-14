"""Real-time (on-robot / live) helpers for the streaming pipeline (Refs #35).

Two small, stdlib-only building blocks:

* :class:`LatestFrameSource` wraps any frame source (a camera, a video file,
  the synthetic world, ...) in a reader thread that keeps ONLY the newest
  frame. A consumer that is slower than the source therefore always processes
  the most recent frame and older frames are dropped (and counted) instead of
  queueing up latency. File sources can be paced at their native frame rate to
  emulate a live camera.
* :class:`AsyncStageRunner` runs one slow stage (normally segmentation) on a
  single worker thread. ``submit`` never blocks and refuses new work while a
  job is pending; ``poll`` returns the newest finished result. The frame loop
  keeps going with the previous result in the meantime.

Times are seconds from ``time.perf_counter`` (or the injected clock) unless a
name ends in ``_ms``. Every thread is a daemon and every join has a timeout,
so a stuck camera or model can never hang interpreter shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional

__all__ = ["LatestFrameSource", "AsyncStageRunner"]

logger = logging.getLogger(__name__)

# Condition wait granularity (s): keeps blocking waits responsive to Ctrl-C.
_POLL_S = 0.1


class LatestFrameSource:
    """Frame source wrapper that always yields the newest available frame.

    A daemon reader thread ("vfoa-latest-frame") iterates ``source`` and puts
    each frame into a single slot; a frame still in the slot when the next one
    arrives is overwritten and counted in :attr:`dropped_frames`. Iterating
    this object (single pass) returns frames in strictly increasing index
    order. The last frame produced by the source is always delivered before
    the end of the stream, and an exception raised by the source is re-raised
    once by ``__next__`` after the frames read before it.

    Args:
        source: any iterable of :class:`vfoa.types.Frame` with optional ``fps``,
            ``width``, ``height`` and ``close()`` (a ``FrameSource``).
        pace: sleep so that frames are published at ``frame.t`` relative to
            the first frame (emulates a camera for file sources). Ignored for
            live sources (``source.live`` true or ``source.kind == "camera"``).
        clock: monotonic clock in seconds used for pacing and capture times.
        sleep: ``sleep(seconds)`` used for pacing; default waits on the stop
            event so that :meth:`close` interrupts it.

    Attributes:
        fps, width, height: copied from ``source``.
        n_frames: always ``None`` (the number of delivered frames is unknown).
        fingerprint: always ``""``: a frame-dropping stream is not reproducible,
            so the pipeline does not use the stage cache with it.
        frames_read: frames published by the reader thread.
        frames_delivered: frames returned to the consumer.
        dropped_frames: frames overwritten before the consumer took them.
        last_capture_time: ``clock()`` value at which the frame most recently
            returned by ``__next__`` was published (``None`` before the first).
    """

    def __init__(
        self,
        source: Any,
        pace: bool = True,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Optional[Callable[[float], Any]] = None,
    ) -> None:
        self.source = source
        self.fps = getattr(source, "fps", 30.0)
        self.width = getattr(source, "width", 0)
        self.height = getattr(source, "height", 0)
        self.n_frames: Optional[int] = None
        self.fingerprint = ""
        self.live = bool(getattr(source, "live", False)) or getattr(source, "kind", "") == "camera"
        self.pace = bool(pace) and not self.live
        self.dropped_frames = 0
        self.frames_read = 0
        self.frames_delivered = 0
        self.last_capture_time: Optional[float] = None
        self._clock = clock
        self._stop = threading.Event()
        self._sleep = sleep if sleep is not None else self._stop.wait
        self._cv = threading.Condition()
        self._slot: Optional[tuple[Any, float]] = None
        self._done = False
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def __iter__(self) -> "LatestFrameSource":
        """Start the reader thread (once) and return ``self``."""
        with self._cv:
            if self._thread is None and not self._closed:
                self._thread = threading.Thread(target=self._reader, name="vfoa-latest-frame", daemon=True)
                self._thread.start()
        return self

    def __next__(self) -> Any:
        """Block until a new frame is available and return it (the newest).

        Raises the source's exception once (after the frames read before it),
        then ``StopIteration``; also ``StopIteration`` after :meth:`close`.
        """
        if self._thread is None:
            iter(self)
        with self._cv:
            while self._slot is None and not self._done and not self._closed:
                self._cv.wait(_POLL_S)
            if self._slot is not None and not self._closed:
                frame, cap = self._slot
                self._slot = None
                self.frames_delivered += 1
                self.last_capture_time = cap
                return frame
            err, self._error = self._error, None
        if err is not None:
            raise err
        raise StopIteration

    def _reader(self) -> None:
        """Reader thread: publish every source frame into the single slot."""
        it: Any = None
        try:
            it = iter(self.source)
            t0: Optional[float] = None  # clock at the first frame
            ft0 = 0.0  # frame.t of the first frame (s)
            for frame in it:
                if self._stop.is_set():
                    break
                if self.pace:
                    now = self._clock()
                    if t0 is None:
                        t0, ft0 = now, float(frame.t)
                    delay = t0 + (float(frame.t) - ft0) - now
                    if delay > 0:
                        self._sleep(delay)
                        if self._stop.is_set():
                            break
                with self._cv:
                    if self._slot is not None:
                        self.dropped_frames += 1
                    self._slot = (frame, self._clock())
                    self.frames_read += 1
                    self._cv.notify_all()
        except Exception as exc:  # surfaced to the consumer by __next__
            logger.exception("frame reader failed")
            with self._cv:
                self._error = exc
        finally:
            # closing a generator here releases e.g. a VideoCapture in the thread that uses it
            close = getattr(it, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("closing the source iterator failed")
            with self._cv:
                self._done = True
                self._cv.notify_all()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the reader (interrupting pacing), join it, then close the source.

        Idempotent. The source is closed only after the reader thread has
        stopped (or ``timeout`` seconds elapsed, with a warning), so a camera
        is never released under a blocked read.
        """
        with self._cv:
            already = self._closed
            self._closed = True
            self._stop.set()
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("frame reader thread did not stop within %.1f s", timeout)
        if not already:
            close = getattr(self.source, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> "LatestFrameSource":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncStageRunner:
    """Run ``fn(frame)`` on one background worker thread without ever blocking.

    Contract:

    * :meth:`submit` returns immediately: ``True`` when the job was accepted,
      ``False`` (counted as ``skipped``) while a job is queued or running, and
      ``False`` after :meth:`close`.
    * :meth:`poll` returns ``(frame_index, result)`` of the newest finished job
      exactly once (an unpolled older result is overwritten), else ``None``.
    * An exception raised by ``fn`` is logged in the worker and re-raised ONCE
      in the caller's thread by the next :meth:`poll`; the runner stays usable.
    * :meth:`close` stops the worker after the current job and joins it.

    The worker thread ("vfoa-async-<name>", daemon) starts lazily on the first
    accepted submit. ``fn`` always runs outside the internal lock.
    """

    def __init__(self, fn: Callable[[Any], Any], name: str = "stage") -> None:
        self.fn = fn
        self.name = name
        self._cv = threading.Condition()
        self._job: Any = None
        self._has_job = False
        self._running = False
        self._closed = False
        self._result: Optional[tuple[int, Any]] = None
        self._error: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None
        self._n = {"submitted": 0, "skipped": 0, "completed": 0, "failed": 0}
        self._total_s = 0.0
        self._max_s = 0.0
        self._last_s = 0.0

    @property
    def busy(self) -> bool:
        """True while a job is queued or running."""
        with self._cv:
            return self._has_job or self._running

    def submit(self, frame: Any) -> bool:
        """Queue ``frame`` for ``fn`` if the worker is idle; never blocks."""
        with self._cv:
            if self._closed:
                return False
            if self._has_job or self._running:
                self._n["skipped"] += 1
                return False
            self._job, self._has_job = frame, True
            self._n["submitted"] += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._worker, name=f"vfoa-async-{self.name}", daemon=True)
                self._thread.start()
            self._cv.notify_all()
            return True

    def poll(self) -> Optional[tuple[int, Any]]:
        """Return and clear the newest ``(frame_index, result)``, or ``None``.

        Re-raises (once) the exception of a failed job instead.
        """
        with self._cv:
            err, self._error = self._error, None
            if err is None:
                out, self._result = self._result, None
                return out
        raise err

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the worker is idle; ``False`` if ``timeout`` (s) expired."""
        with self._cv:
            return self._cv.wait_for(lambda: not (self._has_job or self._running), timeout)

    def stats(self) -> dict:
        """Counters and worker compute times (ms) as plain JSON-safe numbers.

        Keys: submitted, skipped, completed, failed, mean_ms (over finished
        jobs), max_ms, last_ms.
        """
        with self._cv:
            done = self._n["completed"] + self._n["failed"]
            return {
                **self._n,
                "mean_ms": 1000.0 * self._total_s / done if done else 0.0,
                "max_ms": 1000.0 * self._max_s,
                "last_ms": 1000.0 * self._last_s,
            }

    def _worker(self) -> None:
        """Worker loop: take the pending job, run ``fn`` unlocked, store the outcome."""
        while True:
            with self._cv:
                while not self._has_job and not self._closed:
                    self._cv.wait()
                if self._closed:
                    self._job, self._has_job = None, False
                    self._cv.notify_all()
                    return
                frame, self._job, self._has_job, self._running = self._job, None, False, True
            t0 = time.perf_counter()
            result: Any = None
            err: Optional[BaseException] = None
            try:
                result = self.fn(frame)
            except Exception as exc:
                logger.exception("async stage %s failed on frame %s", self.name, getattr(frame, "index", "?"))
                err = exc
            dt = time.perf_counter() - t0
            with self._cv:
                self._running = False
                self._total_s += dt
                self._max_s = max(self._max_s, dt)
                self._last_s = dt
                if err is None:
                    self._result = (int(frame.index), result)
                    self._n["completed"] += 1
                else:
                    self._error = err
                    self._n["failed"] += 1
                self._cv.notify_all()

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker after its current job and join it (idempotent)."""
        with self._cv:
            self._closed = True
            self._cv.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                logger.warning("async stage %s is still running after %.1f s", self.name, timeout)
