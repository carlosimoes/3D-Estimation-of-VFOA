"""Per-stage wall-clock timing of the pipeline.

``StageTimer`` accumulates seconds per stage for the current frame, closes the
frame with ``frame_done()`` and produces per-stage statistics in milliseconds.
``total`` is derived (sum of the frame's stages) and reserved. Thread-safe.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import numpy as np

__all__ = ["CANONICAL_STAGES", "StageTimer"]

#: Canonical stage order used in reports.
CANONICAL_STAGES: tuple[str, ...] = (
    "read",
    "depth",
    "persons",
    "tracking",
    "headpose",
    "gaze",
    "segmentation",
    "scene",
    "smoothing",
    "attention",
    "output",
    "total",
)

_TOTAL = "total"


class StageTimer:
    """Accumulates per-stage durations per frame and summarises them.

    Parameters
    ----------
    enabled:
        A disabled timer never calls the clock and returns empty results.
    clock:
        Monotonic clock returning seconds (injectable for tests).
    """

    def __init__(self, enabled: bool = True, clock: Callable[[], float] = time.perf_counter) -> None:
        self.enabled = bool(enabled)
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Forget every recorded frame, setup cost and counter."""
        with self._lock:
            self._current: dict[str, float] = {}
            self._samples: dict[str, list[float]] = {}
            self._order: list[str] = []
            self._totals: list[float] = []
            self._frames = 0
            self._setup: dict[str, float] = {}
            self._counters: dict[str, int] = {}

    # -- recording -------------------------------------------------------
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the enclosed block and add it to stage ``name`` of the current frame."""
        if not self.enabled:
            yield
            return
        self._check_name(name)
        start = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - start)

    def add(self, name: str, seconds: float) -> None:
        """Add ``seconds`` to stage ``name`` of the current frame."""
        if not self.enabled:
            return
        self._check_name(name)
        with self._lock:
            self._current[name] = self._current.get(name, 0.0) + float(seconds)
            if name not in self._order:
                self._order.append(name)

    def add_setup(self, name: str, seconds: float) -> None:
        """Record a one-off cost (e.g. ``"load:persons"``), excluded from per-frame stats."""
        if not self.enabled:
            return
        with self._lock:
            self._setup[name] = self._setup.get(name, 0.0) + float(seconds) * 1000.0

    def count(self, name: str, n: int = 1) -> None:
        """Increment counter ``name`` (e.g. ``dropped_frames``) by ``n``."""
        if not self.enabled:
            return
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + int(n)

    def frame_done(self) -> dict[str, float]:
        """Close the current frame; return its per-stage milliseconds including ``total``."""
        if not self.enabled:
            return {}
        with self._lock:
            out: dict[str, float] = {}
            for name, sec in self._current.items():
                ms = sec * 1000.0
                out[name] = ms
                self._samples.setdefault(name, []).append(ms)
            total = float(sum(out.values()))
            out[_TOTAL] = total
            self._totals.append(total)
            self._frames += 1
            self._current = {}
            return out

    @staticmethod
    def _check_name(name: str) -> None:
        if name == _TOTAL:
            raise ValueError("'total' is derived from the other stages and cannot be recorded")

    # -- reporting -------------------------------------------------------
    def _ordered_names(self) -> list[str]:
        canon = [s for s in CANONICAL_STAGES if s != _TOTAL and s in self._samples]
        others = [s for s in self._order if s not in CANONICAL_STAGES and s in self._samples]
        return canon + others

    @staticmethod
    def _stats(values: list[float], frames: int, grand_total: float) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        total = float(arr.sum())
        return {
            "runs": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
            "total_ms": total,
            "per_frame_ms": total / frames if frames else 0.0,
            "share": total / grand_total if grand_total > 0 else 0.0,
        }

    def summary(self) -> dict:
        """Statistics over all finished frames (plain Python numbers, JSON-serialisable).

        ``mean_ms``/``p50_ms``/``p95_ms``/``max_ms`` are over the frames where the
        stage ran; ``per_frame_ms`` is ``total_ms / frames`` (amortised over strides);
        ``share`` is the fraction of the total time.
        """
        if not self.enabled:
            return {}
        with self._lock:
            frames = self._frames
            grand = float(sum(self._totals))
            stages: dict[str, dict] = {}
            for name in self._ordered_names():
                stages[name] = self._stats(self._samples[name], frames, grand)
            if frames:
                stages[_TOTAL] = self._stats(self._totals, frames, grand)
            mean_total = grand / frames if frames else 0.0
            return {
                "frames": frames,
                "fps": 1000.0 / mean_total if mean_total > 0 else 0.0,
                "stages": stages,
                "setup_ms": dict(self._setup),
                "counters": dict(self._counters),
            }

    def report(self) -> str:
        """Fixed-width table of :meth:`summary` in canonical stage order."""
        if not self.enabled:
            return ""
        s = self.summary()
        header = (
            f"{'stage':<16}{'runs':>7}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}"
            f"{'max ms':>10}{'ms/frame':>10}{'share':>8}"
        )
        lines = [header, "-" * len(header)]
        for name, st in s["stages"].items():
            lines.append(
                f"{name:<16}{st['runs']:>7d}{st['mean_ms']:>10.2f}{st['p50_ms']:>10.2f}"
                f"{st['p95_ms']:>10.2f}{st['max_ms']:>10.2f}{st['per_frame_ms']:>10.2f}"
                f"{st['share'] * 100:>7.1f}%"
            )
        lines.append(f"frames: {s['frames']}  fps: {s['fps']:.2f}")
        if s["setup_ms"]:
            lines.append(
                "setup: " + ", ".join(f"{k}={v:.1f} ms" for k, v in s["setup_ms"].items())
            )
        if s["counters"]:
            lines.append("counters: " + ", ".join(f"{k}={v}" for k, v in s["counters"].items()))
        return "\n".join(lines)
