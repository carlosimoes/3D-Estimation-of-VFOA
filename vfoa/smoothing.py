"""Temporal smoothing: 3D Kalman filter, direction smoothing and target filtering.

* :class:`KalmanFilter3D` - the thesis constant-velocity filter, corrected
  (issue #23): no ``is 0`` comparisons, ``R_atual``/``Ratual`` are distinct
  arrays, process noise scaled by ``dt``, the state is never overwritten from
  earlier results. The legacy API ``KalmanFilter3D(dt=...).Kalman(z)`` is kept.
* :class:`DirectionSmoother` - per-key (``(track_id, "gaze")`` /
  ``(track_id, "head")``) smoothing of unit direction vectors in the camera
  frame (kalman, ema or none), with predict/hold on missing measurements.
* :class:`TargetSmoother` - per-person temporal filtering of the attention
  target (HMM forward filter with a NONE state, hysteresis, or none).

Times are seconds (float64); directions are unit float64 (3,) vectors.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Hashable, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["KalmanFilter3D", "DirectionSmoother", "TargetSmoother"]


def _section(cfg: Optional[Mapping[str, Any]], name: str) -> dict:
    if cfg is None or not hasattr(cfg, "get"):
        return {}
    sec = cfg.get(name, {})
    return dict(sec or {})


def _cv_F(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F


def _cv_Q(dt: float, q: float) -> np.ndarray:
    I3 = np.eye(3)
    return q * np.block([
        [dt ** 3 / 3.0 * I3, dt ** 2 / 2.0 * I3],
        [dt ** 2 / 2.0 * I3, dt * I3],
    ])


class KalmanFilter3D:
    """Constant-velocity Kalman filter on 3D positions; state ``[p (3), v (3)]``.

    Positions are in the caller's units (metres for 3D points, unitless for
    direction vectors); ``dt`` in seconds. Only the position is measured
    (``H = [I3 | 0]``). ``Q`` defaults to the white-acceleration model scaled by
    ``q`` and ``dt``; ``R`` defaults to ``r * I3`` and, when ``adaptive_r`` is
    true and ``R`` was not given, follows the thesis velocity-sigmoid rule
    (:meth:`real_time_R_update`).
    """

    def __init__(self, F: Optional[np.ndarray] = None, B: Optional[np.ndarray] = None,
                 H: Optional[np.ndarray] = None, Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None, P: Optional[np.ndarray] = None,
                 x0: Optional[np.ndarray] = None, dt: Optional[float] = None, *,
                 q: float = 1.0, r: float = 100.0, adaptive_r: bool = True) -> None:
        if F is None and dt is None:
            raise ValueError("KalmanFilter3D needs either F or dt")
        self.dt = float(dt) if dt is not None else None
        self.q = float(q)
        self.r = float(r)
        self._auto_F = F is None
        self._auto_Q = Q is None
        self.F = _cv_F(self.dt) if F is None else np.asarray(F, dtype=np.float64)
        self.n = self.F.shape[1]
        self.H = np.hstack([np.eye(3), np.zeros((3, self.n - 3))]) if H is None \
            else np.asarray(H, dtype=np.float64)
        self.B = 0.0 if B is None else np.asarray(B, dtype=np.float64)
        if Q is None:
            self.Q = _cv_Q(self.dt, self.q) if self.dt is not None else np.eye(self.n)
        else:
            self.Q = np.asarray(Q, dtype=np.float64)
        m = self.H.shape[0]
        self.R = self.r * np.eye(m) if R is None else np.asarray(R, dtype=np.float64)
        self.adaptive_r = bool(adaptive_r) and R is None
        self.P0 = np.eye(self.n) if P is None else np.asarray(P, dtype=np.float64).copy()
        self.P = self.P0.copy()
        self.x = np.zeros((self.n, 1)) if x0 is None else \
            np.asarray(x0, dtype=np.float64).reshape(self.n, 1).copy()
        self.result = self.x[:3, 0].copy()
        self.velo = self.x[3:6, 0].copy() if self.n >= 6 else np.zeros(3)
        self.initR = 0
        self.R_atual: Optional[np.ndarray] = None
        self.Ratual: Optional[np.ndarray] = None
        self.R_prev: Optional[np.ndarray] = None
        self.s_prev: Optional[np.ndarray] = None
        self._r_updates = 0

    def set_dt(self, dt: float) -> None:
        """Change the time step (seconds); rebuilds the auto-built F and Q."""
        dt = float(dt)
        if self.dt is not None and dt == self.dt:
            return
        self.dt = dt
        if self._auto_F:
            self.F = _cv_F(dt)
        if self._auto_Q:
            self.Q = _cv_Q(dt, self.q)

    def initialise(self, z: Any) -> None:
        """Start the filter at position ``z`` with zero velocity."""
        z3 = np.asarray(z, dtype=np.float64).reshape(-1)[:3]
        self.x = np.zeros((self.n, 1))
        self.x[:3, 0] = z3
        self.P = self.P0.copy()
        self.initR = 1
        self.result = z3.copy()
        self.velo = np.zeros(3)

    def real_time_R_update(self) -> None:
        """Thesis adaptive measurement noise from the estimated velocity (per axis)."""
        if not self.adaptive_r:
            return
        s = 1.0 / (1.0 + np.exp(-np.abs(np.asarray(self.velo, dtype=np.float64).reshape(3))))
        self.R_atual = self.r * np.where(s > 0.94, 0.1, s)
        if self.R_prev is None or self.s_prev is None:
            self.Ratual = self.R_atual.copy()
        else:
            self.Ratual = np.where(s < self.s_prev, 0.3 * self.R_atual + 0.7 * self.R_prev,
                                   self.R_atual)
        self.s_prev = s
        self.R_prev = self.Ratual.copy()
        self.R = np.diag(self.Ratual)
        self._r_updates += 1

    def predict(self, u: Any = 0) -> np.ndarray:
        """Time update ``x = F x + B u``, ``P = F P F^T + Q``."""
        Bu = np.dot(self.B, u)
        self.x = self.F @ self.x + (np.asarray(Bu, dtype=np.float64).reshape(-1, 1)
                                    if np.ndim(Bu) else Bu)
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.initR += 1
        self.result = self.x[:3, 0].copy()
        return self.x

    def update(self, z: Any) -> np.ndarray:
        """Measurement update with the first 3 values of ``z`` (Joseph form)."""
        m = self.H.shape[0]
        zz = np.asarray(z, dtype=np.float64).reshape(-1)[:m].reshape(m, 1)
        y = zz - self.H @ self.x
        S = self.R + self.H @ self.P @ self.H.T
        K = np.linalg.solve(S.T, (self.P @ self.H.T).T).T
        self.x = self.x + K @ y
        IKH = np.eye(self.n) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ self.R @ K.T
        self.result = self.x[:3, 0].copy()
        self.velo = self.x[3:6, 0].copy()
        return self.x

    def Kalman(self, z: Any) -> list:
        """Legacy one-call filter step: returns the filtered position as a list."""
        if self.initR == 0:
            self.initialise(z)
            self.real_time_R_update()
            return np.asarray(z, dtype=np.float64).reshape(-1)[:3].tolist()
        self.real_time_R_update()
        self.predict()
        self.update(z)
        return self.result.tolist()


def _unit(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        return None
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return None
    return a / n


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.atan2(np.linalg.norm(np.cross(a, b)), float(np.dot(a, b))))


class _DirState:
    __slots__ = ("kf", "vec", "t_last", "t_meas", "pending")

    def __init__(self) -> None:
        self.kf: Optional[KalmanFilter3D] = None
        self.vec: Optional[np.ndarray] = None
        self.t_last = 0.0
        self.t_meas = 0.0
        self.pending = 0


class DirectionSmoother:
    """Smooths unit direction vectors per key (camera frame, times in seconds).

    Config ``smoothing``: ``direction`` (kalman|ema|none), ``ema_alpha``,
    ``kalman_q``, ``kalman_r`` (measurement std, unit-vector units),
    ``max_hold_s``; optional ``jump_deg`` (default 30, 0/None disables the
    outlier gate of the kalman mode).
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None, fps: Optional[float] = None) -> None:
        s = _section(cfg, "smoothing")
        self.mode = str(s.get("direction", "kalman"))
        if self.mode not in ("kalman", "ema", "none"):
            raise ValueError(f"smoothing.direction must be kalman|ema|none, got {self.mode!r}")
        self.fps = float(fps) if fps is not None and fps > 0 else 30.0
        self.alpha = float(s.get("ema_alpha", 0.5))
        self.kalman_q = float(s.get("kalman_q", 1.0))
        self.kalman_r = float(s.get("kalman_r", 0.05))
        self.max_hold_s = float(s.get("max_hold_s", 0.5))
        jump = s.get("jump_deg", 30.0)
        self.jump_rad = math.radians(float(jump)) if jump else 0.0
        self._state: dict[Hashable, _DirState] = {}

    def reset(self) -> None:
        self._state.clear()

    def drop(self, key: Hashable) -> None:
        self._state.pop(key, None)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._state

    def _new_filter(self, m: np.ndarray) -> KalmanFilter3D:
        r2 = self.kalman_r ** 2
        kf = KalmanFilter3D(dt=1.0 / self.fps, q=self.kalman_q, r=r2, adaptive_r=False,
                            P=np.diag([r2] * 3 + [1.0] * 3))
        kf.initialise(m)
        return kf

    def _init(self, key: Hashable, m: np.ndarray, t: float) -> np.ndarray:
        st = _DirState()
        st.t_last = st.t_meas = t
        if self.mode == "kalman":
            st.kf = self._new_filter(m)
        st.vec = m.copy()
        self._state[key] = st
        return m.copy()

    def _write_back(self, st: _DirState) -> Optional[np.ndarray]:
        p = st.kf.x[:3, 0]
        n = float(np.linalg.norm(p))
        if not np.isfinite(n) or n < 1e-6:
            return None
        st.kf.x[:3, 0] = p / n
        st.vec = st.kf.x[:3, 0].copy()
        return st.vec.copy()

    def update(self, key: Hashable, direction: Any, t: float,
               confidence: float = 1.0) -> Optional[np.ndarray]:
        """Feed a measurement (or None) for ``key`` at time ``t`` (s); returns unit (3,) or None."""
        t = float(t)
        if self.mode == "none":
            return None if direction is None else np.array(direction, dtype=np.float64).reshape(-1)
        conf = float(confidence) if confidence is not None else 1.0
        m = _unit(direction)
        if m is not None and not conf > 0:
            m = None
        st = self._state.get(key)
        if st is None:
            return None if m is None else self._init(key, m, t)
        dt = t - st.t_last
        if not dt > 0:
            dt = 1.0 / self.fps
        stale = t - st.t_meas > self.max_hold_s + 1e-9
        if m is None:
            if stale:
                self.drop(key)
                return None
            st.t_last = t
            if self.mode == "kalman":
                kf = st.kf
                kf.set_dt(dt)
                kf.predict()
                kf.x[3:6, 0] *= 0.5 ** (dt / 0.1)
                out = self._write_back(st)
                return out if out is not None else st.vec.copy()
            return st.vec.copy()
        if stale:
            return self._init(key, m, t)
        st.t_meas = t
        st.t_last = t
        if self.mode == "kalman":
            kf = st.kf
            kf.set_dt(dt)
            kf.predict()
            p = _unit(kf.x[:3, 0])
            if p is None:
                return self._init(key, m, t)
            if self.jump_rad and _angle(p, m) > self.jump_rad:
                st.pending += 1
                if st.pending >= 2:
                    return self._init(key, m, t)
                kf.x[:3, 0] = p
                st.vec = p.copy()
                return p.copy()
            st.pending = 0
            kf.R = self.kalman_r ** 2 / float(np.clip(conf, 0.05, 1.0)) * np.eye(3)
            kf.update(m)
            out = self._write_back(st)
            return out if out is not None else self._init(key, m, t)
        # ema
        c = float(np.clip(conf, 0.0, 1.0))
        den = self.alpha * c + 1.0 - self.alpha
        a = self.alpha * c / den if den > 0 else 1.0
        out = _unit((1.0 - a) * st.vec + a * m)
        st.vec = m.copy() if out is None else out
        return st.vec.copy()


class TargetSmoother:
    """Temporal filter of each person's attention target (object id or None).

    Config: ``smoothing.target`` (hmm|hysteresis|none), ``smoothing.hmm_stay_prob``,
    ``smoothing.hysteresis_frames`` and ``attention.min_score``.
    """

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        s = _section(cfg, "smoothing")
        a = _section(cfg, "attention")
        self.mode = str(s.get("target", "hmm"))
        if self.mode not in ("hmm", "hysteresis", "none"):
            raise ValueError(f"smoothing.target must be hmm|hysteresis|none, got {self.mode!r}")
        self.p = float(np.clip(float(s.get("hmm_stay_prob", 0.9)), 0.5, 0.999))
        self.hysteresis_frames = max(int(s.get("hysteresis_frames", 3)), 1)
        self.min_score = float(a.get("min_score", 0.1))
        self.eps = 2.0 * (1.0 - self.p) / self.p
        self._belief: dict[int, dict[Optional[int], float]] = {}
        self._current: dict[int, Optional[int]] = {}
        self._cand: dict[int, tuple[Optional[int], int]] = {}
        self._t: dict[int, float] = {}

    def reset(self) -> None:
        self._belief.clear()
        self._current.clear()
        self._cand.clear()
        self._t.clear()

    def drop(self, person_id: int) -> None:
        for d in (self._belief, self._current, self._cand, self._t):
            d.pop(person_id, None)

    def _winner(self, scores: Mapping[int, float]) -> Optional[int]:
        best: Optional[int] = None
        best_s = -math.inf
        for oid in sorted(scores):
            sc = float(scores[oid])
            if np.isfinite(sc) and sc > best_s:
                best, best_s = oid, sc
        if best is None or best_s < self.min_score:
            return None
        return best

    def update(self, person_id: int, scores: Mapping[int, float], t: float) -> Optional[int]:
        """Feed this frame's ``{object_id: score_total}`` for ``person_id``; returns the target."""
        scores = {int(k): float(v) for k, v in (scores or {}).items()
                  if v is not None and np.isfinite(float(v))}
        self._t[person_id] = float(t)
        winner = self._winner(scores)
        if self.mode == "none":
            return winner
        if self.mode == "hysteresis":
            return self._hysteresis(person_id, scores, winner)
        return self._hmm(person_id, scores)

    def _hysteresis(self, pid: int, scores: Mapping[int, float], winner: Optional[int]) -> Optional[int]:
        if pid not in self._current:
            self._current[pid] = winner
            self._cand[pid] = (None, 0)
            return winner
        cur = self._current[pid]
        if cur is not None and cur not in scores:
            cur = None
            self._current[pid] = None
            self._cand[pid] = (None, 0)
        if winner == cur:
            self._cand[pid] = (None, 0)
            return cur
        cand, n = self._cand.get(pid, (None, 0))
        n = n + 1 if (cand == winner and n > 0) else 1
        if n >= self.hysteresis_frames:
            self._current[pid] = winner
            self._cand[pid] = (None, 0)
            return winner
        self._cand[pid] = (winner, n)
        return cur

    def _hmm(self, pid: int, scores: Mapping[int, float]) -> Optional[int]:
        states: list[Optional[int]] = [None] + sorted(scores)
        n = len(states)
        prev = self._belief.get(pid)
        b = np.array([prev.get(s, 0.0) for s in states]) if prev else np.zeros(n)
        total = float(b.sum())
        b = np.full(n, 1.0 / n) if not total > 0 else b / total
        p = self.p
        prior = np.ones(1) if n == 1 else p * b + (1.0 - p) * (1.0 - b) / (n - 1)
        max_score = max(scores.values()) if scores else 0.0
        e = np.empty(n)
        e[0] = self.eps + (self.min_score if max_score >= self.min_score else 1.0 - max_score)
        for i, s in enumerate(states[1:], start=1):
            e[i] = self.eps + float(np.clip(scores[s], 0.0, 1.0))
        post = prior * e
        post = post / post.sum()
        post = np.maximum(post, 1e-9)
        post = post / post.sum()
        self._belief[pid] = {s: float(v) for s, v in zip(states, post)}
        best_i = 0
        for i in range(1, n):
            if post[i] > post[best_i]:
                best_i = i
        return states[best_i]
