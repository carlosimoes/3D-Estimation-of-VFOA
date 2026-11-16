#!/usr/bin/env python
"""CPU/GPU benchmark of the streaming VFOA pipeline (Refs #35, #32, #30).

Builds the pipeline from a profile/config, runs ``--warmup`` + ``--frames``
frames of a video, an image directory or the synthetic world, and reports per
stage: runs, mean/p50/p95/max milliseconds, amortised milliseconds per frame,
share of the total, and the overall FPS (warm-up frames excluded). The result
is printed as a table (and as Markdown with ``--markdown``) and written as
JSON::

    {"status": "ok"|"skipped", "reason": str|null,
     "meta": {profile, input, frames, warmup, resolution, config_hash, python,
              numpy, opencv, torch, ultralytics, onnxruntime, open3d,
              cpu_count, date, backends, fallbacks, loadavg_start,
              loadavg_end, argv, env, realtime, segmentation_async,
              torch_threads},
     "timing": StageTimer.summary() of the measured frames,
     "run": {wall_s, frames_total, warmup, latency, latency_all, async, cache},
     "micro": {ply_roundtrip, slab_test, stereo_bm}}

``meta.realtime`` and ``meta.segmentation_async`` record ``input.realtime`` and
``segmentation.async``; ``meta.torch_threads`` is torch's intra-op thread count
at the end of the run (only when torch was loaded). ``run.latency`` is the
capture-to-result latency of the measured frames only (warm-up excluded),
measured by the benchmark itself; ``run.latency_all`` is the pipeline's own
figure over every frame, warm-up included. The ``async`` worker statistics
cover the whole run. With ``input.realtime`` true, ``--frames`` and
``--warmup`` count SOURCE frames: frames are dropped while the pipeline is
busy, so fewer frames are measured (a note is printed).

Default output: ``benchmarks/results/<profile>-<input stem>.json`` (git-ignored);
a micro-only run (``--frames 0 --micro``) writes ``benchmarks/results/micro.json``.

When a backend or its weights are unavailable (e.g. ultralytics not installed,
no network for the weights) it prints ``SKIPPED: <reason>``, writes status
``skipped`` and exits 0. Configuration/input errors exit 2.

Examples (run from the repository root)::

    python benchmarks/bench_pipeline.py --profile synthetic --frames 60 --warmup 5
    python benchmarks/bench_pipeline.py --profile fast --input Final_videos/museum.mp4 --frames 30
    python benchmarks/bench_pipeline.py --profile fast --input Final_videos/mpi_sintel.mp4 \\
        --camera-dir Final_videos/stereo/camdata_left/bandage_2 \\
        --depth-dir Final_videos/stereo/training/bandage_2_depth --set depth.backend=sintel_gt
    python benchmarks/bench_pipeline.py --profile synthetic --input Final_videos/mpi_sintel.mp4 \\
        --camera-dir ... --depth-dir ... --stereo-synth --set depth.backend=stereo_bm --set depth.scale=0.5
    python benchmarks/bench_pipeline.py --frames 0 --micro

``--stereo-synth`` renders the right view of every frame from the ground-truth
depth of ``--depth-dir`` (before the timed run, so synthesis is not timed) so
that the ``stereo_bm`` depth backend can be measured without stereo footage.
``--micro`` adds micro-benchmarks of the #32 fixes: the legacy ASCII-PLY write
plus open3d read versus a NumPy AABB, the legacy pure-Python slab test versus
:func:`vfoa.geometry.ray_aabb`, and the legacy double full-resolution StereoBM
versus one compute at half resolution.

Only the standard library, NumPy and OpenCV are imported at module import
time; vfoa, open3d, torch and ultralytics are imported lazily.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.metadata
import json
import logging
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

__all__ = [
    "build_parser",
    "build_config",
    "run_pipeline",
    "micro_benchmarks",
    "collect_meta",
    "format_markdown",
    "main",
]

MODEL_KINDS = ("depth", "persons", "headpose", "gaze", "segmentation")
SINTEL_VIDEO = "Final_videos/mpi_sintel.mp4"
SINTEL_CAM_DIR = "Final_videos/stereo/camdata_left/bandage_2"
SINTEL_DEPTH_DIR = "Final_videos/stereo/training/bandage_2_depth"
#: Default directory of the JSON results (read at call time, so tests can patch it).
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"


# ---------------------------------------------------------------------- arguments
def build_parser() -> argparse.ArgumentParser:
    """Command-line parser of the benchmark."""
    p = argparse.ArgumentParser(
        prog="bench_pipeline",
        description="Per-stage timing benchmark of the VFOA pipeline",
    )
    p.add_argument("--profile", default=None,
                   help="fast|accurate|thesis|synthetic (default: synthetic, or the --config file's profile)")
    p.add_argument("--config", default=None, help="YAML configuration file")
    p.add_argument("--input", default=None, help="video, image directory, camera index or 'synthetic'")
    p.add_argument("--camera-dir", default=None, help="Sintel .cam directory (input.camera_dir)")
    p.add_argument("--depth-dir", default=None, help="Sintel .dpt directory (input.depth_dir)")
    p.add_argument("--frames", type=int, default=100, help="measured frames (0 = no pipeline run)")
    p.add_argument("--warmup", type=int, default=10, help="warm-up frames excluded from the statistics")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set gaze.stride=3 (repeatable)")
    p.add_argument("--stereo-synth", action="store_true",
                   help="synthesise the right view from the --depth-dir ground truth (exercises stereo_bm)")
    p.add_argument("--micro", action="store_true", help="also run the #32 micro-benchmarks")
    p.add_argument("--micro-repeats", type=int, default=5, help="repeats per micro-benchmark (median)")
    p.add_argument("--markdown", action="store_true", help="also print the result as Markdown tables")
    p.add_argument("--out", default=None, help="JSON result file (default benchmarks/results/<profile>-<input>.json)")
    return p


def _set_keys(sets: Sequence[str]) -> set[str]:
    return {s.split("=", 1)[0].strip() for s in sets}


def build_config(args: argparse.Namespace) -> Any:
    """Resolved :class:`vfoa.config.Config` for the benchmark arguments.

    Same precedence as ``python -m vfoa run`` (profile <- file <- ``--set`` <-
    flags); timing is forced on, periodic logging off, and ``input.max_frames``
    is ``warmup + frames``. When a real input feeds synthetic backends, the
    synthetic world takes the source resolution (unless ``--set
    synthetic.width/height`` is given) so that every stage works on the same
    image size.
    """
    from vfoa.config import load_config  # noqa: PLC0415

    sets = list(args.set or [])
    keys = _set_keys(sets)
    profile = args.profile if (args.profile or args.config) else "synthetic"
    cfg = load_config(args.config, profile, overrides=sets or None)
    if args.input is not None:
        cfg.input.path = str(args.input)
        if "input.source" not in keys:
            cfg.input.source = "auto"
    if args.camera_dir is not None:
        cfg.input.camera_dir = str(args.camera_dir)
    if args.depth_dir is not None:
        cfg.input.depth_dir = str(args.depth_dir)
    n_total = max(int(args.warmup), 0) + max(int(args.frames), 0)
    cfg.timing.enabled = True
    cfg.runtime.log_every = 0
    cfg.input.max_frames = n_total
    cfg.synthetic.n_frames = max(int(cfg.synthetic.n_frames), n_total)

    uses_synthetic = any(cfg[k].get("backend") == "synthetic" for k in MODEL_KINDS)
    if uses_synthetic and str(cfg.input.get("path")) != "synthetic" and cfg.input.get("source") != "synthetic" \
            and int(args.frames) > 0:
        from vfoa.io.sources import open_source  # noqa: PLC0415

        src = open_source(cfg)
        try:
            w, h = int(src.width), int(src.height)
        finally:
            src.close()
        if "synthetic.width" not in keys:
            cfg.synthetic.width = w
        if "synthetic.height" not in keys:
            cfg.synthetic.height = h
    return cfg


# ---------------------------------------------------------------------- pipeline run
class _Collector:
    """Sink that keeps the per-frame stage timings and capture-to-result latencies.

    ``capture`` maps a frame index to its capture time (``time.perf_counter()``
    seconds, filled by :class:`_CaptureTimes`); ``latency_s[i]`` is the latency
    of ``rows[i]`` in seconds, or ``None`` when the frame's capture time is
    unknown.
    """

    def __init__(self, capture: Optional[dict[int, float]] = None) -> None:
        self.rows: list[dict[str, float]] = []
        self.latency_s: list[Optional[float]] = []
        self._capture = capture if capture is not None else {}

    def write(self, frame: Any, result: Any) -> None:
        now = time.perf_counter()
        self.rows.append(dict(result.timings_ms or {}))
        captured = self._capture.pop(int(result.index), None)
        self.latency_s.append(None if captured is None else max(now - captured, 0.0))

    def close(self) -> None:
        pass


class _CaptureTimes:
    """Delegating source wrapper that records each frame's capture time by frame index.

    The capture time is the source's ``last_capture_time`` (real-time sources)
    when it is a number, else ``time.perf_counter()`` when the frame is handed
    out; either is no later than the time the pipeline itself records.
    """

    def __init__(self, source: Any) -> None:
        self._source = source
        self.times: dict[int, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def __iter__(self) -> Iterator[Any]:
        for frame in self._source:
            captured = getattr(self._source, "last_capture_time", None)
            if isinstance(captured, bool) or not isinstance(captured, (int, float)):
                captured = time.perf_counter()
            self.times[int(frame.index)] = float(captured)
            yield frame

    def close(self) -> None:
        self._source.close()


def _latency_summary(samples: Sequence[Optional[float]]) -> dict:
    """``{"frames", "mean_ms", "p50_ms", "p95_ms", "max_ms"}`` of latency samples (seconds); ``{}`` if none."""
    from vfoa.timing import StageTimer  # noqa: PLC0415

    timer = StageTimer(enabled=True)
    for sample in samples:
        if sample is None:
            continue
        timer.add("latency", float(sample))
        timer.frame_done()
    lat = (timer.summary().get("stages") or {}).get("latency")
    if lat is None:
        return {}
    return {"frames": lat["runs"], "mean_ms": lat["mean_ms"], "p50_ms": lat["p50_ms"],
            "p95_ms": lat["p95_ms"], "max_ms": lat["max_ms"]}


class _WithRightView:
    """Delegating source wrapper that attaches pre-rendered right views (by frame index)."""

    def __init__(self, source: Any, rights: dict[int, np.ndarray]) -> None:
        self._source = source
        self._rights = rights

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    def __iter__(self) -> Iterator[Any]:
        for frame in self._source:
            yield dataclasses.replace(frame, right=self._rights.get(int(frame.index)))

    def close(self) -> None:
        self._source.close()


def _synth_right_views(cfg: Any) -> dict[int, np.ndarray]:
    """Right view of every source frame rendered from the ground-truth depth (not timed)."""
    from vfoa.backends.depth_stereo import synthesize_right_view  # noqa: PLC0415
    from vfoa.io.sintel import depth_read  # noqa: PLC0415
    from vfoa.io.sources import open_source  # noqa: PLC0415

    depth_dir = Path(cfg.input.depth_dir)
    rights: dict[int, np.ndarray] = {}
    src = open_source(cfg)
    try:
        for frame in src:
            path = depth_dir / f"frame_{int(frame.index) + 1:04d}.dpt"
            if not path.exists():
                raise FileNotFoundError(f"--stereo-synth: missing ground-truth depth {path}")
            if frame.intrinsics is None:
                raise ValueError("--stereo-synth: the source gives no intrinsics")
            rights[int(frame.index)] = synthesize_right_view(frame.image, depth_read(path), frame.intrinsics)
    finally:
        src.close()
    return rights


def run_pipeline(cfg: Any, frames: int, warmup: int, stereo_synth: bool = False) -> dict:
    """Run the pipeline on ``cfg``'s input; statistics exclude the first ``warmup`` frames.

    Returns ``{"timing": StageTimer.summary(), "table": report text, "run": {...},
    "backends": {kind: name}, "fallbacks": {kind: {"from", "to"}},
    "resolution": [W, H], "frames": measured frames}``. Raises
    :class:`vfoa.backends.base.BackendUnavailable` when a model is unavailable
    and ``ValueError`` when the input yields no frame.
    """
    from vfoa.pipeline import Pipeline, open_input  # noqa: PLC0415
    from vfoa.timing import StageTimer  # noqa: PLC0415

    rights = _synth_right_views(cfg) if stereo_synth else None
    src = open_input(cfg)
    if rights is not None:
        src = _WithRightView(src, rights)
    src = _CaptureTimes(src)
    resolution = [int(getattr(src, "width", 0) or 0), int(getattr(src, "height", 0) or 0)]
    collector = _Collector(src.times)
    pipe = Pipeline(cfg)
    summary = pipe.run(src, [collector])

    rows = collector.rows
    n = len(rows)
    if n == 0:
        raise ValueError("the input produced no frames")
    w = min(max(int(warmup), 0), n - 1)
    timer = StageTimer(enabled=True)
    for row in rows[w:]:
        for name, ms in row.items():
            if name in ("total", "output"):
                continue
            timer.add(name, float(ms) / 1000.0)
        timer.frame_done()
    for name, ms in (summary.get("setup_ms") or {}).items():
        timer.add_setup(name, float(ms) / 1000.0)
    for name, value in (summary.get("counters") or {}).items():
        timer.count(name, int(value))

    fallbacks: dict[str, dict[str, Any]] = {}
    for kind in MODEL_KINDS:
        sec = pipe.cfg.get(kind) or {}
        if sec.get("fallback_from"):
            fallbacks[kind] = {"from": sec.get("fallback_from"), "to": sec.get("backend")}
    return {
        "timing": timer.summary(),
        "table": timer.report(),
        "run": {
            "wall_s": summary.get("wall_s"),
            "frames_total": n,
            "warmup": w,
            "latency": _latency_summary(collector.latency_s[w:]),
            "latency_all": summary.get("latency") or {},
            "async": summary.get("async") or {},
            "cache": summary.get("cache") or {},
        },
        "backends": dict(summary.get("backends") or {}),
        "fallbacks": fallbacks,
        "resolution": resolution,
        "frames": n - w,
    }


# ---------------------------------------------------------------------- micro-benchmarks
def _median_ms(fn: Callable[[], Any], repeats: int) -> float:
    times = []
    for _ in range(max(int(repeats), 1)):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _entry(legacy_ms: float, new_ms: float, agree: bool, **extra: Any) -> dict:
    return {
        "status": "ok",
        "legacy_ms": legacy_ms,
        "new_ms": new_ms,
        "speedup": legacy_ms / new_ms if new_ms > 0 else float("inf"),
        "agree": bool(agree),
        **extra,
    }


def _legacy_write_ply(vertices: np.ndarray, colors: np.ndarray, filename: str) -> None:
    """The thesis ``create_output``: ASCII PLY with np.savetxt (Depth_Estimator / Reconstruction_scene)."""
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    header = (
        "ply\nformat ascii 1.0\nelement vertex %(vert_num)d\nproperty float x\nproperty float y\n"
        "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(filename, "w") as fh:
        fh.write(header % dict(vert_num=len(vertices)))
        np.savetxt(fh, vertices, "%f %f %f %d %d %d")


def _micro_ply(repeats: int, small: bool) -> dict:
    try:
        import open3d as o3d  # noqa: PLC0415
    except Exception as exc:  # ImportError, or a broken install
        return {"status": "skipped", "reason": f"open3d not available ({type(exc).__name__}: {exc})"}
    from vfoa.geometry import aabb_from_points  # noqa: PLC0415

    rng = np.random.default_rng(0)
    n = 2000 if small else 20000
    pts = rng.normal(size=(n, 3)).astype(np.float32)
    colors = rng.integers(0, 255, size=(n, 3))
    with tempfile.TemporaryDirectory(prefix="vfoa-bench-") as d:
        path = os.path.join(d, "object.ply")

        def legacy() -> tuple[np.ndarray, np.ndarray]:
            _legacy_write_ply(pts, colors, path)
            box = o3d.io.read_point_cloud(path).get_axis_aligned_bounding_box()
            return np.asarray(box.min_bound), np.asarray(box.max_bound)

        def new() -> tuple[np.ndarray, np.ndarray]:
            return aabb_from_points(pts)

        a, b = legacy(), new()
        agree = bool(np.allclose(a[0], b[0], atol=1e-4) and np.allclose(a[1], b[1], atol=1e-4))
        lm = _median_ms(legacy, repeats)
        nm = _median_ms(new, max(repeats, 5) * 4)
    return _entry(lm, nm, agree, points=n)


def _legacy_slab_hits(origin: np.ndarray, dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> int:
    """Pure-Python port of the thesis ``Attention.intersection`` loop (rays x boxes)."""
    hits = 0
    for dvec in dirs:
        dir_inv = 1.0 / dvec
        for minimo, maximo in zip(bmin, bmax):
            t1 = (minimo[0] - origin[0]) * dir_inv[0]
            t2 = (maximo[0] - origin[0]) * dir_inv[0]
            tmin = min(t1, t2)
            tmax = max(t1, t2)
            for i in range(0, 3):
                t1 = (minimo[i] - origin[i]) * dir_inv[i]
                t2 = (maximo[i] - origin[i]) * dir_inv[i]
                tmin = max(tmin, min(t1, t2))
                tmax = min(tmax, max(t1, t2))
            if tmax > tmin:
                hits += 1
    return hits


def _micro_slab(repeats: int, small: bool) -> dict:
    from vfoa.geometry import cone_rays, ray_aabb  # noqa: PLC0415

    rng = np.random.default_rng(1)
    n_persons, n_boxes = (1, 10) if small else (5, 20)
    rings = (11.3, 21.8, 12.5, 24.0)  # thesis gaze + head rings, 20 rays each
    dirs = np.concatenate([
        cone_rays(np.array([0.1 * k + 0.03, 0.05, 1.0]), a, 20) for k in range(n_persons) for a in rings
    ])
    dirs[np.abs(dirs) < 1e-9] = 1e-9  # the legacy code divides by every component
    centres = rng.uniform(-2.0, 2.0, size=(n_boxes, 3)) + np.array([0.0, 0.0, 4.0])
    half = rng.uniform(0.1, 0.6, size=(n_boxes, 3))
    bmin, bmax = centres - half, centres + half
    origin = np.zeros(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        legacy_hits = _legacy_slab_hits(origin, dirs, bmin, bmax)
        new_hits = int(ray_aabb(origin, dirs, bmin, bmax)[0].sum())
        lm = _median_ms(lambda: _legacy_slab_hits(origin, dirs, bmin, bmax), repeats)
        nm = _median_ms(lambda: ray_aabb(origin, dirs, bmin, bmax), max(repeats, 5) * 4)
    return _entry(lm, nm, legacy_hits == new_hits, rays=int(dirs.shape[0]), boxes=n_boxes,
                  hits=new_hits)


def _stereo_pair(small: bool) -> tuple[np.ndarray, np.ndarray, str]:
    """Grey rectified pair: Sintel frame 1 + synthesised right view, else a random texture."""
    if not small:
        try:
            from vfoa.backends.depth_stereo import synthesize_right_view  # noqa: PLC0415
            from vfoa.io.sintel import depth_read, intrinsics_from_cam  # noqa: PLC0415

            video = REPO_ROOT / SINTEL_VIDEO
            dpt = REPO_ROOT / SINTEL_DEPTH_DIR / "frame_0001.dpt"
            cam = REPO_ROOT / SINTEL_CAM_DIR / "frame_0001.cam"
            if video.exists() and dpt.exists() and cam.exists():
                cap = cv2.VideoCapture(str(video))
                ok, img = cap.read()
                cap.release()
                if ok and img is not None:
                    h, w = img.shape[:2]
                    intr = intrinsics_from_cam(cam, w, h, baseline=0.1)
                    right = synthesize_right_view(img, depth_read(dpt), intr)
                    return (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY),
                            "sintel bandage_2 frame 1 + synthesised right view")
        except Exception:  # fall back to the texture below
            logging.getLogger(__name__).debug("Sintel stereo pair unavailable", exc_info=True)
    rng = np.random.default_rng(2)
    w, h = (320, 160) if small else (1024, 436)
    tex = rng.integers(0, 256, size=(h // 4 + 1, (w + 64) // 4 + 1), dtype=np.uint8)
    tex = cv2.resize(tex, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    left = np.ascontiguousarray(tex[:h, 32:32 + w])
    right = np.ascontiguousarray(tex[:h, 32 + 24:32 + 24 + w])  # constant 24 px disparity
    return left, right, f"random texture {w}x{h}, 24 px disparity"


def _micro_stereo(repeats: int, small: bool) -> dict:
    gl, gr, desc = _stereo_pair(small)
    h, w = gl.shape[:2]
    nd = 160 if w >= 512 else 64
    full = cv2.StereoBM_create(numDisparities=nd, blockSize=15)
    half = cv2.StereoBM_create(numDisparities=max(16, nd // 2), blockSize=15)
    sw, sh = max(1, w // 2), max(1, h // 2)

    def legacy() -> np.ndarray:
        # legacy Depth_Estimator: StereoBM over the full-resolution pair, computed twice per frame
        full.compute(gl, gr)
        return full.compute(gl, gr).astype(np.float32) / 16.0

    def new() -> np.ndarray:
        a = cv2.resize(gl, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(gr, (sw, sh), interpolation=cv2.INTER_AREA)
        return half.compute(a, b).astype(np.float32) / 16.0 * (w / sw)

    d_full, d_half = legacy(), new()
    vf, vh = d_full[d_full > 0], d_half[d_half > 0]
    agree = bool(vf.size and vh.size and abs(float(np.median(vf)) - float(np.median(vh)))
                 <= max(1.5, 0.1 * float(np.median(vf))))
    lm = _median_ms(legacy, repeats)
    nm = _median_ms(new, repeats)
    return _entry(lm, nm, agree, image=desc, size=[w, h],
                  median_disparity_full=float(np.median(vf)) if vf.size else None,
                  median_disparity_half=float(np.median(vh)) if vh.size else None)


def micro_benchmarks(repeats: int = 5, small: bool = False) -> dict:
    """Micro-benchmarks of the #32 fixes (median of ``repeats`` runs, milliseconds).

    Each entry is ``{"status": "ok", "legacy_ms", "new_ms", "speedup" (legacy/new),
    "agree", ...}`` or ``{"status": "skipped", "reason"}`` (missing dependency).
    ``small`` uses tiny inputs (tests).
    """
    out: dict[str, dict] = {}
    for name, fn in (("ply_roundtrip", _micro_ply), ("slab_test", _micro_slab), ("stereo_bm", _micro_stereo)):
        try:
            out[name] = fn(repeats, small)
        except Exception as exc:  # a micro-benchmark never breaks the benchmark
            out[name] = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
    return out


# ---------------------------------------------------------------------- metadata / output
def _version(dist: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _loadavg() -> Optional[list[float]]:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):
        return None


THREAD_ENV = ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "MKL_NUM_THREADS")


def thread_env() -> dict:
    """The thread-related environment variables (``None`` when unset)."""
    return {k: os.environ.get(k) for k in THREAD_ENV}


def collect_meta(args: argparse.Namespace, cfg: Any, run: Optional[dict] = None,
                 loadavg_start: Optional[list[float]] = None, argv: Optional[Sequence[str]] = None,
                 env: Optional[dict] = None) -> dict:
    """Environment and configuration metadata of a benchmark result (never imports heavy packages)."""
    from vfoa.config import config_hash  # noqa: PLC0415

    run = run or {}
    meta = {
        "profile": cfg.get("profile"),
        "input": cfg.input.get("path"),
        "frames": int(run.get("frames", 0)) if run else 0,
        "frames_requested": int(args.frames),
        "warmup": int(run["run"]["warmup"]) if run else (int(args.warmup) if args.frames > 0 else 0),
        "resolution": run.get("resolution"),
        "config_hash": config_hash(cfg),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": _version("torch"),
        "ultralytics": _version("ultralytics"),
        "onnxruntime": _version("onnxruntime"),
        "open3d": _version("open3d"),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "backends": run.get("backends", {}),
        "fallbacks": run.get("fallbacks", {}),
        "stereo_synth": bool(args.stereo_synth),
        "overrides": list(args.set or []),
        "loadavg_start": loadavg_start,
        "loadavg_end": _loadavg(),
        "argv": list(argv) if argv is not None else sys.argv[1:],
        # Thread settings change CPU results by more than 20x; always record them. Taken
        # at start-up by main(): importing ultralytics sets OMP_NUM_THREADS=1 in-process.
        "env": dict(env) if env is not None else thread_env(),
        "realtime": bool((cfg.get("input") or {}).get("realtime")),
        "segmentation_async": bool((cfg.get("segmentation") or {}).get("async")),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            meta["torch_threads"] = int(torch.get_num_threads())
        except Exception:  # pragma: no cover - defensive
            pass
    return meta


def _fmt(x: Any, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def format_markdown(result: dict) -> str:
    """Markdown rendering of a benchmark result (stage table, FPS, setup, micro-benchmarks)."""
    meta = result.get("meta") or {}
    lines: list[str] = []
    res = meta.get("resolution")
    lines.append(
        f"profile `{meta.get('profile')}`, input `{meta.get('input')}`"
        + (f" ({res[0]}x{res[1]})" if res else "")
        + f", {meta.get('frames')} measured frames after {meta.get('warmup')} warm-up, "
        f"status **{result.get('status')}**"
    )
    if result.get("status") == "skipped":
        lines.append("")
        lines.append(f"SKIPPED: {result.get('reason')}")
    timing = result.get("timing") or {}
    stages = timing.get("stages") or {}
    if stages:
        lines += ["", "| stage | runs | mean ms | p50 ms | p95 ms | max ms | ms/frame | share |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, st in stages.items():
            lines.append(
                f"| {name} | {st['runs']} | {st['mean_ms']:.2f} | {st['p50_ms']:.2f} | {st['p95_ms']:.2f} | "
                f"{st['max_ms']:.2f} | {st['per_frame_ms']:.2f} | {st['share'] * 100:.1f}% |"
            )
        lines += ["", f"FPS (1000 / mean total): **{timing.get('fps', 0.0):.2f}**"]
        setup = timing.get("setup_ms") or {}
        if setup:
            lines.append("setup: " + ", ".join(f"{k} {v:.0f} ms" for k, v in setup.items()))
        counters = timing.get("counters") or {}
        if counters:
            lines.append("counters: " + ", ".join(f"{k}={v}" for k, v in counters.items()))
        if meta.get("fallbacks"):
            lines.append("fallbacks: " + ", ".join(
                f"{k}: {v['from']} -> {v['to']}" for k, v in meta["fallbacks"].items()))
    micro = result.get("micro") or {}
    if micro:
        lines += ["", "| micro-benchmark | legacy ms | new ms | speed-up | agree |", "|---|---:|---:|---:|---|"]
        for name, m in micro.items():
            if m.get("status") != "ok":
                lines.append(f"| {name} | - | - | skipped: {m.get('reason')} | - |")
            else:
                lines.append(f"| {name} | {m['legacy_ms']:.2f} | {m['new_ms']:.3f} | {m['speedup']:.1f}x | "
                             f"{m['agree']} |")
    return "\n".join(lines)


def _micro_lines(micro: dict) -> list[str]:
    out = []
    for name, m in micro.items():
        if m.get("status") != "ok":
            out.append(f"micro {name}: SKIPPED ({m.get('reason')})")
        else:
            out.append(f"micro {name}: legacy {m['legacy_ms']:.2f} ms  new {m['new_ms']:.3f} ms  "
                       f"speed-up {m['speedup']:.1f}x  agree {m['agree']}")
    return out


def _default_out(cfg: Any, micro_only: bool = False) -> Path:
    """Default result file: ``RESULTS_DIR/micro.json`` for a micro-only run, else ``<profile>-<input stem>.json``."""
    if micro_only:
        return RESULTS_DIR / "micro.json"
    path = cfg.input.get("path")
    stem = "synthetic" if path in (None, "synthetic") else (Path(str(path)).stem or "input")
    return RESULTS_DIR / f"{cfg.get('profile')}-{stem}.json"


def _realtime_note(args: argparse.Namespace, run: dict) -> str:
    """One-line explanation of the frame counts of an ``input.realtime`` run."""
    info = run["run"]
    counters = (run.get("timing") or {}).get("counters") or {}
    dropped = counters.get("dropped_frames")
    return (
        f"NOTE: input.realtime=true: --frames and --warmup count source frames "
        f"({int(args.warmup)}+{int(args.frames)} read); frames are dropped while the pipeline is busy, "
        f"so {info['frames_total']} were processed ({dropped if dropped is not None else '?'} dropped) "
        f"and {run['frames']} measured after {info['warmup']} warm-up results."
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _sanitize(obj: Any) -> Any:
    """Replace non-finite floats with None so the JSON is strict."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the benchmark; returns the exit code (0 ok/skipped, 2 configuration or input error)."""
    from vfoa.backends.base import BackendUnavailable  # noqa: PLC0415
    from vfoa.cli import live_lines  # noqa: PLC0415

    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.frames < 0 or args.warmup < 0:
        print("bench_pipeline: error: --frames and --warmup must be >= 0", file=sys.stderr)
        return 2
    if args.frames == 0 and not args.micro:
        print("bench_pipeline: error: nothing to do (--frames 0 without --micro)", file=sys.stderr)
        return 2
    if args.stereo_synth and not args.depth_dir:
        print("bench_pipeline: error: --stereo-synth needs --depth-dir (ground-truth depth)", file=sys.stderr)
        return 2

    load_start = _loadavg()
    env_start = thread_env()
    try:
        cfg = build_config(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"bench_pipeline: error: {exc}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"status": "ok", "reason": None, "meta": {}, "timing": {}, "run": {}, "micro": {}}
    run: Optional[dict] = None
    if args.frames > 0:
        try:
            run = run_pipeline(cfg, int(args.frames), int(args.warmup), stereo_synth=bool(args.stereo_synth))
        except BackendUnavailable as exc:
            result["status"] = "skipped"
            result["reason"] = str(exc)
            print(f"SKIPPED: {exc}")
        except (ValueError, FileNotFoundError) as exc:
            print(f"bench_pipeline: error: {exc}", file=sys.stderr)
            return 2
    if run is not None:
        result["timing"] = run["timing"]
        result["run"] = run["run"]
        print(run["table"])
        for kind, fb in run["fallbacks"].items():
            print(f"NOTE: {kind} backend {fb['from']!r} unavailable; measured the fallback {fb['to']!r}")
        for line in live_lines(run["run"]):
            if line.startswith("latency"):
                print(line + "  [measured frames, warm-up excluded]")
            else:
                print(line + "  [whole run, warm-up included]")
        if (cfg.get("input") or {}).get("realtime"):
            print(_realtime_note(args, run))
    if args.micro:
        result["micro"] = micro_benchmarks(repeats=int(args.micro_repeats))
        for line in _micro_lines(result["micro"]):
            print(line)
    result["meta"] = collect_meta(args, cfg, run, load_start, argv, env_start)
    if args.markdown:
        print()
        print(format_markdown(result))

    out = Path(args.out) if args.out else _default_out(cfg, micro_only=(args.frames == 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_sanitize(result), fh, indent=1, default=_json_default)
    print(f"result written to {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
