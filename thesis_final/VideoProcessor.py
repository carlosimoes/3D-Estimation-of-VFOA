"""Legacy video processing entry point, a thin wrapper over :class:`vfoa.pipeline.Pipeline`.

The thesis ``VideoProcessor`` read the whole video into RAM, split it into
batches (with a broken multi-batch path), reloaded every model per call and
encoded the output videos with ffmpeg. It now streams the input frame by frame
through the vfoa pipeline (models loaded once, ids consistent over the whole
video) and writes, incrementally, into ``<output.dir>/<video stem>/``:

* ``results.jsonl`` / ``results.csv``  structured per-frame outputs (``t = frame / fps`` seconds),
* ``gazes.mp4``, ``keypoints.mp4``, ``headpose.mp4``, ``overlay.mp4``, ``mosaic.mp4``
  feature videos (``writeVideo``; OpenCV ``mp4v``, no ffmpeg),
* ``timing.json`` (per-stage timing summary), ``config.yaml`` (resolved config) and
  ``features.desc`` (Width, Height, NumberFrames, FrameRate).

The source frames are never modified: every drawing helper returns a new image.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import copy
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from GazeFrameRenderer import draw_gaze_arrow
from Reconstruction_scene import depth_from_disparity, q_for_frame
from vfoa.backends.base import DepthBackend
from vfoa.backends.depth_simple import frame_intrinsics
from vfoa.config import Config, dump_config, load_config
from vfoa.io.outputs import CsvSink, JsonlSink, VideoSink, compose_mosaic, draw_head_axes, draw_overlay
from vfoa.pipeline import Pipeline, open_input
from vfoa.types import DepthFrame, Frame, FrameResult

__all__ = [
    "VideoProcessor",
    "PrecomputedDepth",
    "FeatureVideoSink",
    "config_from_legacy_args",
    "draw_keypoints",
    "draw_gazes",
    "draw_headpose",
    "COCO_LIMBS",
]

logger = logging.getLogger(__name__)

DEFAULT_PROFILE = "thesis"

#: COCO-17 skeleton (pairs of keypoint indices).
COCO_LIMBS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)
_LIMB_COLORS: tuple[tuple[int, int, int], ...] = tuple(
    tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(180 * k / len(COCO_LIMBS)), 255, 255]]]),
                                       cv2.COLOR_HSV2BGR)[0, 0])
    for k in range(len(COCO_LIMBS))
)

#: legacy flag -> config key path ("section.key" or "section.key[index]").
LEGACY_FLAGS: tuple[tuple[str, str], ...] = (
    ("cfg", "persons.openpose_cfg"),
    ("opM", "persons.openpose_weights"),
    ("dpCfg", "persons.densepose_cfg"),
    ("dpM", "persons.densepose_weights"),
    ("dpT", "persons.densepose_thr"),
    ("PpCfg", "segmentation.panoptic_cfg"),
    ("PpM", "segmentation.panoptic_weights"),
    ("PpT", "segmentation.panoptic_thr"),
    ("gM", "gaze.weights"),
    ("gB", "gaze.batch_size"),
    ("gT", "gaze.uncertainty_thr"),
    ("od", "output.dir"),
    ("max_frames", "input.max_frames"),
)


# ------------------------------------------------------------------ config
def _profile_of_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if isinstance(data, Mapping) and data.get("profile"):
        return str(data["profile"])
    return None


def config_from_legacy_args(args: Any) -> Config:
    """Resolved :class:`Config` of legacy ``FinalTest`` arguments (or a config mapping).

    A :class:`Config` is deep-copied; another mapping is merged over its
    ``profile`` (default ``thesis``). For an argparse namespace (every attribute
    optional) the profile is ``args.profile``, else the ``profile`` key of
    ``args.config``, else ``thesis``; ``args.set`` are ``a.b=c`` overrides; then
    every non-None legacy flag (``cfg, opM, dpCfg, dpM, dpT, PpCfg, PpM, PpT, gM,
    gB, gT, hM1, hM2, od, max_frames, dev``) is written into its config key.
    """
    if isinstance(args, Config):
        return copy.deepcopy(args)
    if isinstance(args, Mapping):
        data = dict(args)
        profile = data.pop("profile", None) or DEFAULT_PROFILE
        return load_config(profile=profile, overrides=data or None)
    config_path = getattr(args, "config", None)
    profile = getattr(args, "profile", None) or _profile_of_file(config_path) or DEFAULT_PROFILE
    sets = list(getattr(args, "set", None) or [])
    cfg = load_config(config_path, profile, overrides=sets or None)
    for flag, key in LEGACY_FLAGS:
        value = getattr(args, flag, None)
        if value is None:
            continue
        section, name = key.split(".")
        cfg[section][name] = value
    models = list(cfg.headpose.get("models") or [])
    for i, flag in enumerate(("hM1", "hM2")):
        value = getattr(args, flag, None)
        if value is not None:
            while len(models) <= i:
                models.append(None)
            models[i] = value
    cfg.headpose.models = models
    dev = getattr(args, "dev", None)
    if isinstance(dev, str) and (dev == "cpu" or dev.startswith("cuda")):
        cfg.models.device = dev
    return cfg


# ------------------------------------------------------------------ depth
class PrecomputedDepth(DepthBackend):
    """Depth served from legacy precomputed disparities (``Depth_Estimator`` output).

    ``disp[k]`` (pixels, or a :class:`DepthFrame`) belongs to frame
    ``input.start + k``; ``matrix_Q`` is one 4x4 Q matrix or one per frame.
    Depth is ``Z = fx * B / d`` metres; frames outside the list get an
    all-invalid depth.
    """

    name = "precomputed"

    def __init__(self, cfg: Mapping[str, Any], matrix_Q: Any = None, disp: Optional[Sequence[Any]] = None) -> None:
        super().__init__(cfg)
        self.matrix_Q = matrix_Q
        self.disp = disp if disp is not None else []
        self.calls = 0

    def compute(self, frame: Frame) -> DepthFrame:
        self.calls += 1
        start = int((self.cfg.get("input") or {}).get("start") or 0)
        k = int(frame.index) - start
        h, w = frame.image.shape[:2]
        if 0 <= k < len(self.disp):
            df = depth_from_disparity(self.disp[k], q_for_frame(self.matrix_Q, k))
            if df.depth.shape[:2] == (h, w):
                return df
            sx, sy = w / df.depth.shape[1], h / df.depth.shape[0]
            depth = cv2.resize(df.depth, (w, h), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(df.valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            intr = df.intrinsics.scaled(sx, sy)
            intr = dataclasses.replace(intr, width=w, height=h)
            return DepthFrame(depth=depth.astype(np.float32), valid=valid, intrinsics=intr)
        return DepthFrame(
            depth=np.zeros((h, w), np.float32), valid=np.zeros((h, w), bool),
            intrinsics=frame_intrinsics(self.cfg, frame),
        )


# ------------------------------------------------------------------ drawing
def _scale(size: Sequence[int], src_size: Optional[Sequence[int]]) -> tuple[float, float]:
    if not src_size or not src_size[0] or not src_size[1]:
        return 1.0, 1.0
    return float(size[0]) / float(src_size[0]), float(size[1]) / float(src_size[1])


def draw_keypoints(result: FrameResult, size: Sequence[int], src_size: Optional[Sequence[int]] = None) -> np.ndarray:
    """New black (h, w, 3) image (``size`` = (w, h)) with the COCO-17 skeletons of ``result``.

    Keypoints (pixels of an image of ``src_size`` = (w, h)) are rescaled; only
    joints with confidence > 0 are drawn.
    """
    w, h = int(size[0]), int(size[1])
    out = np.zeros((h, w, 3), np.uint8)
    sx, sy = _scale(size, src_size)
    for person in result.persons:
        det = person.detection
        kp = None if det is None else det.keypoints
        if kp is None:
            continue
        kp = np.asarray(kp, dtype=np.float64)
        if kp.shape != (17, 3):
            continue
        ok = (kp[:, 2] > 0) & np.isfinite(kp).all(axis=1)
        pts = np.round(kp[:, :2] * (sx, sy)).astype(np.int64)
        for (a, b), color in zip(COCO_LIMBS, _LIMB_COLORS):
            if ok[a] and ok[b]:
                cv2.line(out, (int(pts[a, 0]), int(pts[a, 1])), (int(pts[b, 0]), int(pts[b, 1])), color, 2,
                         cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            cv2.circle(out, (int(pts[j, 0]), int(pts[j, 1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def _eye_px(person: Any) -> Optional[np.ndarray]:
    if person.eyes_px is not None:
        e = np.asarray(person.eyes_px, dtype=np.float64).reshape(-1)
        if e.shape == (2,) and np.isfinite(e).all():
            return e
    det = person.detection
    box = None if det is None else det.head_box
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.shape != (4,) or not np.isfinite(b).all():
        return None
    return np.array([(b[0] + b[2]) / 2.0, 0.65 * b[1] + 0.35 * b[3]])


def draw_gazes(result: FrameResult, size: Sequence[int], src_size: Optional[Sequence[int]] = None,
               length: float = 0.1) -> np.ndarray:
    """New black (h, w, 3) image with one gaze arrow per person (origin = eye pixel).

    ``length`` is the arrow length as a fraction of the output width.
    """
    w, h = int(size[0]), int(size[1])
    out = np.zeros((h, w, 3), np.uint8)
    sx, sy = _scale(size, src_size)
    for person in result.persons:
        if person.gaze is None:
            continue
        eye = _eye_px(person)
        if eye is None:
            continue
        out = draw_gaze_arrow(out, (eye[0] * sx, eye[1] * sy), person.gaze.direction, length * w)
    return out


def draw_headpose(image: np.ndarray, result: FrameResult) -> np.ndarray:
    """COPY of ``image`` with the head pose of every person (FSA-Net axes, or an arrow)."""
    out = np.array(image, dtype=np.uint8, copy=True)
    for person in result.persons:
        if person.head is None:
            continue
        eye = _eye_px(person)
        if eye is None:
            continue
        eul = person.head.euler_deg
        if eul is not None and np.isfinite(np.asarray(eul, dtype=np.float64)).all():
            e = np.asarray(eul, dtype=np.float64).reshape(-1)
            out = draw_head_axes(out, e[0], e[1], e[2], origin=eye, size=50.0)
        else:
            out = draw_gaze_arrow(out, eye, person.head.direction, 50.0, color=(255, 255, 255))
    return out


class FeatureVideoSink:
    """Streams the five feature videos of the legacy pipeline (sink protocol).

    ``gazes.mp4`` and ``keypoints.mp4`` are drawn on black ``size`` = (w, h)
    canvases, ``headpose.mp4`` and ``overlay.mp4`` on copies of the frame, and
    ``mosaic.mp4`` = 2x2 [frame, headpose, gazes, keypoints] at (2w, 2h).
    """

    NAMES = ("gazes", "keypoints", "headpose", "overlay", "mosaic")

    def __init__(self, out_dir: Any, fps: float, size: Sequence[int] = (960, 720)) -> None:
        self.out_dir = Path(out_dir)
        self.size = (int(size[0]), int(size[1]))
        w, h = self.size
        self.sinks = {
            "gazes": VideoSink(self.out_dir / "gazes.mp4", fps, self.size, overlay=False),
            "keypoints": VideoSink(self.out_dir / "keypoints.mp4", fps, self.size, overlay=False),
            "headpose": VideoSink(self.out_dir / "headpose.mp4", fps, None, overlay=False),
            "overlay": VideoSink(self.out_dir / "overlay.mp4", fps, None, overlay=False),
            "mosaic": VideoSink(self.out_dir / "mosaic.mp4", fps, (2 * w, 2 * h), overlay=False),
        }
        self._src_size: Optional[tuple[int, int]] = None

    def write(self, frame: Optional[Frame], result: FrameResult) -> None:
        if frame is not None:
            image = frame.image
            self._src_size = (int(image.shape[1]), int(image.shape[0]))
        else:
            sw, sh = self._src_size or self.size
            image = np.zeros((sh, sw, 3), np.uint8)
        src = (int(image.shape[1]), int(image.shape[0]))
        gazes = draw_gazes(result, self.size, src)
        keypoints = draw_keypoints(result, self.size, src)
        headpose = draw_headpose(image, result)
        overlay = draw_overlay(image, result, None if frame is None else frame.intrinsics)
        w, h = self.size
        mosaic = compose_mosaic([image, headpose, gazes, keypoints], (2, 2), (2 * w, 2 * h))
        for name, img in (("gazes", gazes), ("keypoints", keypoints), ("headpose", headpose),
                          ("overlay", overlay), ("mosaic", mosaic)):
            self.sinks[name].write_image(img)

    def close(self) -> None:
        for s in self.sinks.values():
            s.close()


# ------------------------------------------------------------------ processor
class VideoProcessor:
    """Runs the VFOA pipeline over a video and writes the legacy outputs.

    ``args`` is a legacy argparse namespace, a :class:`Config` or a config
    mapping (see :func:`config_from_legacy_args`). ``matrix_Q``/``disp`` are
    optional precomputed Q matrices and disparities (``Depth_Estimator``);
    ``w``/``h`` the size of the gaze/keypoint feature videos.
    """

    def __init__(self, args: Any, matrix_Q: Any = None, disp: Optional[Sequence[Any]] = None,
                 w: int = 960, h: int = 720) -> None:
        self.config = args
        self.cfg: Config = config_from_legacy_args(args)
        self.matrix_Q = matrix_Q
        self.disp = disp
        self.outputWidth = int(w)
        self.outputHeight = int(h)
        self.inputW = self.inputH = 0
        self.extraSinks: list = []
        self.directory: Optional[Path] = None
        self.summary: dict = {}
        self.report: str = ""
        self.pipeline: Optional[Pipeline] = None

    # -------------------------------------------------------------- main entry
    def processVideo(self, filePath: Any, imagesBatchLength: int = 10000, extractGaze: bool = True,
                     writeVideo: bool = True, matchPointsBox: bool = False, *,
                     outputDir: Any = None) -> dict:
        """Stream ``filePath`` (video, image directory, camera index or ``synthetic``) through the pipeline.

        ``imagesBatchLength`` is accepted for compatibility and ignored (the
        input is streamed, so any length works with consistent track ids).
        ``extractGaze=False`` disables the head-pose and gaze stages;
        ``matchPointsBox`` sets ``persons.match_keypoints``. Outputs go to
        ``outputDir`` or ``<output.dir>/<input stem>/`` (created, never wiped).
        Returns the run summary (also in ``self.summary``).
        """
        cfg = copy.deepcopy(self.cfg)
        return self._process(cfg, filePath, extractGaze, writeVideo, matchPointsBox, outputDir)

    def _process(self, cfg: Config, filePath: Any, extractGaze: bool, writeVideo: bool,
                 matchPointsBox: bool, outputDir: Any) -> dict:
        if filePath is not None:
            cfg.input.path = str(filePath)
            cfg.input.source = "auto"
        if not extractGaze:
            cfg.gaze.backend = "none"
            cfg.headpose.backend = "none"
        cfg.persons.match_keypoints = bool(matchPointsBox)
        if cfg.depth.get("backend") == "stereo_bm" and not cfg.input.get("right_path") and self.disp is None:
            fallback = "sintel_gt" if cfg.input.get("depth_dir") else "constant"
            logger.warning("depth.backend stereo_bm needs a right view (input.right_path); using %s", fallback)
            cfg.depth.backend = fallback
        if cfg.input.get("path") is None:
            raise ValueError("no input: pass a video path or set input.path")
        stem = Path(str(cfg.input.path).rstrip("/\\")).stem or "input"
        out_dir = Path(outputDir) if outputDir is not None else Path(cfg.output.get("dir") or "outputs") / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.output.dir = str(out_dir)
        self.directory = out_dir
        dump_config(cfg, out_dir / "config.yaml")

        backends = None
        if self.disp is not None:
            backends = {"depth": PrecomputedDepth(cfg, self.matrix_Q, self.disp)}
        source = open_input(cfg)
        fps = float(getattr(source, "fps", None) or 30.0)
        self.inputW = int(getattr(source, "width", 0) or 0)
        self.inputH = int(getattr(source, "height", 0) or 0)
        sinks: list = []
        try:
            if cfg.output.get("jsonl", True):
                sinks.append(JsonlSink(out_dir / "results.jsonl"))
            sinks.append(CsvSink(out_dir / "results.csv"))
            if writeVideo:
                sinks.append(FeatureVideoSink(out_dir, fps, (self.outputWidth, self.outputHeight)))
            sinks.extend(self.extraSinks)
        except Exception:
            for s in sinks:
                s.close()
            source.close()
            raise
        self.pipeline = Pipeline(cfg, backends)
        summary = self.pipeline.run(source, sinks)
        self.summary = summary
        self.report = self.pipeline.timer.report()
        with open(out_dir / "timing.json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True, default=str)
        desc = {
            "Width": self.inputW,
            "Height": self.inputH,
            "NumberFrames": int(summary.get("frames", 0) or 0),
            "FrameRate": fps,
        }
        with open(out_dir / "features.desc", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(desc, fh)
        return summary

    def generateVideosWithFeatures(self, filePath: str, outputPath: Any, indexBegin: int = 0,
                                   numberFrames: int = sys.maxsize) -> dict:
        """Process ``numberFrames`` frames from ``indexBegin`` and write the feature videos to ``outputPath``."""
        cfg = copy.deepcopy(self.cfg)
        cfg.input.start = int(indexBegin)
        if numberFrames is not None and numberFrames < sys.maxsize:
            cfg.input.max_frames = int(numberFrames)
        return self._process(cfg, filePath, True, True, bool(cfg.persons.get("match_keypoints", False)),
                             outputPath)

    # -------------------------------------------------------------- legacy drawing helpers
    def _eye(self, bbox: Any, image: np.ndarray, normalise: bool) -> np.ndarray:
        b = np.asarray(bbox, dtype=np.float64).reshape(-1)[:4]
        eye = np.array([(b[0] + b[2]) / 2.0, 0.65 * b[1] + 0.35 * b[3]])
        if normalise and self.inputW and self.inputH:
            eye = eye * (image.shape[1] / self.inputW, image.shape[0] / self.inputH)
        return eye

    def getFullImage(self, image: np.ndarray, bbox: Any, gaze: Any) -> np.ndarray:
        """New image: ``image`` with the gaze arrow at the eye pixel of ``bbox`` (input pixels)."""
        img = np.asarray(image)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return draw_gaze_arrow(img, self._eye(bbox, img, True), gaze, 0.1 * img.shape[1])

    def getFullImageHP(self, image: np.ndarray, bbox: Any, headpose: Any) -> np.ndarray:
        """New image: ``image`` with the FSA-Net (yaw, pitch, roll) axes at the eye pixel of ``bbox``."""
        eye = self._eye(bbox, np.asarray(image), False)
        return self.draw_axis(image, headpose[0], headpose[1], headpose[2], tdx=eye[0], tdy=eye[1], size=50)

    def draw_axis(self, img: np.ndarray, yaw: float, pitch: float, roll: float, tdx: Optional[float] = None,
                  tdy: Optional[float] = None, size: float = 50, thickness: Any = (2, 2, 2)) -> np.ndarray:
        """New image with the head axes (X red, Y green, Z blue); see ``vfoa.io.outputs.draw_head_axes``."""
        origin = None if tdx is None or tdy is None else (tdx, tdy)
        t = thickness[0] if isinstance(thickness, (tuple, list)) else thickness
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return draw_head_axes(img, yaw, pitch, roll, origin=origin, size=size, thickness=int(t))
