"""Gaze360 (thesis model, 7-frame GazeLSTM) as a streaming backend.

The GazeLSTM is built ONCE in ``load()`` on ``models.device`` (no forced CUDA,
no DataParallel; a checkpoint's ``module.`` prefixes are stripped - Refs #32).

``observe()`` stores, on every frame, one preprocessed head crop per track
(BGR -> RGB, 224x224 uint8 - Refs #9) in bounded per-track buffers.
``estimate()`` builds for every person the 7-frame window centred on the
current frame with spacing ``gaze.window_step`` (default ``max(fps // 8, 1)``);
each slot takes the NEAREST available crop of that track, so the window holds
real temporal context (past frames, and future frames when the pipeline runs
with ``runtime.lookahead_frames``) instead of the same frame 7 times (Refs #7).
``gaze.window: static`` reproduces the thesis behaviour (7 copies of frame t).
All persons go through one batched forward pass; the ImageNet normalisation
happens on the device. Outputs (yaw, pitch in radians) follow the Gaze360
dataset convention (eye frame +x image-left, +y up, +z away from the camera;
``t = p = 0`` looks into the camera, +yaw turns towards the image left, +pitch
looks up) and are converted with :func:`vfoa.geometry.gaze360_to_camera`, i.e.
``(-cos p sin t, -sin p, -cos p cos t)`` in the camera frame. Gaze360 predicts
gaze relative to the camera -> eye ray, so ``gaze.eye_frame_correction`` (on by
default) rotates that vector into the frame of the ray through the eye pixel.
An uncertainty above ``gaze.uncertainty_thr`` gives no estimate.

torch and the gaze360 repository are imported lazily; the model factory is
injectable for tests.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from vfoa.backends.accurate._thirdparty import ENV_VAR, resolve_path, third_party_root
from vfoa.backends.accurate.fsanet import crop_head
from vfoa.backends.base import BackendUnavailable, GazeBackend, resolve_device
from vfoa.geometry import eye_pixel, gaze360_to_camera, unproject
from vfoa.types import DirectionEstimate, Frame, Person

__all__ = ["MEAN", "STD", "WINDOW", "CROP_SIZE", "strip_module_prefix", "Gaze360Gaze"]

logger = logging.getLogger(__name__)

#: ImageNet normalisation in RGB order.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
#: Number of frames in a Gaze360 window (slot 3 is the current frame).
WINDOW = 7
#: Side of the model input crop in pixels.
CROP_SIZE = 224


def strip_module_prefix(state: Mapping[str, Any]) -> dict:
    """Remove a leading ``module.`` (DataParallel) from every key of a state dict."""
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}


def _to_numpy(a: Any) -> np.ndarray:
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)


class Gaze360Gaze(GazeBackend):
    """Gaze360 GazeLSTM with a real temporal window over per-track crop buffers.

    ``model_factory() -> model`` replaces loading the repo + checkpoint (tests).
    The model maps a float tensor (P, 7, 3, 224, 224) to ``(angles (P, 2) [yaw,
    pitch] rad, var (P, 2))``. ``fps`` may be set by the pipeline; otherwise it
    comes from ``input.fps`` or is inferred from ``Frame.t``.
    """

    name = "gaze360"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        model_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().__init__(cfg)
        self._model_factory = model_factory
        self._model: Any = None
        self._device = "cpu"
        self._buffers: dict[int, dict[int, np.ndarray]] = {}
        self._last_index = -1
        self._cfg_fps = (cfg.get("input") or {}).get("fps")
        self.fps: Optional[float] = float(self._cfg_fps) if self._cfg_fps else None
        runtime = cfg.get("runtime") or {}
        self.lookahead = int(runtime.get("lookahead_frames", 0) or 0)

    # ------------------------------------------------------------------ loading

    def _load(self) -> None:
        try:
            import torch  # noqa: F401, PLC0415 (lazy heavy import)
        except ImportError as exc:
            raise BackendUnavailable(
                "Gaze360 needs the 'torch' package; install torch or choose another gaze.backend"
            ) from exc
        self._device = resolve_device(self.cfg)
        factory = self._model_factory or self._default_model
        self._model = factory()

    def _default_model(self) -> Any:
        import torch  # noqa: PLC0415 (lazy heavy import)

        weights_value = self.section.get("weights")
        if not weights_value:
            raise BackendUnavailable("Gaze360: config key gaze.weights is empty")
        try:
            weights = resolve_path(self.cfg, weights_value)
        except BackendUnavailable as exc:
            raise BackendUnavailable(
                f"Gaze360 weights {str(weights_value)!r} (config key gaze.weights) cannot be "
                f"resolved: {exc}"
            ) from exc
        if not weights.is_file():
            raise BackendUnavailable(
                f"Gaze360 checkpoint {str(weights)!r} not found (config key gaze.weights, "
                f"relative paths resolve against models.third_party_root / {ENV_VAR})"
            )
        root = third_party_root(self.cfg)
        candidates = []
        if root is not None:
            candidates = [root / "gaze360" / "code", root / "gaze360" / "gaze360" / "code", root / "gaze360"]
        code_dir = next((c for c in candidates if (c / "model.py").is_file()), None)
        if code_dir is None:
            raise BackendUnavailable(
                "Gaze360 repository not found: expected gaze360/code/model.py under the config key "
                f"models.third_party_root (or env {ENV_VAR}); root = {str(root)!r}"
            )
        for entry in (str(code_dir), str(root)):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        spec = importlib.util.spec_from_file_location("vfoa_gaze360_model", code_dir / "model.py")
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise BackendUnavailable(f"Gaze360: cannot import {code_dir / 'model.py'}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        original = getattr(mod, "resnet18", None)
        if original is not None:
            def _no_download(*args: Any, **kwargs: Any) -> Any:
                kwargs["pretrained"] = False
                return original(*args, **kwargs)

            mod.resnet18 = _no_download
        try:
            model = mod.GazeLSTM()
        finally:
            if original is not None:
                mod.resnet18 = original

        device = self._device
        try:
            ckpt = torch.load(str(weights), map_location=device, weights_only=True)
        except Exception as exc:
            logger.warning(
                "Gaze360: weights_only load of %s failed (%s); retrying with weights_only=False "
                "(trusted local checkpoint)", weights, exc,
            )
            ckpt = torch.load(str(weights), map_location=device, weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, Mapping) else ckpt
        model.load_state_dict(strip_module_prefix(state))
        model = model.to(device).eval()
        logger.info("Gaze360: loaded %s on %s", weights, device)
        return model

    # ---------------------------------------------------------------- buffering

    def _step(self) -> int:
        step = self.section.get("window_step")
        if step:
            return max(int(step), 1)
        fps = self.fps or 30.0
        return max(int(round(float(fps), 3) // 8), 1)

    def _span(self) -> int:
        return 3 * self._step() + self.lookahead

    def _store(self, frame: Frame, persons: Sequence[Person], stream: bool = True) -> None:
        """Buffer crops of ``persons`` at ``frame.index``.

        ``stream=True`` (observe): the frame is the newest of the stream; an index
        going backwards means a restarted stream (buffers cleared) and old
        entries are pruned. ``stream=False`` (estimate filling gaps, possibly
        behind the newest frame with a lookahead): only adds crops.
        """
        if stream:
            if frame.index < self._last_index:
                self._buffers.clear()
            self._last_index = frame.index
        if self.fps is None and frame.index > 0 and frame.t > 0:
            self.fps = round(frame.index / frame.t, 3)
        for p in persons:
            buf = self._buffers.get(p.track_id)
            if buf is not None and frame.index in buf:
                continue
            crop = crop_head(frame.image, p.detection.head_box)
            if crop is None:
                continue
            img = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._buffers.setdefault(p.track_id, {})[frame.index] = img
        if stream:
            oldest = frame.index - self._span()
            for tid in list(self._buffers):
                buf = self._buffers[tid]
                for idx in [i for i in buf if i < oldest]:
                    del buf[idx]
                if not buf:
                    del self._buffers[tid]

    def observe(self, frame: Frame, persons: Sequence[Person]) -> None:
        """Buffer one RGB 224x224 crop per track for ``frame`` (cheap, no model)."""
        self._store(frame, persons)

    def window_indices(self, track_id: int, t: int) -> Optional[np.ndarray]:
        """Frame indices of the 7 window slots of ``track_id`` at frame ``t`` (None if no crop at t)."""
        buf = self._buffers.get(track_id)
        if not buf or t not in buf:
            return None
        if str(self.section.get("window", "causal")) == "static":
            return np.full(WINDOW, t, dtype=np.int64)
        avail = np.array(sorted(buf), dtype=np.int64)
        slots = t + self._step() * np.arange(-(WINDOW // 2), WINDOW // 2 + 1, dtype=np.int64)
        return avail[np.abs(avail[None, :] - slots[:, None]).argmin(axis=1)]

    # ---------------------------------------------------------------- inference

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        """Gaze direction per track id from one batched forward pass over all persons."""
        self.load()
        import torch  # noqa: PLC0415 (lazy heavy import)

        missing = [p for p in persons if frame.index not in self._buffers.get(p.track_id, {})]
        if missing:
            self._store(frame, missing, stream=frame.index > self._last_index)
        t = frame.index
        chosen: list[Person] = []
        windows: list[np.ndarray] = []
        for p in persons:
            idx = self.window_indices(p.track_id, t)
            if idx is None:
                continue
            buf = self._buffers[p.track_id]
            windows.append(np.stack([buf[int(i)] for i in idx]))
            chosen.append(p)
        if not chosen:
            return {}
        batch_all = np.stack(windows)  # (P, 7, 224, 224, 3) uint8 RGB
        bs = int(self.section.get("batch_size") or 0) or len(chosen)
        device = self._device
        mean = torch.tensor(MEAN, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(STD, dtype=torch.float32, device=device).view(1, 1, 3, 1, 1)
        angs, vars_ = [], []
        with torch.no_grad():
            for s in range(0, len(chosen), bs):
                x = torch.from_numpy(np.ascontiguousarray(batch_all[s:s + bs])).to(device)
                x = x.permute(0, 1, 4, 2, 3).float().div_(255.0)
                x = (x - mean) / std
                out = self._model(x)
                angs.append(_to_numpy(out[0]).reshape(-1, 2))
                vars_.append(_to_numpy(out[1]).reshape(len(x), -1))
        ang = np.concatenate(angs).astype(np.float64)
        var = np.concatenate(vars_).astype(np.float64)

        thr = float(self.section.get("uncertainty_thr", 3.0))
        correct = bool(self.section.get("eye_frame_correction", True)) and frame.intrinsics is not None
        kpt_thr = float((self.cfg.get("persons") or {}).get("kpt_thr", 0.35))
        result: dict[int, DirectionEstimate] = {}
        for k, p in enumerate(chosen):
            yaw, pitch, u = ang[k, 0], ang[k, 1], var[k, 0]
            if not (np.isfinite(yaw) and np.isfinite(pitch) and np.isfinite(u)) or u > thr:
                continue
            eye_ray = None
            if correct:
                uv = p.eyes_px
                if uv is None:
                    uv = eye_pixel(p.detection.head_box, p.detection.keypoints, kpt_thr)
                if uv is not None:
                    eye_ray = unproject(np.asarray(uv, dtype=np.float64), 1.0, frame.intrinsics)
            direction = gaze360_to_camera(yaw, pitch, eye_ray)
            denom = min(thr, math.pi) if thr > 0 else math.pi
            conf = float(np.clip(1.0 - u / denom, 0.0, 1.0))
            result[p.track_id] = DirectionEstimate(
                direction=direction, confidence=conf, frame_index=frame.index
            )
        return result

    # ---------------------------------------------------------------- lifecycle

    def reset(self) -> None:
        self._buffers.clear()
        self._last_index = -1
        self.fps = float(self._cfg_fps) if self._cfg_fps else None

    def close(self) -> None:
        self._model = None
        self._loaded = False
