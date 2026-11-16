"""Fast appearance-based gaze: L2CS-style binned yaw/pitch models on ONNX Runtime.

Backend ``gaze.backend: l2cs`` (:class:`BinnedGaze360Backend`). One model call
per face crop gives two heads of 90 logits each (4 degree bins starting at
-180 degrees). Supported checkpoints, all in this output format:

* MobileGaze (https://github.com/yakhyo/gaze-estimation, release ``weights``):
  ``mobileone_s0`` (default, CPU friendly), ``resnet18``, ``resnet34``,
  ``resnet50``, ``mobilenetv2`` as ``<name>_gaze.onnx``. The released ONNX
  files have a static batch of 1; a model re-exported with
  ``onnx_export.py --dynamic`` is batched automatically.
* L2CS-Net ResNet-50 (Gaze360): no ONNX file is published (the official link
  is dead; the py-feat Hugging Face mirror has safetensors). Export it to ONNX
  yourself and point ``gaze.l2cs.path`` at it; the decode is the same.

Pipeline per frame (``estimate``):

1. Gate: the nose and at least one eye keypoint must be confident
   (``gaze.l2cs.kpt_thr``). Persons that fail get no gaze estimate, so the
   attention stage falls back to head pose (after the pipeline's
   ``gaze.max_hold_frames`` hold of the last estimate).
2. Square crop centred on the confident face keypoints, side
   ``crop_margin * max(keypoint extent, 2.3 * eye distance)``, limited by the
   head box; border replication outside the frame.
3. BGR -> RGB, one resize to ``input_size`` (default 448, the training
   size), ``/255``, ImageNet mean/std, NCHW float32. The frame is never mirrored.
4. Decode each head: ``deg = sum(softmax(logits) * arange(90)) * 4 - 180``.
   The outputs are taken BY POSITION (first = ``a``, second = ``b``): the
   L2CS repository names them inconsistently (its ``forward`` returns yaw
   first, ``train.py`` unpacks pitch first), so names are never used.
5. Direction in the OpenCV camera frame (+x right, +y down, +z forward)::

       d = (-cos(b) sin(a), -sin(b), -cos(b) cos(a))

   ``a = b = 0`` looks straight at the camera ``(0, 0, -1)``; positive ``a``
   points image-left (``-x``), positive ``b`` points image-up (``-y``). These
   signs are DERIVED FROM THE REPOSITORIES' CODE (``draw_gaze`` and the
   Gaze360 eye-frame convention), not checked on real footage yet.
6. Optional perspective correction (``perspective_correction``, default off,
   unverified): the model only sees the crop, so ``d`` is rotated by the
   minimal rotation taking the optical axis to the camera ray through the
   crop centre.
7. Confidence = ``sqrt(c_a * c_b) * visibility`` in [0, 1], where
   ``c = 1 - entropy(softmax) / log(90)`` per head and ``visibility`` is the
   face keypoint confidence from the gate.

Weights are never committed. They resolve from ``gaze.l2cs.path``, else
``<weights_dir>/<file>`` (``models.weights_dir`` or env ``VFOA_WEIGHTS_DIR``),
else they are downloaded there when ``models.allow_download`` and
``gaze.l2cs.download`` are true. Otherwise :class:`BackendUnavailable` is
raised (the fast profile then falls back to ``head_proxy``).

Cost: MobileOne-S0 at 448 px takes about 40-65 ms per person on CPU, far over
a real-time budget; use ``gaze.stride`` (or ``input_size: 224``, accuracy
unknown) for real-time CPU runs.

Licences: the MobileGaze and L2CS-Net code is MIT. The weights are trained on
Gaze360, whose terms are research / non-commercial only; the weights are
therefore an optional download and must not be committed or redistributed
for commercial use.

Import hygiene: this module imports only NumPy, OpenCV and vfoa;
``onnxruntime`` is imported lazily in ``load()``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from vfoa.backends.base import BackendUnavailable, GazeBackend, weights_dir
from vfoa.types import DirectionEstimate, Frame, Person

__all__ = [
    "WEIGHTS",
    "face_visible",
    "face_crop_box",
    "crop_square",
    "preprocess",
    "decode_bins",
    "angles_to_direction",
    "head_ray_rotation",
    "resolve_l2cs_weights",
    "BinnedGaze360Backend",
]

log = logging.getLogger(__name__)

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
FACE = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
N_BINS = 90
BIN_DEG = 4.0
OFFSET_DEG = 180.0

_RELEASE = "https://github.com/yakhyo/gaze-estimation/releases/download/weights/"
#: name -> (file name, URL, sha256 or None). Only verified hashes are listed.
WEIGHTS: dict[str, tuple[str, str, Optional[str]]] = {
    "mobileone_s0": (
        "mobileone_s0_gaze.onnx",
        _RELEASE + "mobileone_s0_gaze.onnx",
        "8b4fdc4e3da44733c9a82e7776b411e4a39f94e8e285aee0fc85a548a55f7d9f",
    ),
    "resnet18": ("resnet18_gaze.onnx", _RELEASE + "resnet18_gaze.onnx", None),
    "resnet34": ("resnet34_gaze.onnx", _RELEASE + "resnet34_gaze.onnx", None),
    "resnet50": ("resnet50_gaze.onnx", _RELEASE + "resnet50_gaze.onnx", None),
    "mobilenetv2": ("mobilenetv2_gaze.onnx", _RELEASE + "mobilenetv2_gaze.onnx", None),
}

_HINT = ("set gaze.l2cs.path to an ONNX file, put it in models.weights_dir (or env VFOA_WEIGHTS_DIR), "
         "or use gaze.backend=head_proxy")


# --------------------------------------------------------------------- helpers
def face_visible(kp: Optional[np.ndarray], thr: float) -> float:
    """Face visibility in [0, 1] from COCO-17 keypoints; 0 = gate failed.

    Fails when ``kp`` is None, the nose confidence is below ``thr`` or both
    eye confidences are below ``thr``. Otherwise ``mean(nose, best eye)``.
    """
    if kp is None:
        return 0.0
    kp = np.asarray(kp, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] < 3 or kp.shape[1] < 3:
        return 0.0
    nose = float(kp[NOSE, 2])
    eye = float(max(kp[L_EYE, 2], kp[R_EYE, 2]))
    if not (nose >= thr and eye >= thr):
        return 0.0
    return float(np.clip(0.5 * (nose + eye), 0.0, 1.0))


def _square_of(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (float(v) for v in box)
    side = max(x2 - x1, y2 - y1)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float64)


def face_crop_box(
    kp: Optional[np.ndarray],
    head_box: Optional[np.ndarray],
    thr: float,
    margin: float = 1.1,
    min_side: float = 4.0,
) -> Optional[np.ndarray]:
    """Square face box ``[x1, y1, x2, y2]`` (float64 pixels, may leave the frame).

    Centred on the extent of the confident face keypoints (nose, eyes, ears);
    side ``margin * max(keypoint extent, 2.3 * eye distance if both eyes are
    confident)``. With a head box the side is clamped to
    ``[0.5, 1] * its longer side``. With fewer than 2 confident points the
    head box made square is used. ``None`` when nothing gives at least
    ``min_side`` pixels.
    """
    hb = None
    if head_box is not None:
        hb = np.asarray(head_box, dtype=np.float64).reshape(-1)
        if hb.shape != (4,) or not np.isfinite(hb).all() or hb[2] - hb[0] <= 0 or hb[3] - hb[1] <= 0:
            hb = None
    pts = None
    both_eyes = False
    if kp is not None:
        k = np.asarray(kp, dtype=np.float64)
        ok = np.array([k[i, 2] >= thr and np.isfinite(k[i, :2]).all() for i in FACE])
        if ok.sum() >= 2:
            pts = k[list(FACE), :2][ok]
            both_eyes = bool(ok[L_EYE] and ok[R_EYE])
    if pts is None:
        if hb is None:
            return None
        box = _square_of(hb)
        return box if box[2] - box[0] >= min_side else None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float(max(hi - lo))
    if both_eyes:
        extent = max(extent, 2.3 * float(np.linalg.norm(k[L_EYE, :2] - k[R_EYE, :2])))
    side = float(margin) * extent
    if hb is not None:
        long_side = float(max(hb[2] - hb[0], hb[3] - hb[1]))
        side = min(max(side, 0.5 * long_side), long_side)
    if not side >= min_side:
        return None
    c = 0.5 * (lo + hi)
    return np.array([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2], dtype=np.float64)


def crop_square(image: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Crop ``box`` from ``image`` (border replication outside it) and resize to ``(size, size, 3)``."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    side = max(int(round(max(x2 - x1, y2 - y1))), 1)
    ix1 = int(math.floor(0.5 * (x1 + x2) - side / 2.0 + 0.5))
    iy1 = int(math.floor(0.5 * (y1 + y2) - side / 2.0 + 0.5))
    ix2, iy2 = ix1 + side, iy1 + side
    cx1, cy1 = max(ix1, 0), max(iy1, 0)
    cx2, cy2 = min(ix2, w), min(iy2, h)
    if cx2 <= cx1 or cy2 <= cy1:  # entirely outside: replicate the nearest pixel
        cx1 = min(max(ix1, 0), w - 1)
        cy1 = min(max(iy1, 0), h - 1)
        cx2, cy2 = cx1 + 1, cy1 + 1
    patch = image[cy1:cy2, cx1:cx2]
    top, left = cy1 - iy1, cx1 - ix1
    bottom, right = iy2 - cy2, ix2 - cx2
    if top or left or bottom or right:
        patch = cv2.copyMakeBorder(patch, max(top, 0), max(bottom, 0), max(left, 0), max(right, 0),
                                   cv2.BORDER_REPLICATE)
    return cv2.resize(patch, (int(size), int(size)), interpolation=cv2.INTER_LINEAR)


def preprocess(crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """``(N, 3, S, S)`` float32: BGR -> RGB, ``/255``, ImageNet mean/std (never mirrored)."""
    x = np.stack([np.asarray(c)[..., ::-1] for c in crops_bgr]).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32)


def decode_bins(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Degrees and certainty of ``(N, 90)`` logits.

    ``deg = sum(softmax * arange(90)) * 4 - 180``; certainty
    ``1 - entropy / log(90)`` in [0, 1] (1 = one-hot, 0 = uniform).
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1, N_BINS)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    deg = (p * np.arange(N_BINS)).sum(axis=1) * BIN_DEG - OFFSET_DEG
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1)
    conf = np.clip(1.0 - ent / math.log(N_BINS), 0.0, 1.0)
    return deg, conf


def angles_to_direction(a_rad: Any, b_rad: Any) -> np.ndarray:
    """``(N, 3)`` unit camera-frame directions ``(-cos b sin a, -sin b, -cos b cos a)``."""
    a = np.asarray(a_rad, dtype=np.float64).reshape(-1)
    b = np.asarray(b_rad, dtype=np.float64).reshape(-1)
    d = np.stack([-np.cos(b) * np.sin(a), -np.sin(b), -np.cos(b) * np.cos(a)], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def head_ray_rotation(u: float, v: float, K: np.ndarray) -> np.ndarray:
    """Minimal rotation (3x3 float64) taking the optical axis ``(0, 0, 1)`` to the ray through ``(u, v)``."""
    r = np.linalg.solve(np.asarray(K, dtype=np.float64), np.array([u, v, 1.0]))
    r /= np.linalg.norm(r)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, r)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(r @ z, -1.0, 1.0))
    if s < 1e-12:
        return np.eye(3)
    rvec = axis / s * math.atan2(s, c)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return np.asarray(R, dtype=np.float64)


# --------------------------------------------------------------------- weights
def _l2cs_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    return (cfg.get("gaze") or {}).get("l2cs") or {}


def _download(url: str, target: Path, sha256: Optional[str] = None, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``target`` (via ``<target>.part``), checking ``sha256`` when given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "vfoa-weights-downloader"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                fh.write(chunk)
        if sha256 and digest.hexdigest().lower() != str(sha256).lower():
            raise ValueError(f"sha256 mismatch: expected {sha256}, got {digest.hexdigest()}")
        os.replace(part, target)
    finally:
        if part.exists():
            part.unlink()
    return target


def resolve_l2cs_weights(cfg: Mapping[str, Any]) -> Path:
    """Path of the ONNX weights of ``gaze.l2cs`` (see the module docstring for the order)."""
    l2 = _l2cs_section(cfg)
    model = str(l2.get("model") or "mobileone_s0")
    entry = WEIGHTS.get(model)
    path = l2.get("path")
    if path:
        p = Path(str(path)).expanduser()
        if p.is_file():
            return p
        filename = p.name
    elif entry is not None:
        filename = entry[0]
    else:
        filename = f"{model}_gaze.onnx"
    target = weights_dir(cfg) / filename
    if target.is_file():
        return target
    url = l2.get("url") or (entry[1] if entry else None)
    sha = l2.get("sha256") or (entry[2] if entry and url == entry[1] else None)
    models = cfg.get("models") or {}
    if not (models.get("allow_download", True) and l2.get("download", True)):
        raise BackendUnavailable(
            f"L2CS gaze weights {filename!r} not found (looked for {path or '-'} and {target}); "
            f"downloads are disabled (models.allow_download / gaze.l2cs.download). {_HINT}."
        )
    if not url:
        raise BackendUnavailable(
            f"L2CS gaze weights {filename!r} not found at {target} and no download URL is known "
            f"for gaze.l2cs.model={model!r}; set gaze.l2cs.url or gaze.l2cs.path. {_HINT}."
        )
    log.info("downloading L2CS gaze weights %s -> %s (Gaze360-trained: research/non-commercial use)",
             url, target)
    try:
        return _download(str(url), target, sha)
    except BackendUnavailable:
        raise
    except Exception as exc:
        raise BackendUnavailable(
            f"Cannot download L2CS gaze weights from {url} to {target}: {exc!r}. "
            f"Check the network, or {_HINT} (keys gaze.l2cs.path/url/download)."
        ) from exc


def _import_ort() -> Any:
    """Import onnxruntime lazily (monkeypatched in tests)."""
    import onnxruntime as ort  # noqa: PLC0415 (lazy optional dependency)

    return ort


def _providers(cfg: Mapping[str, Any], ort: Any, configured: Any) -> list[str]:
    if configured:
        return [str(p) for p in configured]
    device = str((cfg.get("models") or {}).get("device", "auto") or "auto")
    try:
        avail = list(ort.get_available_providers())
    except Exception:  # pragma: no cover - defensive
        avail = []
    if device != "cpu" and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# --------------------------------------------------------------------- backend
class BinnedGaze360Backend(GazeBackend):
    """L2CS-style binned gaze (MobileGaze / L2CS-Net weights) on ONNX Runtime.

    Config ``gaze.l2cs``: ``model``, ``path``, ``url``, ``sha256``,
    ``download``, ``input_size``, ``kpt_thr``, ``crop_margin``,
    ``perspective_correction``, ``providers``. ``session`` may be injected
    (an object with ``get_inputs``/``get_outputs``/``run``); then neither
    onnxruntime nor weights are needed.
    """

    name = "l2cs"

    def __init__(self, cfg: Mapping[str, Any], session: Any = None) -> None:
        super().__init__(cfg)
        self._session = session
        self._session_injected = session is not None
        self._input_name = "input"
        self._dynamic = False
        self._size = 448
        self.stats = {"gated_out": 0, "estimated": 0, "runs": 0}

    @property
    def l2cs(self) -> Mapping[str, Any]:
        return self.section.get("l2cs") or {}

    def _load(self) -> None:
        l2 = self.l2cs
        if self._session is None:
            try:
                ort = _import_ort()
            except ImportError as exc:
                raise BackendUnavailable(
                    "The l2cs gaze backend needs the 'onnxruntime' package: pip install onnxruntime "
                    "(requirements-fast.txt) or set gaze.backend=head_proxy"
                ) from exc
            path = resolve_l2cs_weights(self.cfg)
            options = ort.SessionOptions()
            # Disable intra-op spin-waiting: spinning ONNX Runtime threads contend
            # with torch's OpenMP threads (YOLO persons/segmentation stages); on a
            # shared CPU this measured about 2x on the persons stage.
            add_entry = getattr(options, "add_session_config_entry", None)
            if callable(add_entry):
                try:
                    add_entry("session.intra_op.allow_spinning", "0")
                except Exception as exc:  # pragma: no cover - defensive
                    log.debug("l2cs gaze: cannot disable ORT spinning: %r", exc)
            threads = (self.cfg.get("runtime") or {}).get("num_threads")
            if threads:
                options.intra_op_num_threads = int(threads)
            providers = _providers(self.cfg, ort, l2.get("providers"))
            try:
                self._session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            except Exception as exc:
                raise BackendUnavailable(
                    f"Cannot load the L2CS gaze ONNX model {path}: {exc!r} (gaze.l2cs.path; "
                    f"delete the file to download it again)"
                ) from exc
            log.info("l2cs gaze: loaded %s with %s", path, providers)
        sess = self._session
        inp = sess.get_inputs()[0]
        self._input_name = inp.name
        shape = list(getattr(inp, "shape", None) or [None, 3, None, None])
        self._dynamic = not isinstance(shape[0], int)
        size = int(l2.get("input_size", 448) or 448)
        model_size = shape[2] if len(shape) > 2 else None
        if isinstance(model_size, int) and model_size > 0 and model_size != size:
            log.warning("l2cs gaze: gaze.l2cs.input_size=%d but the model expects %d; using %d",
                        size, model_size, model_size)
            size = model_size
        self._size = size
        outputs = sess.get_outputs()
        if len(outputs) != 2:
            raise BackendUnavailable(
                f"The L2CS gaze model must have exactly 2 outputs (two 90-bin heads), got {len(outputs)}"
            )

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sess = self._session
        if self._dynamic:
            bs = max(int(self.section.get("batch_size", 16) or 16), 1)
            chunks = [x[i:i + bs] for i in range(0, len(x), bs)]
        else:
            chunks = [x[i:i + 1] for i in range(len(x))]
        a_parts, b_parts = [], []
        for chunk in chunks:
            outs = sess.run(None, {self._input_name: np.ascontiguousarray(chunk)})
            self.stats["runs"] += 1
            a_parts.append(np.asarray(outs[0], dtype=np.float64).reshape(len(chunk), -1))
            b_parts.append(np.asarray(outs[1], dtype=np.float64).reshape(len(chunk), -1))
        return np.concatenate(a_parts), np.concatenate(b_parts)

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        self.load()
        l2 = self.l2cs
        thr = float(l2.get("kpt_thr", 0.5))
        margin = float(l2.get("crop_margin", 1.1))
        items: list[tuple[int, np.ndarray, float]] = []
        crops: list[np.ndarray] = []
        for p in persons:
            det = p.detection
            vis = face_visible(det.keypoints, thr)
            if vis <= 0.0:
                self.stats["gated_out"] += 1
                continue
            box = face_crop_box(det.keypoints, det.head_box, thr, margin)
            if box is None:
                self.stats["gated_out"] += 1
                continue
            items.append((int(p.track_id), box, vis))
            crops.append(crop_square(frame.image, box, self._size))
        if not items:
            return {}
        la, lb = self._run(preprocess(crops))
        deg_a, conf_a = decode_bins(la)
        deg_b, conf_b = decode_bins(lb)
        dirs = angles_to_direction(np.radians(deg_a), np.radians(deg_b))
        K = None
        if l2.get("perspective_correction"):
            from vfoa.backends.depth_simple import frame_intrinsics  # noqa: PLC0415

            K = frame_intrinsics(self.cfg, frame).K  # frame.intrinsics first, rescaled to the image
        out: dict[int, DirectionEstimate] = {}
        for j, (tid, box, vis) in enumerate(items):
            d = dirs[j]
            if K is not None:
                u, v = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
                d = head_ray_rotation(u, v, K) @ d
                d = d / np.linalg.norm(d)
            conf = float(np.clip(math.sqrt(conf_a[j] * conf_b[j]) * vis, 0.0, 1.0))
            out[tid] = DirectionEstimate(
                direction=d, confidence=conf,
                euler_deg=np.array([deg_a[j], deg_b[j], 0.0]), frame_index=frame.index,
            )
        self.stats["estimated"] += len(out)
        return out

    def close(self) -> None:
        if not self._session_injected:
            self._session = None
            self._loaded = False
