"""Fast person backend: YOLO11-pose (ultralytics, optional) -> persons + keypoints + head box.

One network pass gives, per person, the body box, a detection score and the
17 COCO keypoints. The head box is derived from the face keypoints
(:func:`head_box_from_keypoints`). Tracking is NOT done here (no
``model.track``); see :mod:`vfoa.tracking`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vfoa.backends.base import PersonBackend, resolve_device
from vfoa.types import Frame, PersonDetection

from ._ultra import load_yolo, predict, to_numpy

__all__ = ["YoloPosePersons", "head_box_from_keypoints"]

log = logging.getLogger(__name__)

_L_EYE, _R_EYE = 1, 2
_L_SHOULDER, _R_SHOULDER = 5, 6


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float64).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, max(width - 1, 0))
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, max(height - 1, 0))
    return out


def head_box_from_keypoints(
    keypoints: Optional[np.ndarray],
    body_box: Optional[np.ndarray],
    image_size: Sequence[int],
    kpt_thr: float = 0.35,
    head_scale: float = 1.3,
) -> Optional[np.ndarray]:
    """Square head box ``[x1, y1, x2, y2]`` (float32 pixels) from COCO-17 face keypoints.

    * Face keypoints = nose, eyes, ears (rows 0..4) with confidence >= ``kpt_thr``.
    * Head width candidates: max pairwise distance of the face keypoints (>= 2
      of them), ``2.3 * eye distance`` (both eyes), ``0.4 * shoulder width``
      (both shoulders >= ``kpt_thr``). ``side = head_scale * max(candidates)``,
      at least half of the body-based size ``s_fb = min(0.45 * body_w,
      0.3 * body_h)``; centred on the mean face keypoint and enlarged to contain
      every face keypoint, then clipped to the image.
    * Fallback when no face keypoint is confident: a square of side ``s_fb``
      whose top edge is the body box top, horizontally centred on the body box.
    * ``None`` when neither gives a box of at least 1x1 pixel.

    ``image_size`` is ``(width, height)``.
    """
    width, height = int(image_size[0]), int(image_size[1])

    s_fb = 0.0
    body = None
    if body_box is not None:
        body = np.asarray(body_box, dtype=np.float64).reshape(-1)
        if body.shape == (4,) and np.isfinite(body).all():
            bw, bh = body[2] - body[0], body[3] - body[1]
            if bw > 0 and bh > 0:
                s_fb = float(min(0.45 * bw, 0.3 * bh))
        else:
            body = None

    if keypoints is not None:
        kp = np.asarray(keypoints, dtype=np.float64)
        if kp.ndim == 2 and kp.shape[0] >= 7 and kp.shape[1] >= 3:
            finite = np.isfinite(kp[:, :2]).all(axis=1)
            ok = finite & (kp[:, 2] >= kpt_thr)
            face_ok = ok[:5]
            pts = kp[:5][face_ok, :2]
            if len(pts) >= 1:
                cands = []
                if len(pts) >= 2:
                    cands.append(float(np.linalg.norm(pts[:, None] - pts[None], axis=-1).max()))
                if ok[_L_EYE] and ok[_R_EYE]:
                    cands.append(2.3 * float(np.linalg.norm(kp[_L_EYE, :2] - kp[_R_EYE, :2])))
                if ok[_L_SHOULDER] and ok[_R_SHOULDER]:
                    cands.append(0.4 * float(np.linalg.norm(kp[_L_SHOULDER, :2] - kp[_R_SHOULDER, :2])))
                side = head_scale * max(cands) if cands and max(cands) > 0 else s_fb
                side = max(side, 0.5 * s_fb)
                c = pts.mean(axis=0)
                half = side / 2.0
                box = np.array([c[0] - half, c[1] - half, c[0] + half, c[1] + half])
                box[:2] = np.minimum(box[:2], pts.min(axis=0))
                box[2:] = np.maximum(box[2:], pts.max(axis=0))
                box = _clip_box(box, width, height)
                if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0:
                    return box.astype(np.float32)

    # Documented fallback: top-centre square of the body box.
    if body is not None and s_fb > 0:
        cx = 0.5 * (body[0] + body[2])
        box = np.array([cx - s_fb / 2.0, body[1], cx + s_fb / 2.0, body[1] + s_fb])
        box = _clip_box(box, width, height)
        if box[2] - box[0] >= 1.0 and box[3] - box[1] >= 1.0:
            return box.astype(np.float32)
    return None


class YoloPosePersons(PersonBackend):
    """Persons, COCO-17 keypoints and head boxes from an ultralytics YOLO pose model.

    Config (section ``persons``): ``model`` (weights, default ``yolo11n-pose.pt``),
    ``imgsz``, ``score_thr``, ``kpt_thr``, ``head_scale``. ``model`` may be an
    injected object with a ``predict`` method (tests); then nothing is imported.
    """

    name = "yolo_pose"

    def __init__(self, cfg: Mapping[str, Any], model: Any = None) -> None:
        super().__init__(cfg)
        self._model = model
        self._device: Optional[str] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        sec = self.section
        self._device = resolve_device(self.cfg)
        self._model = load_yolo(
            self.cfg, sec.get("model", "yolo11n-pose.pt"), "persons.model",
            imgsz=int(sec.get("imgsz", 640)), device=self._device,
        )

    def detect(self, frame: Frame) -> list[PersonDetection]:
        """Detect persons in ``frame.image`` (uint8 BGR, passed to the model unchanged)."""
        self.load()
        sec = self.section
        score_thr = float(sec.get("score_thr", 0.4))
        kpt_thr = float(sec.get("kpt_thr", 0.35))
        head_scale = float(sec.get("head_scale", 1.3))
        height, width = frame.image.shape[:2]

        result = predict(
            self._model, frame.image, device=self._device,
            imgsz=int(sec.get("imgsz", 640)), conf=score_thr,
        )
        boxes = getattr(result, "boxes", None) if result is not None else None
        if boxes is None:
            return []
        xyxy = to_numpy(getattr(boxes, "xyxy", None)).astype(np.float64).reshape(-1, 4)
        if len(xyxy) == 0:
            return []
        conf = to_numpy(getattr(boxes, "conf", None)).astype(np.float64).reshape(-1)
        if conf.shape[0] != xyxy.shape[0]:
            conf = np.ones(len(xyxy), dtype=np.float64)

        kps = None
        kobj = getattr(result, "keypoints", None)
        if kobj is not None:
            data = to_numpy(getattr(kobj, "data", None)).astype(np.float32)
            if data.ndim == 3 and data.shape[0] == len(xyxy) and data.shape[1] == 17:
                if data.shape[2] == 3:
                    kps = data
                elif data.shape[2] == 2:
                    kps = np.concatenate([data, np.ones(data.shape[:2] + (1,), np.float32)], axis=2)

        out: list[PersonDetection] = []
        for i in range(len(xyxy)):
            score = float(conf[i])
            if not np.isfinite(score) or score < score_thr:
                continue
            if not np.isfinite(xyxy[i]).all():
                continue
            body = _clip_box(xyxy[i], width, height)
            if body[2] - body[0] <= 0 or body[3] - body[1] <= 0:
                continue
            kp = None
            if kps is not None:
                kp = kps[i].copy()
                bad = ~np.isfinite(kp).all(axis=1)
                kp[bad] = 0.0
                # ultralytics 8.3.0-8.3.140 zero the x,y of every keypoint with
                # confidence < 0.5 in Keypoints.__init__ but keep its confidence,
                # and requirements-fast.txt allows ultralytics>=8.3. With
                # persons.kpt_thr = 0.35 such a point would arrive as (0, 0, conf)
                # and be used as real (head boxes stretched to the image origin).
                # Mark it missing here so every consumer of the stored keypoints
                # (eye pixel, head pose, tracking) is protected.
                kp[(kp[:, 0] == 0) & (kp[:, 1] == 0), 2] = 0.0
            head = head_box_from_keypoints(kp, body, (width, height), kpt_thr, head_scale)
            out.append(PersonDetection(head_box=head, body_box=body, keypoints=kp, score=score))
        return out
