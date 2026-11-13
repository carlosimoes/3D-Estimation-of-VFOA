"""Fast segmentation backend: YOLO11-seg (ultralytics, optional) -> instance label map.

Every detected instance becomes one segment (``is_thing=True``); overlapping
masks are resolved in favour of the higher score. YOLO returns masks at the
letterboxed network resolution; they are mapped back to the frame with
:func:`_unletterbox` once, on the combined label map.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from vfoa.backends.base import SegmentationBackend, resolve_device
from vfoa.types import Frame, SegmentInfo, Segmentation

from ._ultra import load_yolo, predict, to_numpy

__all__ = ["YoloSegSegmentation"]

log = logging.getLogger(__name__)


def _unletterbox(lab: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Map a label map at letterboxed network size ``(mh, mw)`` to the frame size ``(H, W)``.

    Crops the symmetric letterbox padding (ultralytics rounding) and resizes
    with nearest neighbour. Same shape -> returned as is.
    """
    H, W = int(shape[0]), int(shape[1])
    mh, mw = lab.shape[:2]
    if (mh, mw) == (H, W):
        return lab
    gain = min(mh / H, mw / W)
    pw = (mw - W * gain) / 2.0
    ph = (mh - H * gain) / 2.0
    top, left = int(round(ph - 0.1)), int(round(pw - 0.1))
    bottom, right = mh - int(round(ph + 0.1)), mw - int(round(pw + 0.1))
    crop = lab[top:bottom, left:right]
    if crop.size == 0:
        crop = lab
    return cv2.resize(np.ascontiguousarray(crop), (W, H), interpolation=cv2.INTER_NEAREST)


def _names_lookup(names: Any, cls: int) -> str:
    if isinstance(names, Mapping):
        value = names.get(cls, names.get(str(cls)))
        return str(value) if value is not None else str(cls)
    if isinstance(names, (list, tuple)) and 0 <= cls < len(names):
        return str(names[cls])
    return str(cls)


class YoloSegSegmentation(SegmentationBackend):
    """Instance segmentation with an ultralytics YOLO seg model.

    Config (section ``segmentation``): ``model`` (default ``yolo11n-seg.pt``),
    ``imgsz``, ``score_thr``. An injected ``model`` object (with ``predict``
    and optionally ``names``) bypasses ultralytics (tests).
    """

    name = "yolo_seg"

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
            self.cfg, sec.get("model", "yolo11n-seg.pt"), "segmentation.model",
            imgsz=int(sec.get("imgsz", 640)), device=self._device,
        )

    def segment(self, frame: Frame) -> Segmentation:
        """Label map (frame shape, int32, ids 1..N by decreasing score) and segments."""
        self.load()
        sec = self.section
        score_thr = float(sec.get("score_thr", 0.35))
        H, W = frame.image.shape[:2]
        empty = Segmentation(np.zeros((H, W), dtype=np.int32), [], frame_index=frame.index)

        result = predict(
            self._model, frame.image, device=self._device,
            imgsz=int(sec.get("imgsz", 640)), conf=score_thr,
        )
        if result is None:
            return empty
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None:
            return empty
        m = to_numpy(getattr(masks, "data", None))
        if m.ndim != 3 or m.shape[0] == 0:
            return empty
        m = m > 0.5
        n = m.shape[0]
        scores = to_numpy(getattr(boxes, "conf", None)).astype(np.float64).reshape(-1)
        classes = to_numpy(getattr(boxes, "cls", None)).reshape(-1)
        if scores.shape[0] != n:
            scores = np.ones(n, dtype=np.float64)
        if classes.shape[0] != n:
            classes = np.zeros(n, dtype=np.int64)

        keep = np.flatnonzero(np.isfinite(scores) & (scores >= score_thr))
        if keep.size == 0:
            return empty
        order = keep[np.argsort(-scores[keep], kind="stable")]  # best first

        small = np.zeros(m.shape[1:], dtype=np.int32)
        for rank in range(len(order) - 1, -1, -1):  # lowest score painted first
            small[m[order[rank]]] = rank + 1
        lab = _unletterbox(small, (H, W))

        present = np.zeros(len(order) + 1, dtype=bool)
        present[np.unique(lab)] = True
        present[0] = False
        lut = np.zeros(len(order) + 1, dtype=np.int32)
        names = getattr(result, "names", None) or getattr(self._model, "names", None) or {}
        segments: list[SegmentInfo] = []
        for rank, idx in enumerate(order):
            if not present[rank + 1]:
                continue
            new_id = len(segments) + 1
            lut[rank + 1] = new_id
            cls = int(classes[idx]) if np.isfinite(classes[idx]) else -1
            segments.append(SegmentInfo(new_id, _names_lookup(names, cls), True, float(scores[idx])))
        label_map = lut[lab].astype(np.int32, copy=False)
        return Segmentation(label_map, segments, frame_index=frame.index)
