"""Panoptic FPN segmentation (thesis model, detectron2) and shared detectron2 helpers.

The detectron2 model is built ONCE in ``load()`` on the configured device
(``models.device``; no forced CUDA placement, no model wrappers, no multi-GPU worker
processes - Refs #13, #32). Frames are uint8 BGR; the image handed to the model
follows ``cfg.INPUT.FORMAT`` of the detectron2 config (flipped only when the
model wants RGB - Refs #9).

detectron2 and torch are imported lazily; this module imports cleanly without
them. Pre/post-processing is in pure functions testable without detectron2, and
the model runtime is injectable (``runtime_factory``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from vfoa.backends.accurate import _thirdparty
from vfoa.backends.base import BackendUnavailable, SegmentationBackend, resolve_device
from vfoa.types import Frame, Segmentation, SegmentInfo

__all__ = [
    "Detectron2Runtime",
    "to_model_format",
    "run_detectron2",
    "build_detectron2_runtime",
    "build_panoptic_runtime",
    "panoptic_to_segmentation",
    "PanopticFPNSegmentation",
]

logger = logging.getLogger(__name__)


@dataclass
class Detectron2Runtime:
    """A loaded detectron2 model plus what is needed to feed it (DefaultPredictor semantics).

    ``model`` maps ``list[{"image": CHW float tensor, "height", "width"}]`` to a
    list of output dicts. ``input_format`` is ``cfg.INPUT.FORMAT`` ("BGR"/"RGB").
    ``resize`` maps an HWC image (already in model channel order) to the resized
    model input, or is None.
    """

    model: Callable[[list], list]
    input_format: str = "BGR"
    resize: Optional[Callable[[np.ndarray], np.ndarray]] = None
    device: str = "cpu"
    thing_classes: Sequence[str] = ()
    stuff_classes: Sequence[str] = ()


def to_model_format(image_bgr: np.ndarray, input_format: str) -> np.ndarray:
    """Return the BGR frame in the channel order the model expects (never mutates it).

    "BGR" -> the same array; "RGB" -> a channel-flipped view; other -> ValueError.
    """
    fmt = str(input_format).upper()
    if fmt == "BGR":
        return image_bgr
    if fmt == "RGB":
        return image_bgr[:, :, ::-1]
    raise ValueError(f"Unsupported detectron2 INPUT.FORMAT {input_format!r} (expected 'BGR' or 'RGB')")


def run_detectron2(rt: Detectron2Runtime, image_bgr: np.ndarray) -> dict:
    """Run ``rt.model`` on one BGR frame like detectron2's DefaultPredictor.

    ``height``/``width`` in the input dict are the ORIGINAL frame size, so the
    model rescales its outputs to frame pixels. Returns the model's output dict.
    """
    import torch  # noqa: PLC0415 (lazy heavy import)

    h, w = image_bgr.shape[:2]
    img = to_model_format(image_bgr, rt.input_format)
    if rt.resize is not None:
        img = rt.resize(img)
    tensor = torch.as_tensor(np.ascontiguousarray(img.astype("float32").transpose(2, 0, 1)))
    with torch.no_grad():
        return rt.model([{"image": tensor, "height": int(h), "width": int(w)}])[0]


def _unavailable_detectron2(exc: BaseException) -> BackendUnavailable:
    return BackendUnavailable(
        f"detectron2 is not importable ({exc}); install detectron2 or point the config key "
        f"models.third_party_root (or the environment variable {_thirdparty.ENV_VAR}) to the "
        f"thesis third-party checkout holding detectron2"
    )


def _model_file(cfg: Mapping[str, Any], value: Any, key: str) -> Any:
    if not value:
        raise BackendUnavailable(f"No file configured for {key}")
    path = _thirdparty.resolve_path(cfg, value)
    if not path.is_file():
        raise BackendUnavailable(
            f"File {str(path)!r} (config key {key}; relative paths resolve against "
            f"models.third_party_root / {_thirdparty.ENV_VAR}) does not exist"
        )
    return path


def build_detectron2_runtime(
    cfg: Mapping[str, Any],
    config_file: Any,
    weights: Any,
    *,
    keys: tuple[str, str] = ("<config key>", "<weights key>"),
    configure: Optional[Callable[[Any], None]] = None,
    add_config: Optional[Callable[[Any], None]] = None,
    extra_sys_paths: Sequence[str] = (),
) -> Detectron2Runtime:
    """Build a detectron2 model ONCE on ``models.device`` and wrap it in a runtime.

    ``keys`` name the config keys of ``config_file``/``weights`` for error
    messages. ``add_config(d2cfg)`` extends the detectron2 config before merging
    the file (e.g. DensePose); ``configure(d2cfg)`` adjusts it after.
    Raises BackendUnavailable when detectron2 or a file is missing.
    """
    if _thirdparty.third_party_root(cfg) is not None:
        _thirdparty.add_sys_paths(cfg, "", *extra_sys_paths)
    try:
        from detectron2.checkpoint import DetectionCheckpointer  # noqa: PLC0415
        from detectron2.config import get_cfg  # noqa: PLC0415
        from detectron2.data import MetadataCatalog  # noqa: PLC0415
        from detectron2.data.transforms import ResizeShortestEdge  # noqa: PLC0415
        from detectron2.modeling import build_model  # noqa: PLC0415
    except ImportError as exc:
        raise _unavailable_detectron2(exc) from exc

    cfg_path = _model_file(cfg, config_file, keys[0])
    weights_path = _model_file(cfg, weights, keys[1])

    d2 = get_cfg()
    if add_config is not None:
        add_config(d2)
    d2.merge_from_file(str(cfg_path))
    d2.MODEL.WEIGHTS = str(weights_path)
    device = resolve_device(cfg)
    d2.MODEL.DEVICE = device
    if configure is not None:
        configure(d2)
    d2.freeze()

    # build_model places the model on MODEL.DEVICE (Refs #13: single device).
    model = build_model(d2)
    model.eval()
    DetectionCheckpointer(model).load(d2.MODEL.WEIGHTS)

    aug = ResizeShortestEdge([d2.INPUT.MIN_SIZE_TEST, d2.INPUT.MIN_SIZE_TEST], d2.INPUT.MAX_SIZE_TEST)

    def _resize(img: np.ndarray) -> np.ndarray:
        return aug.get_transform(img).apply_image(img)

    thing_classes: Sequence[str] = ()
    stuff_classes: Sequence[str] = ()
    try:
        meta = MetadataCatalog.get(d2.DATASETS.TEST[0])
        thing_classes = tuple(getattr(meta, "thing_classes", ()) or ())
        stuff_classes = tuple(getattr(meta, "stuff_classes", ()) or ())
    except Exception:  # metadata is optional (labels fall back to thing_<id>/stuff_<id>)
        logger.warning("detectron2 metadata unavailable; using numeric labels")
    logger.info("Loaded detectron2 model %s on %s", cfg_path, device)
    return Detectron2Runtime(
        model=model,
        input_format=str(d2.INPUT.FORMAT),
        resize=_resize,
        device=device,
        thing_classes=thing_classes,
        stuff_classes=stuff_classes,
    )


def build_panoptic_runtime(cfg: Mapping[str, Any]) -> Detectron2Runtime:
    """Panoptic FPN runtime from ``segmentation.panoptic_cfg`` / ``panoptic_weights``."""
    seg = cfg["segmentation"]
    thr = float(seg.get("panoptic_thr", 0.5))

    def _configure(d2: Any) -> None:
        d2.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = thr

    return build_detectron2_runtime(
        cfg,
        seg.get("panoptic_cfg"),
        seg.get("panoptic_weights"),
        keys=("segmentation.panoptic_cfg", "segmentation.panoptic_weights"),
        configure=_configure,
    )


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _class_name(classes: Sequence[str], cid: int, prefix: str) -> str:
    if 0 <= cid < len(classes):
        return str(classes[cid])
    return f"{prefix}_{cid}"


def panoptic_to_segmentation(
    panoptic_seg: Any,
    segments_info: Sequence[Mapping[str, Any]],
    thing_classes: Sequence[str],
    stuff_classes: Sequence[str],
    frame_index: int = -1,
    out_shape: Optional[tuple[int, int]] = None,
) -> Segmentation:
    """Convert detectron2 panoptic output to a :class:`Segmentation`.

    ``panoptic_seg`` is an (H, W) id map (tensor or ndarray); ``segments_info``
    lists ``{"id", "isthing", "category_id", "score"?}``. ``out_shape`` (H, W)
    resizes the map with nearest neighbour. Pixels whose id is not > 0, not
    listed or absent from the map become 0. Unknown categories get the labels
    ``thing_<id>`` / ``stuff_<id>``.
    """
    label_map = _to_numpy(panoptic_seg).astype(np.int32, copy=False)
    if label_map.ndim != 2:
        label_map = label_map.reshape(label_map.shape[-2:])
    if out_shape is not None and tuple(label_map.shape) != tuple(out_shape[:2]):
        h, w = int(out_shape[0]), int(out_shape[1])
        label_map = cv2.resize(label_map, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    present = set(np.unique(label_map).tolist())
    segments: list[SegmentInfo] = []
    keep: list[int] = []
    for info in segments_info or ():
        sid = int(info["id"])
        if sid <= 0 or sid not in present or sid in keep:
            continue
        is_thing = bool(info.get("isthing", True))
        cid = int(info.get("category_id", -1))
        label = _class_name(thing_classes, cid, "thing") if is_thing else _class_name(stuff_classes, cid, "stuff")
        segments.append(SegmentInfo(id=sid, label=label, is_thing=is_thing, score=float(info.get("score", 1.0))))
        keep.append(sid)
    out = np.where(np.isin(label_map, np.asarray(keep, dtype=np.int32)), label_map, 0).astype(np.int32)
    return Segmentation(label_map=np.ascontiguousarray(out), segments=segments, frame_index=int(frame_index))


class PanopticFPNSegmentation(SegmentationBackend):
    """Thesis Panoptic FPN segmentation (detectron2), model built once.

    ``runtime_factory(cfg) -> Detectron2Runtime`` is injectable (tests); the
    default builds the detectron2 model from ``segmentation.panoptic_cfg`` /
    ``panoptic_weights`` with ``panoptic_thr``.
    """

    name = "panoptic_fpn"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        runtime_factory: Optional[Callable[[Mapping[str, Any]], Detectron2Runtime]] = None,
    ) -> None:
        super().__init__(cfg)
        self._factory = runtime_factory or build_panoptic_runtime
        self.runtime: Optional[Detectron2Runtime] = None

    def _load(self) -> None:
        if self.runtime is None:
            self.runtime = self._factory(self.cfg)

    def segment(self, frame: Frame) -> Segmentation:
        """Panoptic segmentation of ``frame`` (label map at frame resolution)."""
        self.load()
        rt = self.runtime
        assert rt is not None
        out = run_detectron2(rt, frame.image)
        panoptic_seg, segments_info = out["panoptic_seg"]
        return panoptic_to_segmentation(
            panoptic_seg,
            segments_info,
            rt.thing_classes,
            rt.stuff_classes,
            frame_index=frame.index,
            out_shape=frame.image.shape[:2],
        )

    def close(self) -> None:
        self.runtime = None
        self._loaded = False
