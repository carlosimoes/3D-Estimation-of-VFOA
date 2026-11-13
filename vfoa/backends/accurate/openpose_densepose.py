"""Thesis person backend: DensePose head boxes + OpenPose keypoints.

DensePose (detectron2) gives person boxes and per-pixel body-part labels; the
head box is the extent of the head parts (IUV labels 23/24) expanded by 15%.
OpenPose (rtpose VGG19) gives skeletons that are converted to COCO-17 pixel
keypoints and matched one-to-one to the head boxes (Refs #18).

Both models are built ONCE in ``load()`` on ``models.device`` (Refs #32, #13).
The BGR frame is handed to detectron2 in ``cfg.INPUT.FORMAT`` order (Refs #9);
OpenPose's rtpose preprocessing works on BGR as in the original repository.
torch, detectron2, densepose and the OpenPose ``lib`` package are imported
lazily; pure pre/post-processing functions are testable without them.

Coordinates: pixels of the full frame, x = column, y = row.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from vfoa.backends.accurate import _thirdparty
from vfoa.backends.accurate.panoptic_fpn import (
    Detectron2Runtime,
    build_detectron2_runtime,
    run_detectron2,
)
from vfoa.backends.base import BackendUnavailable, PersonBackend, resolve_device
from vfoa.types import Frame, PersonDetection

__all__ = [
    "OPENPOSE_TO_COCO17",
    "LOWER_BODY_COCO",
    "HEAD_COCO",
    "HEAD_PART_IDS",
    "head_box_from_parts",
    "densepose_part_labels",
    "openpose_preprocess",
    "openpose_humans_to_coco17",
    "match_skeletons_to_head_boxes",
    "OpenPoseRuntime",
    "build_openpose_runtime",
    "build_densepose_runtime",
    "OpenPoseDensePosePersons",
]

logger = logging.getLogger(__name__)

#: Position = COCO-17 index, value = OpenPose-18 part index (the neck, part 1, is dropped).
#: OpenPose-18: 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder,
#: 6 l_elbow, 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee,
#: 13 l_ankle, 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear.
OPENPOSE_TO_COCO17: tuple[int, ...] = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
#: COCO-17 knees and ankles.
LOWER_BODY_COCO: tuple[int, ...] = (13, 14, 15, 16)
#: COCO-17 nose, eyes and ears.
HEAD_COCO: tuple[int, ...] = (0, 1, 2, 3, 4)
#: DensePose IUV part labels of the head.
HEAD_PART_IDS: tuple[int, ...] = (23, 24)

_INELIGIBLE = 1e3


# --------------------------------------------------------------------------- DensePose


def head_box_from_parts(
    labels: np.ndarray,
    origin_xy: Sequence[float],
    image_size: Sequence[int],
    expand: float = 0.15,
) -> Optional[np.ndarray]:
    """Head box (float32 [x1, y1, x2, y2] frame pixels) from a DensePose part-label crop.

    ``labels`` is the (h, w) part-label image of one instance whose top-left
    corner is at ``origin_xy`` (frame pixels); ``image_size`` = (W, H). The head
    extent (labels >= 23, inclusive first/last column/row as in the thesis) is
    expanded by ``expand`` of its size on each side and clipped to the image.
    Returns None without head pixels or when the clipped box is narrower or
    shorter than 1 pixel.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2 or lab.size == 0:
        return None
    mask = lab >= min(HEAD_PART_IDS)
    if not mask.any():
        return None
    xs = np.flatnonzero(mask.any(axis=0))
    ys = np.flatnonzero(mask.any(axis=1))
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])
    dw = (x1 - x0) * expand
    dh = (y1 - y0) * expand
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    width, height = float(image_size[0]), float(image_size[1])
    box = np.array([x0 - dw + ox, y0 - dh + oy, x1 + dw + ox, y1 + dh + oy], dtype=np.float64)
    box[[0, 2]] = np.clip(box[[0, 2]], 0.0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0.0, height)
    if box[2] - box[0] < 1.0 or box[3] - box[1] < 1.0:
        return None
    return box.astype(np.float32)


def densepose_part_labels(dp: Any, j: int, height: int, width: int) -> Optional[np.ndarray]:
    """Part-label image (height, width) int64 of instance ``j`` of a DensePose output.

    Uses ``dp.S``/``dp.I`` (legacy chart output) or ``dp.coarse_segm``/``dp.fine_segm``;
    both are bilinearly resized to the box size, the coarse segmentation masks
    the fine part labels (0 = background). None for an empty box.
    """
    if int(height) <= 0 or int(width) <= 0:
        return None
    import torch.nn.functional as F  # noqa: PLC0415 (lazy heavy import)

    S = getattr(dp, "S", None)
    if S is None:
        S = getattr(dp, "coarse_segm")
    I = getattr(dp, "I", None)  # noqa: E741
    if I is None:
        I = getattr(dp, "fine_segm")  # noqa: E741
    size = (int(height), int(width))
    s = F.interpolate(S[[j]], size, mode="bilinear", align_corners=False).argmax(dim=1)
    i = F.interpolate(I[[j]], size, mode="bilinear", align_corners=False).argmax(dim=1)
    return (i * (s > 0).long()).squeeze(0).detach().cpu().numpy()


@dataclass
class _Instance:
    head_box: Optional[np.ndarray]
    body_box: np.ndarray
    score: float


# --------------------------------------------------------------------------- OpenPose


def openpose_preprocess(
    image_bgr: np.ndarray, size: int = 368, factor: int = 8
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """rtpose preprocessing of a BGR frame.

    The shorter side is scaled to ``size``, the image zero-padded at the
    bottom/right to multiples of ``factor`` and normalised as ``x / 256 - 0.5``
    (padding becomes -0.5); channels stay BGR. Returns (CHW float32, scale,
    (net_h, net_w)).
    """
    h, w = image_bgr.shape[:2]
    scale = float(size) / float(min(h, w))
    resized = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    rh, rw = resized.shape[:2]
    net_h = int(math.ceil(rh / factor) * factor)
    net_w = int(math.ceil(rw / factor) * factor)
    padded = np.zeros((net_h, net_w, 3), dtype=np.float32)
    padded[:rh, :rw] = resized
    chw = (padded / 256.0 - 0.5).transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32), scale, (net_h, net_w)


def openpose_humans_to_coco17(
    humans: Sequence[Any],
    net_size: Sequence[int],
    scale: float = 1.0,
    upper_body_only: bool = True,
    image_size: Optional[Sequence[int]] = None,
) -> list[np.ndarray]:
    """Convert OpenPose humans to (17, 3) float32 COCO-17 keypoints in frame pixels.

    ``human.body_parts`` maps OpenPose-18 part index -> part with ``x``/``y``
    normalised by the PADDED network input ``net_size`` = (net_w, net_h) and a
    ``score``. Pixels: x = part.x * net_w / scale, y = part.y * net_h / scale.
    Missing parts (and parts outside ``image_size`` = (W, H)) have confidence 0;
    ``upper_body_only`` zeroes knees and ankles. Humans with no part left are
    skipped.
    """
    net_w, net_h = float(net_size[0]), float(net_size[1])
    inv = 1.0 / float(scale)
    coco_of_op = {op: coco for coco, op in enumerate(OPENPOSE_TO_COCO17)}
    out: list[np.ndarray] = []
    for human in humans or ():
        parts = getattr(human, "body_parts", None)
        if parts is None and isinstance(human, Mapping):
            parts = human
        kp = np.zeros((17, 3), dtype=np.float32)
        for op_idx, part in (parts or {}).items():
            coco = coco_of_op.get(int(op_idx))
            if coco is None:
                continue
            x = float(part.x) * net_w * inv
            y = float(part.y) * net_h * inv
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if image_size is not None and not (0.0 <= x <= image_size[0] and 0.0 <= y <= image_size[1]):
                continue
            conf = float(np.clip(float(getattr(part, "score", 1.0)), 1e-3, 1.0))
            kp[coco] = (x, y, conf)
        if upper_body_only:
            kp[list(LOWER_BODY_COCO)] = 0.0
        if np.any(kp[:, 2] > 0):
            out.append(kp)
    return out


def match_skeletons_to_head_boxes(
    skeletons: Sequence[Optional[np.ndarray]],
    head_boxes: Sequence[Optional[np.ndarray]],
    min_inside: float = 0.5,
) -> dict[int, int]:
    """One-to-one assignment of skeletons to head boxes -> {box index: skeleton index}.

    A pair is eligible when the skeleton has head keypoints (COCO 0..4 with
    conf > 0), the box exists and at least ``min_inside`` of those keypoints lie
    inside the box (inclusive). Cost = 1 - inside fraction + 0.01 * normalised
    distance of the head-keypoint centroid to the box centre; the Hungarian
    assignment keeps only eligible pairs, so a skeleton without head keypoints
    matches nothing and no box is shared.
    """
    n_s, n_b = len(skeletons), len(head_boxes)
    if n_s == 0 or n_b == 0:
        return {}
    cost = np.full((n_s, n_b), _INELIGIBLE, dtype=np.float64)
    for s, kp in enumerate(skeletons):
        if kp is None:
            continue
        head = np.asarray(kp, dtype=np.float64)[list(HEAD_COCO)]
        head = head[head[:, 2] > 0, :2]
        if len(head) == 0:
            continue
        centroid = head.mean(axis=0)
        for b, box in enumerate(head_boxes):
            if box is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in np.asarray(box).reshape(4))
            inside = (head[:, 0] >= x1) & (head[:, 0] <= x2) & (head[:, 1] >= y1) & (head[:, 1] <= y2)
            frac = float(inside.mean())
            if frac < min_inside or frac <= 0.0:
                continue
            diag = max(math.hypot(x2 - x1, y2 - y1), 1e-6)
            dist = math.hypot(centroid[0] - (x1 + x2) / 2.0, centroid[1] - (y1 + y2) / 2.0)
            cost[s, b] = 1.0 - frac + 0.01 * min(dist / diag, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return {int(b): int(s) for s, b in zip(rows, cols) if cost[s, b] < _INELIGIBLE}


@dataclass
class OpenPoseRuntime:
    """A loaded rtpose model. ``paf_to_pose(heatmaps HWC19, pafs HWC38) -> humans``."""

    model: Any
    paf_to_pose: Callable[[np.ndarray, np.ndarray], Sequence[Any]]
    image_size: int = 368
    downsample: int = 8
    device: str = "cpu"


def _openpose_unavailable(exc: BaseException) -> BackendUnavailable:
    return BackendUnavailable(
        f"The OpenPose repository (Pose_Estimation, package 'lib') is not importable ({exc}); "
        f"set the config key models.third_party_root or the environment variable "
        f"{_thirdparty.ENV_VAR} to the thesis checkout containing Pose_Estimation "
        f"(weights: persons.openpose_weights)"
    )


def build_openpose_runtime(cfg: Mapping[str, Any]) -> OpenPoseRuntime:
    """Build the rtpose VGG19 model ONCE on ``models.device`` (single device, no model wrappers)."""
    persons = cfg["persons"]
    if _thirdparty.third_party_root(cfg) is not None:
        _thirdparty.add_sys_paths(cfg, "Pose_Estimation")
    try:
        import torch  # noqa: PLC0415
        from lib.config.default import _C  # noqa: PLC0415
        from lib.network.rtpose_vgg import get_model  # noqa: PLC0415
        from lib.utils.paf_to_pose import paf_to_pose_cpp  # noqa: PLC0415
    except ImportError as exc:
        raise _openpose_unavailable(exc) from exc

    def _file(key: str) -> Any:
        value = persons.get(key)
        if not value:
            raise BackendUnavailable(f"No file configured for persons.{key}")
        path = _thirdparty.resolve_path(cfg, value)
        if not path.is_file():
            raise BackendUnavailable(
                f"File {str(path)!r} (config key persons.{key}; relative paths resolve against "
                f"models.third_party_root / {_thirdparty.ENV_VAR}) does not exist"
            )
        return path

    cfg_path = _file("openpose_cfg")
    weights_path = _file("openpose_weights")
    op = _C.clone()
    op.merge_from_file(str(cfg_path))
    op.freeze()
    device = resolve_device(cfg)
    model = get_model("vgg19")
    state = torch.load(str(weights_path), map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model = model.to(device).float().eval()
    logger.info("Loaded OpenPose rtpose_vgg19 from %s on %s", weights_path, device)

    def _paf_to_pose(heatmaps: np.ndarray, pafs: np.ndarray) -> Sequence[Any]:
        return paf_to_pose_cpp(heatmaps, pafs, op)

    return OpenPoseRuntime(
        model=model,
        paf_to_pose=_paf_to_pose,
        image_size=int(op.DATASET.IMAGE_SIZE),
        downsample=int(op.MODEL.DOWNSAMPLE),
        device=device,
    )


def build_densepose_runtime(cfg: Mapping[str, Any]) -> Detectron2Runtime:
    """DensePose R-CNN runtime from ``persons.densepose_cfg`` / ``densepose_weights``."""
    persons = cfg["persons"]
    thr = float(persons.get("densepose_thr", 0.85))

    def _add_config(d2: Any) -> None:
        try:
            from densepose.config import add_densepose_config  # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailable(
                f"DensePose (projects/DensePose, package 'densepose') is not importable ({exc}); "
                f"set models.third_party_root or {_thirdparty.ENV_VAR} to the detectron2 "
                f"checkout holding projects/DensePose"
            ) from exc
        add_densepose_config(d2)

    def _configure(d2: Any) -> None:
        d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thr

    return build_detectron2_runtime(
        cfg,
        persons.get("densepose_cfg"),
        persons.get("densepose_weights"),
        keys=("persons.densepose_cfg", "persons.densepose_weights"),
        configure=_configure,
        add_config=_add_config,
        extra_sys_paths=("projects/DensePose",),
    )


# --------------------------------------------------------------------------- backend


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class OpenPoseDensePosePersons(PersonBackend):
    """Thesis person detector: DensePose head boxes + OpenPose COCO-17 keypoints.

    Factories ``densepose_factory(cfg) -> Detectron2Runtime`` and
    ``openpose_factory(cfg) -> OpenPoseRuntime`` are injectable (tests). Each
    model is built once. Optional config keys: ``persons.upper_body_only``
    (True), ``persons.match_min_inside`` (0.5), ``persons.keep_headless`` (False).
    """

    name = "openpose_densepose"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        densepose_factory: Optional[Callable[[Mapping[str, Any]], Detectron2Runtime]] = None,
        openpose_factory: Optional[Callable[[Mapping[str, Any]], OpenPoseRuntime]] = None,
    ) -> None:
        super().__init__(cfg)
        self._densepose_factory = densepose_factory or build_densepose_runtime
        self._openpose_factory = openpose_factory or build_openpose_runtime
        self.densepose_rt: Optional[Detectron2Runtime] = None
        self.openpose_rt: Optional[OpenPoseRuntime] = None

    # -- runtimes -----------------------------------------------------------
    def _densepose(self) -> Detectron2Runtime:
        if self.densepose_rt is None:
            self.densepose_rt = self._densepose_factory(self.cfg)
        return self.densepose_rt

    def _openpose(self) -> OpenPoseRuntime:
        if self.openpose_rt is None:
            self.openpose_rt = self._openpose_factory(self.cfg)
        return self.openpose_rt

    def _load(self) -> None:
        self._densepose()
        if self.section.get("match_keypoints", True):
            self._openpose()

    # -- stages -------------------------------------------------------------
    def _instances(self, image: np.ndarray) -> list[_Instance]:
        rt = self._densepose()
        out = run_detectron2(rt, image)
        instances = out["instances"]
        boxes = _to_numpy(instances.pred_boxes.tensor).reshape(-1, 4).astype(np.float64)
        scores_raw = getattr(instances, "scores", None)
        scores = _to_numpy(scores_raw).reshape(-1) if scores_raw is not None else np.ones(len(boxes))
        dp = getattr(instances, "pred_densepose", None)
        H, W = image.shape[:2]
        result: list[_Instance] = []
        for j, box in enumerate(boxes):
            x0 = int(np.clip(round(box[0]), 0, W))
            y0 = int(np.clip(round(box[1]), 0, H))
            x1 = int(np.clip(round(box[2]), 0, W))
            y1 = int(np.clip(round(box[3]), 0, H))
            head = None
            if dp is not None:
                labels = densepose_part_labels(dp, j, y1 - y0, x1 - x0)
                if labels is not None:
                    head = head_box_from_parts(labels, (x0, y0), (W, H))
            body = np.array(
                [np.clip(box[0], 0, W), np.clip(box[1], 0, H), np.clip(box[2], 0, W), np.clip(box[3], 0, H)],
                dtype=np.float32,
            )
            score = float(scores[j]) if j < len(scores) else 1.0
            result.append(_Instance(head, body, score))
        return result

    def head_instances(self, image: np.ndarray) -> list[tuple[Optional[np.ndarray], np.ndarray, float]]:
        """DensePose instances of a BGR image: (head_box | None, body_box float32, score)."""
        return [(i.head_box, i.body_box, i.score) for i in self._instances(image)]

    def skeletons(self, image: np.ndarray) -> list[np.ndarray]:
        """OpenPose skeletons of a BGR image as (17, 3) COCO-17 pixel keypoints."""
        import torch  # noqa: PLC0415

        rt = self._openpose()
        chw, scale, (net_h, net_w) = openpose_preprocess(image, rt.image_size, rt.downsample)
        x = torch.from_numpy(chw)
        with torch.no_grad():
            (a, b), _ = rt.model(x[None].to(rt.device))
        # rtpose returns (PAF 38 channels, heatmap 19 channels); pick by channel count.
        if a.shape[1] == 19 and b.shape[1] != 19:
            heat, paf = a, b
        else:
            heat, paf = b, a
        heat_np = heat[0].detach().permute(1, 2, 0).cpu().numpy()
        paf_np = paf[0].detach().permute(1, 2, 0).cpu().numpy()
        humans = rt.paf_to_pose(heat_np, paf_np)
        H, W = image.shape[:2]
        return openpose_humans_to_coco17(
            humans,
            (net_w, net_h),
            scale=scale,
            upper_body_only=bool(self.section.get("upper_body_only", True)),
            image_size=(W, H),
        )

    def detect(self, frame: Frame) -> list[PersonDetection]:
        """Untracked persons of ``frame`` in DensePose instance order.

        Instances without a head box are dropped unless ``persons.keep_headless``.
        Skeletons are matched one-to-one to head boxes; unmatched boxes keep
        ``keypoints=None`` and unmatched skeletons are dropped.
        """
        self.load()
        section = self.section
        image = frame.image
        instances = self._instances(image)
        if not section.get("keep_headless", False):
            instances = [i for i in instances if i.head_box is not None]
        if not instances:
            return []
        matches: dict[int, int] = {}
        skels: list[np.ndarray] = []
        if section.get("match_keypoints", True) and any(i.head_box is not None for i in instances):
            skels = self.skeletons(image)
            matches = match_skeletons_to_head_boxes(
                skels,
                [i.head_box for i in instances],
                min_inside=float(section.get("match_min_inside", 0.5)),
            )
        return [
            PersonDetection(
                head_box=inst.head_box,
                body_box=inst.body_box,
                keypoints=skels[matches[b]] if b in matches else None,
                score=inst.score,
            )
            for b, inst in enumerate(instances)
        ]

    def close(self) -> None:
        self.densepose_rt = None
        self.openpose_rt = None
        self._loaded = False
