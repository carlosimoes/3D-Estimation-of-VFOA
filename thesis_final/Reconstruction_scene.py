"""Legacy scene reconstruction + attention scoring, a thin wrapper over :mod:`vfoa`.

``Reconstruction(metadata, matrix_Q, dir)(frames, disparity, panoptic_results,
mapFramesOutputs)`` keeps the thesis call shape but builds the metric scene
with :class:`vfoa.scene.SceneBuilder` (no PLY round-trip, no open3d on the
default path) and scores attention with :func:`vfoa.attention.score_frame`.

* Coordinates: camera frame of the left camera (OpenCV: +X right, +Y down,
  +Z forward), metres. The legacy ``Q[3,2] = -1/B`` sign is tolerated: the
  depth is always ``Z = fx * B / d > 0``.
* ``"g"`` (gaze) is a camera-frame direction; ``"hp"`` is either a
  camera-frame unit direction or FSA-Net (yaw, pitch, roll) degrees.
* The CSV (``final_data_Attention.csv``, columns
  :data:`vfoa.io.outputs.CSV_COLUMNS`) is written through ONE file handle.
* The Open3D window is opened only with ``visualize=True``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import logging
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vfoa.attention import AttentionParams, best_targets, score_frame
from vfoa.config import load_config
from vfoa.geometry import disparity_to_depth, euler_to_forward
from vfoa.io.outputs import CsvSink
from vfoa.scene import SceneBuilder
from vfoa.types import (
    DepthFrame,
    DirectionEstimate,
    FrameResult,
    Intrinsics,
    Person,
    PersonDetection,
    SegmentInfo,
    Segmentation,
)

__all__ = ["Reconstruction", "intrinsics_from_Q", "q_for_frame", "depth_from_disparity",
           "segmentation_from_legacy", "persons_from_legacy", "draw_geometries"]

logger = logging.getLogger(__name__)

CSV_NAME = "final_data_Attention.csv"


def draw_geometries(pcds: list, index: int) -> None:
    """Open the Open3D window (delegates to ``Arrow_line.draw_geometries``; patchable)."""
    from Arrow_line import draw_geometries as _draw  # noqa: PLC0415

    _draw(pcds, index)


# ------------------------------------------------------------------ conversions
def intrinsics_from_Q(Q: Any, width: int, height: int) -> Intrinsics:
    """Intrinsics (pixels, baseline metres) encoded in a 4x4 reprojection matrix.

    ``fx = Q[2,3]``, ``r = Q[1,1] = fx/fy``, ``cx = -Q[0,3]``, ``cy = -Q[1,3]/r``,
    ``baseline = 1/|Q[3,2]|`` (the legacy negative ``Q[3,2]`` gives the same result).
    """
    Q = np.asarray(Q, dtype=np.float64)
    if Q.shape != (4, 4):
        raise ValueError(f"Q must be 4x4, got {Q.shape}")
    fx = float(Q[2, 3])
    r = float(Q[1, 1]) if Q[1, 1] != 0 else 1.0
    q32 = abs(float(Q[3, 2]))
    if not (fx > 0 and q32 > 0 and math.isfinite(fx) and math.isfinite(q32)):
        raise ValueError(f"Q does not encode a valid focal length/baseline: fx={fx}, Q[3,2]={Q[3, 2]}")
    return Intrinsics(
        fx=fx, fy=fx / r, cx=-float(Q[0, 3]), cy=-float(Q[1, 3]) / r,
        width=int(width), height=int(height), baseline=1.0 / q32,
    )


def q_for_frame(matrix_Q: Any, i: int) -> Any:
    """Q matrix of frame ``i``: ``matrix_Q`` itself when it is one 4x4 matrix, else ``matrix_Q[i]``."""
    if matrix_Q is None:
        return None
    if not isinstance(matrix_Q, Mapping):
        try:
            a = np.asarray(matrix_Q, dtype=np.float64)
        except (TypeError, ValueError):
            a = None
        if a is not None and a.shape == (4, 4):
            return a
    return matrix_Q[i]


def depth_from_disparity(disparity: Any, Q: Any) -> DepthFrame:
    """Metric :class:`DepthFrame` (``Z = fx*B/d`` metres, positive) of a disparity map (pixels).

    A :class:`DepthFrame` passes through unchanged. ``d <= 0`` is invalid.
    """
    if isinstance(disparity, DepthFrame):
        return disparity
    d = np.asarray(disparity, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"disparity must be HxW, got shape {d.shape}")
    if Q is None:
        raise ValueError("a disparity map needs its Q matrix")
    h, w = d.shape
    intr = intrinsics_from_Q(Q, w, h)
    depth, valid = disparity_to_depth(d, intr.fx, float(intr.baseline))
    return DepthFrame(depth=depth, valid=valid, intrinsics=intr, disparity=d)


def _to_numpy(x: Any) -> np.ndarray:
    """``tensor.cpu().numpy()`` duck-typed; arrays pass through."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.asarray(x)


def _segment_label(info: Mapping[str, Any], metadata: Any) -> tuple[str, bool]:
    is_thing = bool(info.get("isthing", info.get("is_thing", True)))
    if "label" in info and info["label"] is not None:
        return str(info["label"]), is_thing
    cat = int(info.get("category_id", 0))
    classes = None
    if metadata is not None:
        classes = getattr(metadata, "thing_classes" if is_thing else "stuff_classes", None)
        if classes is None and isinstance(metadata, Mapping):
            classes = metadata.get("thing_classes" if is_thing else "stuff_classes")
    if classes is not None and 0 <= cat < len(classes):
        return str(classes[cat]), is_thing
    return str(cat), is_thing


def segmentation_from_legacy(panoptic: Any, metadata: Any = None, frame_index: int = -1) -> Segmentation:
    """:class:`Segmentation` of a legacy ``(label_map, segments_info)`` panoptic result.

    A :class:`Segmentation` passes through. Labels come from
    ``metadata.thing_classes`` / ``metadata.stuff_classes`` indexed by
    ``category_id``; segment id 0 ("not identified") is skipped wherever it is.
    """
    if isinstance(panoptic, Segmentation):
        return panoptic
    label_map, infos = panoptic[0], panoptic[1]
    L = _to_numpy(label_map).astype(np.int32, copy=False)
    segments: list[SegmentInfo] = []
    for info in infos or []:
        if isinstance(info, SegmentInfo):
            if int(info.id) == 0:
                continue
            segments.append(info)
            continue
        sid = int(info["id"])
        if sid == 0:
            continue
        label, is_thing = _segment_label(info, metadata)
        segments.append(SegmentInfo(id=sid, label=label, is_thing=is_thing,
                                    score=float(info.get("score", 1.0) or 1.0)))
    return Segmentation(label_map=L, segments=segments, frame_index=int(frame_index))


def _unit(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else None


def _direction(value: Any, frame_index: int, euler: bool) -> Optional[DirectionEstimate]:
    """DirectionEstimate of a legacy cue (unit vector, or FSA-Net Euler degrees when ``euler``)."""
    if value is None:
        return None
    if isinstance(value, DirectionEstimate):
        return value
    a = np.asarray(value, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    if euler and abs(float(np.linalg.norm(a)) - 1.0) > 1e-3:
        d = euler_to_forward(a[0], a[1], a[2])
        return DirectionEstimate(direction=d, confidence=1.0, euler_deg=a.copy(), frame_index=frame_index)
    u = _unit(a)
    if u is None:
        return None
    return DirectionEstimate(direction=u, confidence=1.0, frame_index=frame_index)


def persons_from_legacy(entries: Optional[Mapping[Any, Any]], frame_index: int = -1) -> list[Person]:
    """Persons of one legacy ``mapFramesOutputs`` entry ``{id: {"b", "kp", "g", "hp"}}``.

    Missing keys mean a missing cue/box (never KeyError).
    """
    persons: list[Person] = []
    for pid, e in (entries or {}).items():
        e = e or {}
        box = e.get("b")
        box = None if box is None else np.asarray(box, dtype=np.float32).reshape(-1)
        if box is not None and (box.shape != (4,) or not np.isfinite(box).all()):
            box = None
        kp = e.get("kp")
        kp = None if kp is None else np.asarray(kp, dtype=np.float32)
        if kp is not None and kp.shape != (17, 3):
            kp = None
        persons.append(
            Person(
                track_id=int(pid),
                detection=PersonDetection(head_box=box, keypoints=kp),
                gaze=_direction(e.get("g"), frame_index, euler=False),
                head=_direction(e.get("hp"), frame_index, euler=True),
            )
        )
    return persons


# ------------------------------------------------------------------ reconstruction
class Reconstruction:
    """Legacy entry: per-frame 3D scene + attention values (see the module docstring).

    Parameters
    ----------
    Metadata_panoptic:
        detectron2 metadata (``thing_classes``/``stuff_classes``) of the panoptic labels.
    matrix_Q:
        One 4x4 Q matrix or a sequence of them (one per frame).
    dir:
        Output directory of ``final_data_Attention.csv``.
    visualize, visualize_frame:
        Open the Open3D window for frame ``visualize_frame`` (off by default).
    cfg:
        :class:`vfoa.config.Config`; None -> the ``thesis`` profile.
    fps:
        Frame rate used for the CSV timestamps (``t = i / fps`` seconds).
    """

    def __init__(self, Metadata_panoptic: Any, matrix_Q: Any, dir: Any, visualize: bool = False,
                 cfg: Optional[Mapping[str, Any]] = None, fps: float = 30.0, visualize_frame: int = 0) -> None:
        self.metadata = Metadata_panoptic
        self.matrix_Q = matrix_Q
        self.output_file = dir
        self.visualize = bool(visualize)
        self.visualize_frame = int(visualize_frame)
        self.cfg = cfg if cfg is not None else load_config(profile="thesis")
        self.fps = float(fps) if fps and float(fps) > 0 else 30.0
        self.scene = SceneBuilder(self.cfg)
        self.params = AttentionParams.from_config(self.cfg)
        self.results: list[FrameResult] = []

    def __call__(self, frames: Sequence[Any], disparity: Sequence[Any], panoptic_results: Sequence[Any],
                 mapFramesOutputs: Any) -> dict[int, dict]:
        """Attention values ``{frame: {person_id: {object_id: {...}}}}`` (frames are not modified)."""
        out: dict[int, dict] = {}
        self.results = []
        entries = list(mapFramesOutputs.values()) if mapFramesOutputs is not None else []
        csv_path = Path(self.output_file) / CSV_NAME
        sink = CsvSink(csv_path)
        try:
            for i in range(len(frames)):
                depth = depth_from_disparity(disparity[i], q_for_frame(self.matrix_Q, i))
                seg = segmentation_from_legacy(panoptic_results[i], self.metadata, i)
                persons = persons_from_legacy(entries[i] if i < len(entries) else {}, i)
                objects = self.scene.build_objects(seg, depth)
                self.scene.locate_eyes(persons, depth)
                self.scene.match_own_segments(persons, seg, objects)
                scores = score_frame(persons, objects, self.params, depth, seg)
                targets = best_targets(scores, self.params.min_score)
                for p in persons:
                    targets.setdefault(int(p.track_id), None)
                result = FrameResult(index=i, t=i / self.fps, persons=persons, objects=objects,
                                     scores=scores, targets=targets)
                self.results.append(result)
                sink.write(None, result)
                attention: dict[int, dict] = {int(p.track_id): {} for p in persons}
                for s in scores:
                    attention.setdefault(int(s.person_id), {})[int(s.object_id)] = {
                        "object": s.label,
                        "attentionG": None,
                        "total_Attention_G": s.score_gaze,
                        "attentionH": None,
                        "total_Attention_H": s.score_head,
                        "total_Attention": s.score_total,
                        "occluded": bool(s.occluded),
                    }
                out[i] = attention
                if self.visualize and i == self.visualize_frame:
                    self.visualization(result, i)
                logger.debug("frame %d: %d persons, %d objects", i, len(persons), len(objects))
        finally:
            sink.close()
        return out

    def visualization(self, result: FrameResult, index: int) -> None:
        """Open3D view of the objects (points + AABBs) and the gaze/head cones of ``result``."""
        try:
            import open3d as o3d  # noqa: PLC0415
            from Arrow_line import create_cone_arrow  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("visualisation needs open3d (%s)", exc)
            return
        params = self.params
        geoms: list = []
        for o in result.objects:
            pts = np.asarray(o.points, dtype=np.float64).reshape(-1, 3)
            if pts.size:
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
                geoms.append(pcd)
            box = o3d.geometry.AxisAlignedBoundingBox(
                np.asarray(o.aabb_min, dtype=np.float64), np.asarray(o.aabb_max, dtype=np.float64)
            )
            geoms.append(box)
        for p in result.persons:
            if p.eyes_3d is None:
                continue
            for est, kind, half in ((p.gaze, "gaze", params.half_angles_gaze_deg[0]),
                                    (p.head, "headpose", params.half_angles_head_deg[0])):
                if est is None:
                    continue
                d = np.asarray(est.direction, dtype=np.float64)
                # ring of radius R at distance 300 along d -> cone half-angle ``half``
                radius = 300.0 * math.tan(math.radians(float(half)))
                geoms.extend(create_cone_arrow(p.eyes_3d, d, None, radius, params.rays_n, kind))
        draw_geometries(geoms, index)
