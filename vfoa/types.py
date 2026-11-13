"""Shared data types of the VFOA pipeline (dataclasses, NumPy only).

Conventions (see the architecture contract, section 2):

* The only 3D frame is the OpenCV pinhole frame of the left camera:
  origin at the optical centre, +X right, +Y down, +Z forward, units metres.
  ``u = fx * X / Z + cx`` (column), ``v = fy * Y / Z + cy`` (row).
* Images are uint8 BGR, boxes are float32 ``[x1, y1, x2, y2]`` pixels,
  keypoints are float32 (17, 3) COCO-17 ``(x, y, confidence)``.
* Directions are float64 unit vectors in the camera frame, pointing from the
  eyes towards what is looked at.

Dataclasses holding arrays use ``eq=False`` (identity comparison) because the
generated ``__eq__`` would raise on array fields.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "COCO_KEYPOINTS",
    "Intrinsics",
    "Frame",
    "DepthFrame",
    "PersonDetection",
    "DirectionEstimate",
    "Person",
    "SegmentInfo",
    "Segmentation",
    "SceneObject",
    "AttentionScore",
    "FrameResult",
]

#: COCO-17 keypoint names, in the order of the (17, 3) keypoint arrays.
COCO_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
)


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of the (left) camera.

    ``fx, fy`` focal lengths and ``cx, cy`` principal point in pixels;
    ``width, height`` image size in pixels; ``baseline`` stereo baseline in
    metres (``None`` for a monocular camera).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    baseline: Optional[float] = None

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("width", "height"):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.baseline is not None:
            object.__setattr__(self, "baseline", float(self.baseline))

    @property
    def K(self) -> np.ndarray:
        """3x3 float64 camera matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def Q(self) -> np.ndarray:
        """4x4 float64 disparity-to-depth matrix for ``cv2.reprojectImageTo3D``.

        With disparity ``d = fx * B / Z`` (pixels, positive), the reprojected
        points are camera-frame metres with POSITIVE ``Z``. Requires a baseline.
        """
        if self.baseline is None or not self.baseline > 0:
            raise ValueError(f"Intrinsics.Q() needs a positive baseline, got {self.baseline!r}")
        r = self.fx / self.fy
        return np.array(
            [
                [1.0, 0.0, 0.0, -self.cx],
                [0.0, r, 0.0, -self.cy * r],
                [0.0, 0.0, 0.0, self.fx],
                [0.0, 0.0, 1.0 / self.baseline, 0.0],
            ],
            dtype=np.float64,
        )

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Intrinsics":
        """Intrinsics of the image resized by ``(sx, sy)`` (pixel-centre convention).

        ``cx' = (cx + 0.5) * sx - 0.5``; width/height are rounded. Baseline kept.
        """
        sx = float(sx)
        sy = sx if sy is None else float(sy)
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=(self.cx + 0.5) * sx - 0.5,
            cy=(self.cy + 0.5) * sy - 0.5,
            width=int(round(self.width * sx)),
            height=int(round(self.height * sy)),
            baseline=self.baseline,
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project camera-frame points ``(..., 3)`` (metres) to pixels ``(..., 2)``.

        Returns float64 ``(u, v)`` without clipping; points with ``Z <= 0`` or
        non-finite coordinates give NaN (no warning).
        """
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1] != 3:
            raise ValueError(f"points must have shape (..., 3), got {p.shape}")
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        ok = np.isfinite(p).all(axis=-1) & (z > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            zs = np.where(ok, z, 1.0)
            u = self.fx * x / zs + self.cx
            v = self.fy * y / zs + self.cy
        out = np.stack([u, v], axis=-1)
        out[~ok] = np.nan
        return out

    @classmethod
    def from_K(
        cls, K: np.ndarray, width: int, height: int, baseline: Optional[float] = None
    ) -> "Intrinsics":
        """Build from a 3x3 camera matrix (skew ignored)."""
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"K must be 3x3, got {K.shape}")
        return cls(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
            width=width, height=height, baseline=baseline,
        )

    @classmethod
    def from_hfov(
        cls, width: int, height: int, hfov_deg: float = 60.0, baseline: Optional[float] = None
    ) -> "Intrinsics":
        """Square-pixel intrinsics from a horizontal field of view in degrees.

        ``fx = fy = (width / 2) / tan(hfov / 2)``, principal point at the image centre.
        """
        f = (width / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)
        return cls(
            fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
            width=width, height=height, baseline=baseline,
        )


@dataclass(eq=False)
class Frame:
    """One input frame. ``image`` is HxWx3 uint8 BGR; ``t = index / fps`` seconds."""

    index: int
    t: float
    image: np.ndarray
    right: Optional[np.ndarray] = None
    intrinsics: Optional[Intrinsics] = None


@dataclass(eq=False)
class DepthFrame:
    """Depth of a frame: ``depth`` HxW float32 metres (0 where invalid), ``valid`` HxW bool.

    ``disparity`` (optional) is HxW float32 pixels, ``d = fx * B / Z``.
    """

    depth: np.ndarray
    valid: np.ndarray
    intrinsics: Intrinsics
    disparity: Optional[np.ndarray] = None


def _as_box(value: object, name: str) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have 4 values [x1, y1, x2, y2], got shape {np.shape(value)}")
    return arr


@dataclass(eq=False)
class PersonDetection:
    """Untracked person detection: boxes float32 [x1,y1,x2,y2] px, keypoints (17,3) COCO-17."""

    head_box: Optional[np.ndarray]
    body_box: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    score: float = 1.0

    def __post_init__(self) -> None:
        self.head_box = _as_box(self.head_box, "head_box")
        self.body_box = _as_box(self.body_box, "body_box")
        if self.keypoints is not None:
            kp = np.asarray(self.keypoints, dtype=np.float32)
            if kp.shape != (17, 3):
                raise ValueError(f"keypoints must have shape (17, 3), got {kp.shape}")
            self.keypoints = kp
        self.score = float(self.score)


@dataclass(eq=False)
class DirectionEstimate:
    """Gaze or head direction: unit float64 (3,) vector in the camera frame.

    ``confidence`` in [0, 1]; ``euler_deg`` = (yaw, pitch, roll) degrees for head pose.
    The direction is coerced to float64 but not renormalised.
    """

    direction: np.ndarray
    confidence: float = 1.0
    euler_deg: Optional[np.ndarray] = None
    frame_index: int = -1

    def __post_init__(self) -> None:
        d = np.asarray(self.direction, dtype=np.float64).reshape(-1)
        if d.shape != (3,):
            raise ValueError(f"direction must have 3 values, got shape {np.shape(self.direction)}")
        self.direction = d
        self.confidence = float(self.confidence)
        if self.euler_deg is not None:
            self.euler_deg = np.asarray(self.euler_deg, dtype=np.float64).reshape(-1)


@dataclass(eq=False)
class Person:
    """A tracked person with its cues. ``eyes_3d`` is camera-frame metres."""

    track_id: int
    detection: PersonDetection
    gaze: Optional[DirectionEstimate] = None
    head: Optional[DirectionEstimate] = None
    eyes_px: Optional[np.ndarray] = None
    eyes_3d: Optional[np.ndarray] = None
    own_object_id: Optional[int] = None


@dataclass
class SegmentInfo:
    """Metadata of one segment (``id > 0`` as used in ``Segmentation.label_map``)."""

    id: int
    label: str
    is_thing: bool = True
    score: float = 1.0


@dataclass(eq=False)
class Segmentation:
    """Panoptic-style segmentation: ``label_map`` HxW int32, 0 = no segment."""

    label_map: np.ndarray
    segments: list[SegmentInfo]
    frame_index: int = -1


@dataclass(eq=False)
class SceneObject:
    """A 3D scene object (camera frame, metres) built from one segment and the depth."""

    object_id: int
    segment_id: int
    label: str
    is_thing: bool
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    centroid: np.ndarray
    points: np.ndarray
    n_points: int


@dataclass
class AttentionScore:
    """Attention of one person to one object; scores in [0, 1], ``None`` = cue unavailable."""

    person_id: int
    object_id: int
    label: str
    score_gaze: Optional[float]
    score_head: Optional[float]
    score_total: float
    occluded: bool = False


@dataclass(eq=False)
class FrameResult:
    """Everything the pipeline produced for one frame.

    ``targets`` maps track id -> attended object id (or None); ``timings_ms``
    holds per-stage milliseconds; ``ran`` tells whether a stage executed on this
    frame (False = result held/reused from an earlier frame).
    """

    index: int
    t: float
    persons: list[Person] = field(default_factory=list)
    objects: list[SceneObject] = field(default_factory=list)
    scores: list[AttentionScore] = field(default_factory=list)
    targets: dict[int, Optional[int]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    ran: dict[str, bool] = field(default_factory=dict)
