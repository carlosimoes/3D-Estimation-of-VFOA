"""Fast head pose from 2D face keypoints with a rigid head model and PnP.

The five COCO face keypoints (nose, eyes, ears) are matched to a canonical
head model (:data:`HEAD_MODEL`, metres) and ``cv2.solvePnP`` (SQPNP) recovers
the head rotation in the camera frame. Cost: a fraction of a millisecond per
person, no network.

Head frame: aligned with the camera frame when the person faces the camera
(+X towards image right, +Y down, +Z away from the camera), so the facing
direction is ``-Z`` and ``forward = R @ (0, 0, -1)``. Euler angles follow the
thesis ``draw_axis`` convention ``R = Rx(-pitch) @ Ry(yaw) @ Rz(roll)``,
giving ``forward = (-sin yaw, -cos yaw sin pitch, -cos yaw cos pitch)``.

Limitations: at least 4 face keypoints including the nose are required (eyes
and ears are coplanar in the model and 3 points are mirror-ambiguous), so
persons seen from the back (only ears visible) get no estimate.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from vfoa.backends.base import HeadPoseBackend
from vfoa.types import DirectionEstimate, Frame, Intrinsics, Person

__all__ = ["HEAD_MODEL", "solve_head_pose", "KeypointHeadPose"]

log = logging.getLogger(__name__)

#: Canonical head model (metres, head frame): nose, left eye, right eye, left ear, right ear.
HEAD_MODEL = np.array(
    [
        [0.0, 0.02, -0.105],
        [0.032, -0.02, -0.08],
        [-0.032, -0.02, -0.08],
        [0.075, 0.0, 0.0],
        [-0.075, 0.0, 0.0],
    ],
    dtype=np.float64,
)

_MIN_SPAN_PX = 8.0


def solve_head_pose(
    keypoints: Optional[np.ndarray],
    K: np.ndarray,
    kpt_thr: float = 0.35,
    relaxed_thr: float = 0.1,
    fallback_confidence: float = 0.5,
    max_reproj_frac: float = 0.15,
) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    """Head pose from COCO-17 keypoints ``(17, 3)`` (x, y pixels, confidence).

    Returns ``(forward, confidence, euler_deg)``: ``forward`` a unit float64
    camera-frame vector, ``confidence`` in [0, 1], ``euler_deg`` =
    (yaw, pitch, roll) degrees; or ``None`` when no reliable estimate exists.

    * Fewer than 3 face keypoints with confidence >= ``kpt_thr`` -> ``None``.
    * Normal case: >= 4 strong face keypoints including the nose.
    * Fallback (lower confidence, scaled by ``fallback_confidence``): only 3
      are strong, so keypoints with confidence >= ``relaxed_thr`` are used,
      still requiring >= 4 points including the nose; otherwise ``None``.
    * Rejected (``None``): point span < 8 px, solver failure, head behind the
      camera, or reprojection RMS > ``max_reproj_frac`` * span.
    * ``confidence = mean(used conf) * factor * clip(span / 40, 0.25, 1)``.
    """
    if keypoints is None:
        return None
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] < 5 or kp.shape[1] < 3:
        return None
    face = kp[:5]
    conf = np.where(np.isfinite(face[:, 2]), face[:, 2], 0.0)
    conf = np.where(np.isfinite(face[:, :2]).all(axis=1), conf, 0.0)
    strong = conf >= kpt_thr
    if int(strong.sum()) < 3:
        return None
    if int(strong.sum()) >= 4 and strong[0]:
        use, factor = strong, 1.0
    else:
        use, factor = conf >= relaxed_thr, float(fallback_confidence)
        if int(use.sum()) < 4 or not use[0]:
            return None

    pts = np.ascontiguousarray(face[use, :2], dtype=np.float64)
    obj = np.ascontiguousarray(HEAD_MODEL[use])
    span = float(np.linalg.norm(pts[:, None] - pts[None], axis=-1).max())
    if not math.isfinite(span) or span < _MIN_SPAN_PX:
        return None
    K = np.asarray(K, dtype=np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, pts, K, None, flags=cv2.SOLVEPNP_SQPNP)
    except cv2.error:
        return None
    if not ok or rvec is None or tvec is None:
        return None
    R, _ = cv2.Rodrigues(rvec)
    f = -R[:, 2]
    n = float(np.linalg.norm(f))
    tz = float(np.asarray(tvec).ravel()[2])
    if not np.isfinite(R).all() or not np.isfinite(tvec).all() or n < 1e-9 or tz <= 0:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
    if not math.isfinite(rms) or rms > max_reproj_frac * span:
        return None
    f = f / n
    yaw = math.degrees(math.asin(float(np.clip(-f[0], -1.0, 1.0))))
    pitch = math.degrees(math.atan2(-f[1], -f[2]))
    roll = math.degrees(math.atan2(-R[0, 1], R[0, 0]))
    confidence = (
        float(np.clip(conf[use].mean(), 0.0, 1.0)) * factor * float(np.clip(span / 40.0, 0.25, 1.0))
    )
    return f.astype(np.float64), confidence, np.array([yaw, pitch, roll], dtype=np.float64)


class KeypointHeadPose(HeadPoseBackend):
    """Head direction per track from the person's COCO face keypoints (PnP).

    Config: ``persons.kpt_thr`` (strong keypoint threshold), optional
    ``headpose.relaxed_kpt_thr`` (0.1), ``headpose.fallback_confidence`` (0.5),
    ``headpose.max_reproj_frac`` (0.15); intrinsics from ``frame.intrinsics``,
    else ``Intrinsics.from_hfov(W, H, camera.hfov_deg)``. Persons without
    keypoints or without a reliable solution are absent from the result.
    """

    name = "keypoints"

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg)

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        self.load()
        intr = frame.intrinsics
        if intr is None:
            h, w = frame.image.shape[:2]
            camera = self.cfg.get("camera") or {}
            intr = Intrinsics.from_hfov(w, h, float(camera.get("hfov_deg", 60.0) or 60.0))
        K = intr.K
        persons_cfg = self.cfg.get("persons") or {}
        sec = self.section
        kpt_thr = float(persons_cfg.get("kpt_thr", 0.35))
        relaxed = float(sec.get("relaxed_kpt_thr", 0.1))
        fb_conf = float(sec.get("fallback_confidence", 0.5))
        max_frac = float(sec.get("max_reproj_frac", 0.15))

        out: dict[int, DirectionEstimate] = {}
        for person in persons:
            try:
                kp = person.detection.keypoints if person.detection is not None else None
                if kp is None:
                    continue
                sol = solve_head_pose(kp, K, kpt_thr, relaxed, fb_conf, max_frac)
                if sol is None:
                    continue
                forward, confidence, euler = sol
                out[person.track_id] = DirectionEstimate(
                    direction=forward, confidence=confidence, euler_deg=euler, frame_index=frame.index
                )
            except Exception as exc:  # never raise for one bad person
                log.debug("keypoint head pose failed for track %s: %r", getattr(person, "track_id", "?"), exc)
        return out
