"""Legacy OpenPose keypoint predictor (thin wrapper over the vfoa accurate backend).

The implementation lives in :mod:`vfoa.backends.accurate.openpose_densepose`;
the model is built once on ``models.device``. Results are (17, 3) float32
COCO-17 pixel keypoints per person (lower body removed by default).
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Mapping, Sequence

import numpy as np

from vfoa.backends.accurate.openpose_densepose import OpenPoseDensePosePersons
from vfoa.config import load_config

__all__ = ["OpenPosePredictor"]


class OpenPosePredictor(OpenPoseDensePosePersons):
    """Legacy API: ``OpenPosePredictor(args)(frames)``.

    ``args`` is a vfoa config mapping or a namespace with ``cfg`` (OpenPose
    experiment yaml) and ``opM`` (weights).
    """

    #: OpenPose-18 indices of r_knee, r_ankle, l_knee, l_ankle and the background slot.
    bottomBodyPartsIndex = [9, 10, 12, 13, 18]

    def __init__(self, args: Any, **kw: Any) -> None:
        if isinstance(args, Mapping):
            cfg = args
        else:
            persons: dict = {}
            if getattr(args, "cfg", None):
                persons["openpose_cfg"] = str(args.cfg)
            if getattr(args, "opM", None):
                persons["openpose_weights"] = str(args.opM)
            cfg = load_config(profile="accurate", overrides={"persons": persons})
        super().__init__(cfg, **kw)
        self.batchSize = getattr(args, "opB", 1)

    def __call__(self, frames: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        """COCO-17 keypoints of every person of every BGR frame (loads OpenPose only)."""
        self._openpose()
        return [self.skeletons(frame) for frame in frames]

    def initModelForLoop(self) -> None:
        """Build the OpenPose model (once)."""
        self._openpose()

    def predictIteration(self, inputs: Sequence[np.ndarray]) -> list[list[np.ndarray]]:
        """Alias of ``__call__``."""
        return self(inputs)
