"""Legacy DensePose head-box predictor (thin wrapper over the vfoa accurate backend).

The implementation lives in :mod:`vfoa.backends.accurate.openpose_densepose`;
the model is built once on ``models.device`` (no multi-GPU worker processes).
Images are BGR uint8.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vfoa.backends.accurate.openpose_densepose import (
    OpenPoseDensePosePersons,
    head_box_from_parts,
)
from vfoa.config import load_config

__all__ = ["DensePosePredictor"]


class DensePosePredictor(OpenPoseDensePosePersons):
    """Legacy API: ``DensePosePredictor(configFile, poseModelFile, accuracyThreshold)(images)``.

    ``configFile`` may also be a vfoa config mapping. ``batchSize`` is ignored
    (frames are processed one by one with a model loaded once).
    """

    def __init__(
        self,
        configFile: Any,
        poseModelFile: Optional[str] = None,
        accuracyThreshold: float = 0.85,
        batchSize: int = 1,
        **kw: Any,
    ) -> None:
        if isinstance(configFile, Mapping):
            cfg = configFile
        else:
            persons: dict = {"densepose_thr": float(accuracyThreshold), "match_keypoints": False}
            if configFile is not None:
                persons["densepose_cfg"] = str(configFile)
            if poseModelFile is not None:
                persons["densepose_weights"] = str(poseModelFile)
            cfg = load_config(profile="accurate", overrides={"persons": persons})
        super().__init__(cfg, **kw)
        self.batchSize = batchSize

    def __call__(self, images: Sequence[np.ndarray]) -> list[list[Optional[np.ndarray]]]:
        """Head boxes (float32 [x1, y1, x2, y2] or None) of every instance of every BGR image."""
        self._densepose()
        return [[head for head, _body, _score in self.head_instances(img)] for img in images]

    @staticmethod
    def extractHeadBoxFromOutput(box: Sequence[float], instance: np.ndarray) -> Optional[np.ndarray]:
        """Head box from a part-label crop whose top-left corner is ``box[0:2]`` (clipped to the crop)."""
        h, w = np.asarray(instance).shape[:2]
        size = (float(box[0]) + w, float(box[1]) + h)
        return head_box_from_parts(instance, (box[0], box[1]), size)
