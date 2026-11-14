"""Legacy Panoptic FPN predictor (thin wrapper over the vfoa accurate backend).

The implementation lives in :mod:`vfoa.backends.accurate.panoptic_fpn`; the
model is built once on ``models.device`` (no multi-GPU worker processes).
Images are BGR uint8.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vfoa.backends.accurate.panoptic_fpn import Detectron2Runtime, PanopticFPNSegmentation
from vfoa.config import load_config
from vfoa.types import Frame, Segmentation

__all__ = ["PanopticPredictor"]


class PanopticPredictor(PanopticFPNSegmentation):
    """Legacy API: ``PanopticPredictor(configFile, modelFile, threshold)(images)``.

    ``configFile`` may also be a vfoa config mapping. ``batchSize`` is ignored.
    """

    def __init__(
        self,
        configFile: Any,
        poseModelFile: Optional[str] = None,
        accuracyThreshold: float = 0.5,
        batchSize: int = 1,
        **kw: Any,
    ) -> None:
        if isinstance(configFile, Mapping):
            cfg = configFile
        else:
            seg: dict = {"panoptic_thr": float(accuracyThreshold)}
            if configFile is not None:
                seg["panoptic_cfg"] = str(configFile)
            if poseModelFile is not None:
                seg["panoptic_weights"] = str(poseModelFile)
            cfg = load_config(profile="accurate", overrides={"segmentation": seg})
        super().__init__(cfg, **kw)
        self.batchSize = batchSize

    def __call__(self, images: Sequence[np.ndarray]) -> tuple[list[Segmentation], Optional[Detectron2Runtime]]:
        """Segmentations of the BGR images (frame_index = position) and the runtime (class names)."""
        results = [self.segment(Frame(index=i, t=0.0, image=img)) for i, img in enumerate(images)]
        return results, self.runtime
