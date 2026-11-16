"""Legacy FSA-Net head-pose predictor (thin alias of the vfoa accurate backend).

The implementation lives in :mod:`vfoa.backends.accurate.fsanet`: both ONNX
sessions are created once, crops are normalised with ``(x - 127.5) / 128``
(Refs #8) and the averaged (yaw, pitch, roll) become a unit camera-frame
direction (Refs #2). The old batch ``__call__`` over a whole video is gone; the
legacy VideoProcessor runs on the vfoa Pipeline.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Mapping, Optional

from vfoa.backends.accurate.fsanet import FSANetHeadPose
from vfoa.config import load_config

__all__ = ["HeadPredictor"]


class HeadPredictor(FSANetHeadPose):
    """Legacy API: ``HeadPredictor(headModelFile1, headModelFile2, batchSize)``.

    A vfoa config mapping may be passed as ``cfg`` or as the first argument;
    otherwise the ``accurate`` profile is used with the model files mapped to
    ``headpose.models``. ``batchSize`` is kept for compatibility only.
    """

    def __init__(
        self,
        headModelFile1: Any = None,
        headModelFile2: Optional[Any] = None,
        batchSize: Optional[int] = None,
        cfg: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> None:
        if cfg is None and isinstance(headModelFile1, Mapping):
            cfg = headModelFile1
        if cfg is None:
            files = [str(f) for f in (headModelFile1, headModelFile2) if f is not None]
            overrides = {"headpose": {"models": files}} if files else None
            cfg = load_config(profile="accurate", overrides=overrides)
        super().__init__(cfg, **kw)
        self.batchSize = batchSize
