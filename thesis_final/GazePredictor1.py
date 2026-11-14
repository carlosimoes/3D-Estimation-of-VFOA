"""Legacy Gaze360 predictor (thin alias of the vfoa accurate backend).

The implementation lives in :mod:`vfoa.backends.accurate.gaze360`: the GazeLSTM
is loaded once on ``models.device``, crops are RGB (Refs #9) and every person
gets a real 7-frame temporal window (Refs #7). The old batch ``__call__`` over
a whole video is gone; the legacy VideoProcessor runs on the vfoa Pipeline.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Mapping, Optional

from vfoa.backends.accurate.gaze360 import Gaze360Gaze
from vfoa.config import load_config

__all__ = ["GazePredictor"]


class GazePredictor(Gaze360Gaze):
    """Legacy API: ``GazePredictor(gazeModelFile, batchSize, uncertaintyThreshold)``.

    A vfoa config mapping may be passed as ``cfg`` or as the first argument;
    otherwise the ``accurate`` profile is used with the legacy arguments mapped
    to ``gaze.weights``, ``gaze.batch_size`` and ``gaze.uncertainty_thr``.
    """

    def __init__(
        self,
        gazeModelFile: Any = None,
        batchSize: Optional[int] = 16,
        uncertaintyThreshold: float = 3.0,
        cfg: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> None:
        if cfg is None and isinstance(gazeModelFile, Mapping):
            cfg = gazeModelFile
        if cfg is None:
            gaze: dict = {"uncertainty_thr": float(uncertaintyThreshold)}
            if batchSize is not None:
                gaze["batch_size"] = int(batchSize)
            if gazeModelFile is not None:
                gaze["weights"] = str(gazeModelFile)
            cfg = load_config(profile="accurate", overrides={"gaze": gaze})
        super().__init__(cfg, **kw)
        self.batchSize = batchSize
