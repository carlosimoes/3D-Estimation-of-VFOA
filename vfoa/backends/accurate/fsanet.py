"""FSA-Net head pose (thesis model, two ONNX sessions) as a streaming backend.

The two onnxruntime sessions are created ONCE in ``load()`` (Refs #32). Per
person the head box is clipped to the image, the BGR crop is resized to 64x64,
laid out as CHW float32 and normalised with ``(x - 127.5) / 128`` exactly like
the upstream FSA-Net demo (Refs #8; the thesis used ``(x - 447.5) / 448``). The
(yaw, pitch, roll) outputs of both models are averaged and converted to a unit
camera-frame forward vector with :func:`vfoa.geometry.euler_to_forward`
(Refs #2; the thesis used the raw angles as a 3D vector).

onnxruntime is imported lazily; the session factory is injectable for tests.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from vfoa.backends.accurate._thirdparty import ENV_VAR, resolve_path
from vfoa.backends.base import BackendUnavailable, HeadPoseBackend, resolve_device
from vfoa.geometry import euler_to_forward
from vfoa.types import DirectionEstimate, Frame, Person

__all__ = ["crop_head", "preprocess_head", "FSANetHeadPose", "INPUT_SIZE"]

logger = logging.getLogger(__name__)

#: FSA-Net input side length in pixels.
INPUT_SIZE = 64


def crop_head(image: np.ndarray, box: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Crop ``box`` = [x1, y1, x2, y2] pixels (rounded, clipped to the image) from ``image``.

    Returns a view of the image, or None when the box is None/non-finite or the
    clipped region has zero area.
    """
    if box is None:
        return None
    b = np.asarray(box, dtype=np.float64).reshape(-1)
    if b.shape != (4,) or not np.isfinite(b).all():
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.rint(b).astype(int)
    x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return image[y1:y2, x1:x2]


def preprocess_head(crop: np.ndarray) -> np.ndarray:
    """BGR uint8 head crop -> FSA-Net input blob (1, 3, 64, 64) float32 in ~[-1, 1].

    Stays in BGR (the model's native order); ``(x - 127.5) / 128``.
    """
    img = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    chw = img.transpose(2, 0, 1)[None].astype(np.float32)
    return (chw - np.float32(127.5)) / np.float32(128.0)


class FSANetHeadPose(HeadPoseBackend):
    """FSA-Net (two ONNX models averaged) head pose -> unit camera-frame direction.

    ``session_factory(path) -> session`` (an object with ``run(output_names,
    feeds)``) replaces onnxruntime for tests.
    """

    name = "fsanet"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        session_factory: Optional[Callable[[str], Any]] = None,
    ) -> None:
        super().__init__(cfg)
        self._session_factory = session_factory
        self._sessions: list[tuple[Any, str, str]] = []

    # ------------------------------------------------------------------ loading

    def _model_paths(self) -> list[str]:
        entries = self.section.get("models") or []
        if isinstance(entries, (str, bytes)):
            entries = [entries]
        if not entries:
            raise BackendUnavailable("FSA-Net: config key headpose.models is empty")
        paths = []
        for entry in entries:
            try:
                path = resolve_path(self.cfg, entry)
            except BackendUnavailable as exc:
                raise BackendUnavailable(
                    f"FSA-Net weights {str(entry)!r} (config key headpose.models) cannot be "
                    f"resolved: {exc}"
                ) from exc
            if not path.is_file():
                raise BackendUnavailable(
                    f"FSA-Net ONNX model {str(path)!r} not found (config key headpose.models, "
                    f"relative paths resolve against models.third_party_root / {ENV_VAR})"
                )
            paths.append(str(path))
        return paths

    def _default_factory(self) -> Callable[[str], Any]:
        try:
            import onnxruntime as ort  # noqa: PLC0415 (lazy heavy import)
        except ImportError as exc:
            raise BackendUnavailable(
                "FSA-Net needs the 'onnxruntime' package (pip install onnxruntime); "
                "or choose another headpose.backend"
            ) from exc
        providers = ["CPUExecutionProvider"]
        if resolve_device(self.cfg).startswith("cuda"):
            try:
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception:  # pragma: no cover - defensive
                pass
        options = ort.SessionOptions()
        threads = (self.cfg.get("runtime") or {}).get("num_threads")
        if threads:
            options.intra_op_num_threads = int(threads)

        def factory(path: str) -> Any:
            return ort.InferenceSession(path, sess_options=options, providers=providers)

        return factory

    def _load(self) -> None:
        paths = self._model_paths()
        factory = self._session_factory or self._default_factory()
        sessions = []
        for path in paths:
            sess = factory(path)
            in_name, out_name = "input", "output"
            try:
                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name
            except Exception:
                pass
            sessions.append((sess, in_name, out_name))
        self._sessions = sessions
        logger.info("FSA-Net: loaded %d ONNX session(s)", len(sessions))

    # ---------------------------------------------------------------- inference

    def estimate(self, frame: Frame, persons: Sequence[Person]) -> dict[int, DirectionEstimate]:
        """Head direction per track id; persons without a usable head box are absent."""
        self.load()
        out: dict[int, DirectionEstimate] = {}
        for p in persons:
            crop = crop_head(frame.image, p.detection.head_box)
            if crop is None:
                continue
            blob = preprocess_head(crop)
            results = []
            for sess, in_name, out_name in self._sessions:
                res = sess.run([out_name], {in_name: blob})[0]
                results.append(np.asarray(res, dtype=np.float64).reshape(1, -1))
            ypr = np.mean(np.vstack(results), axis=0)[:3]
            if ypr.shape != (3,) or not np.isfinite(ypr).all():
                continue
            out[p.track_id] = DirectionEstimate(
                direction=euler_to_forward(ypr[0], ypr[1], ypr[2]),
                confidence=1.0,
                euler_deg=ypr,
                frame_index=frame.index,
            )
        return out

    def close(self) -> None:
        self._sessions = []
        self._loaded = False
