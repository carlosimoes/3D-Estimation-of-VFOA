"""Backend registry.

``REGISTRY[kind][name]`` holds ``"module:Class"`` strings (or classes registered
at runtime) that are resolved lazily with importlib: importing this package
imports no backend module. ``create_backend(kind, cfg)`` reads
``cfg[kind].backend`` and returns an (unloaded) instance, or ``None`` for
``"none"``. ``create_backend_with_fallback(kind, cfg)`` additionally honours
``cfg[kind].fallback``: the primary backend is loaded eagerly and, when it is
unavailable, the fallback backend is created and loaded instead.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Mapping, Optional, Union

from .base import KINDS, Backend, BackendUnavailable

__all__ = [
    "REGISTRY",
    "register",
    "resolve",
    "available",
    "create_backend",
    "create_backend_with_fallback",
    "Backend",
    "BackendUnavailable",
    "KINDS",
]

log = logging.getLogger(__name__)

REGISTRY: dict[str, dict[str, Union[str, type]]] = {
    "depth": {
        "stereo_bm": "vfoa.backends.depth_stereo:StereoBMDepth",
        "sintel_gt": "vfoa.backends.depth_simple:SintelGTDepth",
        "constant": "vfoa.backends.depth_simple:ConstantDepth",
        "synthetic": "vfoa.backends.synthetic:SyntheticDepth",
    },
    "persons": {
        "openpose_densepose": "vfoa.backends.accurate.openpose_densepose:OpenPoseDensePosePersons",
        "yolo_pose": "vfoa.backends.fast.yolo_pose:YoloPosePersons",
        "synthetic": "vfoa.backends.synthetic:SyntheticPersons",
    },
    "headpose": {
        "fsanet": "vfoa.backends.accurate.fsanet:FSANetHeadPose",
        "keypoints": "vfoa.backends.fast.keypoints_headpose:KeypointHeadPose",
        "synthetic": "vfoa.backends.synthetic:SyntheticHeadPose",
    },
    "gaze": {
        "gaze360": "vfoa.backends.accurate.gaze360:Gaze360Gaze",
        "head_proxy": "vfoa.backends.fast.head_proxy_gaze:HeadProxyGaze",
        "l2cs": "vfoa.backends.fast.l2cs_gaze:BinnedGaze360Backend",
        "synthetic": "vfoa.backends.synthetic:SyntheticGaze",
    },
    "segmentation": {
        "panoptic_fpn": "vfoa.backends.accurate.panoptic_fpn:PanopticFPNSegmentation",
        "yolo_seg": "vfoa.backends.fast.yolo_seg:YoloSegSegmentation",
        "synthetic": "vfoa.backends.synthetic:SyntheticSegmentation",
    },
}


def _check_kind(kind: str) -> None:
    if kind not in REGISTRY:
        raise ValueError(f"Unknown backend kind {kind!r}; valid kinds: {', '.join(KINDS)}")


def register(kind: str, name: str, target: Union[str, type]) -> None:
    """Register ``target`` (a class or a ``"module:Class"`` string) as ``kind``/``name``."""
    _check_kind(kind)
    if not name or name == "none":
        raise ValueError(f"Invalid backend name {name!r}")
    if isinstance(target, str):
        if ":" not in target:
            raise ValueError(f"Backend target {target!r} must look like 'package.module:Class'")
    elif not isinstance(target, type):
        raise TypeError(f"Backend target must be a class or a 'module:Class' string, got {target!r}")
    REGISTRY[kind][name] = target


def available(kind: str) -> list[str]:
    """Sorted registered names of ``kind`` (without importing anything)."""
    _check_kind(kind)
    return sorted(REGISTRY[kind])


def resolve(kind: str, name: str) -> type:
    """Return the backend class registered as ``kind``/``name`` (imports its module)."""
    _check_kind(kind)
    try:
        target = REGISTRY[kind][name]
    except KeyError:
        raise ValueError(
            f"Unknown {kind} backend {name!r}; valid names: {', '.join(available(kind) + ['none'])}"
        ) from None
    if isinstance(target, type):
        return target
    module_name, _, attr = target.partition(":")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError:
        raise ImportError(f"Module {module_name!r} has no backend class {attr!r}") from None


def create_backend(kind: str, cfg: Mapping[str, Any]) -> Optional[Backend]:
    """Instantiate (without loading) the backend named by ``cfg[kind].backend``.

    Returns ``None`` when the name is ``"none"`` or null; unknown names raise
    ``ValueError`` listing the valid names.
    """
    _check_kind(kind)
    section = cfg.get(kind) or {}
    name = section.get("backend")
    if name is None or name == "none":
        return None
    cls = resolve(kind, name)
    return cls(cfg)


def create_backend_with_fallback(kind: str, cfg: Mapping[str, Any]) -> tuple[Optional[Backend], Optional[str]]:
    """Create the backend of ``kind``, falling back to ``cfg[kind].fallback`` when it is unavailable.

    Without a fallback (or when it equals the primary name) this is
    :func:`create_backend` and the backend is NOT loaded; returns
    ``(backend, None)``. With a fallback the primary backend is loaded
    eagerly; on :class:`BackendUnavailable` a warning with the reason is
    logged and the fallback backend is created and loaded instead, returning
    ``(fallback_backend, fallback_name)``. Errors of the fallback propagate.
    """
    _check_kind(kind)
    section = cfg.get(kind) or {}
    name = section.get("backend")
    fallback = section.get("fallback")
    backend = create_backend(kind, cfg)
    if backend is None or not fallback or fallback == name:
        return backend, None
    try:
        backend.load()
        return backend, None
    except BackendUnavailable as exc:
        log.warning("%s backend %r unavailable (%s); falling back to %r", kind, name, exc, fallback)
        try:
            backend.close()
        except Exception:  # pragma: no cover - best effort
            pass
    if fallback == "none":
        return None, "none"
    fb = resolve(kind, str(fallback))(cfg)
    fb.load()
    return fb, str(fallback)
