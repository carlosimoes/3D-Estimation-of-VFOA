"""Configuration: defaults, profiles, YAML loading, overrides and hashing.

``load_config`` merges ``DEFAULTS <- PROFILES[profile] <- YAML file <- overrides``
and returns a :class:`Config` (a nested dict with attribute access).

Note: ``segmentation.async`` is a Python keyword; read it as
``cfg.segmentation["async"]``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np
import yaml

__all__ = [
    "Config",
    "DEFAULTS",
    "PROFILES",
    "SECTIONS",
    "load_config",
    "config_hash",
    "canonical_json",
    "dump_config",
]


class Config(dict):
    """Nested dict with attribute access: ``cfg.attention.method == cfg["attention"]["method"]``.

    Nested plain dicts are wrapped into ``Config`` on construction and assignment.
    Missing attributes raise ``AttributeError`` (so copy/pickle work).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.update(*args, **kwargs)

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, Mapping) and not isinstance(value, Config):
            return Config(value)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key: Any, default: Any = None) -> Any:  # type: ignore[override]
        if key not in self:
            self[key] = default
        return self[key]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config(copy.deepcopy(dict(self), memo))

    def __reduce__(self):  # pickle support
        return (Config, (dict(self),))

    def to_dict(self) -> dict:
        """Deep copy as plain dicts/lists."""
        return _to_plain(self)


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    return copy.deepcopy(obj)


DEFAULTS: dict = {
    "profile": "fast",
    "input": {
        "source": "auto",  # auto|video|images|camera|synthetic
        "path": None,
        "right_path": None,
        "camera_dir": None,
        "depth_dir": None,
        "start": 0,
        "max_frames": None,
        "fps": None,
        "realtime": False,
    },
    "camera": {
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        "hfov_deg": 60.0,
        "baseline_m": 0.1,
    },
    "depth": {
        "backend": "constant",  # stereo_bm|sintel_gt|constant|synthetic
        "stride": 1,
        "phase": 0,
        "scale": 1.0,
        "num_disparities": 160,
        "block_size": 15,
        "wls": "auto",
        "wls_lambda": 8000000.0,
        "wls_sigma": 1.5,
        "min_depth_m": 0.05,
        "max_depth_m": 100.0,
        "constant_depth_m": 2.5,
    },
    "persons": {
        "backend": "yolo_pose",  # openpose_densepose|yolo_pose|synthetic
        "stride": 1,
        "phase": 0,
        "score_thr": 0.4,
        "kpt_thr": 0.35,
        "model": "yolo11n-pose.pt",
        "imgsz": 640,
        "head_scale": 1.3,
        "match_keypoints": True,
        "openpose_cfg": "Pose_Estimation/experiments/vgg19_368x368_sgd.yaml",
        "openpose_weights": "Pose_Estimation/pose_model.pth",
        "densepose_cfg": "projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml",
        "densepose_weights": "projects/DensePose/configs/model_final_d366fa.pkl",
        "densepose_thr": 0.85,
    },
    "tracking": {
        "type": "byte",  # byte|greedy_iou
        "box": "head",
        "iou_thr": 0.3,
        "max_age": 30,
        "high_thr": 0.5,
        "low_thr": 0.1,
    },
    "headpose": {
        "backend": "keypoints",  # fsanet|keypoints|synthetic|none
        "stride": 1,
        "phase": 0,
        "max_hold_frames": 15,
        "models": [
            "headpose-fsanet-pytorch/pretrained/fsanet-1x1-iter-688590.onnx",
            "headpose-fsanet-pytorch/pretrained/fsanet-var-iter-688590.onnx",
        ],
    },
    "gaze": {
        "backend": "none",  # gaze360|head_proxy|synthetic|none
        "stride": 1,
        "phase": 0,
        "max_hold_frames": 15,
        "window": "causal",  # causal|static
        "window_step": None,  # null -> max(int(fps // 8), 1)
        "uncertainty_thr": 3.0,
        "eye_frame_correction": True,
        "weights": "gaze360/gaze360_model.pth.tar",
        "batch_size": 16,
        "proxy_confidence": 0.5,
    },
    "segmentation": {
        "backend": "yolo_seg",  # panoptic_fpn|yolo_seg|synthetic|none
        "stride": 10,
        "phase": 0,
        "async": False,
        "score_thr": 0.35,
        "model": "yolo11n-seg.pt",
        "imgsz": 640,
        "panoptic_cfg": "configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml",
        "panoptic_weights": "configs/COCO-PanopticSegmentation/model_final_cafdb1.pkl",
        "panoptic_thr": 0.5,
    },
    "scene": {
        "point_stride": 2,
        "max_points_per_object": 256,
        "min_points": 20,
        "outlier": "percentile",  # percentile|radius|none
        "outlier_percentiles": [2.0, 98.0],
        "ignore_labels": [],
        "person_labels": ["person"],
        "self_min_overlap": 0.3,
        "eye_patch_px": 3,
        "match_dist_m": 0.75,
        "object_max_age": 5,
    },
    "attention": {
        "method": "angular",  # angular|rays
        "sigma_gaze_deg": 10.0,
        "sigma_head_deg": 20.0,
        "fusion": "confidence",  # fixed|agreement|confidence
        "w_gaze": 0.6,
        "w_head": 0.4,
        "agreement_deg": 45.0,
        "target": "surface",  # surface|centroid
        "occlusion": "depth",  # depth|aabb|none
        "occlusion_margin_m": 0.15,
        "occlusion_samples": 24,
        "min_target_distance_m": 0.15,
        "distance_full_m": None,
        "distance_decay_m": 2.0,
        "min_score": 0.1,
        "rays": {
            "n": 20,
            "half_angles_gaze_deg": [11.3, 21.8],
            "half_angles_head_deg": [12.5, 24.0],
        },
    },
    "smoothing": {
        "direction": "kalman",  # kalman|ema|none
        "ema_alpha": 0.5,
        "kalman_q": 1.0,
        "kalman_r": 0.05,
        "max_hold_s": 0.5,
        "target": "hmm",  # hmm|hysteresis|none
        "hmm_stay_prob": 0.9,
        "hysteresis_frames": 3,
    },
    "cache": {
        "enabled": False,
        "dir": None,  # null -> <output.dir>/cache
        "mode": "readwrite",  # readwrite|read|write
        "stages": ["depth", "persons", "headpose", "gaze", "segmentation"],
        "chunk_size": 64,
    },
    "output": {
        "dir": "outputs",
        "jsonl": True,
        "csv": False,
        "video": False,
        "show": False,
    },
    "models": {
        "device": "auto",
        "weights_dir": "weights",  # env VFOA_WEIGHTS_DIR overrides
        "allow_download": True,
        "third_party_root": None,  # null -> env VFOA_THIRD_PARTY
    },
    "runtime": {
        "lookahead_frames": 0,
        "num_threads": None,
        "seed": 0,
        "log_every": 100,
    },
    "timing": {"enabled": True},
    "synthetic": {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "n_frames": 120,
        "n_persons": 2,
        "n_objects": 4,
        "seed": 0,
        "noise_deg": 0.0,
        "occluder": False,
        "cost_ms": {},
    },
}

#: Top-level section names (every top-level key of DEFAULTS).
SECTIONS: tuple[str, ...] = tuple(DEFAULTS)

_ACCURATE: dict = {
    "depth": {"backend": "stereo_bm"},
    "persons": {"backend": "openpose_densepose"},
    "headpose": {"backend": "fsanet"},
    "gaze": {"backend": "gaze360"},
    "segmentation": {"backend": "panoptic_fpn", "stride": 1},
}

_THESIS_EXTRA: dict = {
    "tracking": {"type": "greedy_iou", "iou_thr": 0.5, "max_age": 0},
    "gaze": {"window": "static"},
    "scene": {"outlier": "radius", "ignore_labels": ["wall"], "point_stride": 1},
    "attention": {"method": "rays", "fusion": "agreement", "occlusion": "none"},
    "smoothing": {"direction": "none", "target": "none"},
}


def _deep_merge(base: Mapping, override: Mapping) -> dict:
    """Return a deep copy of ``base`` with ``override`` merged in.

    Mappings merge recursively; any other value (including lists) replaces.
    Neither input is mutated.
    """
    out = _to_plain(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = _to_plain(value)
    return out


PROFILES: dict = {
    "fast": {},
    "accurate": _deep_merge({}, _ACCURATE),
    "thesis": _deep_merge(_ACCURATE, _THESIS_EXTRA),
    "synthetic": {
        "input": {"source": "synthetic", "path": "synthetic"},
        "depth": {"backend": "synthetic"},
        "persons": {"backend": "synthetic"},
        "headpose": {"backend": "synthetic"},
        "gaze": {"backend": "synthetic"},
        "segmentation": {"backend": "synthetic", "stride": 5},
    },
}

_SCI_FLOAT = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+$")


def _fix_floats(obj: Any) -> Any:
    """Convert strings like '1e-3' (which PyYAML keeps as str) to float, recursively."""
    if isinstance(obj, str):
        return float(obj) if _SCI_FLOAT.match(obj.strip()) else obj
    if isinstance(obj, Mapping):
        return {k: _fix_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fix_floats(v) for v in obj]
    return obj


def _validate_sections(data: Mapping, origin: str) -> None:
    for key, value in data.items():
        if key == "profile":
            continue
        if key not in DEFAULTS:
            raise ValueError(
                f"Unknown config section {key!r} in {origin}; valid sections: {', '.join(SECTIONS)}"
            )
        if isinstance(DEFAULTS[key], dict) and value is not None and not isinstance(value, Mapping):
            raise ValueError(
                f"Config section {key!r} in {origin} must be a mapping, got {type(value).__name__}"
            )


def _parse_override_strings(items: Iterable[str]) -> dict:
    out: dict = {}
    for item in items:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError(f"Override {item!r} must look like 'section.key=value'")
        key, raw = item.split("=", 1)
        key = key.strip()
        parts = [p for p in key.split(".")]
        if not key or any(not p for p in parts):
            raise ValueError(f"Override {item!r} has an invalid key")
        value = yaml.safe_load(raw) if raw.strip() != "" else None
        nested: Any = value
        for part in reversed(parts):
            nested = {part: nested}
        out = _deep_merge(out, nested)
    return out


def load_config(
    path: Optional[Union[str, Path]] = None,
    profile: Optional[str] = None,
    overrides: Optional[Union[Mapping, Iterable[str]]] = None,
) -> Config:
    """Load a configuration.

    Merge order: ``DEFAULTS <- PROFILES[profile] <- YAML file <- overrides``.
    ``profile`` = argument, else the file's ``profile`` key, else ``"fast"``.
    ``overrides`` is a nested dict or a list of ``"a.b.c=value"`` strings
    (value parsed with ``yaml.safe_load``). Unknown top-level sections raise
    ``ValueError``; unknown keys inside a section are accepted.
    """
    file_data: dict = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Config file {path} must contain a mapping, got {type(loaded).__name__}")
        file_data = _fix_floats(dict(loaded))
        _validate_sections(file_data, f"config file {path}")

    if overrides is None:
        ov: dict = {}
    elif isinstance(overrides, Mapping):
        ov = _fix_floats(_to_plain(overrides))
    elif isinstance(overrides, str):
        ov = _fix_floats(_parse_override_strings([overrides]))
    else:
        ov = _fix_floats(_parse_override_strings(list(overrides)))
    if "profile" in ov:
        raise ValueError("'profile' cannot be overridden; pass profile=... to load_config instead")
    _validate_sections(ov, "overrides")

    name = profile if profile is not None else file_data.get("profile") or "fast"
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; valid profiles: {', '.join(PROFILES)}")

    merged = _deep_merge(DEFAULTS, PROFILES[name])
    # An empty section in YAML ("attention:") loads as None: treat it as "no change".
    file_data = {
        k: v for k, v in file_data.items()
        if k != "profile" and not (v is None and isinstance(DEFAULTS[k], dict))
    }
    ov = {k: v for k, v in ov.items() if not (v is None and isinstance(DEFAULTS[k], dict))}
    merged = _deep_merge(merged, file_data)
    merged = _deep_merge(merged, ov)
    merged["profile"] = name
    return Config(merged)


def _canonical(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    if isinstance(obj, np.generic):
        return _canonical(obj.item())
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Canonical JSON for hashing: sorted keys, compact, every non-bool number as float."""
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), allow_nan=True)


def config_hash(cfg: Mapping, sections: Optional[Iterable[str]] = None) -> str:
    """16-hex-char sha1 of the canonical JSON of the chosen sections (default: all)."""
    names = list(cfg.keys()) if sections is None else list(sections)
    payload = {s: cfg.get(s) for s in names}
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def dump_config(cfg: Mapping, path: Union[str, Path]) -> Path:
    """Write the config as YAML (creating parent directories); returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else _to_plain(cfg)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return path
