"""Per-stage output cache: re-run attention/smoothing without the models (#28).

Layout::

    <cache dir>/<stage>-<key>/meta.json
    <cache dir>/<stage>-<key>/frames_<first index:06d>.npz

Each chunk is an ``np.savez_compressed`` archive whose arrays are named
``"<frame index>/<field>"``; a chunk is written every ``chunk_size`` stored
frames and on close. ``meta.json`` holds ``format_version, stage, key,
fingerprint, config, chunk_size, n_frames, complete``; ``complete`` becomes
true only on :meth:`StageCache.close` after a full run.

The key (:func:`stage_key`) hashes the input fingerprint, the stage's config
section, ``models``, ``camera`` and the keys of upstream stages, so changing
attention / smoothing / scene / output settings never invalidates a cache.

Payloads are stored with native dtypes (no pickle), so integer/bool data round
trip exactly and floats keep their precision:

* depth -> :class:`DepthFrame` (intrinsics, optional disparity)
* persons -> ``list[PersonDetection]`` (None boxes/keypoints round-trip)
* headpose / gaze -> ``dict[track_id, DirectionEstimate]`` (raw, pre-smoothing)
* segmentation -> :class:`Segmentation` (segments as a JSON string)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from vfoa.config import _to_plain, canonical_json
from vfoa.types import (
    DepthFrame,
    DirectionEstimate,
    Intrinsics,
    PersonDetection,
    SegmentInfo,
    Segmentation,
)

__all__ = [
    "FORMAT_VERSION",
    "STAGES",
    "MODES",
    "stage_key",
    "stage_meta",
    "cache_root",
    "encode",
    "decode",
    "StageCache",
]

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
#: Cacheable stages.
STAGES: tuple[str, ...] = ("depth", "persons", "headpose", "gaze", "segmentation")
MODES: tuple[str, ...] = ("readwrite", "read", "write")
_CHUNK_RE = re.compile(r"^frames_(\d+)(?:_(\d+))?\.npz$")
_META = "meta.json"


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown cache stage {stage!r}; valid stages: {', '.join(STAGES)}")


def _section(cfg: Mapping[str, Any], name: str) -> Any:
    value = cfg.get(name)
    return _to_plain(value) if value is not None else None


def _upstream(cfg: Mapping[str, Any], stage: str, fingerprint: str) -> dict:
    if stage not in ("headpose", "gaze"):
        return {}
    up: dict = {
        "persons": stage_key(cfg, "persons", fingerprint),
        "tracking": _section(cfg, "tracking"),
    }
    if stage == "gaze":
        runtime = cfg.get("runtime") or {}
        inp = cfg.get("input") or {}
        up["lookahead"] = runtime.get("lookahead_frames", 0)
        up["fps"] = inp.get("fps")
        if (cfg.get("gaze") or {}).get("backend") == "head_proxy":
            up["headpose"] = stage_key(cfg, "headpose", fingerprint)
    return up


def stage_key(cfg: Mapping[str, Any], stage: str, fingerprint: str) -> str:
    """16-hex-char cache key of ``stage`` for the input ``fingerprint``.

    Hashes ``{format_version, stage, fingerprint, config: cfg[stage], models,
    camera, upstream}``. Depth also includes the auxiliary input paths
    (``input.right_path/camera_dir/depth_dir``). Upstream: headpose and gaze
    depend on the persons key and ``tracking``; gaze also on
    ``runtime.lookahead_frames`` and ``input.fps`` and, for the ``head_proxy``
    backend, on the headpose key. Unknown stage -> ValueError.
    """
    _check_stage(stage)
    payload: dict = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "fingerprint": str(fingerprint),
        "config": _section(cfg, stage),
        "models": _section(cfg, "models"),
        "camera": _section(cfg, "camera"),
        "upstream": _upstream(cfg, stage, fingerprint),
    }
    if stage == "depth":
        inp = cfg.get("input") or {}
        payload["inputs"] = {k: inp.get(k) for k in ("right_path", "camera_dir", "depth_dir")}
    return hashlib.sha1(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def stage_meta(cfg: Mapping[str, Any], stage: str, fingerprint: str) -> dict:
    """Descriptive metadata stored in meta.json: ``{fingerprint, config}``."""
    _check_stage(stage)
    return {"fingerprint": str(fingerprint), "config": _section(cfg, stage)}


def cache_root(cfg: Mapping[str, Any]) -> Path:
    """Cache directory: ``cache.dir`` or ``<output.dir>/cache``."""
    c = cfg.get("cache") or {}
    if c.get("dir"):
        return Path(c["dir"]).expanduser()
    out = (cfg.get("output") or {}).get("dir") or "outputs"
    return Path(out).expanduser() / "cache"


# ------------------------------------------------------------------ encode / decode
def _encode_directions(payload: Mapping[int, DirectionEstimate]) -> dict[str, np.ndarray]:
    items = list(payload.items())
    n = len(items)
    ids = np.array([int(k) for k, _ in items], dtype=np.int64)
    direction = np.zeros((n, 3), np.float64)
    confidence = np.zeros(n, np.float64)
    frame_index = np.zeros(n, np.int64)
    has_euler = np.zeros(n, bool)
    euler = np.zeros((n, 3), np.float64)
    for r, (_k, est) in enumerate(items):
        direction[r] = np.asarray(est.direction, np.float64).reshape(3)
        confidence[r] = float(est.confidence)
        frame_index[r] = int(est.frame_index)
        if est.euler_deg is not None:
            e = np.asarray(est.euler_deg, np.float64).reshape(-1)
            if e.shape != (3,):
                raise ValueError(f"euler_deg must have 3 values, got {e.shape}")
            has_euler[r] = True
            euler[r] = e
    return {
        "ids": ids, "direction": direction, "confidence": confidence,
        "frame_index": frame_index, "has_euler": has_euler, "euler_deg": euler,
    }


def _decode_directions(a: Mapping[str, np.ndarray]) -> dict[int, DirectionEstimate]:
    out: dict[int, DirectionEstimate] = {}
    for r, tid in enumerate(a["ids"].tolist()):
        out[int(tid)] = DirectionEstimate(
            direction=np.array(a["direction"][r], dtype=np.float64),
            confidence=float(a["confidence"][r]),
            euler_deg=np.array(a["euler_deg"][r], dtype=np.float64) if bool(a["has_euler"][r]) else None,
            frame_index=int(a["frame_index"][r]),
        )
    return out


def encode(stage: str, payload: Any) -> dict[str, np.ndarray]:
    """Encode a stage payload into named NumPy arrays (native dtypes, no objects)."""
    _check_stage(stage)
    if stage == "depth":
        d: DepthFrame = payload
        intr = d.intrinsics
        if intr is None:
            iv = np.full(7, np.nan, np.float64)
        else:
            iv = np.array(
                [intr.fx, intr.fy, intr.cx, intr.cy, intr.width, intr.height,
                 np.nan if intr.baseline is None else intr.baseline],
                dtype=np.float64,
            )
        out = {
            "depth": np.ascontiguousarray(d.depth, dtype=np.float32),
            "valid": np.ascontiguousarray(d.valid, dtype=bool),
            "intrinsics": iv,
        }
        if d.disparity is not None:
            out["disparity"] = np.ascontiguousarray(d.disparity, dtype=np.float32)
        return out
    if stage == "persons":
        dets = list(payload)
        n = len(dets)
        has = np.zeros((n, 3), bool)
        head = np.zeros((n, 4), np.float32)
        body = np.zeros((n, 4), np.float32)
        kps = np.zeros((n, 17, 3), np.float32)
        score = np.zeros(n, np.float64)
        for r, det in enumerate(dets):
            score[r] = float(det.score)
            if det.head_box is not None:
                has[r, 0] = True
                head[r] = det.head_box
            if det.body_box is not None:
                has[r, 1] = True
                body[r] = det.body_box
            if det.keypoints is not None:
                has[r, 2] = True
                kps[r] = det.keypoints
        return {
            "n": np.array(n, dtype=np.int64), "score": score, "has": has,
            "head_box": head, "body_box": body, "keypoints": kps,
        }
    if stage in ("headpose", "gaze"):
        return _encode_directions(payload or {})
    # segmentation
    s: Segmentation = payload
    segs = [
        {"id": int(g.id), "label": str(g.label), "is_thing": bool(g.is_thing), "score": float(g.score)}
        for g in s.segments
    ]
    return {
        "label_map": np.ascontiguousarray(s.label_map, dtype=np.int32),
        "frame_index": np.array(int(s.frame_index), dtype=np.int64),
        "segments": np.array(json.dumps(segs)),
    }


def decode(stage: str, arrays: Mapping[str, np.ndarray]) -> Any:
    """Inverse of :func:`encode`."""
    _check_stage(stage)
    if stage == "depth":
        iv = np.asarray(arrays["intrinsics"], np.float64)
        intr = None
        if np.isfinite(iv[:6]).all():
            intr = Intrinsics(
                fx=iv[0], fy=iv[1], cx=iv[2], cy=iv[3], width=int(iv[4]), height=int(iv[5]),
                baseline=None if np.isnan(iv[6]) else float(iv[6]),
            )
        disp = arrays.get("disparity") if hasattr(arrays, "get") else None
        return DepthFrame(
            depth=np.array(arrays["depth"], dtype=np.float32),
            valid=np.array(arrays["valid"], dtype=bool),
            intrinsics=intr,  # type: ignore[arg-type]
            disparity=None if disp is None else np.array(disp, dtype=np.float32),
        )
    if stage == "persons":
        n = int(arrays["n"])
        has = arrays["has"]
        out = []
        for r in range(n):
            out.append(PersonDetection(
                head_box=np.array(arrays["head_box"][r]) if has[r, 0] else None,
                body_box=np.array(arrays["body_box"][r]) if has[r, 1] else None,
                keypoints=np.array(arrays["keypoints"][r]) if has[r, 2] else None,
                score=float(arrays["score"][r]),
            ))
        return out
    if stage in ("headpose", "gaze"):
        return _decode_directions(arrays)
    segs = json.loads(str(np.asarray(arrays["segments"]).item()))
    return Segmentation(
        label_map=np.array(arrays["label_map"], dtype=np.int32),
        segments=[SegmentInfo(id=int(g["id"]), label=str(g["label"]), is_thing=bool(g["is_thing"]),
                              score=float(g["score"])) for g in segs],
        frame_index=int(arrays["frame_index"]),
    )


# ------------------------------------------------------------------ StageCache
class StageCache:
    """Chunked on-disk cache of one stage's per-frame outputs.

    ``mode``: ``readwrite`` (read hits, store misses), ``read`` (never creates
    or modifies files) or ``write`` (ignore existing data; stale chunks are
    deleted before the first write). The constructor never writes. Missing,
    truncated or corrupt chunks are treated as misses (a warning is logged and
    the chunk is dropped from the index), never as errors.
    """

    def __init__(
        self,
        root: Union[str, Path],
        stage: str,
        key: str,
        mode: str = "readwrite",
        chunk_size: int = 64,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> None:
        _check_stage(stage)
        if mode not in MODES:
            raise ValueError(f"Unknown cache mode {mode!r}; valid modes: {', '.join(MODES)}")
        self.root = Path(root)
        self.stage = stage
        self.key = str(key)
        self.mode = mode
        self.chunk_size = max(int(chunk_size or 1), 1)
        self.extra_meta = dict(meta or {})
        self.dir = self.root / f"{stage}-{self.key}"
        self._index: dict[int, tuple[Path, tuple[str, ...]]] = {}
        self._pending: dict[int, dict[str, np.ndarray]] = {}
        self._open_path: Optional[Path] = None
        self._open_npz: Any = None
        self._stale = mode == "write"     # existing files must be purged before writing
        self._meta: Optional[dict] = None
        self._complete = False
        self._closed = False
        if mode != "write":
            self._scan()

    # -------------------------------------------------------------- index
    def _scan(self) -> None:
        if not self.dir.is_dir():
            return
        meta = None
        try:
            with open(self.dir / _META, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except FileNotFoundError:
            meta = None
        except Exception as exc:  # corrupt meta
            logger.warning("cache %s: unreadable meta.json (%s); ignoring cached data", self.dir, exc)
            meta = None
        if (
            not isinstance(meta, dict)
            or meta.get("format_version") != FORMAT_VERSION
            or meta.get("stage") != self.stage
            or meta.get("key") != self.key
        ):
            if meta is not None or any(self.dir.glob("frames_*.npz")):
                logger.warning("cache %s: missing or mismatching meta.json; treating as empty", self.dir)
            self._stale = True
            return
        self._meta = meta
        chunks = sorted(
            (p for p in self.dir.iterdir() if _CHUNK_RE.match(p.name)),
            key=lambda p: (int(_CHUNK_RE.match(p.name).group(1)), int(_CHUNK_RE.match(p.name).group(2) or 0)),
        )
        for path in chunks:
            try:
                with np.load(path, allow_pickle=False) as npz:
                    names = list(npz.files)
            except Exception as exc:
                logger.warning("cache %s: skipping unreadable chunk %s (%s)", self.dir, path.name, exc)
                continue
            fields: dict[int, list[str]] = {}
            for name in names:
                idx, sep, fld = name.partition("/")
                if not sep:
                    continue
                try:
                    fields.setdefault(int(idx), []).append(fld)
                except ValueError:
                    continue
            for idx, flds in fields.items():
                self._index[idx] = (path, tuple(flds))
        self._complete = bool(meta.get("complete")) and int(meta.get("n_frames", -1)) == len(self._index)

    def _close_npz(self) -> None:
        if self._open_npz is not None:
            try:
                self._open_npz.close()
            except Exception:
                pass
        self._open_npz = None
        self._open_path = None

    def _drop_chunk(self, path: Path) -> None:
        if self._open_path == path:
            self._close_npz()
        for idx in [i for i, (p, _f) in self._index.items() if p == path]:
            del self._index[idx]
        self._complete = False

    # -------------------------------------------------------------- queries
    @property
    def complete(self) -> bool:
        """True when meta.json says complete and every recorded frame is indexed."""
        return self._complete and not self._pending

    @property
    def n_frames(self) -> int:
        """Number of frames available (stored or pending)."""
        return len(set(self._index) | set(self._pending))

    def indices(self) -> list[int]:
        """Sorted frame indices available in the cache."""
        return sorted(set(self._index) | set(self._pending))

    def has(self, index: int) -> bool:
        index = int(index)
        return index in self._pending or index in self._index

    def get(self, index: int) -> Any:
        """Decoded payload of frame ``index`` or None (miss or unreadable data)."""
        index = int(index)
        if index in self._pending:
            return decode(self.stage, self._pending[index])
        entry = self._index.get(index)
        if entry is None:
            return None
        path, fields = entry
        try:
            if self._open_path != path:
                self._close_npz()
                self._open_npz = np.load(path, allow_pickle=False)
                self._open_path = path
            arrays = {f: self._open_npz[f"{index}/{f}"] for f in fields}
            return decode(self.stage, arrays)
        except Exception as exc:
            logger.warning("cache %s: dropping unreadable chunk %s (%s)", self.dir, path.name, exc)
            self._drop_chunk(path)
            return None

    # -------------------------------------------------------------- writes
    def put(self, index: int, payload: Any) -> None:
        """Store the payload of frame ``index`` (no-op in read mode)."""
        if self.mode == "read":
            return
        if self._closed:
            raise RuntimeError(f"cache {self.dir} is closed")
        arrays = {k: np.array(v, copy=True) for k, v in encode(self.stage, payload).items()}
        self._pending[int(index)] = arrays
        self._complete = False
        if len(self._pending) >= self.chunk_size:
            self.flush()

    def _purge(self) -> None:
        if self.dir.is_dir():
            self._close_npz()
            for p in self.dir.iterdir():
                if _CHUNK_RE.match(p.name) or p.name == _META or p.name.endswith(".tmp"):
                    try:
                        p.unlink()
                    except FileNotFoundError:
                        pass
        self._index.clear()
        self._meta = None
        self._stale = False

    def _write_meta(self, complete: bool) -> None:
        meta = {
            **self.extra_meta,
            "format_version": FORMAT_VERSION,
            "stage": self.stage,
            "key": self.key,
            "fingerprint": self.extra_meta.get("fingerprint"),
            "config": self.extra_meta.get("config"),
            "chunk_size": self.chunk_size,
            "n_frames": len(self._index),
            "complete": bool(complete),
        }
        tmp = self.dir / (_META + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(meta, fh, indent=1, sort_keys=True, default=str)
        os.replace(tmp, self.dir / _META)
        self._meta = meta

    def flush(self) -> None:
        """Write pending frames to a new chunk and update meta.json (complete false)."""
        if self.mode == "read" or not self._pending:
            return
        if self._stale:
            self._purge()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._close_npz()
        first = min(self._pending)
        path = self.dir / f"frames_{first:06d}.npz"
        n = 0
        while path.exists():
            n += 1
            path = self.dir / f"frames_{first:06d}_{n}.npz"
        arrays = {
            f"{idx}/{fld}": arr for idx, flds in sorted(self._pending.items()) for fld, arr in flds.items()
        }
        tmp = self.dir / (path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
        for idx, flds in self._pending.items():
            self._index[idx] = (path, tuple(flds))
        self._pending.clear()
        self._complete = False
        self._write_meta(False)

    def close(self, complete: bool = True) -> None:
        """Flush and write meta.json with ``complete``; read mode writes nothing."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.mode == "read":
                return
            self.flush()
            if self._stale:
                if not self.dir.is_dir():
                    return
                self._purge()
            if not self.dir.is_dir():
                return
            self._write_meta(complete)
            self._complete = bool(complete)
        finally:
            self._close_npz()

    def __enter__(self) -> "StageCache":
        return self

    def __exit__(self, exc_type: Any, *exc: object) -> None:
        self.close(complete=exc_type is None)

    def __repr__(self) -> str:
        return f"StageCache({str(self.dir)!r}, mode={self.mode!r}, frames={self.n_frames})"
