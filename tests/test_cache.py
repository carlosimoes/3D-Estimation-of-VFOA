"""Tests of the per-stage output cache (#28)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vfoa.backends import create_backend
from vfoa.backends.synthetic import SyntheticWorld
from vfoa.cache import (
    FORMAT_VERSION,
    STAGES,
    StageCache,
    cache_root,
    decode,
    encode,
    stage_key,
    stage_meta,
)
from vfoa.config import load_config
from vfoa.types import (
    DepthFrame,
    DirectionEstimate,
    Intrinsics,
    Person,
    PersonDetection,
    SegmentInfo,
    Segmentation,
)

FP = "video:/data/x.mp4:123:456:0:None"


# ------------------------------------------------------------------ helpers
def _intr(baseline=0.1):
    return Intrinsics(fx=500.5, fy=501.25, cx=319.5, cy=179.5, width=64, height=36, baseline=baseline)


def _depth_payload(rng, disparity=False, baseline=0.1):
    d = rng.uniform(0.3, 9.0, size=(36, 64)).astype(np.float32)
    valid = rng.uniform(size=(36, 64)) > 0.2
    d[~valid] = 0
    disp = (500.5 * 0.1 / np.maximum(d, 1e-3)).astype(np.float32) if disparity else None
    return DepthFrame(depth=d, valid=valid, intrinsics=_intr(baseline), disparity=disp)


def _persons_payload(rng):
    kp = rng.uniform(0, 60, size=(17, 3)).astype(np.float32)
    return [
        PersonDetection(head_box=[1.5, 2.25, 10.0, 12.0], body_box=[0, 0, 20, 30], keypoints=kp, score=0.875),
        PersonDetection(head_box=None, body_box=[3, 4, 5, 6], keypoints=None, score=0.3),
        PersonDetection(head_box=[4, 4, 8, 8], body_box=None, keypoints=None),
    ]


def _dirs_payload(with_euler=True):
    d1 = np.array([0.1, -0.2, -0.97])
    d1 /= np.linalg.norm(d1)
    return {
        7: DirectionEstimate(d1, 0.8, euler_deg=np.array([1.5, -2.0, 0.25]) if with_euler else None,
                             frame_index=12),
        3: DirectionEstimate([0.0, 0.0, -1.0], 1.0, frame_index=12),
    }


def _seg_payload(rng):
    lab = rng.integers(0, 4, size=(36, 64)).astype(np.int32)
    segs = [SegmentInfo(1, "cup", True, 0.9), SegmentInfo(2, "wall", False, 0.55), SegmentInfo(3, "person ü", True)]
    return Segmentation(label_map=lab, segments=segs, frame_index=5)


def assert_same(stage, a, b):
    if stage == "depth":
        assert a.depth.dtype == np.float32 and np.array_equal(a.depth, b.depth)
        assert a.valid.dtype == bool and np.array_equal(a.valid, b.valid)
        assert a.intrinsics == b.intrinsics
        if a.disparity is None:
            assert b.disparity is None
        else:
            assert np.array_equal(a.disparity, b.disparity)
    elif stage == "persons":
        assert len(a) == len(b)
        for x, y in zip(a, b):
            for f in ("head_box", "body_box", "keypoints"):
                xa, ya = getattr(x, f), getattr(y, f)
                if xa is None:
                    assert ya is None
                else:
                    assert ya.dtype == np.float32 and np.array_equal(xa, ya)
            assert x.score == y.score
    elif stage in ("headpose", "gaze"):
        assert list(a) == list(b)
        for k in a:
            assert np.array_equal(a[k].direction, b[k].direction) and b[k].direction.dtype == np.float64
            assert a[k].confidence == b[k].confidence and a[k].frame_index == b[k].frame_index
            if a[k].euler_deg is None:
                assert b[k].euler_deg is None
            else:
                assert np.array_equal(a[k].euler_deg, b[k].euler_deg)
    else:
        assert b.label_map.dtype == np.int32 and np.array_equal(a.label_map, b.label_map)
        assert a.frame_index == b.frame_index
        assert [vars(s) for s in a.segments] == [vars(s) for s in b.segments]


def _variants(rng):
    return [
        ("depth", _depth_payload(rng)),
        ("depth", _depth_payload(rng, disparity=True, baseline=None)),
        ("persons", _persons_payload(rng)),
        ("persons", []),
        ("headpose", _dirs_payload()),
        ("headpose", {}),
        ("gaze", _dirs_payload(with_euler=False)),
        ("gaze", {}),
        ("segmentation", _seg_payload(rng)),
        ("segmentation", Segmentation(np.zeros((4, 5), np.int32), [], -1)),
    ]


def _snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        st = p.stat()
        h = hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else None
        out[str(p.relative_to(root))] = (st.st_size, st.st_mtime_ns, h)
    return out


def _fill(root, n=10, chunk_size=4, stage="headpose", key="k" * 16, close=True, **kw):
    c = StageCache(root, stage, key, chunk_size=chunk_size, meta={"fingerprint": FP, "config": {"a": 1}}, **kw)
    for i in range(n):
        c.put(i, {i: DirectionEstimate([0.0, 0.0, -1.0], i / 10.0, frame_index=i)})
    if close:
        c.close()
    return c


# ------------------------------------------------------------------ encode / decode
def test_encode_decode_roundtrip(rng):
    for stage, payload in _variants(rng):
        arrays = encode(stage, payload)
        assert all(isinstance(v, np.ndarray) and v.dtype != object for v in arrays.values())
        assert_same(stage, payload, decode(stage, arrays))


def test_disk_roundtrip_all_variants(tmp_path, rng):
    variants = _variants(rng)
    for n, (stage, payload) in enumerate(variants):
        c = StageCache(tmp_path, stage, f"{n:016d}", chunk_size=2)
        c.put(n, payload)
        c.put(n + 100, payload)
        c.close()
        r = StageCache(tmp_path, stage, f"{n:016d}", mode="read")
        assert r.complete and r.has(n) and r.has(n + 100)
        assert_same(stage, payload, r.get(n))
        assert_same(stage, payload, r.get(n + 100))
        assert r.get(n + 1) is None
        r.close()


def test_unknown_stage_and_mode(tmp_path):
    with pytest.raises(ValueError):
        encode("attention", None)
    with pytest.raises(ValueError):
        StageCache(tmp_path, "scene", "x")
    with pytest.raises(ValueError):
        StageCache(tmp_path, "depth", "x", mode="append")
    with pytest.raises(ValueError):
        stage_key(load_config(profile="synthetic"), "attention", FP)


# ------------------------------------------------------------------ keys
def _keys(cfg, fp=FP):
    return {s: stage_key(cfg, s, fp) for s in STAGES}


def test_stage_key_sensitivity():
    base = load_config(profile="synthetic")
    k0 = _keys(base)
    assert all(len(k) == 16 and int(k, 16) >= 0 for k in k0.values())
    assert len(set(k0.values())) == len(STAGES)
    assert _keys(load_config(profile="synthetic")) == k0

    def changed(overrides, fp=FP):
        k = _keys(load_config(profile="synthetic", overrides=overrides), fp)
        return {s for s in STAGES if k[s] != k0[s]}

    assert changed([], fp="other") == set(STAGES)
    for stage, ov in [("depth", "depth.num_disparities=64"), ("segmentation", "segmentation.stride=3"),
                      ("headpose", "headpose.max_hold_frames=3"), ("gaze", "gaze.window=static")]:
        assert changed([ov]) == {stage}, ov
    assert changed(["persons.score_thr=0.9"]) == {"persons", "headpose", "gaze"}
    assert changed(["tracking.iou_thr=0.9"]) == {"headpose", "gaze"}
    assert changed(["models.device=cpu"]) == set(STAGES)
    assert changed(["camera.hfov_deg=70"]) == set(STAGES)
    assert changed(["runtime.lookahead_frames=6"]) == {"gaze"}
    for ov in ["attention.sigma_gaze_deg=3", "attention.method=rays", "smoothing.direction=ema",
               "scene.point_stride=4", "output.dir=elsewhere", "output.csv=true", "cache.mode=read",
               "timing.enabled=false"]:
        assert changed([ov]) == set(), ov
    # head_proxy gaze depends on the headpose key
    hp = load_config(profile="synthetic", overrides=["gaze.backend=head_proxy"])
    hp2 = load_config(profile="synthetic", overrides=["gaze.backend=head_proxy", "headpose.max_hold_frames=2"])
    assert stage_key(hp, "gaze", FP) != stage_key(hp2, "gaze", FP)


def test_stage_meta_and_root():
    cfg = load_config(profile="synthetic", overrides=["output.dir=/x/out"])
    m = stage_meta(cfg, "depth", FP)
    assert m["fingerprint"] == FP and m["config"]["backend"] == "synthetic"
    json.dumps(m)
    assert cache_root(cfg) == Path("/x/out") / "cache"
    cfg2 = load_config(profile="synthetic", overrides=["cache.dir=/y/c"])
    assert cache_root(cfg2) == Path("/y/c")


# ------------------------------------------------------------------ StageCache behaviour
def test_chunks_complete_and_meta(tmp_path):
    c = _fill(tmp_path, close=False)
    d = tmp_path / ("headpose-" + "k" * 16)
    assert sorted(p.name for p in d.glob("frames_*.npz")) == ["frames_000000.npz", "frames_000004.npz"]
    assert not c.complete
    c.close()
    assert c.complete
    assert sorted(p.name for p in d.glob("frames_*.npz")) == [
        "frames_000000.npz", "frames_000004.npz", "frames_000008.npz"]
    assert not list(d.glob("*.tmp"))
    meta = json.loads((d / "meta.json").read_text())
    assert meta["format_version"] == FORMAT_VERSION and meta["stage"] == "headpose"
    assert meta["key"] == "k" * 16 and meta["fingerprint"] == FP
    assert meta["n_frames"] == 10 and meta["complete"] is True and meta["config"] == {"a": 1}
    r = StageCache(tmp_path, "headpose", "k" * 16)
    assert r.complete and r.n_frames == 10 and r.indices() == list(range(10))
    for i in range(10):
        got = r.get(i)
        assert list(got) == [i] and got[i].confidence == pytest.approx(i / 10.0)
    r.close()


def test_incomplete_caches(tmp_path):
    c = _fill(tmp_path / "a", close=False)
    c.close(complete=False)
    r = StageCache(tmp_path / "a", "headpose", "k" * 16, mode="read")
    assert not r.complete and r.n_frames == 10
    # never closed: only the flushed chunks are visible, not complete
    _fill(tmp_path / "b", n=6, close=False)
    r2 = StageCache(tmp_path / "b", "headpose", "k" * 16, mode="read")
    assert not r2.complete and r2.indices() == [0, 1, 2, 3]
    # a put after a complete read makes it incomplete until close
    _fill(tmp_path / "c")
    rw = StageCache(tmp_path / "c", "headpose", "k" * 16)
    assert rw.complete
    rw.put(10, {})
    assert not rw.complete
    rw.close()
    assert rw.complete and StageCache(tmp_path / "c", "headpose", "k" * 16).n_frames == 11


def test_read_mode_never_writes(tmp_path):
    missing = tmp_path / "nothing"
    c = StageCache(missing, "depth", "a" * 16, mode="read")
    assert c.get(0) is None and not c.has(0)
    c.put(0, _depth_payload(np.random.default_rng(0)))
    assert c.get(0) is None
    c.close()
    assert not missing.exists()
    _fill(tmp_path / "x")
    before = _snapshot(tmp_path / "x")
    r = StageCache(tmp_path / "x", "headpose", "k" * 16, mode="read")
    r.put(99, {})
    r.get(3)
    r.flush()
    r.close()
    assert _snapshot(tmp_path / "x") == before


def test_constructor_never_writes(tmp_path):
    for mode in ("readwrite", "write"):
        StageCache(tmp_path / mode, "depth", "a" * 16, mode=mode)
        assert not (tmp_path / mode).exists()


@pytest.mark.parametrize("damage", ["delete", "truncate", "zero_middle", "garbage"])
def test_damaged_chunks_are_misses(tmp_path, damage):
    _fill(tmp_path)
    d = tmp_path / ("headpose-" + "k" * 16)
    victim = d / "frames_000004.npz"
    if damage == "delete":
        victim.unlink()
    elif damage == "truncate":
        data = victim.read_bytes()
        victim.write_bytes(data[: len(data) // 2])
    elif damage == "garbage":
        victim.write_bytes(b"not a zip at all")
    else:
        data = bytearray(victim.read_bytes())
        n = len(data)
        data[n // 4: n // 2] = bytes(n // 2 - n // 4)
        victim.write_bytes(bytes(data))
    r = StageCache(tmp_path, "headpose", "k" * 16, mode="read")
    for i in range(10):
        got = r.get(i)
        if 4 <= i < 8:
            assert got is None or (list(got) == [i] and got[i].confidence == pytest.approx(i / 10.0))
        else:
            assert list(got) == [i]
    assert not r.complete
    r.close()


def test_chunk_deleted_after_open(tmp_path):
    _fill(tmp_path)
    r = StageCache(tmp_path, "headpose", "k" * 16, mode="read")
    (tmp_path / ("headpose-" + "k" * 16) / "frames_000008.npz").unlink()
    assert r.get(8) is None and r.get(9) is None and not r.has(9)
    assert list(r.get(0)) == [0]


def test_corrupt_meta(tmp_path):
    _fill(tmp_path)
    d = tmp_path / ("headpose-" + "k" * 16)
    (d / "meta.json").write_text("{not json")
    r = StageCache(tmp_path, "headpose", "k" * 16, mode="read")
    assert r.get(0) is None and not r.complete
    # readwrite on a corrupt cache rewrites it cleanly
    rw = StageCache(tmp_path, "headpose", "k" * 16)
    rw.put(0, {5: DirectionEstimate([0, 0, -1.0])})
    rw.close()
    r2 = StageCache(tmp_path, "headpose", "k" * 16)
    assert r2.indices() == [0] and list(r2.get(0)) == [5] and r2.complete
    # format version mismatch -> ignored
    meta = json.loads((d / "meta.json").read_text())
    meta["format_version"] = FORMAT_VERSION + 1
    (d / "meta.json").write_text(json.dumps(meta))
    assert StageCache(tmp_path, "headpose", "k" * 16, mode="read").get(0) is None


def test_write_mode_replaces(tmp_path):
    _fill(tmp_path)
    w = StageCache(tmp_path, "headpose", "k" * 16, mode="write", chunk_size=4)
    assert not w.has(0) and w.get(0) is None
    w.put(20, {1: DirectionEstimate([0, 0, -1.0])})
    w.close()
    r = StageCache(tmp_path, "headpose", "k" * 16, mode="read")
    assert r.indices() == [20] and r.complete
    d = tmp_path / ("headpose-" + "k" * 16)
    assert [p.name for p in d.glob("frames_*.npz")] == ["frames_000020.npz"]


def test_readwrite_resume_and_pending(tmp_path):
    _fill(tmp_path, n=6, close=False)  # frames 0..3 flushed, 4..5 lost
    rw = StageCache(tmp_path, "headpose", "k" * 16, chunk_size=4)
    assert rw.indices() == [0, 1, 2, 3]
    for i in range(4, 7):
        assert rw.get(i) is None
        rw.put(i, {i: DirectionEstimate([0, 0, -1.0], frame_index=i)})
    got = rw.get(5)  # pending (not flushed yet)
    assert list(got) == [5] and got[5].frame_index == 5
    rw.close()
    r = StageCache(tmp_path, "headpose", "k" * 16, mode="read")
    assert r.indices() == list(range(7)) and r.complete


def test_context_manager(tmp_path):
    with StageCache(tmp_path, "gaze", "g" * 16) as c:
        c.put(0, {})
    assert StageCache(tmp_path, "gaze", "g" * 16).complete
    with pytest.raises(RuntimeError):
        with StageCache(tmp_path / "e", "gaze", "g" * 16) as c:
            c.put(0, {})
            raise RuntimeError("boom")
    assert not StageCache(tmp_path / "e", "gaze", "g" * 16).complete


# ------------------------------------------------------------------ synthetic replay
def test_replay_synthetic_world(tmp_path):
    cfg = load_config(profile="synthetic", overrides={"synthetic": {"n_frames": 12}})
    world = SyntheticWorld.from_config(cfg)
    fp = world.source().fingerprint
    backends = {s: create_backend(s, cfg) for s in STAGES}
    caches = {s: StageCache(tmp_path, s, stage_key(cfg, s, fp), chunk_size=5, meta=stage_meta(cfg, s, fp))
              for s in STAGES}

    def compute(stage, frame):
        b = backends[stage]
        if stage == "depth":
            return b.compute(frame)
        if stage == "persons":
            return b.detect(frame)
        if stage == "segmentation":
            return b.segment(frame)
        persons = [Person(track_id=10 + n, detection=d) for n, d in enumerate(world.detections(frame.index))]
        return b.estimate(frame, persons)

    fresh = {s: [] for s in STAGES}
    for frame in world.source():
        for s in STAGES:
            out = compute(s, frame)
            fresh[s].append(out)
            caches[s].put(frame.index, out)
    for c in caches.values():
        c.close()
    for s in STAGES:
        r = StageCache(tmp_path, s, stage_key(cfg, s, fp), mode="read")
        assert r.complete and r.n_frames == 12
        for i in range(12):
            assert_same(s, fresh[s][i], r.get(i))
