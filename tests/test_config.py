from __future__ import annotations

import copy
import pickle
import re

import pytest

from vfoa.config import (
    DEFAULTS,
    PROFILES,
    Config,
    canonical_json,
    config_hash,
    dump_config,
    load_config,
)

SECTIONS = [
    "profile", "input", "camera", "depth", "persons", "tracking", "headpose", "gaze",
    "segmentation", "scene", "attention", "smoothing", "cache", "output", "models",
    "runtime", "timing", "synthetic",
]
PROFILE_NAMES = ["fast", "accurate", "thesis", "synthetic"]


def test_default_sections_and_spot_values():
    cfg = load_config()
    assert list(cfg.keys()) == SECTIONS
    assert cfg.profile == "fast"
    assert cfg.depth.backend == "constant"
    assert cfg.persons.backend == "yolo_pose"
    assert cfg.headpose.backend == "keypoints"
    assert cfg.gaze.backend == "none"
    assert cfg.segmentation.backend == "yolo_seg"
    assert cfg.segmentation.stride == 10
    assert cfg.segmentation["async"] is False
    assert cfg.tracking.type == "byte"
    assert cfg.attention.rays.half_angles_head_deg == [12.5, 24.0]
    assert cfg.depth.wls_lambda == 8000000.0
    assert cfg.cache.stages == ["depth", "persons", "headpose", "gaze", "segmentation"]
    assert cfg.synthetic.cost_ms == {}
    assert cfg.models.third_party_root is None


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_profiles(profile):
    cfg = load_config(profile=profile)
    assert cfg.profile == profile
    backends = {k: cfg[k].backend for k in ("depth", "persons", "headpose", "gaze", "segmentation")}
    if profile == "fast":
        assert backends == {
            "depth": "constant", "persons": "yolo_pose", "headpose": "keypoints",
            "gaze": "none", "segmentation": "yolo_seg",
        }
    elif profile in ("accurate", "thesis"):
        assert backends == {
            "depth": "stereo_bm", "persons": "openpose_densepose", "headpose": "fsanet",
            "gaze": "gaze360", "segmentation": "panoptic_fpn",
        }
        assert cfg.segmentation.stride == 1
    else:
        assert set(backends.values()) == {"synthetic"}
        assert cfg.input.source == "synthetic" and cfg.input.path == "synthetic"
        assert cfg.segmentation.stride == 5

    if profile == "thesis":
        assert cfg.tracking.type == "greedy_iou"
        assert cfg.tracking.iou_thr == 0.5
        assert cfg.tracking.max_age == 0
        assert cfg.gaze.window == "static"
        assert cfg.scene.outlier == "radius"
        assert cfg.scene.ignore_labels == ["wall"]
        assert cfg.scene.point_stride == 1
        assert cfg.attention.method == "rays"
        assert cfg.attention.fusion == "agreement"
        assert cfg.attention.occlusion == "none"
        assert cfg.smoothing.direction == "none"
        assert cfg.smoothing.target == "none"
        # untouched keys keep their defaults
        assert cfg.attention.sigma_gaze_deg == 10.0
        assert cfg.tracking.box == "head"
    else:
        assert cfg.tracking.type == "byte"
        assert cfg.attention.method == "angular"
        assert cfg.smoothing.direction == "kalman"


def test_attribute_access_and_to_dict():
    cfg = load_config()
    assert cfg.attention.method == cfg["attention"]["method"]
    assert isinstance(cfg.attention, Config) and isinstance(cfg.attention.rays, Config)
    assert cfg.get("nope", 3) == 3
    with pytest.raises(AttributeError):
        cfg.nope  # noqa: B018
    cfg.output.dir = "elsewhere"
    assert cfg["output"]["dir"] == "elsewhere"
    cfg["extra"] = {"a": {"b": 1}}
    assert cfg.extra.a.b == 1
    cfg.update({"more": {"x": 2}})
    assert isinstance(cfg.more, Config)
    cfg.setdefault("other", {"y": 1})
    assert cfg.other.y == 1

    plain = cfg.to_dict()
    assert type(plain) is dict and type(plain["attention"]["rays"]) is dict
    plain["attention"]["method"] = "rays"
    assert cfg.attention.method == "angular"

    dc = copy.deepcopy(cfg)
    assert isinstance(dc, Config) and dc == cfg and dc.attention is not cfg.attention
    pk = pickle.loads(pickle.dumps(cfg))
    assert isinstance(pk.attention, Config) and pk == cfg
    assert copy.copy(cfg) == cfg


def test_overrides_strings():
    cfg = load_config(overrides=["attention.sigma_gaze_deg=5", "scene.ignore_labels=[wall]"])
    assert cfg.attention.sigma_gaze_deg == 5
    assert cfg.scene.ignore_labels == ["wall"]
    cfg = load_config(overrides=["attention.rays.n=30", "input.path=null", "depth.wls_lambda=1e-1",
                                 "output.dir=a=b"])
    assert cfg.attention.rays.n == 30
    assert cfg.attention.rays.half_angles_gaze_deg == [11.3, 21.8]
    assert cfg.input.path is None
    assert isinstance(cfg.depth.wls_lambda, float) and cfg.depth.wls_lambda == pytest.approx(0.1)
    assert cfg.output.dir == "a=b"
    assert load_config(overrides="gaze.backend=none").gaze.backend == "none"


def test_overrides_nested_dict():
    cfg = load_config(overrides={"attention": {"sigma_gaze_deg": 5, "rays": {"n": 7}},
                                 "scene": {"ignore_labels": ["wall"]}})
    assert cfg.attention.sigma_gaze_deg == 5
    assert cfg.attention.rays.n == 7
    assert cfg.attention.rays.half_angles_head_deg == [12.5, 24.0]
    assert cfg.scene.ignore_labels == ["wall"]
    assert cfg.attention.sigma_head_deg == 20.0


def test_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("foo:\n  x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="foo"):
        load_config(bad)
    with pytest.raises(ValueError, match="foo"):
        load_config(overrides=["foo.x=1"])
    with pytest.raises(ValueError, match="foo"):
        load_config(overrides={"foo": {"x": 1}})
    with pytest.raises(ValueError):
        load_config(overrides=["attention.sigma_gaze_deg"])
    with pytest.raises(ValueError):
        load_config(profile="turbo")
    with pytest.raises(ValueError):
        load_config(overrides=["profile=thesis"])
    with pytest.raises(ValueError):
        load_config(overrides={"attention": 3})
    notmap = tmp_path / "list.yaml"
    notmap.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(notmap)
    # unknown key inside a known section is accepted
    assert load_config(overrides=["attention.new_knob=1"]).attention.new_knob == 1


def test_merge_order(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text(
        "profile: thesis\nattention:\n  method: angular\n  sigma_head_deg: 1e1\n"
        "segmentation:\n", encoding="utf-8",
    )
    cfg = load_config(f)
    assert cfg.profile == "thesis"
    assert cfg.attention.method == "angular"  # file beats profile
    assert cfg.attention.fusion == "agreement"  # profile beats defaults
    assert cfg.attention.sigma_head_deg == 10.0 and isinstance(cfg.attention.sigma_head_deg, float)
    assert cfg.segmentation.backend == "panoptic_fpn"  # empty section = no change
    cfg = load_config(f, overrides=["attention.method=rays"])
    assert cfg.attention.method == "rays"  # overrides beat file
    cfg = load_config(f, profile="fast")
    assert cfg.profile == "fast"  # argument beats file profile
    assert cfg.attention.fusion == "confidence"
    assert cfg.attention.method == "angular"


@pytest.mark.parametrize("profile", PROFILE_NAMES)
def test_yaml_mirrors_profile(repo_root, profile):
    from_file = load_config(repo_root / "configs" / f"{profile}.yaml")
    assert from_file.to_dict() == load_config(profile=profile).to_dict()


def test_all_config_files_load(repo_root):
    files = sorted((repo_root / "configs").glob("*.yaml"))
    assert {p.stem for p in files} >= {"fast", "accurate", "thesis", "synthetic", "sintel_gt"}
    for path in files:
        load_config(path)
    cfg = load_config(repo_root / "configs" / "sintel_gt.yaml")
    assert cfg.profile == "fast"
    assert cfg.depth.backend == "sintel_gt"
    for key in ("path", "camera_dir", "depth_dir"):
        assert (repo_root / cfg.input[key]).exists(), key


def test_config_hash():
    cfg = load_config()
    h = config_hash(cfg)
    assert re.fullmatch(r"[0-9a-f]{16}", h)
    assert config_hash(load_config()) == h
    other = load_config(overrides=["attention.sigma_gaze_deg=5"])
    assert config_hash(other) != h
    assert config_hash(other, ["depth", "persons"]) == config_hash(cfg, ["depth", "persons"])
    assert config_hash(other, ["attention"]) != config_hash(cfg, ["attention"])
    a = load_config(overrides=["depth.stride=5"])
    b = load_config(overrides=["depth.stride=5.0"])
    assert config_hash(a, ["depth"]) == config_hash(b, ["depth"])
    assert canonical_json({"b": 1, "a": True}) == '{"a":true,"b":1.0}'


def test_dump_roundtrip(tmp_path):
    cfg = load_config(profile="thesis", overrides=["attention.sigma_gaze_deg=5"])
    path = dump_config(cfg, tmp_path / "sub" / "config.yaml")
    assert path.exists()
    assert load_config(path).to_dict() == cfg.to_dict()


def test_defaults_not_mutated():
    before = copy.deepcopy(DEFAULTS)
    profiles_before = copy.deepcopy(PROFILES)
    cfg = load_config(profile="thesis", overrides={"scene": {"ignore_labels": ["x"]}})
    cfg.scene.ignore_labels.append("y")
    cfg.attention.rays.half_angles_gaze_deg.append(1.0)
    assert DEFAULTS == before
    assert PROFILES == profiles_before
