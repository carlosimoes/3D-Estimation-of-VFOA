"""Tests of the legacy thesis_final/Attention.py wrapper (section 14 contract)."""

from __future__ import annotations

import math
import os
import subprocess
import sys

import numpy as np
import pytest

import Attention as A

NAMES = ["vect2cone", "intersection", "inside_box", "distance", "calcangle2value",
         "calc_intersection_points", "CalcAttention", "calc_ratio",
         "calc_normalize_Attention", "Attention"]


def _env():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_legacy_import_exposes_names_without_heavy_modules(repo_root):
    code = (
        "import sys, Attention\n"
        f"missing = [n for n in {NAMES!r} if not callable(getattr(Attention, n, None))]\n"
        "assert not missing, missing\n"
        "heavy = [m for m in ('torch', 'open3d', 'matplotlib') if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=repo_root / "thesis_final", env=_env(),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr


def test_legacy_script_demo_runs(repo_root):
    r = subprocess.run([sys.executable, str(repo_root / "thesis_final" / "Attention.py")],
                       cwd=repo_root, env=_env(), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "total_Attention" in r.stdout


def test_vect2cone_along_x_unique_finite():
    rays = A.vect2cone([0, 0, 0], [500, 0, 0], 20, 20)
    assert rays.shape == (20, 3) and np.isfinite(rays).all()
    assert len(np.unique(np.round(rays, 9), axis=0)) == 20


def test_intersection_behind_and_inf_components():
    bmax, bmin = np.array([1.0, 1, -4]), np.array([-1.0, -1, -6])
    with np.errstate(divide="ignore"):
        behind = 1.0 / np.array([0.0, 0, 1])
        dir_inv = 1.0 / np.array([0.0, 0.0, -1.0])
    assert A.intersection(bmax, bmin, behind, np.zeros(3)) == (False, False)
    p1, p2 = A.intersection(bmax, bmin, dir_inv, np.zeros(3))
    assert np.isfinite(p1).all() and np.isfinite(p2).all()
    assert p1 == pytest.approx([0, 0, -4]) and p2 == pytest.approx([0, 0, -6])


def test_calc_ratio_translation_invariant():
    bbox = [np.array([2.0, 4, 4]), np.array([0.0, 0, 0])]
    assert A.calc_ratio(bbox) == pytest.approx(1 / 3)
    moved = [bbox[0] + 5000, bbox[1] + 5000]
    assert A.calc_ratio(moved) == pytest.approx(A.calc_ratio(bbox), abs=1e-12)


def test_inside_box_and_distance():
    assert A.inside_box([1, 1, 1], [0, 0, 0], [0.5, 0.5, 1.0])
    assert not A.inside_box([1, 1, 1], [0, 0, 0], [0.5, 1.5, 0.5])
    assert A.distance(np.zeros(3), np.array([3.0, 4, 0])) == pytest.approx(5.0)


def _rot_y(v, deg):
    a = math.radians(deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return R @ np.asarray(v, float)


def _objects():
    return {1: {"name": "box", "bbox": [np.array([0.5, 0.5, 5.5]), np.array([-0.3, -0.5, 4.5])]},
            2: {"name": "side", "bbox": [np.array([3.0, 0.5, 5.5]), np.array([2.0, -0.5, 4.5])]}}


@pytest.mark.parametrize("deg, w", [(30.0, (1.9, 1.1)), (60.0, (1.55, 1.45))])
def test_attention_total_is_weighted_mean(deg, w):
    g = np.array([0.0, 0.0, 1.0])
    rays = A.vect2cone([0, 0, 0], [0, 0, 5], 0.6, 20)
    person = {0: {"eyes": [0, 0, 0], "g": g, "hp": _rot_y(g, deg), "vg": rays, "vhp": rays}}
    out = A.Attention(person, _objects())[0][1]
    G, H = out["total_Attention_G"], out["total_Attention_H"]
    assert G == pytest.approx(H) and 0 < G <= 1
    assert out["total_Attention"] == pytest.approx((w[0] * G + w[1] * H) / 3, abs=1e-12)
    # with different per-cue rays the weights matter (they no longer cancel)
    person[0]["vhp"] = A.vect2cone([0, 0, 0], [0, 0, 5], 2.5, 20)
    out = A.Attention(person, _objects())[0][1]
    G, H = out["total_Attention_G"], out["total_Attention_H"]
    assert abs(G - H) > 1e-3
    assert out["total_Attention"] == pytest.approx((w[0] * G + w[1] * H) / 3, abs=1e-12)


@pytest.mark.parametrize("drop", [["g"], ["hp"], ["vg"], ["vhp"], ["eyes"], ["g", "vg"],
                                  ["hp", "vhp"], ["g", "hp", "vg", "vhp"]])
def test_attention_missing_keys_never_raise(drop):
    g = np.array([0.0, 0.0, 1.0])
    p = {"eyes": [0, 0, 0], "g": g, "hp": g, "vg": A.vect2cone([0, 0, 0], [0, 0, 5], 0.6, 20),
         "vhp": A.vect2cone([0, 0, 0], [0, 0, 5], 0.6, 20)}
    for k in drop:
        p.pop(k)
    out = A.Attention({7: p}, _objects())[7][1]
    assert set(out) == {"object", "attentionG", "total_Attention_G", "attentionH",
                        "total_Attention_H", "total_Attention"}
    if "eyes" in drop or {"g", "vg"} <= set(drop):
        assert out["attentionG"] is None and out["total_Attention_G"] is None
    if "eyes" in drop or {"hp", "vhp"} <= set(drop):
        assert out["attentionH"] is None and out["total_Attention_H"] is None
    if "eyes" in drop or {"g", "hp", "vg", "vhp"} <= set(drop):
        assert out["total_Attention"] is None
    else:
        assert 0.0 <= out["total_Attention"] <= 1.0


def test_attention_own_object_skipped_and_behind_scores_zero():
    g = np.array([0.0, 0.0, -1.0])  # looking away from both boxes
    out = A.Attention({0: {"eyes": [0, 0, 0], "g": g, "hp": g, "own": 2}}, _objects())[0]
    assert set(out) == {1}
    assert out[1]["total_Attention_G"] == 0.0 and out[1]["total_Attention"] == 0.0


def test_attention_far_boxes_stay_in_unit_range():
    shift = np.array([5000.0, -5000.0, 5000.0])
    base = dict(_objects())
    base[3] = {"name": "wall", "bbox": [np.array([2.0, 2.0, 6.0]), np.array([-2.0, -2.0, 5.0])]}
    objs = {k: {"name": v["name"], "bbox": [v["bbox"][0] + shift, v["bbox"][1] + shift]}
            for k, v in base.items()}
    g = np.array([0.05, 0.0, 1.0])
    out = A.Attention({0: {"eyes": shift, "g": g, "hp": g}}, objs)[0]
    for v in out.values():
        for key in ("total_Attention_G", "total_Attention_H", "total_Attention"):
            assert math.isfinite(v[key]) and 0.0 <= v[key] <= 1.0
    assert out[3]["total_Attention"] > 0.1
    assert A.calc_normalize_Attention(5.0, 0, 0.3, [1.9, 1.1], 1) == 0.0
