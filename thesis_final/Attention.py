"""Legacy thesis attention API (thin wrapper over :mod:`vfoa.attention`).

All public names of the thesis module are kept; the geometry and scoring live
once in :mod:`vfoa.attention` / :mod:`vfoa.geometry`. Fixed relative to the
thesis: rays only count in front of the eyes (#5), ``calc_ratio`` uses
``bbox[0] - bbox[1]`` (#4), the gaze/head weights no longer cancel - per-cue
scores are normalised to [0, 1] first and ``total_Attention`` is their weighted
mean (#3), missing cues give ``None`` instead of KeyError (#14), and a person's
own object (optional key ``"own"``) is skipped (#6).

Boxes are ``bbox = [max_bound, min_bound]`` in camera-frame units.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

from typing import Any, Optional

import numpy as np

from Arrow_line import vect2cone
from vfoa.attention import (
    AttentionParams,
    agreement_weights,
    box_ratio,
    ray_box_values,
    rays_score,
)
from vfoa.geometry import cone_rays, normalize, ray_aabb

__all__ = [
    "vect2cone",
    "intersection",
    "inside_box",
    "distance",
    "calcangle2value",
    "calc_intersection_points",
    "CalcAttention",
    "calc_ratio",
    "calc_normalize_Attention",
    "Attention",
]


def intersection(maximo, minimo, dir_inv, origin):
    """Slab test of one ray (given by its inverse direction) against one box.

    Returns ``(entry_point, exit_point)`` or ``(False, False)`` when the ray
    misses or the box is behind ``origin``. Infinite/zero entries of
    ``dir_inv`` mean a zero direction component.
    """
    inv = np.asarray(dir_inv, dtype=np.float64).reshape(3)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.isfinite(inv) & (inv != 0), 1.0 / np.where(inv != 0, inv, 1.0), 0.0)
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    hit, t_in, t_out = ray_aabb(o, d, np.asarray(minimo, float), np.asarray(maximo, float))
    if not bool(hit[0, 0]):
        return (False, False)
    return (o + t_in[0, 0] * d, o + t_out[0, 0] * d)


def inside_box(maximo, minimo, point) -> bool:
    """Whether ``point`` lies inside the box ``[minimo, maximo]`` (bounds included)."""
    p = np.asarray(point, dtype=np.float64).reshape(3)
    lo = np.minimum(np.asarray(minimo, float), np.asarray(maximo, float))
    hi = np.maximum(np.asarray(minimo, float), np.asarray(maximo, float))
    return bool(np.all(p >= lo) and np.all(p <= hi))


def distance(center_box, point):
    """Euclidean distance between two points."""
    return np.linalg.norm(np.asarray(point, float) - np.asarray(center_box, float), axis=0)


def calcangle2value(vectgaze, vecthead):
    """Thesis weights [gaze, head]: [1.9, 1.1] if the vectors are < 45 degrees apart, else [1.55, 1.45]."""
    return list(agreement_weights(np.asarray(vectgaze, float), np.asarray(vecthead, float)))


def calc_intersection_points(vects_gaze, bbox, origin):
    """Per-ray thesis value ``1.5 - |entry - centre| / half_diagonal`` (0 when the ray misses)."""
    rays = np.asarray(vects_gaze, dtype=np.float64).reshape(-1, 3)
    values, _, _ = ray_box_values(np.asarray(origin, float).reshape(3), rays, bbox[1], bbox[0])
    return values[:, 0].tolist()


def CalcAttention(list_inter, angles, type_gaze):
    """Sum of the per-ray values times the gaze (``type_gaze`` truthy) or head weight."""
    list_inter = np.array(list_inter, dtype=np.float64)
    w = angles[0] if type_gaze else angles[1]
    return float(np.sum(list_inter * w))


def calc_ratio(bbox) -> float:
    """Min half-extent over half diagonal of ``bbox = [max, min]`` (translation invariant)."""
    return float(box_ratio(bbox[1], bbox[0])[0])


def calc_normalize_Attention(valueAttention, number_rays, dist_ratio, angles, gaze_or_head):
    """Cue score in [0, 1]: the weight applied by :func:`CalcAttention` is divided out here."""
    w = float(angles[0] if gaze_or_head else angles[1])
    if w == 0 or not np.isfinite(w):
        return 0.0
    return rays_score(float(valueAttention) / w, number_rays, dist_ratio)


def _vec3(v: Any) -> Optional[np.ndarray]:
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.shape != (3,) or not np.isfinite(a).all():
        return None
    return a


def _cue(person: dict, dkey: str, rkey: str, half_angles, n: int):
    """``(direction, rays)`` of one cue, or ``(None, None)`` when unavailable."""
    d = _vec3(person.get(dkey))
    if d is not None and np.linalg.norm(d) <= 1e-12:
        d = None
    rays = person.get(rkey)
    r = None
    if rays is not None and np.size(rays) > 0:
        r = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
        r = r[np.isfinite(r).all(axis=1) & (np.linalg.norm(r, axis=1) > 1e-12)]
        if r.shape[0] == 0:
            r = None
    if r is None and d is not None:
        r = np.concatenate([cone_rays(d, a, n) for a in half_angles], axis=0)
    if d is None and r is not None:
        m = normalize(r).mean(axis=0)
        d = m if np.linalg.norm(m) > 1e-12 else None
    if r is None:
        return None, None
    return d, r


def Attention(ListPerson, ListObject):
    """Thesis attention dictionary.

    ``ListPerson``: ``{id: {"eyes": (3,), "g": vec, "hp": vec, "vg": rays, "vhp": rays,
    "own": object id to skip}}`` (every key optional). ``ListObject``:
    ``{id: {"name": str, "bbox": [max_bound, min_bound]}}``. Returns
    ``{person_id: {object_id: {"object", "attentionG", "total_Attention_G",
    "attentionH", "total_Attention_H", "total_Attention"}}}``; missing parts are None.
    """
    params = AttentionParams()
    result = {}
    for id_person, person in (ListPerson or {}).items():
        person = person or {}
        eyes = _vec3(person.get("eyes"))
        gd, gr = _cue(person, "g", "vg", params.half_angles_gaze_deg, params.rays_n)
        hd, hr = _cue(person, "hp", "vhp", params.half_angles_head_deg, params.rays_n)
        if gd is not None and hd is not None:
            angle = calcangle2value(gd, hd)
        else:
            angle = [1.0, 1.0]
        own = person.get("own")
        dici_object = {}
        for id_object, obj in (ListObject or {}).items():
            if own is not None and id_object == own:
                continue
            bbox = [np.asarray(obj["bbox"][0], float), np.asarray(obj["bbox"][1], float)]
            ratio = calc_ratio(bbox)
            entry: dict = {"object": obj.get("name")}
            totals = []
            for key, rays, flag in (("G", gr, 1), ("H", hr, 0)):
                if eyes is None or rays is None:
                    entry["attention" + key] = None
                    entry["total_Attention_" + key] = None
                    continue
                inter = calc_intersection_points(rays, bbox, eyes)
                value = CalcAttention(inter, angle, flag)
                score = calc_normalize_Attention(value, len(inter), ratio, angle, flag)
                entry["attention" + key] = inter
                entry["total_Attention_" + key] = score
                totals.append((angle[0] if flag else angle[1], score))
            if not totals:
                entry["total_Attention"] = None
            else:
                wsum = sum(w for w, _ in totals)
                entry["total_Attention"] = sum(w * s for w, s in totals) / wsum
            dici_object[id_object] = entry
        result[id_person] = dici_object
    return result


if __name__ == "__main__":
    # aabb max and min[ 124.698792   91.741074 -351.372559] [-145.305161  -83.609955 -420.657288]
    # vec = [-16.83, 5.605, -5.277]
    # init = [-30,60,-380]

    # need bird aabb
    list_objects={}
    list_box={}
    list_box["name"]="person"
    maximo=np.array([ 124.698792, 91.741074, -351.372559])
    minimo=np.array([-145.305161,  -83.609955, -420.657288])
    list_box["bbox"]=[maximo, minimo]
    list_objects[1]=list_box
    origin=np.array([-30,60,-380])
    dir_inv=np.array([-16.83, 5.605, -5.277])
    list_box={}
    list_box["name"]="bird"
    list_box["bbox"]=[np.array([ 143.248947,  28.42742 , -351.372559]), np.array([  21.062992 , -44.68504 , -378.059082])]
    list_objects[2]=list_box
    origin=np.array([-30,60,-380])
    dir_inv=np.array([-16.83, 5.605, -5.277])
    # vectors=vect2cone([-30,60,-380],np.array([-30,60,-380])+3*np.array([-16.83, 5.605, -5.277]), 20,20)
    list_person={}
    list_person2={}
    list_person2["g"]=[-16.83, 5.605, -5.277]
    list_person2["vg"]=vect2cone([-30,60,-380],np.array([-30,60,-380])+500*np.array([-16.83, 5.605, -5.277]), 500,20)
    list_person2["hp"]=[-0.1588, -0.2883, -0.9443]
    list_person2["vhp"]=vect2cone([-30,60,-380],np.array([-30,60,-380])+500*np.array([-16.83, 5.605, -5.277]), 20,20)
    list_person2["eyes"]=[-30,60,-380]
    list_person[0]=list_person2
    attention_vlues=Attention(list_person, list_objects)
    #print(list_objects)
    #print(list_person)
    print(attention_vlues[0][1])
    print(attention_vlues[0][2])
