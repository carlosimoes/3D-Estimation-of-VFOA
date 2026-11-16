"""Open3D arrow/cone helpers of the thesis visualisation (legacy module).

The cone geometry is delegated to :func:`vfoa.geometry.cone_rays`; open3d is
imported lazily inside the functions that build meshes, so importing this
module needs neither open3d nor a display. Units are those of the camera frame
(metres); arrow sizes are metric.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import math

import numpy as np

from vfoa.geometry import cone_rays, normalize

_TINY = 1e-12


def _o3d():
    """Import open3d lazily (it is only needed for visualisation)."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Arrow_line visualisation needs the optional 'open3d' package (pip install open3d)"
        ) from exc
    return o3d


def draw_geometries(pcds, index):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d = _o3d()
    o3d.visualization.draw_geometries(pcds, window_name="Frame " + str(index))


def get_o3d_FOR(origin=(0, 0, 0), size=0.5):
    """
    Create a FOR that can be added to the open3d point cloud
    """
    o3d = _o3d()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    mesh_frame.translate(np.asarray(origin, dtype=np.float64))
    return mesh_frame


def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec ():
    """
    vec = np.asarray(vec, dtype=np.float64)
    return float(np.sqrt(np.sum(vec ** 2)))


def caculate_align_mat(pVec_Arr):
    """Proper rotation matrix taking +z to the direction of ``pVec_Arr``.

    Zero or non-finite input gives the identity; -z gives diag(1, -1, -1).
    """
    v = np.asarray(pVec_Arr, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.isfinite(v).all():
        return np.eye(3)
    scale = float(np.linalg.norm(v))
    if scale < _TINY:
        return np.eye(3)
    vx, vy, c = v / scale
    s2 = vx * vx + vy * vy
    if s2 < 1e-24:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, 0.0, vx], [0.0, 0.0, vy], [-vx, -vy, 0.0]])
    return np.eye(3) + K + (K @ K) * ((1.0 - c) / s2)


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def _safe_atan_ratio(a: float, b: float) -> float:
    """``atan(a / b)`` without dividing by zero."""
    if abs(b) > _TINY:
        return math.atan(a / b)
    if abs(a) <= _TINY:
        return 0.0
    return math.copysign(math.pi / 2.0, a)


def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the
    z axis vector of the original FOR. The first rotation that is
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis.

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec ():
    """
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    # Rotation over z axis of the FOR
    gamma = _safe_atan_ratio(vec[1], vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    # Rotate vec to calculate next rotation
    vec = (Rz.T @ vec.reshape(-1, 1)).reshape(-1)
    # Rotation over y axis of the FOR
    beta = _safe_atan_ratio(vec[0], vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return (Rz, Ry)


def create_arrow(scale=1.0):
    """
    Create an Open3D arrow along +z, starting at the origin, of total length ``scale``
    metres (cylinder 0.8 * scale, cone 0.2 * scale).
    """
    o3d = _o3d()
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 1.0
    if not (math.isfinite(scale) and scale > 0):
        scale = 1.0
    return o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02 * scale,
        cone_radius=0.04 * scale,
        cylinder_height=0.8 * scale,
        cone_height=0.2 * scale,
    )


def get_arrow(origin=(0, 0, 0), end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    The arrows have the length of the vector (metres); without ``end``/``vec``
    a 1 m arrow along +z is returned. Arguments are never mutated.
    """
    origin = np.array(origin, dtype=np.float64).reshape(3)
    rot = np.eye(3)
    length = 1.0
    if end is not None:
        vec = np.array(end, dtype=np.float64).reshape(3) - origin
    elif vec is not None:
        vec = np.array(vec, dtype=np.float64).reshape(3)
    if vec is not None:
        length = vector_magnitude(vec)
        rot = caculate_align_mat(vec)
    mesh = create_arrow(length)
    mesh.rotate(rot, center=np.array([0.0, 0.0, 0.0]))
    mesh.translate(origin)
    return mesh


def create_arrow_vis(init, direction):
    """Arrow representing ``direction`` starting at ``init``."""
    return get_arrow(init, vec=direction)


def vect2cone(X0, X1, R, n):
    """Legacy cone: ``n`` vectors from ``X0`` to a ring of radius ``R`` around ``X1``.

    The ring lies in the plane through ``X1`` orthogonal to ``X1 - X0``; each
    returned vector (``(n, 3)``) has length ``hypot(|X1 - X0|, R)`` and makes the
    angle ``atan2(R, |X1 - X0|)`` with the axis. Works for any axis direction
    and returns ``n`` distinct vectors (no duplicated endpoint).
    """
    axis = np.asarray(X1, dtype=np.float64).reshape(3) - np.asarray(X0, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(axis))
    if not (math.isfinite(length) and length >= _TINY):
        raise ValueError("vect2cone needs X1 != X0")
    R = float(R)
    return cone_rays(axis, math.degrees(math.atan2(R, length)), int(n)) * math.hypot(length, R)


def create_cone_arrow(origin, first_gaze, vectors, radius, numArrowCone, str_type, length=0.5):
    """Arrows for a direction (red) plus its cone (green for gaze, cyan for head pose).

    ``vectors`` are the cone ray directions; when ``None``/empty/all-zero they are
    built with :func:`vect2cone` (ring of ``radius`` at distance 300 along
    ``first_gaze``). Every arrow is ``length`` metres long. A zero or
    non-finite ``first_gaze`` gives ``[]``.
    """
    g = np.asarray(first_gaze, dtype=np.float64).reshape(-1) if first_gaze is not None else None
    if g is None or g.shape != (3,) or not np.isfinite(g).all() or np.linalg.norm(g) < _TINY:
        return []
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if not radius and not numArrowCone:
        numArrowCone = 20
        radius = 20 if str_type == "gaze" else 500
    ring_color = [0.1, 0.9, 0.1] if str_type == "gaze" else [0.1, 0.9, 0.9]

    arrow = []
    arrow_center = get_arrow(origin, vec=normalize(g) * length)
    arrow_center.paint_uniform_color([0.9, 0.1, 0.1])
    arrow.append(arrow_center)

    vecs = None if vectors is None else np.asarray(vectors, dtype=np.float64)
    if vecs is None or vecs.size == 0 or not np.any(vecs):
        vecs = vect2cone(origin, origin + 300.0 * g, radius, numArrowCone)
    for vect in vecs.reshape(-1, 3):
        if not np.isfinite(vect).all() or np.linalg.norm(vect) < _TINY:
            continue
        arrow_color = get_arrow(origin, vec=normalize(vect) * length)
        arrow_color.paint_uniform_color(ring_color)
        arrow.append(arrow_color)
    return arrow
