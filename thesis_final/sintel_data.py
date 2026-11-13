"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark (Python 3 port).

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release
v1.1: Python 3; depth/cam readers and writers delegate to ``vfoa.io.sintel``;
      PIL imported lazily.

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import numpy as np

from vfoa.io.sintel import TAG_FLOAT, cam_read, cam_write, depth_read, depth_write

__all__ = [
    "TAG_FLOAT", "TAG_CHAR",
    "flow_read", "flow_write", "depth_read", "depth_write", "cam_read", "cam_write",
    "disparity_read", "disparity_write", "segmentation_read", "segmentation_write",
]

# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_CHAR = b"PIEH"


def flow_read(filename):
    """ Read optical flow from file, return (U,V) tuple.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype="<f4", count=1)[0]
        if check != np.float32(TAG_FLOAT):
            raise ValueError(f"flow_read: wrong tag in flow file (should be {TAG_FLOAT}, is {check})")
        width = int(np.fromfile(f, dtype="<i4", count=1)[0])
        height = int(np.fromfile(f, dtype="<i4", count=1)[0])
        size = width * height
        if not (width > 0 and height > 0 and 1 < size < 100000000):
            raise ValueError(f"flow_read: wrong input size (width = {width}, height = {height})")
        tmp = np.fromfile(f, dtype="<f4", count=2 * size).astype(np.float32).reshape((height, width * 2))
    u = tmp[:, 0::2]
    v = tmp[:, 1::2]
    return u, v


def flow_write(filename, uv, v=None):
    """ Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.

    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    if v is None:
        uv = np.asarray(uv)
        if uv.ndim != 3 or uv.shape[2] != 2:
            raise ValueError(f"flow_write: uv must have shape (H, W, 2), got {uv.shape}")
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = np.asarray(uv)
        v = np.asarray(v)
    if u.shape != v.shape:
        raise ValueError(f"flow_write: u and v shapes differ: {u.shape} vs {v.shape}")
    height, width = u.shape
    tmp = np.empty((height, width * 2), dtype="<f4")
    tmp[:, 0::2] = u
    tmp[:, 1::2] = v
    with open(filename, "wb") as f:
        f.write(TAG_CHAR)
        f.write(np.array([width, height], dtype="<i4").tobytes())
        f.write(tmp.tobytes())


def disparity_write(filename, disparity, bitdepth=16):
    """ Write disparity to file.

    bitdepth can be either 16 (default) or 32.

    The maximum disparity is 1024, since the image width in Sintel
    is 1024.
    """
    from PIL import Image  # lazy optional dependency

    d = np.array(disparity, dtype=np.float64, copy=True)

    # Clip disparity.
    d[d > 1024] = 1024
    d[d < 0] = 0

    d_r = (d / 4.0).astype('uint8')
    d_g = ((d * (2.0 ** 6)) % 256).astype('uint8')

    out = np.zeros((d.shape[0], d.shape[1], 3), dtype='uint8')
    out[:, :, 0] = d_r
    out[:, :, 1] = d_g

    if bitdepth > 16:
        d_b = (d * (2 ** 14) % 256).astype('uint8')
        out[:, :, 2] = d_b

    Image.fromarray(out).save(filename, 'PNG')


def disparity_read(filename):
    """ Return disparity read from filename. """
    from PIL import Image  # lazy optional dependency

    f_in = np.array(Image.open(filename))
    d_r = f_in[:, :, 0].astype('float64')
    d_g = f_in[:, :, 1].astype('float64')
    d_b = f_in[:, :, 2].astype('float64')

    depth = d_r * 4 + d_g / (2 ** 6) + d_b / (2 ** 14)
    return depth


def segmentation_write(filename, segmentation):
    """ Write segmentation to file. """
    from PIL import Image  # lazy optional dependency

    segmentation_ = np.asarray(segmentation).astype('int64')
    seg_r = (segmentation_ // (256 ** 2)).astype('uint8')
    seg_g = ((segmentation_ % (256 ** 2)) // 256).astype('uint8')
    seg_b = (segmentation_ % 256).astype('uint8')

    out = np.zeros((segmentation_.shape[0], segmentation_.shape[1], 3), dtype='uint8')
    out[:, :, 0] = seg_r
    out[:, :, 1] = seg_g
    out[:, :, 2] = seg_b

    Image.fromarray(out).save(filename, 'PNG')


def segmentation_read(filename):
    """ Return segmentation read from filename. """
    from PIL import Image  # lazy optional dependency

    f_in = np.array(Image.open(filename))
    seg_r = f_in[:, :, 0].astype('int32')
    seg_g = f_in[:, :, 1].astype('int32')
    seg_b = f_in[:, :, 2].astype('int32')

    segmentation = (seg_r * 256 + seg_g) * 256 + seg_b
    return segmentation
