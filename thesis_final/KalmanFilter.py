"""Legacy thesis Kalman filter module: thin wrapper over :mod:`vfoa.smoothing`.

``KalmanFilter3D(dt=...).Kalman(z)`` keeps the thesis API; the implementation
(with the issue #23 fixes) lives in :class:`vfoa.smoothing.KalmanFilter3D`.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

import logging

import numpy as np

from vfoa.smoothing import KalmanFilter3D

__all__ = ["KalmanFilter3D", "example"]

logger = logging.getLogger(__name__)


def example(path: str = "data.txt", dt: float = 0.033) -> np.ndarray:
    """Filter the 3D points in ``path`` (one ``x y z`` row per frame) and plot them."""
    kf = KalmanFilter3D(dt=dt)
    measurements = np.loadtxt(path)
    predictions = np.array([kf.Kalman(z) for z in measurements])
    error = np.sum((predictions - measurements[:, :3]) ** 2, axis=0)
    logger.info("squared error per axis: %s", error)
    import matplotlib.pyplot as plt  # lazy: optional dependency

    plt.plot(range(len(measurements)), measurements, label="Measurements")
    plt.plot(range(len(predictions)), predictions, label="Kalman Filter Prediction")
    plt.legend()
    plt.show()
    return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example()
