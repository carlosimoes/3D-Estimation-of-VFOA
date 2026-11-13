"""Legacy stereo depth stage (thin wrapper over :class:`vfoa.backends.depth_stereo.StereoBMDepth`).

``Depth_Estimator(videoL, videoR, camfile, type_video)(out_video, init, fim)``
returns ``(list of 4x4 Q matrices, list of float32 disparity maps)`` for the
1-based frames ``init < i <= fim``. Disparities are REAL pixels
(``d = fx * B / Z``, 0 = invalid) and ``cv2.reprojectImageTo3D(d, Q)`` gives
camera-frame metres with positive Z. The written video holds the clean
(rectified) left frames.
"""

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import glob
import logging
import os
from time import perf_counter

import cv2
import numpy as np
import yaml

from vfoa.backends.depth_stereo import StereoBMDepth
from vfoa.config import load_config
from vfoa.io.sintel import cam_read
from vfoa.types import Intrinsics

logger = logging.getLogger(__name__)

#: Baseline of the robot stereo head in metres (legacy T = 204.089 mm).
ROBOT_BASELINE_M = 0.204089
#: Sintel stereo baseline in metres.
SINTEL_BASELINE_M = 0.1
#: Robot frames are processed at 75% of the calibration resolution.
ROBOT_SCALE = 0.75

_DEPTH_PARAMS = {
    "mpi_sintel": {
        "num_disparities": 160, "block_size": 15, "wls": "auto",
        "wls_lambda": 8000000.0, "wls_sigma": 1.5, "scale": 1.0,
    },
    "robot_video": {
        "num_disparities": 256, "block_size": 5, "wls": "auto",
        "wls_lambda": 80000.0, "wls_sigma": 1.3, "scale": 1.0,
    },
}


class Depth_Estimator:
    def __init__(self, videoL, videoR, camfile, type_video):
        if type_video not in _DEPTH_PARAMS:
            raise ValueError(f"type_video must be one of {sorted(_DEPTH_PARAMS)}, got {type_video!r}")
        self.type_video = type_video
        cfg = load_config(profile="thesis", overrides={"depth": dict(_DEPTH_PARAMS[type_video])})
        self._backend = StereoBMDepth(cfg)
        self.intrinsics = None
        self.Q = None
        if type_video == "robot_video":
            self.imagesL = videoL
            self.imagesR = videoR
            cam_dir = camfile if os.path.isdir(camfile) else os.path.dirname(camfile)
            C1, D1, R1, P1, width, height = self.read_config_camera(os.path.join(cam_dir, "left.yaml"))
            C2, D2, R2, P2, _, _ = self.read_config_camera(os.path.join(cam_dir, "right.yaml"))
            R = np.eye(3)
            T = np.array([ROBOT_BASELINE_M * 1000.0, 0.0, 0.0]).reshape(3, 1)  # millimetres, as calibrated
            # legacy naming: inputH holds the WIDTH and inputW the HEIGHT
            self.inputH = round(width * ROBOT_SCALE)
            self.inputW = round(height * ROBOT_SCALE)
            dim = (self.inputH, self.inputW)  # (width, height)
            K1 = np.array(C1, dtype=np.float64)
            K2 = np.array(C2, dtype=np.float64)
            K1[:2] *= ROBOT_SCALE
            K2[:2] *= ROBOT_SCALE
            R1, R2, P1, P2, Q = cv2.stereoRectify(K1, D1, K2, D2, dim, R, T, alpha=0)[:5]
            self.leftMapX, self.leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, dim, cv2.CV_16SC2)
            self.rightMapX, self.rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, dim, cv2.CV_16SC2)
            self.intrinsics = Intrinsics(
                P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2], dim[0], dim[1], baseline=ROBOT_BASELINE_M
            )
            self.Q = self.intrinsics.Q()
        else:
            self.dirL = videoL
            self.dirR = videoR
            self.cam = camfile

    def __call__(self, filepath_newfinalvideo, init, fim):
        """Disparities of the 1-based frames ``init < i <= fim``; writes the clean left frames."""
        matrices, output_disp = [], []
        writer = None
        t0 = perf_counter()
        try:
            for i, left, right, Q in self._frames(init, fim):
                clean_left, disp = self._depth_pair(left, right)
                matrices.append(Q)
                output_disp.append(disp)
                if writer is None:
                    h, w = clean_left.shape[:2]
                    writer = cv2.VideoWriter(
                        filepath_newfinalvideo, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
                    )
                writer.write(clean_left)
                logger.debug("DISP_STEREO frame %d", i)
        finally:
            if writer is not None:
                writer.release()
        dt = perf_counter() - t0
        n = len(output_disp)
        logger.info("Disparity of %d frames in %.3f s (%.1f fps)", n, dt, n / dt if dt > 0 else 0.0)
        return matrices, output_disp

    # ------------------------------------------------------------------ frames
    def _frames(self, init, fim):
        """Yield ``(i, left_bgr, right_bgr, Q)`` for the 1-based frames ``init < i <= fim``."""
        if self.type_video == "mpi_sintel":
            lefts = sorted(glob.glob(os.path.join(self.dirL, "*.png")))
            rights = sorted(glob.glob(os.path.join(self.dirR, "*.png")))
            cams = sorted(glob.glob(os.path.join(self.cam, "*.cam")))
            if not lefts or not rights:
                raise FileNotFoundError(
                    f"no .png stereo images found in {self.dirL!r} / {self.dirR!r}"
                )
            if not cams:
                raise FileNotFoundError(f"no .cam files found in {self.cam!r}")
            for i, (pl, pr, pc) in enumerate(zip(lefts, rights, cams), start=1):
                if i <= init or i > fim:
                    continue
                left, right = self._imread(pl), self._imread(pr)
                K, _ = cam_read(pc)
                h, w = left.shape[:2]
                Q = Intrinsics.from_K(K, w, h, baseline=SINTEL_BASELINE_M).Q()
                yield i, left, right, Q
            return
        capL = cv2.VideoCapture(self.imagesL)
        capR = cv2.VideoCapture(self.imagesR)
        try:
            if not capL.isOpened() or not capR.isOpened():
                raise FileNotFoundError(f"cannot open stereo videos {self.imagesL!r} / {self.imagesR!r}")
            i = 0
            while i < fim:
                okL, left = capL.read()
                okR, right = capR.read()
                if not okL or not okR:
                    break
                i += 1
                if i <= init:
                    continue
                yield i, left, right, self.Q.copy()
        finally:
            capL.release()
            capR.release()

    @staticmethod
    def _imread(path):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cannot read image {path!r}")
        return image

    def _depth_pair(self, left, right):
        """Rectify (robot) and match one pair; returns (clean left BGR, float32 disparity)."""
        if self.type_video == "robot_video":
            size = (self.inputH, self.inputW)
            left = cv2.resize(left, size)
            right = cv2.resize(right, size)
            left = cv2.remap(left, self.leftMapX, self.leftMapY, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            right = cv2.remap(right, self.rightMapX, self.rightMapY, cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return left, self._backend.disparity(left, right)

    # ------------------------------------------------------------ legacy API
    def main_depth(self, imageL, imageR):
        """Legacy entry: image paths (or arrays) -> (clean left BGR, float32 disparity)."""
        left = self._imread(imageL) if isinstance(imageL, (str, os.PathLike)) else imageL
        right = self._imread(imageR) if isinstance(imageR, (str, os.PathLike)) else imageR
        return self._depth_pair(left, right)

    def calculate_disp(self, imgL, imgR):
        """Rectified pair -> (float32 disparity in pixels, invalid mask)."""
        disp = self._backend.disparity(imgL, imgR)
        return disp, disp <= 0

    def calculate_disp_sintel(self, imgL, imgR):
        """Rectified Sintel pair -> (float32 disparity in pixels, invalid mask)."""
        return self.calculate_disp(imgL, imgR)

    def create_output(self, vertices, colors, filename):
        """ Create the ply file for the objects detected in PointCloud. """
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1, 3), colors])

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        '''

        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')

    def load_calibration(self, path):
        """ Loads stereo matrix coefficients. """
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        C1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        C2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        cv_file.release()
        return C1, D1, C2, D2, R, T

    def read_config_camera(self, input_file):
        """ A function to read YAML file"""
        with open(input_file) as f:
            config = yaml.safe_load(f)
        width = int(config['image_width'])
        height = int(config['image_height'])
        instrisec_matrix = np.array(config['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
        distortion_coefficients = np.array(config['distortion_coefficients']['data'], dtype=np.float64)
        rect_matrix = np.array(config['rectification_matrix']['data'], dtype=np.float64).reshape(3, 3)
        proj_matrix = np.array(config['projection_matrix']['data'], dtype=np.float64).reshape(3, 4)

        return instrisec_matrix, distortion_coefficients, rect_matrix, proj_matrix, width, height
