"""Monocular depth with MiDaS (legacy thesis code; torch and MiDaS imported lazily).

MiDaS is found through env ``VFOA_MIDAS_ROOT`` (the MiDaS checkout), else
``$VFOA_THIRD_PARTY/MiDaS``.
"""

import _bootstrap  # noqa: F401  (adds the repo root to sys.path)

import logging
import os
import sys
from time import time as timeSeconds

import cv2

from vfoa.backends.base import BackendUnavailable

logger = logging.getLogger(__name__)


def _midas():
    """Import MiDaS lazily; returns (MidasNet, Resize, NormalizeImage, PrepareForNet)."""
    root = os.environ.get("VFOA_MIDAS_ROOT")
    if not root and os.environ.get("VFOA_THIRD_PARTY"):
        root = os.path.join(os.environ["VFOA_THIRD_PARTY"], "MiDaS")
    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.append(root)
    try:
        from midas.midas_net import MidasNet
        from midas.transforms import NormalizeImage, PrepareForNet, Resize
    except ImportError as exc:
        raise BackendUnavailable(
            "MiDaS is not available: set VFOA_MIDAS_ROOT to a MiDaS checkout "
            f"(or VFOA_THIRD_PARTY to a directory containing MiDaS) [{exc}]"
        ) from exc
    return MidasNet, Resize, NormalizeImage, PrepareForNet


class DepthPredictor_Mono:
    def __init__(self, depthModelFile, batchSize):
        """
        :param depthModelFile: path to model file.
        """
        import torch

        self.depthmodel = depthModelFile
        self.batchSize = batchSize
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    def __call__(self, inputs):
        """
        Run MonoDepthNN to compute depth maps.

        :param inputs: video frames
        :return: list of predictions of depth
        """
        import torch
        from torch.utils.data import DataLoader

        MidasNet = _midas()[0]
        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # load network
        self.model = MidasNet(self.depthmodel, non_negative=True)
        self.model.to(device)
        self.model.eval()
        t = timeSeconds()
        loader = DataLoader(DepthDataSet(inputs), batch_size=self.batchSize, pin_memory=True)
        depthList = []
        nBatches = len(loader)
        step = max(1, len(loader) // 10)
        with torch.no_grad():
            for i, data in enumerate(loader):
                if i % step == 0:
                    logger.info("Depth batch : %d / %d", i + 1, nBatches)
                image = torch.from_numpy(data.to("cpu").numpy()[0]).to(device).unsqueeze(0)
                prediction = self.model.forward(image)
                prediction = (
                    torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=inputs[0].shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                depthList.append(prediction)
                del data, prediction
                empty_cache()
        t = timeSeconds() - t
        logger.info("Depth estimator : %.3f seconds; %.2f estimated images per second",
                    t, len(loader.dataset) / t if t > 0 else 0.0)
        del loader.dataset, loader, self.model
        empty_cache()
        return depthList


class DepthDataSet:
    """Map-style dataset of frames (DataLoader only needs ``__len__``/``__getitem__``)."""

    def __init__(self, frames):
        from torchvision.transforms import Compose

        _, Resize, NormalizeImage, PrepareForNet = _midas()
        self.frames = frames
        self.transform = Compose(
            [
                Resize(
                    384,
                    384,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="upper_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        img = self.frames[item]
        """Read image and output RGB image (0-1).
        """
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        img_input = self.transform({"image": img})["image"]
        return (img_input)
