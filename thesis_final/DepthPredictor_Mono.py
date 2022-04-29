import sys
from torch import load, no_grad, zeros
from torch.nn import DataParallel
from torch.cuda import empty_cache as cudaEmptyCache
from time import time as timeSeconds
from numpy import array,float16
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from torchvision.transforms import Compose
from matplotlib import pyplot as plt

sys.path.append("/home/carlos/Thesis_GOD/detectron2/MiDaS")
import utils
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet


class DepthPredictor_Mono:
    def __init__(self, depthModelFile, batchSize):
        """
        :param depthModelFile: path to model file.
        """
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
        # select device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load network
        self.model = MidasNet(self.depthmodel, non_negative=True)
        self.model.to(device)
        self.model.eval()
        t = timeSeconds()
        loader = DataLoader(DepthDataSet(inputs), batch_size=self.batchSize, pin_memory=True)
        depthList = []
        nBatches = len(loader)
        step = max(1, len(loader) // 10)
        with no_grad():
            for i, data in enumerate(loader):
                if i % step == 0:
                    print("Depth batch :", i + 1, "/", nBatches)
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
                cudaEmptyCache()
        t = timeSeconds() - t
        print("Depth estimator :", t, "seconds;", len(loader.dataset) / t, "estimated images per second")
        del loader.dataset, loader, self.model
        cudaEmptyCache()
        return depthList

class DepthDataSet(Dataset):
    def __init__(self, frames):
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
