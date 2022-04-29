import sys
from torch import load, no_grad, zeros
from torch.nn import DataParallel
from torch.cuda import empty_cache as cudaEmptyCache
from time import time as timeSeconds
from numpy import array,float16
from math import cos, sin
from torch.utils.data import Dataset, DataLoader

from transforms import Resize, ToTensor, Normalize

sys.path.append("/home/carlos/Thesis_GOD/detectron2/gaze360")
from gaze360.code.model import GazeLSTM


IMAGE_NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class GazePredictor:
    def __init__(self, gazeModelFile, batchSize, uncertaintyThreshold):
        """
        :param gazeModelFile: path to mdel file.
        :param uncertaintyThreshold: number to filter the outputs using model uncertainty score
        """
        self.gazeModel = gazeModelFile
        self.batchSize = batchSize
        self.threshold = uncertaintyThreshold

    def __call__(self, inputs, mapFramesOutputs, fps):
        """
        :param inputs: video frames
        :param mapFramesOutputs: map of frames and features, to build the model inputs by cropping the frames using head boxes.
        :param fps: frames per second rate of the video
        :return: list of float16 3D vector
        """
        self.model = DataParallel(GazeLSTM()).cuda()
        self.model.load_state_dict(load(self.gazeModel)['state_dict'])
        self.model.eval()
        t = timeSeconds()
        loader = DataLoader(GazeDataSet(inputs, mapFramesOutputs, fps), batch_size=self.batchSize, pin_memory=True)
        gazesList = []
        nBatches = len(loader)
        step = max(1, len(loader) // 10)
        with no_grad():
            for i, data in enumerate(loader):
                if i % step == 0:
                    print("Gaze batch :", i + 1, "/", nBatches)
                outputs = self.model(data.cuda())
                gazes, uncertainties = outputs[0], outputs[1][:, 0]
                for gaze, uncertainty in zip(gazes, uncertainties):
                    if uncertainty.item() < self.threshold:
                        cosi = cos(gaze[1])
                        gazesList.append(
                            array([cosi * sin(gaze[0]), sin(gaze[1]), -cosi * cos(gaze[0])], dtype=float16))
                    else:
                        gazesList.append(None)
                del data, gazes, uncertainties, outputs
                cudaEmptyCache()
        t = timeSeconds() - t
        print("Gaze estimation :", t, "seconds;", len(loader.dataset) / t, "detected individuals per second")
        del loader.dataset, loader, self.model
        cudaEmptyCache()
        return gazesList

class GazeDataSet(Dataset):
    def __init__(self, frames, mapFramesOutputs, fps):
        self.mapFramesOutputs = mapFramesOutputs
        self.frames = frames
        self.framesStep = max(int(fps // 8), 1)
        self.inputs = []
        for frame, instances in mapFramesOutputs.items():
            for instance in instances.keys():
                bbox = self.mapFramesOutputs.getHeadBox(frame, instance)
                if bbox is not None:
                    self.inputs.append((frame, instance))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        """
        Returns a (1,7,3, 224,224) tensor : values of the 2nd dimension are resized cropped images of the instance.
        The fourth image of the sequence is the frame considered for input.
        Can be optimized by storing the cropped and transformed frames for further use in the loading loop.
        """
        input_image = zeros(7, 3, 224, 224)
        count = 0
        frame, instance=self.inputs[item]
        for image in range(frame - 3 * self.framesStep, frame+ 4 * self.framesStep, self.framesStep):
            im = self.frames[frame]
            bbox = self.mapFramesOutputs.getHeadBox(frame, instance)
            input_image[count, :, :, :] = IMAGE_NORMALIZE(ToTensor()(Resize((224, 224))(im[bbox[1]:bbox[3], bbox[0]:bbox[2]])))
            count += 1
        return input_image.view(1, 7, 3, 224, 224)
