import numpy as np
import cv2
import onnxruntime
import sys
from torch.utils.data import Dataset, DataLoader
from time import time as timeSeconds
from numpy import array,float16
from torch.cuda import empty_cache as cudaEmptyCache
from matplotlib import pyplot as plt

class HeadPredictor:
    def __init__(self, headModelFile1, headModelFile2, batchSize):
        """
        :param HeadModelFile: path to mdel file.
        """
        self.Head1 = headModelFile1
        self.Head2 = headModelFile2
        self.batchSize = batchSize

    def __call__(self, inputs, mapFramesOutputs, fps):
        """
        :param inputs: video frames
        :param mapFramesOutputs: map of frames and features, to build the model inputs by cropping the frames using head boxes.
        :param fps: frames per second rate of the video
        :return: list of float16 3D vector
        """
        self.sess = onnxruntime.InferenceSession(f'{self.Head1}')
        self.sess2 = onnxruntime.InferenceSession(f'{self.Head2}')
        t = timeSeconds()
        loader = DataLoader(HeadDataSet(inputs, mapFramesOutputs, fps), batch_size=self.batchSize, pin_memory=True)
        HeadList = []
        nBatches = len(loader)
        step = max(1, len(loader) // 10)
        
        for i, data in enumerate(loader):
            if i % step == 0:
                print("Head batch :", i + 1, "/", nBatches)
            #get headpose
            for inst in range(data.shape[0]):
                image = data[inst,:,:,:,:].numpy()
                res1 = self.sess.run(["output"], {"input": image})[0]
                res2 = self.sess2.run(["output"], {"input": image})[0]
                yaw,pitch,roll = np.mean(np.vstack((res1,res2)), axis=0)
                HeadList.append(array([yaw, pitch, roll], dtype=float16))
            del data
            cudaEmptyCache()
        t = timeSeconds() - t
        print("Head Pose estimation :", t, "seconds;", len(loader.dataset) / t, "detected individuals per second")
        del self.sess2, self.sess, loader.dataset, loader
        cudaEmptyCache()
        return HeadList


class HeadDataSet(Dataset):
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
        Returns a (1, 3, 64,64) array: preprocess headpose model input
        frame -> cropped instance
        cropped instance -> resized to 64x64 -> normalized
        """
        frame, instance=self.inputs[item]
        im = self.frames[frame]
        bbox = self.mapFramesOutputs.getHeadBox(frame, instance)
        face_roi = im[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face_roi = cv2.resize(face_roi,(64,64))
        input_image = face_roi.transpose((2,0,1))
        input_image = np.expand_dims(input_image,axis=0)
        input_image = (input_image-447.5)/448
        input_image = input_image.astype(np.float32)
        return input_image