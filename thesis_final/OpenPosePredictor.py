import sys
from time import time as timeSeconds
from torch import load, from_numpy
from torch import no_grad as noGradientComputing
from torch.cuda import empty_cache as cudaEmptyCache
from torch.nn import DataParallel
from numpy import float16

from torch.utils.data import Dataset, DataLoader

sys.path.append("/home/carlos/Thesis_GOD/detectron2/Pose_Estimation")
from lib.network.rtpose_vgg import get_model
from lib.config.default import update_config 
from lib.config.default import _C as cfg
from lib.utils.common import CocoPart
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.datasets.preprocessing import vgg_preprocess, rtpose_preprocess
from lib.network import im_transform


'''
MS COCO annotation order:
0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
14: r knee		15: l ankle		16: r ankle

The order in this work:
(0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
17-'left_ear' )
'''

class OpenPosePredictor:
    bottomBodyPartsIndex = [CocoPart.RKnee.value, CocoPart.RAnkle.value, CocoPart.LKnee.value, CocoPart.LAnkle.value,
                            CocoPart.Background.value]

    def __init__(self, args):
        """
        :param args: attributes:
            - dev: str
            - opM: path to model file
            - opB: batch size
        """
        update_config(cfg, args)
        self.devices = args.dev
        self.weights = args.opM
        self.batchSize = args.opB
        self.model = None

    def initModelForLoop(self):
        """
        Called before a predictIteration().
        """
        self.model = get_model('vgg19')
        self.model.load_state_dict(load(self.weights))
        if len(self.devices) > 0:
            self.model = DataParallel(self.model).cuda()
        self.model.float()
        self.model.eval()

    def predictIteration(self, inputs):
        """
        Returns predicted features from inputs.
        :param inputs: iterable of images.
        """
        loader = DataLoader(OpenPoseDataSet(inputs, cfg), batch_size=self.batchSize, pin_memory=True)
        nBatches = len(loader)
        processedOutputs = []
        with noGradientComputing():
            if len(self.devices) > 0:
                for i,data in enumerate(loader):
                    print("OP batch :", i + 1, "/", nBatches)
                    heatmaps, pafs = self.model(data.cuda())[0][0:2]
                    heatMapsCPU, pafsCPU = heatmaps.cpu().data.numpy().transpose(0, 2, 3,
                                                                                 1), pafs.cpu().data.numpy().transpose(
                        0, 2, 3, 1)
                    for paf, heatMap in zip(pafsCPU, heatMapsCPU):
                        processedOutputs.append((paf_to_pose_cpp(paf, heatMap, cfg)))
                    del heatmaps, pafs, heatMapsCPU, pafsCPU, data
                    cudaEmptyCache()
            else:
                for data in loader:
                    output = self.model(data)
                    self.outputsModel.append(output[0][0:2])
                    self.semaphoreInputs.release()
                    del output
        self.retainUpperKeypoints(processedOutputs)
        del loader.dataset, loader
        return processedOutputs

    def __call__(self, inputs):
        """
        Initializes model and processes inputs, deletes model.
        """
        self.model = get_model('vgg19')
        self.model.load_state_dict(load(self.weights))
        if len(self.devices) > 1:
            self.model = DataParallel(self.model).cuda()
        else:
            self.model.cuda()
        self.model.float()
        self.model.eval()
        loader = DataLoader(OpenPoseDataSet(inputs, cfg), batch_size=self.batchSize, pin_memory=True)
        processedOutputs = []
        nBatches = len(loader)
        step=max(1,nBatches //10)
        ti = timeSeconds()
        with noGradientComputing():
            for i, data in enumerate(loader):
                if i%step==0:
                    print("OP batch :", i+1, "/", nBatches)
                heatmaps, pafs = self.model(data.cuda())[0][0:2]
                heatMapsCPU, pafsCPU = heatmaps.cpu().data.numpy().transpose(0, 2, 3,
                                                                                 1), pafs.cpu().data.numpy().transpose(
                        0, 2, 3, 1)
                for paf, heatMap in zip(pafsCPU, heatMapsCPU):
                    processedOutputs.append((paf_to_pose_cpp(paf, heatMap, cfg)))
                del heatmaps, pafs, heatMapsCPU, pafsCPU, data
                cudaEmptyCache()
        self.retainUpperKeypoints(processedOutputs)
        ti = timeSeconds() - ti
        print("Keypoints detection :", ti, "seconds;", len(inputs) / ti, "frames per second")
        del loader.dataset, loader, self.model
        cudaEmptyCache()
        return processedOutputs

    def retainUpperKeypoints(self,outputs):
        """
        Remove bottom body parts and casts the remaining to float 16-b type.
        """
        for output in outputs:
            for human in output:
                for index, part in human.body_parts.items():
                    if index in OpenPosePredictor.bottomBodyPartsIndex:
                        del part
                    else:
                        part.x = float16(part.x)
                        part.y = float16(part.y)

class OpenPoseDataSet(Dataset):
    def __init__(self, inputs, cfg):
        self.inp_size = cfg.DATASET.IMAGE_SIZE
        self.factor = cfg.MODEL.DOWNSAMPLE
        self.inputs = inputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return from_numpy(
            #vgg_preprocess(im_transform.crop_with_factor(self.inputs[item], self.inp_size, factor=self.factor,
            #                                            is_ceil=True)[0])).float()
            rtpose_preprocess(im_transform.crop_with_factor(self.inputs[item], self.inp_size, factor=self.factor,
                                is_ceil=True)[0])).float()
