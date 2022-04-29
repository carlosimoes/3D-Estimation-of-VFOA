import sys
from time import time as timeSeconds
from numpy import where, sum, array, uint32
from torch import no_grad, as_tensor
from torch.cuda import empty_cache as cudaEmptyCache
from torch.cuda import device_count
from torch import no_grad
from torch.multiprocessing import Process, set_start_method, get_context
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader




sys.path.append("/home/carlos/Thesis_GOD/detectron2")
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.data.transforms import ResizeShortestEdge as transformResize
sys.path.append("/home/carlos/Thesis_GOD/detectron2/projects/DensePose")
from densepose.config import add_densepose_config

"""sys.path.append("/home/carlos/detectron2/projects/DensePose/")
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model"""
	

numberGpus = device_count()

class Predictor:
    def __init__(self, cfg, gpu):
        self.gpu = gpu
        self.cfg = cfg
        self.model = build_model(cfg).cuda(gpu)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

class DensePosePredictor:
    def __init__(self, configFile, poseModelFile, accuracyThreshold, batchSize):
        """
        :param configFile:
        :param poseModelFile:
        :param accuracyThreshold:
        :param batchSize:
        """

        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(configFile)
        cfg.MODEL.WEIGHTS = poseModelFile
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = accuracyThreshold
        cfg.freeze()
        self.cfg = cfg
        self.batchSize = batchSize
        if device_count() > 1:
            set_start_method("spawn", force=True)
            smp = get_context("spawn")
            self.queues = {}
            for device in range(device_count()):
                self.queues[device] = smp.Manager().list()
            self.procs = []
            for gpu in range(device_count()):
                cfgProc = cfg.clone()
                cfgProc.defrost()
                cfgProc.MODEL.DEVICE = "cuda:" + str(gpu)
                self.procs.append(DensePosePredictor.PredictWorker(cfgProc, self.queues[gpu], gpu))

    def __call__(self, images):
        results = []
        if device_count() > 1:
            median = int(round(len(images) / 2))
            batchesGpus = [images[0:median], images[median:]]
            i = 0
            for p in self.procs:
                p.dataLoader = DataLoader(DensePoseDataSet(batchesGpus[i], self.cfg), self.batchSize, num_workers=3*numberGpus)
                p.inputHeight, p.inputWidth = self.inputHeight, self.inputWidth
                p.start()
                i += 1
            t = timeSeconds()
            for p in self.procs:
                p.join()
            t = timeSeconds() - t
            print("Body and head bounding boxes prediction :", t, "seconds ;", len(images) / t, "frames per second")
            for queue in self.queues.values():
                outputs.extend(queue)
        else:
            self.model = build_model(self.cfg).cuda()
            self.model.eval()
            self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(self.cfg.MODEL.WEIGHTS)
            loader = DataLoader(DensePoseDataSet(images, self.cfg), self.batchSize)
            nBatches = len(loader)
            t = timeSeconds()
            self.inputHeight, self.inputWidth = images[0].shape[0:2]
            step=max(1,len(loader)//10)
            with no_grad():
                for i, batch in enumerate(loader):
                    if i % step ==0:
                        print("DP batch :", i + 1, "/", nBatches)
                    data = self.tensorToList(batch)
                    outputs = self.model(data)
                    results.extend(self.postProcess(outputs))
                    del data, outputs, batch
                    cudaEmptyCache()
            t = timeSeconds() - t
            print("Body and head bounding boxes prediction :", t, "seconds ;", len(images) / t, "frames per second")
            del self.model, loader.dataset, loader, self.cfg
            cudaEmptyCache()
        return results

    def postProcess(self, outputs):
        """
        Extracts the head box of each detected individual on each image, and removes tensors from GPU memory.
        :param outputs: iterable of model outputs.
        :return:
        """

        processedOutputs = []
        for output in outputs:
            instances = output['instances']
            boxes = instances.pred_boxes.tensor.to("cpu")
            resultInstances = []
            for j, box in enumerate(boxes.tolist()):
                box = [int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3]))]
                dimensions = box[3] - box[1], box[2] - box[0]
                S = interpolate(instances.pred_densepose.S[[j]], dimensions, mode="bilinear",
                                align_corners=False).argmax(dim=1)
                result = (interpolate(instances.pred_densepose.I[[j]], dimensions, mode="bilinear",
                                      align_corners=False).argmax(dim=1) * (S > 0).long()).squeeze(0)
                resultInstances.append(self.extractHeadBoxFromOutput(box, result.to("cpu").numpy()))
                del S, result, box
            del instances.pred_boxes.tensor, instances.pred_densepose.S, instances.pred_densepose.I, instances.pred_densepose.U, instances.pred_densepose.V, instances, output
            processedOutputs.append(resultInstances)
        return processedOutputs

    class PredictWorker(Process):
        def __init__(self, cfg, queue, i):
            self.predictor = None
            self.cfg = cfg
            self.resultQueue = queue
            self.dataLoader = None
            self.nProc = i
            self.t = 0
            self.inputWidth, self.inputHeight = 0, 0
            super().__init__()

        def run(self):
            self.predictor = Predictor(self.cfg, self.nProc)
            results = []
            with no_grad():
                for batch in self.dataLoader:
                    results.append(self.predictor.model(self.tensorToList(batch, self.nProc)))
            del self.predictor.model, self.dataLoader.dataset, self.dataLoader, self.cfg
            cudaEmptyCache()

        def tensorToList(self, concatTensors, gpu):
            concatTensors.cuda(gpu)
            batch = []
            for tensor in concatTensors:
                batch.append({"image": tensor, "width": self.inputWidth, "height": self.inputHeight})
            return batch

    @staticmethod
    def extractHeadBoxFromOutput(box, instance):
        bbox = None
        xInstance, yInstance = box[0:2]
        iuv_mask_head = instance[:, :] > 22
        if iuv_mask_head.any():
            xs = where(sum(iuv_mask_head, axis=0) > 0)[0]
            ys = where(sum(iuv_mask_head, axis=1) > 0)[0]
            if len(xs) > 0 and len(ys) > 0:
                x0 = xs[0]
                x1 = xs[-1]
                y0 = ys[0]
                y1 = ys[-1]
                w = (x1 - x0) * 0.15
                h = (y1 - y0) * 0.15
                bbox = array((max(0, x0 - w) + xInstance, max(0, y0 - h) + yInstance, max(0, x1 + w) + xInstance,
                              max(0, y1 + h) + yInstance), dtype=uint32)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    return None
        return bbox

    def tensorToList(self, concatTensors):
        """
        Moves a (N,H,W,3) tensor from RAM to GPU memory, and returns a list of dict objects, with an entry of (H,W,3) tensor, 
        to feed the Detectron model.
        """

        concatTensors.cuda()
        batch = []
        for tensor in concatTensors:
            batch.append({"image": tensor, "width": self.inputWidth, "height": self.inputHeight})
        return batch

class DensePoseDataSet(Dataset):
    def __init__(self, frames, cfg):
        self.frames = frames
        self.input_format = cfg.INPUT.FORMAT
        self.transform_gen = transformResize([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                                             cfg.INPUT.MAX_SIZE_TEST)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        img = self.frames[item][:, :, ::-1]
        return as_tensor(self.transform_gen.get_transform(img).apply_image(img).astype("float32").transpose(2, 0, 1))
