import sys
from time import time



from Data import MapFramesOutputs


sys.path.append("/home/carlos/Thesis_GOD/detectron2/Pose_Estimation")
from lib.utils.common import CocoPart



HEAD_PARTS = [CocoPart.REye.value, CocoPart.REar.value, CocoPart.LEye.value, CocoPart.LEar.value, CocoPart.Nose.value]
class Tracker:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.currentState={}

    def setDim(self, w, h):
        self.width, self.height = w, h

    def trackInterFrames(self, listFramesKeypoints: dict, listFramesBoxes):
        t = time()
        id_num = 0
        map = MapFramesOutputs()
        for i,listBoxes in enumerate(listFramesBoxes):
            currentBoxes = dict()
            for headBox in listBoxes:
                if headBox is not None:
                    id_val = self.find_id(headBox, self.currentState)
                    if id_val is None:
                        id_val = id_num
                        id_num += 1
                    currentBoxes[id_val] = {"b":headBox}
            self.currentState = currentBoxes
            map.map[i] = currentBoxes
        t = time() - t
        print("Interframe tracking :", t, " seconds ;", len(listFramesKeypoints) / t,
              " frames per second")
        return map

    def find_id(self, currentBox, boxes):
        id_final = None
        max_iou = 0.5
        for k, box in boxes.items():
            iou = self.intersectionOverUnionRatio(currentBox, box["b"])
            if iou > max_iou:
                id_final = k
                max_iou = iou
        return id_final

    def intersectionOverUnionRatio(self, bb1, bb2):
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        iou = intersection_area / float(
            (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]) + (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - intersection_area)
        if iou <= 0.0 or iou > (1.0 + 1e-8):
            return 0.0
        return iou

class TrackerMatcher(Tracker):
    def __init__(self):
        super().__init__()
        self.currentState = {}

    def trackInterFrames(self, listFramesKeypoints: dict, listFramesBoxes):
        t = time()
        id_num = 0
        tracking_id = MapFramesOutputs()
        for i,(listKeypoints, listBoxes) in enumerate(zip(listFramesKeypoints, listFramesBoxes)):
            mapOutputs = dict()
            mapKeypointsBoxes = self.matchKeypointsBoxes(listKeypoints, listBoxes)
            for keypoints, headBox in mapKeypointsBoxes.items():
                if headBox is not None:
                    id_val = self.findId(headBox)
                    if id_val is None:
                        id_val = id_num
                        id_num += 1
                    mapOutputs[id_val] = {"kp": keypoints, "b": headBox}
                    self.currentState[id_val] = headBox
                else:
                    id_val = id_num
                    id_num += 1
                    mapOutputs[id_val] = {"kp": keypoints}
            tracking_id.addListEntries(i, mapOutputs)
        t = time() - t
        print("Interframe tracking :", t, " seconds ;", len(listFramesKeypoints) / t,
              " frames per second")
        return tracking_id

    def matchKeypointsBoxes(self, listPoints, listBoxes):
        map = {}
        for human in listPoints:
            for headBox in listBoxes:
                boxIncludePoints = True
                if headBox is not None:
                    x1, x2 = headBox[0] / self.width, headBox[2] / self.width
                    y1, y2 = headBox[1] / self.height, headBox[3] / self.height
                    for part in [part for i, part in human.body_parts.items() if i in HEAD_PARTS]:
                        if part.x < x1 or part.x > x2 or part.y < y1 or part.y > y2:
                            boxIncludePoints = False
                            break
                    if boxIncludePoints:
                        map[human] = headBox
                        break
                    else:
                        map[human] = None
                else:
                    map[human] = None
        return map

    def findId(self, currentBox):
        id_final = None
        max_iou = 0.5
        for k, box in self.currentState.items():
            iou = self.intersectionOverUnionRatio(currentBox, box)
            if iou > max_iou:
                id_final = k
                max_iou = iou
        return id_final
