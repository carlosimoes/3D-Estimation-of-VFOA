from typing import Dict

class MapFramesOutputs:
    """
    Contains a map that links frames to its extracted features : each entry is a dictionary of instances (detected individuals), identified by a number, and their associated features, for now keypoitns, head boxes and gaze.
    """

    def __init__(self):
        self.map = dict()

    def addListEntries(self, indexFrame, listEntries):
        self.map[indexFrame] = listEntries

    def keys(self):
        return self.map.keys()

    def values(self):
        return self.map.values()

    def appendMap(self, map):
        """
        Appends a MapFramesOutputs to itself.
        :param map:
        :return:
        """
        currentLength = len(self.map)
        for i, entry in map.map.items():
            self.map[i + currentLength] = entry

    def addGazes(self, gazesList):
        """
        Adds gazes to the map, by order. Checks keypoints and head box entries.
        :param gazesList:
        :return:
        """
        rank = 0
        for frame, detectedInstances in self.map.items():
            for instance in detectedInstances.values():
                if "b" in instance:
                    gaze = gazesList[rank]
                    if gaze is not None:
                        instance["g"] = gaze
                    rank += 1
                    
    def addHeadPose(self, headList):
        """
        Adds headposes to the map, by order. Checks keypoints and head box entries.
        :param headList:
        :return:
        """
        rank = 0
        for frame, detectedInstances in self.map.items():
            for instance in detectedInstances.values():
                if "b" in instance:
                    headpose = headList[rank]
                    if headpose is not None:
                        instance["hp"] = headpose
                    rank += 1

    def getFrame(self, index):
        if index in self.map:
            return self.map[index]
        else:
            return None

    def getHeadBox(self, image, instance):
        instance: Dict = self.map[image][instance]
        if "b" in instance:
            return instance["b"]
        return None

    def getGaze(self, image, instance):
        instance: Dict = self.map[image][instance]
        if "g" in instance:
            return instance["g"]
        return None

    def getHeadPose(self, image, instance):
        instance: Dict = self.map[image][instance]
        if "hp" in instance:
            return instance["hp"]
        return None


    def getKeypoints(self, image, instance):
        instance: Dict = self.map[image][instance]
        if "kp" in instance:
            return instance["kp"]
        return None

    def addFeature(self, iFrame, iDetection, feature, value):
        self.map[iFrame][iDetection][feature] = value

    def __len__(self):
        return len(self.map)

    def items(self):
        return self.map.items()

    def __iter__(self):
        for instances in self.map.values():
            yield instances
