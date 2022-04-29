import cv2 
import os, ffmpeg
from numpy import reshape, concatenate, asarray, uint8, zeros, float16
import numpy as np
from random import randint as randomInt
from time import time as timeSeconds
from sys import maxsize, path
from torch.cuda import empty_cache 
from math import cos, sin
from matplotlib import pyplot as plt

import json

from DensePosePredictor import DensePosePredictor
from Data import MapFramesOutputs
from GazeFrameRenderer import OpenGLVectorRenderer
from GazePredictor1 import GazePredictor
from HeadPredictor import HeadPredictor
from OpenPosePredictor import OpenPosePredictor
from BoxTracker import Tracker, TrackerMatcher
from PanopticPrediction import PanopticPredictor
from Reconstruction_scene import Reconstruction



path.append("/home/carlos/Thesis_GOD/detectron2/Pose_Estimation")
from lib.utils.common import draw_humans
path.append("/home/carlos/Thesis_GOD/detectron2")
from detectron2.utils.visualizer import Visualizer


"""
Extracts information from videos about humans : body keypoints, body pose estimation and gaze estimation ; and serializes it or add it to input frames to produce a video
. Contains a GazeFrameRenderer to visualize the gaze arrow, and predictors. Intermediary and final results are stored in a MapFramesOutputs object, a dictionary mapping each frame to its extracted features.
"""

class VideoProcessor:
    def __init__(self, args, matrix_Q, disp,w=960, h=720):
        """
        :param args: an object with attributes for models configuration and output path.
        :param w:  width of the generated output video(s)
        :param h: height of the generated output video(s)
        Attributes:
        - gazeRenderer : an object that draws a 3D arrow on a blank image from a 3D vector.
        - boxesRandomColors : for coloring the borders of the predicted boxes, on generated images.
        - currentFrames: list of frames, last read
        - inputW, inputH: current video frames dimensions
        - outputW,outputH: dimensions of the output video(s)
        - mapFramesOutputs: results map
        """
        self.gazeRenderer = None
        self.config = args
        self.boxesRandomColors = [[randomInt(0, 254), randomInt(0, 254), randomInt(0, 254)] for i in range(1000)]
        self.currentFrames = None
        self.inputW = self.inputH = 0
        self.outputWidth = w
        self.outputHeight = h
        self.mapFramesOutputs = None
        self.tracker=None
        self.metadata=None
        self.matrix_Q = matrix_Q
        self.disp=disp
        self.directory=None

    def generateVideosWithFeatures(self, filePath: str, outputPath: str, indexBegin: int = 0,
                                   numberFrames: int = maxsize) -> None:
        """
        For test purposes on small videos : depends on the available RAM, processing 10000 frames requires 30 GB.
        :param filePath:
        :param outputPath:
        :param indexBegin:
        :param numberFrames:
        :return:
        """
        fps = self.readFrames(filePath, indexBegin, numberFrames)
        self.mapFramesOutputs = self.extractFeaturesFromCurrentVideo(fps)
        self.writeOutputsToFiles(self.mapFramesOutputs, "test.timestamp", fps)
        videoGazes = cv2.VideoWriter(outputPath + "-gazes.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                 (self.outputWidth, self.outputHeight))
        videoKeypoints = cv2.VideoWriter(outputPath + "-keypoints.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                     (self.outputWidth, self.outputHeight))
        self.gazeRenderer = OpenGLVectorRenderer(self.outputWidth, self.outputHeight, 140)
        self.gazeRenderer.compileShader()
        step = len(self.currentFrames) / 10
        for i in range(len(self.currentFrames)):
            if (i % step) == 0:
                print(i, "frames added to videos with features")
            imageGazes = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            imagePoints = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            if i in self.mapFramesOutputs.map:
                pointsList = []
                for id_t in self.mapFramesOutputs.getFrame(i).keys():
                    points = self.mapFramesOutputs.getKeypoints(i, id_t)
                    if points is not None:
                        pointsList.append(points)
                    bbox = self.mapFramesOutputs.getHeadBox(i, id_t)
                    if bbox is not None:
                        gaze = self.mapFramesOutputs.getGaze(i, id_t)
                        if gaze is not None:
                            imageGazes = self.getFullImage(imageGazes, bbox, gaze)
                if len(pointsList) > 0:
                    imagePoints = draw_humans(imagePoints, pointsList)
            videoKeypoints.write(imagePoints.astype(uint8))
            videoGazes.write(imageGazes.astype(uint8))
        videoGazes.release()
        videoKeypoints.release()

    def processVideo(self, filePath, imagesBatchLength=10000, extractGaze=True, writeVideo=True,matchPointsBox=False):
        """
        Reads a video, passes the frames into models of pose and gaze estimation, write files mapping the frames to the model otputs,
        and writes videos with the model outputs. A mosaic video is finally produced, that contains the original video, and the outputs videos.
        If the original video contains more frames than the value of the parameter imagesBatchLength, this processing is performed in a loop.
        :param filePath: path of the video file.
        :param imagesBatchLength: number of frames to be read and processed in one iteration of the processing loop.
        :param extractGaze: if True, extract head boxes and estimate gaze.
        :param writeVideo: if True, write video(s) of the extracted features, and a final mosaic video.
        :param matchPointsBox:
        :return:
        """
        #! INIT
        detections = MapFramesOutputs()
        cap = cv2.VideoCapture(filePath)
        nFramesVideo = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.inputW, self.inputH = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if matchPointsBox:
            self.tracker=TrackerMatcher()
        else:
            self.tracker=Tracker()
        self.tracker.setDim(self.inputW, self.inputH)
        type=str(cap.read()[1].dtype)
        cap.release()
        dir = self.config.od + filePath[filePath.rfind('/'):filePath.rfind('.')] + "/"
        self.directory=dir
        if nFramesVideo <= imagesBatchLength:
            print("process one run,", nFramesVideo, "frames")
            self.readFrames(filePath,numberFrames=nFramesVideo)
            if extractGaze:
                print("processing with gaze estimation")
                if matchPointsBox:
                    detections, panoptic = self.extractFeaturesFromCurrentVideo(fps, matchPointsBox)
                    self.writeTrackedOutputsToFilesSep(detections, dir, fps)
                else:
                    keypoints, detections, panoptic = self.extractFeaturesFromCurrentVideo(fps, matchPointsBox)
                    self.writeOutputsToFilesSep(keypoints,detections,dir,fps)
            else:
                print("Keypoints estimation")
                keypointsPredictor = OpenPosePredictor(self.config)
                detections = keypointsPredictor(self.currentFrames)
                del keypointsPredictor.model, keypointsPredictor
                empty_cache()
                self.writeKeypointsToFile(detections, dir, fps)
        else:
            print("process",nFramesVideo,"frames", int(round(nFramesVideo / imagesBatchLength)), "runs")
            iter = 0
            if extractGaze:
                print("whole processing, with gaze estimation")
                if matchPointsBox:
                    for indexFrame in range(0, nFramesVideo, imagesBatchLength):
                        print("\nRun",iter+1)
                        self.readFrames(filePath, indexFrame, imagesBatchLength)
                        print(indexFrame)
                        currentDetections = self.extractFeaturesFromCurrentVideo(fps,matchPointsBox)
                        detections.appendMap(currentDetections)
                        #self.writeTempFile(dir,fps,iter,detections=currentDetections)
                        iter+=1
                        self.currentFrames=None
                        print("\n")
                    self.writeTrackedOutputsToFilesSep(detections, dir, fps)
                else:
                    keypoints = []
                    for indexFrame in range(0, nFramesVideo, imagesBatchLength):
                        print("\nRun", iter + 1)
                        self.readFrames(filePath, indexFrame, imagesBatchLength)
                        points, boxesGazes = self.extractFeaturesFromCurrentVideo(fps, matchPointsBox)
                        keypoints.extend(points)
                        detections.appendMap(boxesGazes)
                        #self.writeTempFile(dir,fps,iter,points=points,boxesGazes=boxesGazes)
                        iter += 1
                        self.currentFrames = None
                        print("\n")
                    self.writeOutputsToFilesSep(keypoints,detections, dir, fps)
            else:
                print("Keypoints estimation")
                keypointsPredictor = OpenPosePredictor(self.config)
                keypointsPredictor.initModelForLoop()
                keypoints=[]
                for indexFrame in range(0, nFramesVideo, imagesBatchLength):
                    print("\nRun", iter + 1)
                    self.readFrames(filePath, indexFrame, imagesBatchLength)
                    keypoints.extend(keypointsPredictor.predictIteration(self.currentFrames))
                    #self.writeTempFile(dir,fps,iter,keypointsPredictor.predictIteration(self.currentFrames))
                    iter+=1
                    self.currentFrames = None
                    print("\n")
                del keypointsPredictor.model, keypointsPredictor
                empty_cache()
                self.writeKeypointsToFile(detections, dir, fps)
            #self.mergeTempFiles(dir)
        self.writeVideoDesc(fps,nFramesVideo,type,dir)
        if writeVideo:
            if extractGaze:
                print("Write gaze and keypoints videos")
                if matchPointsBox:
                    self.writeTrackedOutputsToVideos(detections, dir, fps, nFramesVideo, panoptic) #TODO
                else:
                    self.writeOutputsToVideos(keypoints,detections,dir,fps,nFramesVideo, panoptic)
            else:
                print("Write keypoints video")
                self.writeKeypointsToVideo(detections, dir, fps, nFramesVideo)
            print("Write mosaic video")
            self.writeMosaicVideo(filePath,dir,fps,nFramesVideo,extractGaze)

    def writeMosaicVideo(self,rawPath,dir,fps,nFrames,withGaze):
        """
        Writes a video composed of the original on the left side, the predicted keypoints on the right top and the predicted gaze on the right bottom.
        Writes first a blank video to serve as background for the overlaying videos. Uses the python fffmpeg interface.
        :param rawPath: path of the original video file
        :param dir: output directory
        :param fps: frames per second of the original video
        :param nFrames: total number of frames of the original video
        :param withGaze: boolean, if True, add gaze video to mosaic.
        """
        w=cv2.VideoWriter(dir + "null.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),(1920, 1460))
        blank=zeros((1460, 1920, 3), uint8)
        for i in range(nFrames):
            w.write(blank)
        w.release()
        stream=ffmpeg.input(dir + "null.mp4").overlay(ffmpeg.input(rawPath).filter('setpts', "PTS-STARTPTS").filter('scale', 960, 720), x=0,y=0)\
            .overlay(ffmpeg.input(dir + "keypoints.mp4").filter('setpts',"PTS-STARTPTS"), x=960,y=730)
        if withGaze:
            stream=stream.overlay(ffmpeg.input(dir+"gazes.mp4").filter('setpts',"PTS-STARTPTS"),x=0,y=730)
        if 1:
            stream=stream.overlay(ffmpeg.input(dir+"headpose.mp4").filter('setpts',"PTS-STARTPTS"),x=960,y=0)
        stream.output(dir+"mosaic.mp4",vcodec="libx264").run()
        os.remove(dir+"null.mp4")

    def readFrames(self, filePath, indexBegin=0, numberFrames=maxsize):
        """
        Reads a video file and stores the frames from indexBegin to numberFrames into the attribtue self.currentFrames.
        :param filePath: path of the video file
        :param indexBegin: index of the first frame to store
        :param numberFrames: number of frames to store
        """
        self.currentFrames = []
        cap = cv2.VideoCapture(filePath)
        i = 0
        while i < indexBegin:
            read, _ = cap.read()
            if not read:
                break
            i += 1
        i = 0
        while i < numberFrames:
            read, img = cap.read()
            if not read:
                break
            self.currentFrames.append(img)
            i += 1
        cap.release()
        return i

    def extractFeaturesFromCurrentVideo(self, fps,matchPointsBox,frameBegin=0):
        """
        Extracts features from the frames currently stored in self.currentFrames : predicts the upper body keypoints, the head boxes, matches them for the same
        individual and builds a map of each frame and its extracted features. Then estimates gaze and adds it to the map, and returns the map.
        :param fps: frames per second of the video
        :return:
        """
        t = timeSeconds()
        # ! OpenPose 
        # Test NOW
        print("----------------------------")
        print("----------------------------")

        keypointsPredictor = OpenPosePredictor(self.config)
        keypoints = keypointsPredictor(self.currentFrames)
        del keypointsPredictor
        empty_cache()
        # ! DensePose 
        boundingBoxesPredictor = DensePosePredictor(self.config.dpCfg, self.config.dpM, self.config.dpT,
                                                    self.config.dpB)
        boxes = boundingBoxesPredictor(self.currentFrames)
        del boundingBoxesPredictor
        empty_cache()
        # ! Gaze Direction
        mapFramesOutputs = self.tracker.trackInterFrames(keypoints, boxes)
        gazePredictor: GazePredictor = GazePredictor(self.config.gM, self.config.gB,
                                                     uncertaintyThreshold=self.config.gT)
        gazesList = gazePredictor(self.currentFrames, mapFramesOutputs, fps)
        del gazePredictor
        empty_cache()
        # ! Head Pose
        # TODO: validation
        headPredictor: HeadPredictor = HeadPredictor(self.config.hM1,self.config.hM2, self.config.gB)
        headList = headPredictor(self.currentFrames, mapFramesOutputs, fps)
        del headPredictor
        empty_cache()
        mapFramesOutputs.addGazes(gazesList)
        mapFramesOutputs.addHeadPose(headList)
        
        # ! Panoptic
        panopticPredictor = PanopticPredictor(self.config.PpCfg, self.config.PpM, self.config.PpT,
                                                    self.config.PpB)
        all_seg, self.metadata = panopticPredictor(self.currentFrames)
        del panopticPredictor
        empty_cache()
        #! Reconstruction
        reconst_3d = Reconstruction(self.metadata, self.matrix_Q, self.directory)
        points = reconst_3d(self.currentFrames, self.disp, all_seg, mapFramesOutputs)
        del reconst_3d
        t = timeSeconds() - t
        print("Total extraction length :", t, " seconds ; ", len(self.currentFrames) / t, "frames per second")
        # TODO: return the all segmentation of panoptic
        if matchPointsBox:
            return mapFramesOutputs, all_seg
        else:
            return keypoints,mapFramesOutputs, all_seg

    def writeVideoDesc(self,fps,nFrames,type,dir):
        """
        Writes a description file of the processed video : contains the array type of the frames, their width and height, their number and their rate.
        :param fps: frames per second of the original video
        :param nFrames: number of frames of the original video
        :param type: number array type of the video frames
        :param dir: output directory
        :return:
        """
        out=open(dir+"features.desc","w")
        out.write(str({"FrameType":type,"Width":int(self.inputW),"Height":int(self.inputH),"NumberFrames":nFrames,"FrameRate":int(fps)}))
        out.close()

    def writeKeypointsToFile(self, listsKeypoints, dir, fps):
        """
        Writes the predicted keypoints into a file : each files begins with a timestamp, then the JSON reprezentation of the features.
        :param listsKeypoints:
        :param dir:
        :param fps:
        :return:
        """
        fileString = ""
        step = float16(1.0 / fps)
        initTime = timeSeconds()
        for list in listsKeypoints:
            jsonString = {}
            for j, detection in enumerate(list):
                partsDict = {}
                for i, part in detection.body_pparts.items():
                    partsDict[i] = {"x": part.x, "y": part.y}
                jsonString[j] = partsDict
            fileString += str(round(initTime, 3)) + " " + str(jsonString) + "\n"
            initTime += step
        out = open(dir + "keypoints.timestamp", 'w')
        out.write(fileString)
        out.close()

    def writeTrackedOutputsToFile(self, mapFramesOutputs, dir, fps):
        """
        Writes the predicted features (keypoints, head boxes, gaze) in a file : each line begons with a timestamp, then the JSON representation of the features.
        :param mapFramesOutputs:
        :param dir:
        :param fps:
        :return:
        """
        fileString = ""
        step = float16(1.0 / fps)
        initTime = timeSeconds()
        for frame, listInstances in mapFramesOutputs.items():
            jsonString = {}
            if listInstances:
                for instanceId, features in listInstances.items():
                    instanceString = {}
                    if "kp" in features:
                        parts = features["kp"].body_parts
                        if parts:
                            partsDict = {}
                            for i, part in parts.items():
                                partsDict[i] = {"x": part.x, "y": part.y}
                            instanceString["kp"] = partsDict
                    if "b" in features:
                        instanceString["b"] = list(map(int, features["b"]))
                    if "g" in features:
                        instanceString["g"] = list(features["g"])
                    if "hp" in features:
                        instanceString["hp"] = list(features["hp"])
                    jsonString[instanceId] = instanceString
            fileString += str(round(initTime, 3)) + " " + str(jsonString) + "\n"
            initTime += step
        out = open(dir + "features.timestamp", 'w')
        out.write(fileString)
        out.close()
    
    def writeTrackedOutputsToFilesSep(self, mapFramesOutputs, dir, fps):
        """
        Writes the predicted features (keypoints, head boxes, gaze, head pose) in three files. Each line describes a frame, begins with a timestamp, then the string representation of the features map.
        The features are a map of individuals index ID and its associated feature. Across the files, an index ID refers to the same individual.
        :param mapFramesOutputs: MapFramesOutputs object, with keypoints, boxes and gazes as outputs.
        :return:
        """
        print("Write matched features with timestamp in text files")
        fileString1, fileString2, fileString3, fileString4= "", "", "", ""
        step = float16(1.0 / fps)
        initTime = timeSeconds()
        for frame, listInstances in mapFramesOutputs.items():
            jsonString1, jsonString2, jsonString3, jsonString4 = {}, {}, {}, {}
            if listInstances:
                for instanceId, features in listInstances.items():
                    jsonString1[instanceId], jsonString2[instanceId], jsonString3[instanceId], jsonString4[instanceId] = {}, {}, {}, {}
                    if "kp" in features:
                        for i, part in features["kp"].body_parts.items():
                            jsonString1[instanceId][i] = [part.x, part.y]
                    if "b" in features:
                        jsonString2[instanceId]["b"] = [coord for coord in features["b"]]
                        if "g" in features:
                            jsonString3[instanceId]["g"] = [coord for coord in features["g"]]
                        if "hp" in features:
                            jsonString4[instanceId]["hp"] = [coord for coord in features["hp"]]
            fileString1 += str(round(initTime, 3)) + " " + str(jsonString1) + "\n"
            fileString2 += str(round(initTime, 3)) + " " + str(jsonString2) + "\n"
            fileString3 += str(round(initTime, 3)) + " " + str(jsonString3) + "\n"
            fileString4 += str(round(initTime, 3)) + " " + str(jsonString4) + "\n"
            initTime += step
        out1, out2, out3, out4 = open(dir + "keypoints" + ".timestamp", 'w'), open(
            dir + "boxes" + ".timestamp", 'w'), open(dir + "gazes" + ".timestamp",'w'), open(dir + "headpose" + ".timestamp",'w')
        out1.write(fileString1)
        out2.write(fileString2)
        out3.write(fileString3)
        out4.write(fileString4)
        out1.close()
        out2.close()
        out3.close()
        out4.close()

    def writeOutputsToFilesSep(self, keypoints, mapBoxesGazes, dir, fps):
        """
        Writes the predicted features (keypoints, head boxes, gaze) in three files. Each line describes a frame, begins with a timestamp, then the string representation of the features map.
        The boxes and gazes features are a map of individuals index ID and its associated feature. Across the boxes and gazes files, an index ID refers to the same individual.
        :param keypoints: list of lists of Human Object with keypoints attribute (dict).
        :param mapBoxesGazes: MapFramesOutputs object, with boxes and gazes as outputs.
        """
        print("Write features with timestamp in text files")
        #! write outputs
        fileString1, fileString2, fileString3,  fileString4 = "", "", "", ""
        step = float16(1.0 / fps)
        initTime = timeSeconds()
        for listInstances, listBoxesGazes in zip(keypoints, mapBoxesGazes.map.values()):
            jsonString1, jsonString2, jsonString3, jsonString4 = [], {}, {}, {}
            for id, human in enumerate(listInstances):
                pointsList = {}
                for i, part in human.body_parts.items():
                    pointsList[i] = [part.x, part.y]
                jsonString1.append(pointsList)
            for id, features in listBoxesGazes.items():
                jsonString2[id] = [coord for coord in features["b"]]
                if "g" in features:
                    jsonString3[id] = [coord for coord in features["g"]]
                if "g" in features:
                    jsonString4[id] = [coord for coord in features["hp"]]
            fileString1 += str(round(initTime, 3)) + " " + str(jsonString1) + "\n"
            fileString2 += str(round(initTime, 3)) + " " + str(jsonString2) + "\n"
            fileString3 += str(round(initTime, 3)) + " " + str(jsonString3) + "\n"
            fileString4 += str(round(initTime, 3)) + " " + str(jsonString4) + "\n"
            initTime += step
        out1, out2, out3, out4 = open(dir + "keypoints" + ".timestamp", 'w'), open(
            dir + "boxes" + ".timestamp", 'w'), open(dir + "gazes" + ".timestamp",'w'), open(dir + "headpose" + ".timestamp",'w')
        out1.write(fileString1)
        out2.write(fileString2)
        out3.write(fileString3)
        out4.write(fileString4)
        out1.close()
        out2.close()
        out3.close()
        out4.close()

    def writeOutputsToFiles(self,keypoints,mapBoxesGazes,dir,fps,suffix=""):
        #! write to files v1
        fileString1,fileString2 = "",""
        step = float16(1.0 / fps)
        initTime = timeSeconds()
        for listInstances,listBoxesGazes in zip(keypoints,mapBoxesGazes.map.values()):
            jsonString1,jsonString2 = {},{}
            for id, human in enumerate(listInstances):
                jsonString1[id]={}
                for i, part in human.body_parts.items():
                    jsonString1[id][i] = [part.x, part.y]
            for id,features in listBoxesGazes.items():
                jsonString2[id] = {"b":[coord for coord in features["b"]]}
                if "g" in features:
                    jsonString2[id]["g"]=[coord for coord in features["g"]]
                if "hp" in features:
                    jsonString2[id]["hp"]=[coord for coord in features["hp"]]
            fileString1 += str(round(initTime, 3)) + " " + str(jsonString1) + "\n"
            fileString2 += str(round(initTime, 3)) + " " + str(jsonString2) + "\n"
            initTime += step
        out1,out2 = open(dir + "keypoints.timestamp", 'w'),open(dir + "boxes_gazes.timestamp", 'w')
        out1.write(fileString1)
        out2.write(fileString2)
        out1.close()
        out2.close()

    def writeTrackedOutputsToVideos(self, map, dir, fps, nFrames, panoptic):
        """
        Writes the predicted keypoints and gazes of a video on 2 separate videos, on a black background.
        :param map:
        :param dir:
        :param fps:
        :param nFrames:
        :return:
        """
        videoGazes = cv2.VideoWriter(dir + "gazes.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                 (self.outputWidth, self.outputHeight))
        videoKeypoints = cv2.VideoWriter(dir + "keypoints.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                     (self.outputWidth, self.outputHeight))
        videoHeadPose = cv2.VideoWriter(dir + "headpose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                        (self.outputWidth, self.outputHeight))   
        videoPanoptic = cv2.VideoWriter(dir + "panoptic.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                        (self.outputWidth, self.outputHeight))        
        self.gazeRenderer = OpenGLVectorRenderer(self.outputWidth, self.outputHeight, 140)
        self.gazeRenderer.compileShader()
        step = nFrames // 10
        for i in range(nFrames):
            if (i % step) == 0:
                print(i, "frames added to videos with features")
            imageGazes = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            imagePoints = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            imageHead = self.currentFrames[i]
            imagePanoptic = self.currentFrames[i]
            if i in map.map:
                pointsList = []
                for id_t in map.getFrame(i).keys():
                    points = map.getKeypoints(i, id_t)
                    if points is not None:
                        pointsList.append(points)
                    bbox = map.getHeadBox(i, id_t)
                    if bbox is not None:
                        gaze = map.getGaze(i, id_t)
                        head = map.getHeadPose(i, id_t)
                        if gaze is not None:
                            imageGazes = self.getFullImage(imageGazes, bbox, gaze)
                        if head is not None:
                            imageHead = self.getFullImageHP(imageHead, bbox, head)
                if len(pointsList) > 0:
                    imagePoints = draw_humans(imagePoints, pointsList)
                imagePanoptic=self.draw_panoptic(panoptic[i],imagePanoptic)

            videoPanoptic.write(cv2.resize(imagePanoptic, (self.outputWidth, self.outputHeight), interpolation =cv2.INTER_AREA))
            videoHeadPose.write(cv2.resize(imageHead, (self.outputWidth, self.outputHeight), interpolation =cv2.INTER_AREA))
            videoKeypoints.write(imagePoints.astype(uint8))
            videoGazes.write(imageGazes.astype(uint8))
        videoGazes.release()
        videoKeypoints.release()
        videoPanoptic.release()
        videoHeadPose.release()

    def writeOutputsToVideos(self,keypoints,map,dir,fps,nFrames, panoptic):
        """
                Writes the predicted keypoints and gazes of a video on 2 separate videos, on a black background.
                :param map:
                :param dir:
                :param fps:
                :param nFrames:
                :return:
                """
        #listaH, listaG =self.objects2d(keypoints,map, panoptic, nFrames)
        #self.writeobjects(listaG, listaH, dir)
        videoGazes = cv2.VideoWriter(dir + "gazes.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                    (self.outputWidth, self.outputHeight))
        videoKeypoints = cv2.VideoWriter(dir + "keypoints.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                        (self.outputWidth, self.outputHeight))
        videoHeadPose = cv2.VideoWriter(dir + "headpose.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                        (self.outputWidth, self.outputHeight)) 
        videoPanoptic = cv2.VideoWriter(dir + "panoptic.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                        (self.outputWidth, self.outputHeight))                              
        self.gazeRenderer = OpenGLVectorRenderer(self.outputWidth, self.outputHeight, 140)
        self.gazeRenderer.compileShader()
        step = nFrames // 10
        for i,(listHumans,listBoxesGazes) in enumerate(zip(keypoints,map.items())):
            if (i % step) == 0:
                print(i, "frames added to videos with features")
            #! black background or frame
            imageGazes = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            imageHead = np.copy(self.currentFrames[i])
            imagePanoptic = np.copy(self.currentFrames[i])
            for id_t in listBoxesGazes[1].keys():
                bbox = map.getHeadBox(i, id_t)
                if bbox is not None:
                    gaze = map.getGaze(i, id_t)
                    head = map.getHeadPose(i, id_t)
                    if gaze is not None:
                        imageGazes = self.getFullImage(imageGazes, bbox, gaze)
                    if head is not None:
                        imageHead = self.getFullImageHP(imageHead, bbox, head)
            imagePanoptic=self.draw_panoptic(panoptic[i],imagePanoptic)

            videoPanoptic.write(cv2.resize(imagePanoptic, (self.outputWidth, self.outputHeight), interpolation =cv2.INTER_AREA))
            videoHeadPose.write(cv2.resize(imageHead, (self.outputWidth, self.outputHeight), interpolation =cv2.INTER_AREA))
            videoGazes.write(imageGazes.astype(uint8))
            videoKeypoints.write(draw_humans(zeros((self.outputHeight, self.outputWidth, 3), uint8), listHumans).astype(uint8))
        videoGazes.release()
        videoKeypoints.release()
        videoHeadPose.release()
        videoPanoptic.release()

    def writeKeypointsToVideo(self, listsKeypoints, dir, fps, nFrames):
        """
        Writes predicted keypoints in a video file, on a black background.
        :param listsKeypoints:
        :param dir:
        :param fps:
        :param nFrames:
        :return:
        """
        videoKeypoints = cv2.VideoWriter(dir + "keypoints.mp4", cv2.VideoWriter_fourcc(*'mp4v'), round(fps),
                                     (self.outputWidth, self.outputHeight))
        step = nFrames / 10
        for i, listHumans in enumerate(listsKeypoints):
            if (i % step) == 0:
                print(i, "frames added to videos with features")
            imagePoints = zeros((self.outputHeight, self.outputWidth, 3), uint8)
            if len(listHumans) > 0:
                imagePoints = draw_humans(imagePoints, listHumans)
            videoKeypoints.write(imagePoints.astype(uint8))
        videoKeypoints.release()

    def getFullImage(self, image, bbox, gaze):
        """
        Overlays an arrow that represents the gaze estimation on an image.
        :param image:
        :param bbox:
        :param gaze:
        :return:
        """
        bbox = asarray(bbox).astype(float)
        bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] / self.inputW, bbox[1] / self.inputH, bbox[2] / self.inputW, bbox[
            3] / self.inputH
        eyes = [((bbox[0] + bbox[2]) / 2), (0.65 * bbox[1] + 0.35 * bbox[3])]
        arrow = self.gazeRenderer.renderArrow(eyes, gaze, 0.05)
        binary_img = reshape(((arrow[:, :, 0] + arrow[:, :, 1] + arrow[:, :, 2]) == 0.0).astype(float),
                             (self.outputHeight, self.outputWidth, 1))
        binary_img = concatenate((binary_img, binary_img, binary_img), axis=2)
        return (binary_img * image + arrow * (1 - binary_img)).astype(float)

    def getImageWithBox(self, image, bbox, id):
        return cv2.rectangle(image.astype(uint8), (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         self.boxesRandomColors[min(id, 900)]).astype(float)

    def getFullImageHP(self, image, bbox, headpose):
        """
        Overlays the axis of head pose that represents the head estimation on an image.
        :param image:
        :param bbox:
        :param headpose:
        :return:
        """
        bbox = asarray(bbox).astype(float)
        eyes = [((bbox[0] + bbox[2]) / 2), (0.65 * bbox[1] + 0.35 * bbox[3])]
        #yaw, pitch, roll
        #headpose[0],headpose[1],headpose[2]
        
        return  (self.draw_axis(image,headpose[0],headpose[1],headpose[2],tdx=eyes[0],tdy=eyes[1],size=50))

    def draw_axis(self, img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):

        """
        Function used to draw y (headpose label) on Input Image x.
        Implemented by: shamangary
        https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
        Modified by: Omar Hassan
        """
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180

        if tdx != None and tdy != None:
            tdx = tdx
            tdy = tdy
        else:
            height, width = img.shape[:2]
            tdx = width / 2
            tdy = height / 2

        # X-Axis pointing to right. drawn in red
        x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
        y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

        # Y-Axis | drawn in green
        #        v
        x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
        y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

        # Z-Axis (out of the screen) drawn in blue
        x3 = size * (np.sin(yaw)) + tdx
        y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

        cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
        cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
        cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

        return img

    def draw_panoptic(self, panoptic, im):
        panoptic_seg = panoptic[0]
        segments_info = panoptic[1]
        v = Visualizer(im[:, :, ::-1], self.metadata, scale=1.2)
        # TODO :  check how we did to update this
        out = v.draw_panoptic_seg_predictions(panoptic_seg, segments_info) # .to("cpu")
        out = out.get_image()[:, :, ::-1]
        return out

    def objects2d(self, keypoints,map, panoptic, nFrames):
        listhead={}
        listgaze={}
        step = nFrames // 10
        for i,(listHumans,listBoxesGazes) in enumerate(zip(keypoints,map.items())):
            listaingaze={}
            listainhead={}
            if (i % step) == 0:
                print(i, "frames identifying object")
            for id_t in listBoxesGazes[1].keys():
                bbox = map.getHeadBox(i, id_t)
                if bbox is not None:
                    gaze = map.getGaze(i, id_t)
                    head = map.getHeadPose(i, id_t)
                    bbox = asarray(bbox).astype(float)
                    eyes = [((bbox[0] + bbox[2]) / 2), (0.65 * bbox[1] + 0.35 * bbox[3])]
                    if gaze is not None:
                        listaingaze[id_t] = self.findobjectGaze(eyes, gaze,panoptic[i][0],panoptic[i][2])
                    if head is not None:
                        listainhead[id_t] = self.findobjectHP(eyes, head[0],head[1],head[2],panoptic[i][0],panoptic[i][2])
                    #TODO
            listhead[i]=listainhead
            listgaze[i]=listaingaze
        return(listgaze, listhead)

    def findobjectHP(self, eyes, pitch, yaw, roll, panoptic_seg, segments_info):
        panoptic_seg=panoptic_seg.to("cpu").numpy()
        pitch = pitch * np.pi / 180
        yaw = -(yaw * np.pi / 180)
        roll = roll * np.pi / 180
        tdx=eyes[0]
        tdy=eyes[1]
        objects=[]
        prevx=0
        prevy=0
        wi=self.inputW-1
        he=self.inputH-1
        for size in range(100,int(wi*he)):
            x1_aux = size * (np.sin(yaw)) + tdx
            y1_aux = size * (-np.cos(yaw) * np.sin(pitch))  + tdy
            if x1_aux >= wi or y1_aux >= he or x1_aux <= 0 or y1_aux <= 0:
                break
            x1_aux=round(x1_aux)
            y1_aux=round(y1_aux)
            if prevx != x1_aux or prevy != y1_aux:
                aux_object_id=panoptic_seg[y1_aux, x1_aux]
                aux_object=segments_info[aux_object_id]
                objects.append([x1_aux,y1_aux,aux_object])
                prevx=x1_aux
                prevy=y1_aux
            
        return(objects)

    def findobjectGaze(self, eyes, gaze, panoptic_seg, segments_info):
        panoptic_seg=panoptic_seg.to("cpu").numpy()
        tdx=eyes[0]
        tdy=eyes[1]
        
        vectorx=float(gaze[0])
        vectory=float(gaze[1])
        # gaze[2]
        objects=[]
        prevx=0
        prevy=0
        wi=self.inputW-1
        he=self.inputH-1
        for size in range(100,int(wi*he)):
            x1_aux = size * vectorx + tdx
            y1_aux = size * vectory + tdy
            if x1_aux >= wi or y1_aux >= he or x1_aux <= 0 or y1_aux <= 0:
                break
            x1_aux=round(x1_aux)
            y1_aux=round(y1_aux)
            if prevx != x1_aux or prevy != y1_aux:
                aux_object_id=panoptic_seg[y1_aux, x1_aux]
                aux_object=segments_info[aux_object_id]
                objects.append([x1_aux,y1_aux,aux_object])
                prevx=x1_aux
                prevy=y1_aux
            
        return(objects)

    def writeobjects(self, listaG, listaH, dir):
         #! write to files v1
        with open(dir + "objetsH.json", 'w') as fp:
            json.dump(listaH, fp)
        with open(dir + "objetsG.json", 'w') as fp1:
            json.dump(listaG, fp1)
