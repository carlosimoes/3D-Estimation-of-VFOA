import numpy as np
import cv2
import torch
import random

from prediction_depth import compute_scale_and_shift
from DepthPredictor_Mono import DepthPredictor_Mono
from Reconstruction_3D import create_point_cloud



class Reconstruction_Mono_3D:
    def __init__(self, Metadata_panoptic, matrix_Q):
        self.metadata = Metadata_panoptic
        self.matrix_Q = matrix_Q
        

    def __call__(self,frames, disparity, panoptic_results):
        for i in range(len(frames)):
            print("DISPARITY SHAPE")
            print(disparity[i].shape)
            points_3d = self.reconstruction_3D(disparity[i])
            depth_extracted, mask_to_depth = self.point_MonoDepth(panoptic_results[i][0],points_3d, self.createlistofobjects(panoptic_results[i][1]))
            # ? parallel
            disparity_mono = self.get_mono_depth(frames[i])
            final_points, final_colors = self.get_Prediction(points_3d, depth_extracted,mask_to_depth, disparity_mono, frames[i])

            create_point_cloud("/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/video_final2/3d.ply", final_points, final_colors)
            input()
        return(0)

    def createlistofobjects(self,segments_info):
        lista={}
        for info in segments_info:
            id = int(info["category_id"])
            if info["isthing"]:
                text = self.metadata.thing_classes[id]
            else:
                text = self.metadata.stuff_classes[id]
            lista[info["id"]] = text
        lista[0]="object not identified"
        return(lista)

    def point_MonoDepth(self, panoptic_seg, points, lista):
        mask=np.zeros(panoptic_seg.shape)
        destino=np.zeros(panoptic_seg.shape)
        pixel={}
        for object_value in lista.keys():
            # Get indexs on panoptic seg of the specific object
            index2 = np.where(panoptic_seg == object_value)
            # random coordenates withn the indexs of object
            coord=random.sample(range(len(index2[0])), 2)
            coordenadas=[[ index2[1][coord[0]],index2[0][coord[0]]],[index2[1][coord[1]], index2[0][coord[1]]]]
            # eliminate outliers
            while points[coordenadas[0][1]][coordenadas[0][0]][2] == float("inf") or points[coordenadas[1][1]][coordenadas[1][0]][2]  == float("inf"):
                coord=random.sample(range(len(index2[0])), 2) 
                coordenadas=[[ index2[1][coord[0]],index2[0][coord[0]]],[index2[1][coord[1]], index2[0][coord[1]]]]
            # Dic with all the information
            pixel[object_value] = coordenadas
            # Visualize the coordinates obtain in the image
            """
            circ = Circle((coordenadas[0][0], coordenadas[0][1]),10,fc="r")
            ax.add_patch(circ)
            circ = Circle((coordenadas[1][0], coordenadas[1][1]),10, fc="r")
            ax.add_patch(circ) """
            # create mask and target from points 3D
            mask[coordenadas[0][1], coordenadas[0][0]]=True
            mask[coordenadas[1][1], coordenadas[1][0]]=True
            destino[coordenadas[0][1], coordenadas[0][0]]=points[coordenadas[0][1]][coordenadas[0][0]][2]
            destino[coordenadas[1][1], coordenadas[1][0]]=points[coordenadas[1][1]][coordenadas[1][0]][2]
            print(points[coordenadas[0][1]][coordenadas[0][0]][2],points[coordenadas[1][1]][coordenadas[1][0]][2])
        return (destino, mask)
    
    def get_mono_depth(self, frame):
        depth2=DepthPredictor_Mono("/home/carlos/Thesis_GOD/detectron2/MiDaS/model-f46da743.pt",1)
        prediction=depth2([frame])
        del depth2
        return (prediction)

    def get_Prediction(self, points, destino, mask, mono_depth, frame):
        # calculate scale and shift
        print(torch.from_numpy(mono_depth[0]).unsqueeze(0).shape)
        print(torch.from_numpy(destino).unsqueeze(0).shape)
        print(torch.from_numpy(mask).unsqueeze(0).shape)

        scale, shift = compute_scale_and_shift(torch.from_numpy(mono_depth[0]).unsqueeze(0),torch.from_numpy(destino).unsqueeze(0),torch.from_numpy(mask).unsqueeze(0))
        # Obtain the prediction
        __prediction_ssi = scale.view(-1, 1, 1) * torch.from_numpy(mono_depth[0]).unsqueeze(0) + shift.view(-1, 1, 1)
        # replace the Z/depth of points_3D for the new prediction of Z/depth
        print(points[:,:,2].shape)
        print(__prediction_ssi.to("cpu").numpy().shape)
        points[:,:,2]=__prediction_ssi.to("cpu").numpy()
        # Mask colors and points. 
        output_points = points[self.mask_map]
        output_colors = frame[self.mask_map]
        return(output_points ,output_colors)
    
    def reconstruction_3D(self, disparity):
        #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
        print ("\nGenerating the 3D map ...")
        #Reproject points into 3D                  
        points_3D = cv2.reprojectImageTo3D(disparity, self.matrix_Q)

        #! MASKED IMAGE
        #Get rid of points with value 0 (i.e no depth)
        self.mask_map = disparity > disparity.min()

        return (points_3D)
        