import numpy as np
import cv2
import torch
import random
import open3d as o3d
from Attention import Attention, vect2cone
from Arrow_line import get_arrow, create_cone_arrow, draw_geometries
from matplotlib import pyplot as plt
import csv 
import os
import glob

class Reconstruction:
    def __init__(self, Metadata_panoptic, matrix_Q, dir):
        self.metadata = Metadata_panoptic
        self.matrix_Q = matrix_Q
        self.output_file =dir
        

    def __call__(self,frames, disparity, panoptic_results, mapFramesOutputs):
        list_frame_obj={}
        # Get information for each frame of the persons identified
        dict_values = list(mapFramesOutputs.values()) 
        for i in range(len(frames)):
            #print("DISPARITY SHAPE")
            #print(disparity[i].shape)
            frame=cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            # Reconstruction of the Points 3D from disparity
            points_3d = self.reconstruction_3D(disparity[i], self.matrix_Q[i])
            # Creation of the list of Objcets
            lista = self.createlistofobjects(panoptic_results[i][1])
            #print(lista)
            list_person = self.createlistofperson(dict_values[i], points_3d)
            #print(list_person)
            # Creation of the List for the Point Cloud Scene
            pcd=[]
            # dici to all objects
            list_objects={}
            # First object starts with 1 
            count_object=0 
            # aux variable to insert in list new object
            count_aux=0 
            # Visualization of Panoptic segmentation
            #self.visualization_PS(panoptic_results[i][0])
            #self.visualization_PS(frame)
            # For in all objects detected
            for object_value in lista.keys():
                # value == 0 -> unidentified object by panoptic seg
                if object_value == 0 :
                    break
                #print(object_value,lista[object_value])
                # mask with the object
                indices = (panoptic_results[i][0] == object_value)
                # creation of the mask for the disp
                disp_mask = disparity[i] > disparity[i].min()
                # combine two masks
                indices=np.logical_and(disp_mask,indices) 
                # points and colors of the object being analyse
                output_points = points_3d[indices==True]
                output_colors = frame[indices==True]
                #self.visualization_PS(indices)
                print("Done object "+ lista[object_value])
                if lista[object_value] == "wall":
                    continue
                # Create file with the Point Cloud of the object in the respective directory
                output_file = self.output_file + "testobject_" + str(lista[object_value]) + ".ply"
                self.create_output(output_points, output_colors, output_file)
                # Read the file with Point Cloud 
                pcd_aux = o3d.io.read_point_cloud(output_file)
                #TODO: Ajust filter if the data in use is the dataset or videos from robot
                # Filter the Point Cloud of the object with Algorithm Radius Outlier
                cl, ind = pcd_aux.remove_radius_outlier(nb_points=11, radius=1.05)
                # Selection of the Inlier Cloud
                inlier_cloud = pcd_aux.select_by_index(ind)
                # Bounding Box of the inlier cloud
                aabb = inlier_cloud.get_axis_aligned_bounding_box()
                # Create the dici for the objects in this frame
                list_box={}
                #! Carefull Filter 3000
                # Take out the exceptions from the visualization (bbox doesnt exist -> all zeros or inf)
                if (np.any(np.isinf(aabb.get_max_bound())) or not np.any(aabb.get_max_bound()) or np.any(abs(aabb.get_max_bound()) >= 6000))  and ( np.any(np.isinf(aabb.get_min_bound())) or not np.any(aabb.get_min_bound())or np.any(abs(aabb.get_min_bound()) >= 6000)):
                    #if (np.any(np.isinf(aabb.get_max_bound())) or not np.any(aabb.get_max_bound()))  and ( np.any(np.isinf(aabb.get_min_bound())) or not np.any(aabb.get_min_bound())):
                    print("box failed -- not in list")
                else:
                    # Info to the list created (name of the object and the location of the box)
                    list_box["name"]=str(lista[object_value])
                    list_box["bbox"]=[aabb.get_max_bound(), aabb.get_min_bound()]
                    # Appending to the Scene visualization
                    pcd.append(inlier_cloud)
                    pcd.append(aabb)
                    # Increase the number of objects
                    count_object+=1
                    #? only if you want
                    # Put a sphere on the bounding box max bound
                    mesh_sphere_begin = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution= 20)
                    # Translate the sphere to the proper position ([0,0,0] to [x,y,z] of max bound)
                    mesh_sphere_begin.translate(aabb.get_max_bound())
                    # Each sphere has a color (max and min)
                    mesh_sphere_begin.paint_uniform_color([0,1,1])
                    # Put a sphere on the bounding box min bound
                    mesh_sphere_end = o3d.geometry.TriangleMesh.create_sphere(radius=10, resolution= 20)
                    # Translate the sphere to the proper position
                    mesh_sphere_end.translate(aabb.get_min_bound())
                    mesh_sphere_end.paint_uniform_color([1,0,1])
                    # Appending spheres to the Scene visualization
                    pcd.append(mesh_sphere_begin)
                    pcd.append(mesh_sphere_end)
                print(aabb.get_max_bound(), aabb.get_min_bound())
                if count_aux != count_object:
                    list_objects[count_object]=list_box
                    count_aux=count_object
            if i == 16:
                self.visualization(pcd, list_person, i)
            # todo: Return and check if this is right 
            # TODO: Attention if object is closer to origin
            attention_values=Attention(list_person, list_objects)
            #? FRAME CYCLE
            print("FRAME: " + str(i))
            list_frame_obj[i]=attention_values
            self.print2csv( i, attention_values, self.output_file + "final_data_Attention.csv")
            attention_values={}
            self.delete_ply()
            print("DONE")   
        return(0)

    def createlistofobjects(self,segments_info):
        """ Create the list with the information necessary to each object. """
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

    def createlistofperson(self, list_init_person, points_3d):
        """ Create the list with the information necessary to each person. """
        list_person={}
        for person in list_init_person.keys():
            list_person2={}
            if "b" in list_init_person[person].keys() :
                list_person2["b"]=list_init_person[person]['b']
                # GET head bounding box and estimate the eyes postion
                list_person2["eyes"]=self.get_eyes_3d(list_person2["b"], points_3d)
            if "hp" in list_init_person[person].keys():
                # GET head pose vector and ajust to new reference axis
                list_person2["hp"]=list_init_person[person]['hp'] * [-1, 1, 1]
                # Draw in form of cone 20 rays representing the head pose -> diameter 1 = 2000
                list_person2["vhp"]=vect2cone(list_person2["eyes"],list_person2["eyes"]+300*np.array(list_person2["hp"]), 2000,20)
                # Draw in form of cone 20 rays representing the head pose -> diameter 1 = 4000
                vectors = vect2cone(list_person2["eyes"],list_person2["eyes"]+300*np.array(list_person2["hp"]), 4000,20)
                list_person2["vhp"] = np.append(list_person2["vhp"], vectors, axis=0)
            if "g" in list_init_person[person].keys():
                # GET gaze vector and ajust to new reference axis
                list_person2["g"]=list_init_person[person]['g'] * [-1, 1, -1]
                # Draw in form of cone 20 rays representing the gaze -> diameter 1 = 60
                list_person2["vg"]=vect2cone(list_person2["eyes"],list_person2["eyes"]+300*np.array(list_person2["g"]), 60,20)
                # Draw in form of cone 20 rays representing the gaze -> diameter 2 = 120
                vectors = vect2cone(list_person2["eyes"],list_person2["eyes"]+300*np.array(list_person2["g"]), 120,20)
                list_person2["vg"] = np.append(list_person2["vg"], vectors, axis=0)
            list_person[person]=list_person2
        return(list_person)

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
    
    def reconstruction_3D(self, disparity, matrix_Q):
        """ Reconstruction from disparity to 3d Point Cloud. """
        #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
        print ("\nGenerating the 3D map ...")
        #Reproject points into 3D                  
        points_3D = cv2.reprojectImageTo3D(disparity, matrix_Q)

        #! MASKED IMAGE
        #Get rid of points with value 0 (i.e no depth)
        self.mask_map = disparity > disparity.min()

        return (points_3D)

    def create_output(self, vertices, colors, filename):
        """ Create the ply file for the objects detected in PointCloud. """
        colors = colors.reshape(-1, 3)
        vertices = np.hstack([vertices.reshape(-1,3), colors])

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
        '''

        with open(filename, 'w') as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, '%f %f %f %d %d %d')
    
    def get_eyes_3d(self, head_box, points):
        """ Function that estimates the eyes location in 3D. """
        # In the mask get an aproximate position for the eyes
        # Can have some error because it gets only one pixel
        # eyes = points[int((0.65 * head_box[1] + 0.35 * head_box[3])), int(((head_box[0] + head_box[2]) / 2))]

        # Get an box of points close to the eyes for an better estimation for the eyes position in 3D
        eyes=[int((0.65 * head_box[1] + 0.35 * head_box[3])), int(((head_box[0] + head_box[2]) / 2))]
        eyes = points[eyes[0]-3:eyes[0]+3,eyes[1]-3:eyes[1]+3]
        # Reshape the points for the estimation
        """
        You need to use np.transpose to rearrange dimensions. 
        From a n x m x 3 to 3 x (n x m), 
        so send the last axis to the front and shift right the order of the remaining axes (0,1). 
        Finally , reshape to have 3 rows.
        """
        eyes1=eyes.transpose(2,0,1).reshape(3,-1)
        #print("mean eyes:",eyes)
        # Mean aproximation
        #eyes=np.mean(eyes1, axis=1)
        # Median aproximation
        #print("median eyes:",eyes)
        eyes=np.array(np.median(eyes1, axis=1))
        
        return(eyes)

    def visualization(self, pcd, list_person, index):
        """ For each person detected create the visualization for the Gaze and Head Pose arrows. """
        for id_person in list_person.keys():
            # GAZE
            # Vect from gaze to arrow 
            pcd.extend(create_cone_arrow(list_person[id_person]["eyes"],list_person[id_person]["g"],list_person[id_person]["vg"],90,20,"gaze"))
            # HEADPOSE
            # Vect from gaze to arrow
            pcd.extend(create_cone_arrow(list_person[id_person]["eyes"],list_person[id_person]["hp"],list_person[id_person]["vhp"],2000,20,"headpose"))
        draw_geometries(pcd, index)
    
    def visualization_PS(self, panoptic_seg):
        fig = plt.figure("frame sub divided")
        plt.imshow(panoptic_seg)
        plt.show()
    
    def print2csv(self, frame_id, attention_values, name):
        if frame_id == 0:
            w = csv.writer(open(name, "w"))
        else:
            w = csv.writer(open(name, "a"))
        w.writerow(["Frame", frame_id])
        for key, val in attention_values.items():
            w.writerow([frame_id ,"Person", key])
            for key1, val1 in val.items():
                w.writerow([frame_id, key, "object id", key1])

                lista1=[frame_id, key, key1]
                lista1.append('object')
                lista1.append(val1['object'])
                w.writerow(lista1)
                lista1=[]

                lista1=[frame_id, key, key1]
                lista1.append('attentionG')
                lista1.extend(val1['attentionG'])
                w.writerow(lista1)
                lista1=[]

                lista1=[frame_id, key, key1]
                lista1.append('total_Attention_G')
                lista1.append(val1['total_Attention_G'])
                w.writerow(lista1)
                lista1=[]
                
                lista1=[frame_id, key, key1]
                lista1.append('attentionH')
                lista1.extend(val1['attentionH'])
                w.writerow(lista1)
                lista1=[]
                
                lista1=[frame_id, key, key1]
                lista1.append('total_Attention_H')
                lista1.append(val1['total_Attention_H'])
                w.writerow(lista1)
                lista1=[]
                
                lista1=[frame_id, key, key1]
                lista1.append('total_Attention')
                lista1.append(val1['total_Attention'])
                w.writerow(lista1)

    def delete_ply(self):
        file_ply=glob.glob(self.output_file + "/*.ply")
        for plyfile in file_ply:
            os.remove(plyfile)



'''
get HP and GAZE and BBOX
    dict_values = list(mapFramesOutputs.values())
    dict_values[49][1]={'b': np.array([446,  32, 691, 327]), 'g': np.array([ 0.4734, -0.1622, -0.8657]), 'hp': np.array([-30.7  , -12.61 ,   8.086])}
    print(dict_values[49].keys())
    print(dict_values[49][0]['b'])
    print(dict_values[49][0].keys())
    for person in dict_values[49].keys():
        print(person)
        if "b" in dict_values[49][person].keys() :
            print(dict_values[49][person]['b'])
        if "hp" in dict_values[49][person].keys():
            print(dict_values[49][person]['hp'])
        if "g" in dict_values[49][person].keys():
            print(dict_values[49][person]['g'])


'''