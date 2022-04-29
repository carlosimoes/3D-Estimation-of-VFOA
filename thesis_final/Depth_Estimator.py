import cv2
import numpy as np
from matplotlib import pyplot as plt
import yaml
from time import time as timeSeconds
from math import floor
import sintel_data as sio
import glob
import os

import open3d as o3d


class Depth_Estimator:
    def __init__(self, videoL, videoR, camfile, type_video): 
        self.type_video = type_video
        if type_video == "robot_video":
            self.imagesL=videoL
            self.imagesR=videoR
            camfile="/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/Dataset"
            C1, D1, R1, P1, self.inputH, self.inputW= self.read_config_camera(camfile+"/left.yaml")
            C2, D2, R2, P2, self.inputH, self.inputW= self.read_config_camera(camfile+"/right.yaml")
            R = np.array([1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]).reshape(3,3)
            T = np.array([204.089, 0, 0]).reshape(3,1)
            scale_percent = 75/100 # percent of original size
            self.inputW = round(self.inputW * scale_percent )
            self.inputH = round(self.inputH * scale_percent) # floor
            dim = (self.inputH, self.inputW) 
            value  = cv2.stereoRectify(scale_percent*C1, D1, scale_percent*C2, D2, dim, R, T, alpha=0)  # last paramater is alpha, if 0= croped, if 1= not croped
            R1 = value[0]
            R2 = value[1]
            P1 = value[2]
            P2 = value[3]
            self.Q = value[4]
            
            self.leftMapX, self.leftMapY= cv2.initUndistortRectifyMap(scale_percent*C1, D1, R1, P1, dim, cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
            self.rightMapX, self.rightMapY= cv2.initUndistortRectifyMap(scale_percent*C2, D2, R2, P2, dim, cv2.CV_16SC2)
        
        if self.type_video == "mpi_sintel":
            self.dirL=videoL
            self.dirR=videoR
            self.cam =camfile


         
    def __call__(self, filepath_newfinalvideo, init, fim):
        if self.type_video == "robot_video":
            output_disp=[]
            videorect = cv2.VideoWriter(filepath_newfinalvideo, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.inputH, self.inputW))

            capL = cv2.VideoCapture(self.imagesL)
            capR = cv2.VideoCapture(self.imagesR)
            numberFrames = int(round(capL.get(cv2.CAP_PROP_FRAME_COUNT)))
            i = 0
            t = timeSeconds()
            step=max(1,(numberFrames)//10)
            print("DISP_STEREO batch :", str(0) , "/", numberFrames-1)
            while i < numberFrames:
                readL, imageL = capL.read()
                readR, imageR = capR.read()
                if not readL or not readR:
                    break
                i += 1
                #! TEST
                if i > init+1 and i< fim+1:
                    # reduce image size
                    imageL = cv2.resize(imageL, (self.inputH, self.inputW) )
                    imageR = cv2.resize(imageR, (self.inputH, self.inputW) )

                    #disparity
                    imagem_final, disp = self.main_depth(imageL,imageR)            
                    output_disp.append((disp)) 
                    #print("IMAGE SHAPE")
                    #print(imagem_final.shape)
                    #print("DISPARITY SHAPE")
                    #print(disp.shape)
                    videorect.write(imagem_final)
                if i%step==0:
                    print("DISP_STEREO batch :", str(i) , "/", numberFrames)
            t = timeSeconds() - t
            print("Total extraction length of Disparity  :", t, " seconds ; ", numberFrames / t, "frames per second")
            videorect.release()
            capL.release()
            capR.release()
            return self.Q, output_disp# output
        #! SINTEL DATASET
        if self.type_video == "mpi_sintel":
            output_disp=[]
            matrix_final=[]
            # read left camera images
            Leftimage = glob.glob(self.dirL + "/*.png")
            Leftimage.sort()
            # read left camera images
            Rightimage = glob.glob(self.dirR + "/*.png")
            Rightimage.sort()
            # read camera matrix
            Camera_matrix = glob.glob(self.cam + "/*.cam")
            Camera_matrix.sort()
            image_aux=cv2.imread(Rightimage[0])
            self.inputW, self.inputH, c = image_aux.shape
            videorect = cv2.VideoWriter(filepath_newfinalvideo, cv2.VideoWriter_fourcc(*'mp4v'), 30, (self.inputH, self.inputW))

            numberFrames = len(Leftimage)
            t = timeSeconds()
            step=max(1,(numberFrames)//10)
            print("DISP_STEREO batch :", str(0) , "/", numberFrames)
            for i, (imageL,imageR, cam_file) in enumerate(zip(Leftimage, Rightimage, Camera_matrix), start=1): 
                #! TEST
                if i > init and i< fim+1:
                    # Get Intrinsic and Extrinsic matrix
                    I,E = sio.cam_read(cam_file)
                    # The images are rendered from two cameras with a baseline of 10 cm apart.
                    baseline = 100 
                    # Matrix Q - projective matrix
                    Q = np.zeros((4,4))
                    Q[0,0]=Q[1,1]=1
                    Q[0,3]=-I[0,2]
                    Q[1,3]=-I[1,2]
                    Q[2,3]=I[0,0]
                    Q[3,2]=-1/baseline
                    matrix_final.append(Q)
                    #disparity
                    imagem_final, disp = self.main_depth(imageL,imageR)            
                    output_disp.append((disp)) 
                    '''print("IMAGE SHAPE")
                    print(imagem_final.shape)
                    print("DISPARITY SHAPE")
                    print(disp.shape)'''
                    #image to video (blue -> RGB)
                    imagem_final=cv2.cvtColor(imagem_final, cv2.COLOR_RGB2BGR)
                    videorect.write(imagem_final)
                    '''
                    #todo: delete
                    points_3D = cv2.reprojectImageTo3D(output_disp.pop(), Q)
                    output_file = "/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/test/" + "testobject_" + "object" + ".ply"
                    mask_map = disp > disp.min()
                    print(disp.min())
                    points_3D[mask_map] = 0
                    imagem_final = imagem_final[mask_map]
                    self.create_output(points_3D, imagem_final,output_file)
                    # Read the file with Point Cloud 
                    pcd_aux = o3d.io.read_point_cloud(output_file)
                    cl, ind = pcd_aux.remove_radius_outlier(nb_points=11, radius=1.05)
                    # Selection of the Inlier Cloud
                    inlier_cloud = pcd_aux.select_by_index(ind)
                    o3d.visualization.draw_geometries([inlier_cloud], window_name= "Frame "+ str(i))
                    file_ply=glob.glob("/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/test" + "/*.ply")
                    for plyfile in file_ply:
                        os.remove(plyfile)
                    #todo: delete
                    input()
                    '''
                if i%step==0:
                    print("DISP_STEREO batch :", str(i) , "/", numberFrames)
            t = timeSeconds() - t
            print("Total extraction length of Disparity  :", t, " seconds ; ", numberFrames / t, "frames per second")
            videorect.release()
            return matrix_final, output_disp# output self.matrix_Q  

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

    def load_calibration(self, path):
        """ Loads stereo matrix coefficients. """
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        C1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        C2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        cv_file.release()
        return C1, D1, C2, D2, R, T

    def calculate_disp(self, imgL, imgR):

        """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) """
        # SGBM Parameters -----------------

        window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        left_matcher = cv2.StereoBM_create(numDisparities=256, blockSize=window_size)
        disp= left_matcher.compute(imgL,imgR).astype(np.float32) / 16.0
        """
        fig = plt.figure("Disp without filter")
        plt.imshow(disp)
        plt.show()
        """
        #! More precise but more time needed
        """ 
        left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=256,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY) 
        """
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3
        visual_multiplier = 6

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)

        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        
        #! MASKED IMAGE
        mask2= (displ == -16)
        """ 
        fig = plt.figure("Disparity")
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title("disp")
        ax1.imshow(displ)
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title("Filtered")
        ax2.imshow(filteredImg)
        plt.show()
        """
        return filteredImg, mask2
    
    def calculate_disp_sintel(self, imgL, imgR):

        """ Depth map calculation. Works with SGBM and WLS. Need rectified images, returns depth map ( left to right disparity ) 
        # SGBM Parameters -----------------
        SWS = 15  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        PFS = 193.51
        PFC = 8
        MDS = -2.24
        NOD = 242
        TTH = 28
        UR = 11
        SR = 16
        SPWS = 4

        SWS = int(SWS/2)*2+1 #convert to ODD
        PFS = int(PFS/2)*2+1
        PFC = int(PFC/2)*2+1    
        MDS = int(MDS)    
        NOD = int(NOD/16)*16  
        TTH = int(TTH)
        UR = int(UR)
        SR = int(SR)
        SPWS= int(SPWS)
         """
        # Depth map function

        left_matcher = cv2.StereoBM_create(numDisparities= int(160/16)*16, blockSize=15) #16-15
        """
        left_matcher.setPreFilterType(1)
        left_matcher.setPreFilterSize(PFS)
        left_matcher.setPreFilterCap(PFC)
        left_matcher.setMinDisparity(MDS)
        left_matcher.setNumDisparities(NOD)
        left_matcher.setTextureThreshold(TTH)
        left_matcher.setUniquenessRatio(UR)
        left_matcher.setSpeckleRange(SR)
        left_matcher.setSpeckleWindowSize(SPWS)
         
        PFC = 29 # preFilterCap
        MDS = -3 # minDisparity
        NOD = 48 # numDisparities
        TTH = 13 # disp12MaxDiff
        UR = 3 # uniquenessRatio
        SR = 14 # speckleRange
        SPWS = 2 # speckleWindowSize

        dic_MODE = {"0": cv2.STEREO_SGBM_MODE_SGBM, "1": cv2.STEREO_SGBM_MODE_HH , "2": cv2.STEREO_SGBM_MODE_SGBM_3WAY, "3": cv2.STEREO_SGBM_MODE_HH4}

        left_matcher = cv2.StereoSGBM_create(
        minDisparity=MDS,
        numDisparities=NOD,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=SWS,
        P1=8 * 3 * SWS * SWS,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * SWS * SWS,
        disp12MaxDiff=TTH,
        uniquenessRatio=UR,
        speckleWindowSize=SPWS,
        speckleRange=SR,
        preFilterCap=PFC,
        mode=dic_MODE[str(MODE)])""" 
        
        disp= left_matcher.compute(imgL,imgR).astype(np.float32) / 16.0
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        # FILTER Parameters
        lmbda = 8000000 #80000
        sigma = 1.5
        visual_multiplier = 9

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
        wls_filter.setLambda(lmbda)

        wls_filter.setSigmaColor(sigma)
        displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
        dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
        displ = np.int16(displ)
        dispr = np.int16(dispr)

        filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        filteredImg = np.uint8(filteredImg)
        
        #! MASKED IMAGE
        mask2= (displ == -16)
        
        '''fig = plt.figure("Disparity")
        ax1 = fig.add_subplot(1,2,1)
        ax1.set_title("disp")
        ax1.imshow(disp)
        ax2 = fig.add_subplot(1,2,2)
        ax2.set_title("Filtered")
        ax2.imshow(filteredImg)
        plt.show()'''
        
        return filteredImg, mask2
    
    def main_depth(self, imageL, imageR):
        imgL1 = imageL #cv2.imread('/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/Dataset/data52images/171_l_image.jpg', cv2.IMREAD_COLOR)  
        imgR1 = imageR #cv2.imread('/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/Dataset/data52images/171_r_image.jpg', cv2.IMREAD_COLOR) # Using 0 to read image in grayscale mode 

        if self.type_video == "mpi_sintel":
            imgL1 = cv2.imread(imgL1)  
            imgR1 = cv2.imread(imgR1)

        imgL = cv2.cvtColor(imgL1, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imgR1, cv2.COLOR_BGR2GRAY)
        
        
        if self.type_video == "robot_video":
            left_rectified = cv2.remap(imgL, self.leftMapX, self.leftMapY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            right_rectified = cv2.remap(imgR, self.rightMapX, self.rightMapY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        """  
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title("Left")
        ax1.imshow(imgL)
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title("Right")
        ax2.imshow(imgR)
        ax1 = fig.add_subplot(2,2,3)
        ax1.set_title("Left Rect")
        ax1.imshow(left_rectified)
        ax2 = fig.add_subplot(2,2,4)
        ax2.set_title("Right Rect")
        ax2.imshow(right_rectified)
        plt.show() 
        """
        if self.type_video == "mpi_sintel":
            filteredImg, notFiltered = self.calculate_disp_sintel(imgL, imgR)
        else:
            filteredImg, notFiltered = self.calculate_disp(left_rectified, right_rectified)

        #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
        #fig = plt.figure("Disparity")
        #plt.imshow(filteredImg)
        #plt.show()

        #Get color points
        if self.type_video == "robot_video":
            left_rectified = cv2.remap(imgL1, self.leftMapX, self.leftMapY, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        else:
            left_rectified = imgL1
            left_rectified = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)
 
        #! MASKED IMAGE
        img=left_rectified.copy()
        img[notFiltered]=0

        return(img, filteredImg) #, mask_map,  points_3D)

    def read_config_camera(self, input_file):
        """ A function to read YAML file"""
        # image_height
        # image_width
        with open(input_file) as f:
            config = yaml.safe_load(f)
        width = int(config['image_width'])
        height = int(config['image_height'])
        instrisec_matrix = np.array(config['camera_matrix']['data']).reshape(3,3)
        distortion_coefficients = np.array(config['distortion_coefficients']['data'])
        rect_matrix = np.array(config['rectification_matrix']['data']).reshape(3,3)
        proj_matrix = np.array(config['projection_matrix']['data']).reshape(3,4)
        
        return instrisec_matrix, distortion_coefficients, rect_matrix, proj_matrix, width, height

