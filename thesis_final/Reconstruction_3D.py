import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

def reconstruction_3D(disparity, matrix_Q, colors_image):
    #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
    fig = plt.figure("Disparity")
    plt.imshow(disparity)
    plt.show()

    print ("\nGenerating the 3D map ...")
    #Reproject points into 3D                  
    points_3D = cv2.reprojectImageTo3D(disparity, matrix_Q)

    #! MASKED IMAGE
    #Get rid of points with value 0 (i.e no depth)
    mask_map_final = disparity > disparity.min()

    #Mask colors and points. 
    output_points = points_3D[mask_map_final]
    output_colors = colors_image[mask_map_final]
    
    #? Point Cloud 
    # create_point_cloud(file, output_points, output_colors)
    return (output_points, output_colors)

def create_output(vertices, colors, filename):
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

def create_point_cloud(file_output_3D, output_points, output_colors):
    print ("\nCreating the output file ...\n")
    #file_output_3D="/home/carlos/Thesis_GOD/detectron2/Thesis_GOD/Final_videos/StereoImgs/Piano-perfect/new_video_depth_image.ply"
    create_output(output_points, output_colors, file_output_3D)

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(file_output_3D)
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])