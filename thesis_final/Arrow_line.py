import open3d as o3d
import numpy as np
import math

def draw_geometries(pcds, index):
    """
    Draw Geometries
    Args:
        - pcds (): [pcd1,pcd2,...]
    """
    o3d.visualization.draw_geometries(pcds, window_name= "Frame "+ str(index))

def get_o3d_FOR(origin=[0, 0, 0],size=10):
    """ 
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=size)
    mesh_frame.translate(origin)
    return(mesh_frame)

def vector_magnitude(vec):
    """
    Calculates a vector's magnitude.
    Args:
        - vec (): 
    """
    magnitude = np.sqrt(np.sum(vec**2))
    return(magnitude)

def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

    #qTrans_Mat *= scale
    return qTrans_Mat

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat

def calculate_zy_rotation_for_arrow(vec):
    """
    Calculates the rotations required to go from the vector vec to the 
    z axis vector of the original FOR. The first rotation that is 
    calculated is over the z axis. This will leave the vector vec on the
    XZ plane. Then, the rotation over the y axis. 

    Returns the angles of rotation over axis z and y required to
    get the vector vec into the same orientation as axis z
    of the original FOR

    Args:
        - vec (): 
    """
    # Rotation over z axis of the FOR
    gamma = np.arctan(vec[1]/vec[0])
    Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                   [np.sin(gamma),np.cos(gamma),0],
                   [0,0,1]])
    # Rotate vec to calculate next rotation
    vec = Rz.T@vec.reshape(-1,1)
    vec = vec.reshape(-1)
    # Rotation over y axis of the FOR
    beta = np.arctan(vec[0]/vec[2])
    Ry = np.array([[np.cos(beta),0,np.sin(beta)],
                   [0,1,0],
                   [-np.sin(beta),0,np.cos(beta)]])
    return(Rz, Ry)

def create_arrow(scale=3000):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(cone_radius=5,
        cone_height=10,
        cylinder_radius=2,
        cylinder_height=200)
    return(mesh_frame)

def get_arrow(origin=[0, 0, 0], end=None, vec=None):
    """
    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. [x,y,z]
        - vec (): Vector. [i,j,k]
    """
    scale = 10
    Ry = Rz = np.eye(3)
    T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T[:3, -1] = origin
    if end is not None:
        vec = np.array(end) - np.array(origin)
    elif vec is not None:
        vec = np.array(vec)
    if end is not None or vec is not None:
        scale = vector_magnitude(vec)
        #Rz, Ry = calculate_zy_rotation_for_arrow(vec)
        Rot = caculate_align_mat(vec)
    mesh = create_arrow(scale)
    # Create the arrow
    mesh.rotate(Rot, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    return(mesh)

def create_arrow_vis(init, direction):
    # Create a Cartesian Frame of Reference
    FOR = get_o3d_FOR()
    # Create an arrow from point (5,5,5) to point (10,10,10)
    # arrow = get_arrow([5,5,5],[10,10,10])

    # Create an arrow representing vector vec, starting at (5,5,5)
    arrow = get_arrow(init,vec=direction)

    # Create an arrow in the same place as the z axis
    #arrow = get_arrow()

    # Draw everything
    ##draw_geometries([FOR,arrow])
    return(arrow)

def vect2cone(X0, X1,R,n):
    print(X0,X1,R,n)
    vector=np.array(X1)-np.array(X0)
    length_cone = np.linalg.norm(vector)
    t = np.linspace(0,2*np.pi, n)
    xx = R*np.cos(t)
    yy = R*np.sin(t)

    unit_Vx= np.array([1,0,0])
    angle_X0X1=np.arccos(np.dot(unit_Vx,vector)/(np.linalg.norm(unit_Vx)*length_cone))*180/np.pi

    axis_rot=np.cross(unit_Vx,vector)

    u=axis_rot/np.linalg.norm(axis_rot)
    print(u)
    alpha=angle_X0X1*np.pi/180
    cosa = np.cos(alpha)
    sina = np.sin(alpha)
    vera = 1 - cosa
    x = u[0]
    y = u[1]
    z = u[2]
    rot = np.array([[cosa+x**2*vera, x*y*vera-z*sina, x*z*vera+y*sina], 
        [x*y*vera+z*sina, cosa+y**2*vera, y*z*vera-x*sina],
        [x*z*vera-y*sina, y*z*vera+x*sina, cosa+z**2*vera]]).transpose()
    aux=[length_cone]*xx.size
    origin_ref=np.array([0,0,0])
    newxyz = np.array([np.array(aux)-origin_ref[0],xx-origin_ref[1],yy-origin_ref[2]]).transpose()
    newxyz = np.matmul(newxyz,rot)
    newxyz= newxyz + origin_ref
    cone_vect=newxyz-origin_ref
    print(cone_vect)
    return(cone_vect)

def create_cone_arrow(origin,first_gaze, vectors,radius, numArrowCone, str_type):
    arrow=[]
    if str_type == "gaze":
        if not radius and not numArrowCone:
            numArrowCone = 20
            radius = 20
        arrow_center = get_arrow(origin,vec=first_gaze)
        arrow_center.paint_uniform_color([0.9, 0.1, 0.1])
        arrow.append(arrow_center)
        if not vectors.any():
            vectors=vect2cone(origin,np.array(origin)+300*first_gaze, radius, numArrowCone)
        for vect in vectors:
            arrow_color=get_arrow(origin,vec=vect)
            arrow_color.paint_uniform_color([0.1, 0.9, 0.1])
            arrow.append(arrow_color)
    else:
        if not radius and not numArrowCone:
            numArrowCone = 20
            radius = 500
        arrow_center = get_arrow(origin,vec=first_gaze)
        arrow_center.paint_uniform_color([0.9, 0.1, 0.1])
        arrow.append(arrow_center)
        if not vectors.any():
            vectors=vect2cone(origin,np.array(origin)+300*first_gaze, radius, numArrowCone)
        for vect in vectors:
            arrow_color=get_arrow(origin,vec=vect)
            arrow_color.paint_uniform_color([0.1, 0.9, 0.9])
            arrow.append(arrow_color)

    #TODO: extend with point cloud
    return arrow

