import numpy as np

def vect2cone(X0, X1,R,n):
    vector=np.array(X1)-np.array(X0)
    length_cone = np.linalg.norm(vector)
    t = np.linspace(0,2*np.pi, n)
    xx = R*np.cos(t)
    yy = R*np.sin(t)

    unit_Vx= np.array([1,0,0])
    angle_X0X1=np.arccos(np.dot(unit_Vx,vector)/(np.linalg.norm(unit_Vx)*length_cone))*180/np.pi

    axis_rot=np.cross(unit_Vx,vector)

    u=axis_rot/np.linalg.norm(axis_rot)

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
    return(cone_vect)

def intersection(maximo, minimo, dir_inv, origin) :
    t1 = (minimo[0] - origin[0])*dir_inv[0]
    t2 = (maximo[0] - origin[0])*dir_inv[0]

    tmin = min(t1, t2)
    tmax = max(t1, t2)

    for i in range(0,3):
        t1 = (minimo[i] - origin[i])*dir_inv[i]
        t2 = (maximo[i] - origin[i])*dir_inv[i]

        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))

    if tmax > tmin :
        point = tmin * 1/dir_inv + origin
        point2 = tmax * 1/dir_inv + origin
        return (point, point2)
    else:
        return(False, False)

def inside_box(maximo, minimo, point):
    if point[0] <= maximo[0] and point[0] >= minimo[0]:
        if point[1] <= maximo[1] and point[1] >= minimo[1]:
            if point[2] <= maximo[2] and point[2] >= minimo[2]:
                return(True)
            else:
                return(False)
        else:
            return(False)
    else:
        return(False)

def distance(center_box, point):
    squared_dist = np.sum((point-center_box)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return(dist)

def calcangle2value(vectgaze, vecthead):
    """ Returns the angle in radians between vectors 'vectgaze' and 'vecthead'

    all of the other answers here will fail if the two vectors have either 
    the same direction (ex, (1, 0, 0), (1, 0, 0)) or opposite directions 
    (ex, (-1, 0, 0), (1, 0, 0)).

    """
    vectgaze_unit = vectgaze / np.linalg.norm(vectgaze)
    vecthead_unit = vecthead / np.linalg.norm(vecthead)
    angle=np.degrees(np.arccos(np.clip(np.dot(vectgaze_unit, vecthead_unit), -1.0, 1.0)))

    # Information that defines the intersection between the Gaze and Head Pose given values to each:[in:[gaze, head], out:[gaze, head]]
    if angle < 45: 
        # IF intersection exist weights for Gaze and Head Pose
        diffG_HP=[1.9, 1.1]
    else:
        # If the intersection doesnt exist weights for Gaze and Head Pose
        diffG_HP=[1.55, 1.45]

    return diffG_HP

def calc_intersection_points(vects_gaze, bbox, origin):
    #TODO: Distance to origin from intersection point -> Normalization
    list_dist=[]
    centerbox=(bbox[0]+bbox[1])/2
    dist_max=distance(centerbox, bbox[0])
    for vect_gaze in vects_gaze:
        point1, point2 = intersection(bbox[0], bbox[1],np.float64(1)/vect_gaze, origin) # maximo, minimo, np.float64(1)/dir_inv, origin
        if type(point1) != bool and inside_box(bbox[0], bbox[1],point1):
            dist = distance(center_box = centerbox, point=point1)
            dist_O = distance(center_box = origin, point=point1)
            dist=dist/dist_max  # normalization
            list_dist.append(1.5-dist)  #1.5: because we want some attention even when the dist is max-out 
        elif type(point2) != bool and inside_box(bbox[0], bbox[1],point2):
            dist = distance(center_box = centerbox, point=point2)
            dist_O = distance(center_box = origin, point=point2)
            dist=dist/dist_max
            list_dist.append(1.5-dist)
        else:
            list_dist.append(0) # 0 represent that the ray doesnt even touch the box --> no attention
    return(list_dist)

def CalcAttention(list_inter, angles, type_gaze): #TODO: Multiplication
    """ Returns the values of Attention having in account the intersection between the 'Gaze' and 'Head Pose' """
    list_inter=np.array(list_inter)
    if type_gaze:
        result= list_inter * angles[0]
    else:
        result= list_inter * angles[1]
    final_result=np.sum(result)
    return final_result

def calc_ratio(bbox):
    min_dist_box=(min(abs([0]-bbox[1])))/2
    centerbox=(bbox[0]+bbox[1])/2
    dist_max=distance(centerbox, bbox[0])
    return (min_dist_box/dist_max)

def calc_normalize_Attention(valueAttention, number_rays, dist_ratio,angles,gaze_or_head):
    if gaze_or_head:
        return(valueAttention /(number_rays*(1.5-(dist_ratio))* angles[0])) 
    else:
        return(valueAttention /(number_rays*(1.5-(dist_ratio))* angles[1])) 
    
def Attention(ListPerson, ListObject):
    dici_attention={}
    for id_person in ListPerson.keys():
        angle=calcangle2value(ListPerson[id_person]["g"],ListPerson[id_person]["hp"])

        dici_value={}
        dici_object={}

        for id_object in ListObject.keys():
            dist_ratio=calc_ratio(ListObject[id_object]["bbox"])
            #! ATTEntion for GAZE
            list_inter=calc_intersection_points(ListPerson[id_person]["vg"], ListObject[id_object]["bbox"], ListPerson[id_person]["eyes"]) #? [box.max, box.min]
            valueAttention=CalcAttention(list_inter,angle,1)
            dici_value["object"]= ListObject[id_object]["name"]
            dici_value["attentionG"]= list_inter
            norm_valueAttention=calc_normalize_Attention(valueAttention, len(ListPerson[id_person]["vg"]), dist_ratio, angle, 1)
            dici_value["total_Attention_G"]=norm_valueAttention #valueAttention (replace after test)
            #! ATTEntion for HEADPOSE
            list_inter_hp=calc_intersection_points(ListPerson[id_person]["vhp"], ListObject[id_object]["bbox"], ListPerson[id_person]["eyes"]) #? [box.max, box.min]
            valueAttention=CalcAttention(list_inter_hp,angle,0)
            dici_value["attentionH"]= list_inter_hp
            norm_valueAttention=calc_normalize_Attention(valueAttention, len(ListPerson[id_person]["vhp"]), dist_ratio, angle, 0)
            dici_value["total_Attention_H"]= norm_valueAttention #valueAttention (replace after test)
            dici_value["total_Attention"]=dici_value["total_Attention_G"] + dici_value["total_Attention_H"]
            dici_object[id_object]=dici_value
            dici_value={}
        
        dici_attention[id_person]=dici_object
    return(dici_attention)
    


if __name__ == "__main__":
    # aabb max and min[ 124.698792   91.741074 -351.372559] [-145.305161  -83.609955 -420.657288]
    # vec = [-16.83, 5.605, -5.277]
    # init = [-30,60,-380]

    # need bird aabb
    list_objects={}
    list_box={}
    list_box["name"]="person"
    maximo=np.array([ 124.698792, 91.741074, -351.372559])
    minimo=np.array([-145.305161,  -83.609955, -420.657288])
    list_box["bbox"]=[maximo, minimo]
    list_objects[1]=list_box
    origin=np.array([-30,60,-380])
    dir_inv=np.array([-16.83, 5.605, -5.277])
    list_box={}
    list_box["name"]="bird"
    list_box["bbox"]=[np.array([ 143.248947,  28.42742 , -351.372559]), np.array([  21.062992 , -44.68504 , -378.059082])]
    list_objects[2]=list_box
    origin=np.array([-30,60,-380])
    dir_inv=np.array([-16.83, 5.605, -5.277])
    # vectors=vect2cone([-30,60,-380],np.array([-30,60,-380])+3*np.array([-16.83, 5.605, -5.277]), 20,20)
    list_person={}
    list_person2={}
    list_person2["g"]=[-16.83, 5.605, -5.277]
    list_person2["vg"]=vect2cone([-30,60,-380],np.array([-30,60,-380])+500*np.array([-16.83, 5.605, -5.277]), 500,20)
    list_person2["hp"]=[-0.1588, -0.2883, -0.9443]
    list_person2["vhp"]=vect2cone([-30,60,-380],np.array([-30,60,-380])+500*np.array([-16.83, 5.605, -5.277]), 20,20)
    list_person2["eyes"]=[-30,60,-380]
    list_person[0]=list_person2
    attention_vlues=Attention(list_person, list_objects)
    #print(list_objects)
    #print(list_person)
    print(attention_vlues[0][1])
    print(attention_vlues[0][2])
