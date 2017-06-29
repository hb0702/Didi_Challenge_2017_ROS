import numpy as np
import os
import time
from multiprocessing import Pool
from multiprocessing import Process

from sklearn.cluster import DBSCAN

lidar_dir = './data/training_didi_data/car_train_edited/'
gt_box_dir = './data/training_didi_data/car_train_gt_box_edited/'
list_bad_frames = './logs/list_bad_label_frames.txt'

def box_encoder(point, boxes):
    '''

    '''
    
    box_num = in_which_box(point, boxes)
    # print(box_num)
    if box_num == 0:
        return np.zeros(8)

    box = boxes[box_num - 1]
    # print(box.shape)

    theta = np.arctan2(-point[1], point[0])
    # print(theta*180/np.pi)
    # phi = -np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) )
    u0 = point[:3] - box[0]
    ru0 = rotation(-theta, u0)

    u6 = point[:3] - box[6]
    ru6 = rotation(-theta, u6)

    x = np.sqrt(np.sum(np.square(box[1, :2] - box[2, :2])))
    z = np.sqrt(np.sum(np.square(box[0, :2] - box[2, :2])))
    phi = np.arcsin(x / z)

    return np.array([1, ru0[0], ru0[1], ru0[2], ru6[0], ru6[1], ru6[2], phi])


def rotation(theta, point):
    v = np.sin(theta)
    u = np.cos(theta)
    out = np.copy(point)
    out[0] = u * point[0] + v * point[1]
    out[1] = -v * point[0] + u * point[1]
    return out


# can be deleted
def is_in_box(point, box):
    '''
    point: tuple (x,y,z) coordinate
    box: numpy array of shape (8,3)
    return: True or False
    '''
    low = np.min(box[:, 2])
    high = np.max(box[:, 2])
    if (point[2] >= high) or (point[2] <= low):
        return False

    v = point[:2] - box[0, :2]
    v1 = box[1, :2] - box[0, :2]
    v2 = box[3, :2] - box[0, :2]

    det1 = v[0] * v2[1] - v[1] * v2[0]
    if det1 == 0:
        return False

    det2 = v[0] * v1[1] - v[1] * v1[0]
    if det2 == 0:
        return False

    t1 = (v1[0] * v2[1] - v1[1] * v2[0]) / det1
    s1 = (v1[0] * v[1] - v1[1] * v[0]) / det1
    if (t1 <= 1) or (s1 <= 0):
        return False

    t2 = (v2[0] * v1[1] - v2[1] * v1[0]) / det2
    s2 = (v2[0] * v[1] - v2[1] * v[0]) / det2
    if (t2 <= 1) or (s2 <= 0):
        return False

    return True

#############################################################
#### This function is used to replace function is_in_box() 
#############################################################
def near_the_box(point, box):
    '''
    point: (x,y,z) coordinate
    box: numpy array of shape (8,3) inluding 8 corner of 3 coordinate each
    process: the function will ignore the height z coordinate, project the point and the box to xy plane,
        measure the distance of the projected point and the center of projected box. 
        The function will return true if the distance is less than 3/4 diameter of the projected box.   
    '''
    center = np.mean(box[:4,:2], axis = 0)
    d = np.sqrt( np.sum( np.square(point[:2] - center) ) )
    diameter = np.sqrt(np.sum(np.square(box[0,:2] - box[2,:2])))
    if d <= 3.*diameter/4:
        return True
    else:
        return False

# can be deleted
def in_which_box(point, boxes):
    '''
    return in which box the given point belongs to, return 0 if the point doesn't belong to any boxes
    '''
    for i in range(len(boxes)):
        if is_in_box(point, boxes[i]):
            return i + 1
    return 0

###############################################################
#### This function is used to replace function in_which_boxes() 
###############################################################
def near_which_box(point, boxes):
    '''
    return in which box the given point is near to, return 0 if the point isn't near to any boxes.
    By 'near to', we mean the output of near_the_box() is True
    '''
    for i in range(len(boxes)):
        if near_the_box(point, boxes[i]):
            return i + 1
    return 0




def cylindrical_projection_for_training(lidar, gt_box3d, ver_fov=(-24.4, 2.), hor_fov=(-47., 47.), v_res=0.42,
                                        h_res=0.33):
    '''
    lidar: a numpy array of shape N*D, D>=3
    gt_box3d: Ground truth boxes of shape B*8*3 (B : number of boxes)
    ver_fov : angle range of vertical projection in degree
    hor_fov: angle range of horizantal projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution

    return : cylindrical projection (or panorama view) of lidar
    '''

    x = lidar[:, 0]
    y = lidar[:, 1]
    z = lidar[:, 2]
    
    ind = np.where(z>-1.27)
    x=x[ind]
    y=y[ind]
    z=z[ind]
    
    d = np.sqrt(np.square(x) + np.square(y))

    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)

    x_view = np.int16(np.ceil((theta * 180 / np.pi - hor_fov[0]) / h_res))
    y_view = np.int16(np.ceil((phi * 180 / np.pi + ver_fov[1]) / v_res))

    x_max = np.int16(np.ceil((hor_fov[1] - hor_fov[0]) / h_res))
    y_max = 63

    indices = np.logical_and(np.logical_and(x_view >= 0, x_view <= x_max),
                             np.logical_and(y_view >= 0, y_view <= y_max))

    x_view = x_view[indices]
    y_view = y_view[indices]
    z = z[indices]
    d = d[indices]
    d_z = [[d[i], z[i]] for i in range(len(d))]

    view = np.zeros([y_max + 1, x_max + 1, 10], dtype=np.float32)
    view[y_view, x_view, :2] = d_z
    
    
    
    encode_boxes = np.array([box_encoder(lidar[i], gt_box3d) for i in range(len(lidar))])
    encode_boxes = encode_boxes[indices]

    # box = np.zeros([y_max+1, x_max+1, 8],dtype=np.float32)
    view[y_view, x_view, 2:] = encode_boxes

    return view

##########################################################################################
######     Clustering 
##########################################################################################
def cluster(lidar, min_d = 2, min_z = -1.35, max_z = 0.5, max_xrange = 6,
            max_yrange = 6, min_xrange = 0.5, min_yrange = 0.5,  
            min_zrange = 0.2, min_points = 15, z_scale = 1.,eps = 0.8, min_samples = 1):
    '''
    min_z : remove points whose z <= min_z (ground removing)
    min_d : remove points within distance of min_d
    z_scale: scale z coordinate before clustering
    eps, min_smaples: parameters of DBSCAN 
    max_xrange, min_xrange, max_yrange, min_yrange, min_zrange : filter out x,y,z range of clusters 
    '''


    # remove ground points
    lidar = lidar[lidar[:,2]>= min_z]
    # remove near points (can improve)
    d = np.sqrt(np.square(lidar[:,0]) + np.square(lidar[:,1]))
    lidar = lidar[d>=min_d]
    # scale z
    lidar1 = np.copy(lidar)
    lidar1[:,2] = (lidar1[:,2]+min_z)/z_scale
    # Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(lidar1)
    labels = db.labels_
    # filter max_z, max_xrange = 3, max_yrange, min_zrange 
    label_set = list(set(labels))
    cluster_height = np.zeros(len(label_set))
    cluster_zrange = np.zeros(len(label_set))
    cluster_xrange = np.zeros(len(label_set))
    cluster_yrange = np.zeros(len(label_set))
    n_points = np.zeros(len(label_set))
    for i in range(len(label_set)):
        z_cluster = lidar[:,2][labels == label_set[i]]
        cluster_height[i] = np.max(z_cluster)
        cluster_zrange[i] = cluster_height[i] - np.min(z_cluster)
        
        x_cluster = lidar[:,0][labels == label_set[i]]
        cluster_xrange[i] = np.max(x_cluster) - np.min(x_cluster)
        
        y_cluster = lidar[:,1][labels == label_set[i]]
        cluster_yrange[i] = np.max(y_cluster) - np.min(y_cluster)
        
        n_points[i] = np.sum(labels == label_set[i])
        
    features = np.array([[cluster_height[labels[i]], cluster_xrange[labels[i]], cluster_yrange[labels[i]], 
                          cluster_zrange[labels[i]], n_points[labels[i]] ]  for i in range(len(labels))])
    
    index = (features[:,0]<=max_z)*(features[:,1]<=max_xrange)*(features[:,2]<=max_yrange)*(features[:,3]>=min_zrange)*(features[:,4]>=min_points)
    if min_xrange != None:
        index = index*(features[:,1]>= min_xrange)
    if min_yrange != None:
        index = index*(features[:,2]>= min_yrange)
    return lidar[index], labels[index]




#####################################################
####  new vesion of cylindrical_projection_for_train
#####################################################
def fv_cylindrical_projection_for_train(lidar, 
                                        gt_box3d, 
                                        ver_fov = (-22, 4.),#(-24.9, 2.), 
                                        v_res = 1.8,
                                        h_res = 1.13,
                                        angle_offset = 5,
                                        clustering = True):
                                       
    '''
    lidar: a numpy array of shape N*D, D>=3
    gt_box3d: groundtruth boxes of shape B*8*3 (B : number of boxes)
    ver_fov : angle range of vertical projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution
    angle_offset : extend the horizontal view to 360 degree + 2*offset for data augementation    

    return : (360 degree full view + 2*offset) cylindrical projection (or panorama view) of lidar
    '''
    if clustering:
        lidar, _ = cluster(lidar)
    else:
         # remove ground points
        lidar = lidar[lidar[:,2]>= -1.4]

    x = lidar[:,0]
    y = lidar[:,1]
    z = lidar[:,2]
    
    d = np.sqrt(np.square(x)+np.square(y))
    if not clustering:
        # remove near points
        lidar = lidar[d>=2]

    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)
       
    
    x_view = np.int16(np.ceil((theta*180/np.pi + 180)/h_res))
    y_view = np.int16(np.ceil((phi*180/np.pi + ver_fov[1])/v_res))
    
    x_max = np.int16(np.ceil(360/h_res))
    y_max = np.int16(np.ceil((ver_fov[1] - ver_fov[0])/v_res))
    
    view = np.zeros([y_max + 1, x_max + 1, 10], dtype=np.float32)
    if len(lidar) != 0:
        indices = np.logical_and( np.logical_and(x_view >= 0, x_view <= x_max), 
                              np.logical_and(y_view >= 0, y_view <= y_max)  )
        
        x_view = x_view[indices]
        y_view = y_view[indices]
        z = z[indices]
        d = d[indices]
        
        d_z = [[d[i], z[i]] for i in range(len(d))]

        view[y_view, x_view, :2] = d_z
        
        encode_boxes = np.array([box_encoder(lidar[i], gt_box3d) for i in range(len(lidar))])
        encode_boxes = encode_boxes[indices]

        # box = np.zeros([y_max+1, x_max+1, 8],dtype=np.float32)
        view[y_view, x_view, 2:] = encode_boxes
        
    if angle_offset == 0:
        return view
    else:
        pad = int(angle_offset*(x_max + 1)/360)

        out = np.zeros([y_max+1, x_max+1+2*pad, 10],dtype=np.float32)

        #middle = int((x_max+1)/2)
        out[:,:pad,:] = view[:, -pad:,:]
        out[:,pad:pad+x_max+1, :] = view
        out[:, pad+x_max+1:x_max+1+2*pad, :] = view[:,:pad,:]
        return out


#####################################################
####  new vesion of cylindrical_projection_for_test
#####################################################
def fv_cylindrical_projection_for_test(lidar, 
                                        ver_fov = (-22, 4.),#(-24.9, 2.), 
                                        v_res = 1.8,
                                        h_res = 1.13,
                                        clustering = True):
                                       
    '''
    lidar: a numpy array of shape N*D, D>=3
    ver_fov : angle range of vertical projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution
    
    return : (360 degree full view + 2*offset) cylindrical projection (or panorama view) of lidar
    '''
    if clustering:
        lidar, _ = cluster(lidar)
    else:
         # remove ground points
        lidar = lidar[lidar[:,2]>= -1.4]

    x = lidar[:,0]
    y = lidar[:,1]
    z = lidar[:,2]
    
    d = np.sqrt(np.square(x)+np.square(y))
    if not clustering:
        # remove near points
        lidar = lidar[d>=2]


    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)
       
    
    x_view = np.int16(np.ceil((theta*180/np.pi + 180)/h_res))
    y_view = np.int16(np.ceil((phi*180/np.pi + ver_fov[1])/v_res))
    
    x_max = np.int16(np.ceil(360/h_res))
    y_max = np.int16(np.ceil((ver_fov[1] - ver_fov[0])/v_res))
    
    view = np.zeros([y_max+1, x_max+1, 6],dtype=np.float32)
    if len(lidar) == 0:
        return view


    indices = np.logical_and( np.logical_and(x_view >= 0, x_view <= x_max), 
                          np.logical_and(y_view >= 0, y_view <= y_max)  )
    
    x_view = x_view[indices]
    y_view = y_view[indices]
    x = x[indices]
    y = y[indices]
    z = z[indices]
    d = d[indices]
    
    theta = theta[indices]
    phi = phi[indices]
    coord = [[x[i],y[i],z[i],theta[i],phi[i],d[i]] for i in range(len(x))]
    
    view[y_view,x_view] = coord
    
    return view
# Can be deletted
def list_of_paths(lidar_dir, gt_box_dir):
    '''
    return list of lidar, gtbox and training view
    '''
    list_of_lidar = []
    list_of_gtbox = []
    list_of_view = []
    for f in os.listdir(lidar_dir):
        lidar_path = os.path.join(lidar_dir, f, 'lidar')
        gtbox_path = os.path.join(gt_box_dir, f, 'gt_boxes3d')

        view_path = os.path.join(lidar_dir, f, 'view')

        if not os.path.exists(view_path):
            os.makedirs(view_path)

        num_files = len(os.listdir(lidar_path))

        lidar = [os.path.join(lidar_path, 'lidar_' + str(i) + '.npy') for i in range(num_files)]
        gtbox = [os.path.join(gtbox_path, 'gt_boxes3d_' + str(i) + '.npy') for i in range(num_files)]
        view = [os.path.join(view_path, 'view_' + str(i) + '.npy') for i in range(num_files)]

        list_of_lidar += lidar
        list_of_gtbox += gtbox
        list_of_view += view
    return list_of_lidar, list_of_gtbox, list_of_view


############################################################################################
####    Create list of training file and remove bad label frames
############################################################################################
def list_of_training_files(lidar_dir, gt_box_dir, list_bad_frames, remove_bad_frames = True):
    '''
    return list of lidar, gtbox and training view
    '''
    with open(list_bad_frames, 'r') as f:
        bad_frames = f.readlines()
    bad_frames = [frame.rstrip() for frame in bad_frames]
    p = True
    list_of_lidar = []
    list_of_gtbox = []
    list_of_view = []
    for car_type in os.listdir(lidar_dir):
        lidar_path = os.path.join(lidar_dir, car_type, 'lidar')
        gtbox_path = os.path.join(gt_box_dir, car_type, 'gt_boxes3d')
        view_path = os.path.join(lidar_dir, car_type, 'view')

        if not os.path.exists(view_path):
            os.makedirs(view_path)

        for f in os.listdir(lidar_path):
            lidar = os.path.join(lidar_path, f)
            gtbox = os.path.join(gtbox_path, 'gt_boxes3d'+f[5:])
            view = os.path.join(view_path, 'view'+f[5:])
            if remove_bad_frames:
                if car_type + ' ' + f in bad_frames:
                    continue
                else:
                    list_of_lidar.append(lidar)
                    list_of_gtbox.append(gtbox)
                    list_of_view.append(view)
            else:
                list_of_lidar.append(lidar)
                list_of_gtbox.append(gtbox)
                list_of_view.append(view)
        
    return list_of_lidar, list_of_gtbox, list_of_view


##########################################################################
####  correct z value of gt_box
##########################################################################
def correct_z_coord(gt_box, min_z = -1.5):
    out = np.copy(gt_box)
    min_z_box = np.min(out[:,:,2], axis = 1)
    out[:,:,2] = out[:,:,2] - min_z_box + min_z
    return out 


def convert(i):
    lidar = np.load(list_of_lidar[i])
    gt_box = np.load(list_of_gtbox[i])
    
    correct_gtbox = correct_z_coord(gt_box)
        
    view = fv_cylindrical_projection_for_train(lidar, correct_gtbox)
    np.save(list_of_view[i], view)

    return i

if __name__ == '__main__':

    using_pool = True

    start = time.time()
    list_of_lidar, list_of_gtbox, list_of_view = list_of_training_files(lidar_dir, gt_box_dir, list_bad_frames, 
                                                                    remove_bad_frames=True)

    # Adjust num_pool = num of cores in the cpu 
    num_pool = 8
    print('Start converting {} frames'.format(len(list_of_lidar)) )
    if using_pool:
        p = Pool(num_pool)
        p.map(convert, np.arange(len(list_of_lidar)))

    else:
        for i in range(len(list_of_lidar)):
            convert(i)
            if (i+1) % 100 == 0:
                print('Finished {0} over {1} frames'.format(i+1, len(list_of_lidar)))

    print('Done converting - total time = {0}'.format(time.time() - start))

