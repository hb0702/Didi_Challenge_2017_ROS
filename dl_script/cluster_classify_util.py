import numpy as np
from sklearn.cluster import DBSCAN
import time
import os

X_RANGE = 6.4
Y_RANGE = 6.4
MIN_Z = -1.35
MAX_Z = 0.5

SCALE = 1.2

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def discretize(lidar, res = 0.1, x_range = X_RANGE, y_range = Y_RANGE, 
               min_height = MIN_Z, max_height = MAX_Z):
    center_image = np.array([X_RANGE, Y_RANGE])/2
    
    max_lidar = np.max(lidar[:,:2], axis = 0)
    min_lidar = np.min(lidar[:,:2], axis = 0)
    center = (max_lidar + min_lidar)/2

    # Move the cluster to teh origin
    lidar[:,:2] = lidar[:,:2] + np.expand_dims(center_image - center, axis = 0)
    
    
    x_lidar = lidar[:, 0]
    y_lidar = lidar[:, 1]
    z_lidar = lidar[:, 2]
    
    x_img = ((y_range-y_lidar)/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = ((x_range-x_lidar)/res).astype(np.int32)  # y axis is -x in LIDAR

    x_max = int(x_range/res)
    y_max = int(y_range/res)

    index = np.logical_and(np.logical_and(x_img < x_max, x_img >= 0), np.logical_and(y_img < y_max, y_img >= 0))
    x_img = x_img[index]
    y_img = y_img[index] 
    
    pixel_values = np.clip(z_lidar[index], a_min=min_height, a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    
    im = np.zeros([y_max, x_max, 2], dtype=np.uint8)
    for i in range(len(x_img)):
        im[y_img[i], x_img[i], 0] = max(im[y_img[i], x_img[i], 0],pixel_values[i]) # -y because images start from top left
        
        im[y_img[i], x_img[i], 1] += 1
        
    return im, center

def distance(p,q):
    return np.sqrt(np.sum(np.square(p-q)))

def length(v):
    return distance(v,0)

def gt_box_encode(gtbox, center):
    '''
    gtbox has shape 8*3
    '''
    z_range = np.abs(gtbox[4,2] - gtbox[0,2])
    side1 = distance(gtbox[0,:2], gtbox[1,:2])
    side2 = distance(gtbox[1,:2], gtbox[2,:2])
    v = gtbox[1,:2] - gtbox[0,:2]
    yaw_angle = np.arctan2(v[1], v[0])
    box_center = (gtbox[2,:2] + gtbox[0,:2])/2 - center
    return np.array([1, box_center[0], box_center[1], z_range, side1, side2, yaw_angle]) 

def gt_box_decode(features, center, z_min = -1.5):
    z_max = z_min + features[3]
    box_center = center + features[1:3]
    yaw_angle = features[-1]
    side1 = features[4]
    side2 = features[5]
    v10 = np.tan(yaw_angle)
    v0 = np.sqrt(side1*side1/(1+ v10*v10))
    v1 = v0*v10
    v = np.array([v0,v1])
    w = np.array([-v1, v0])*side2/side1
    
    p0 = box_center - (v + w)/2
    p1 = p0 + v
    p2 = p1 + w
    p3 = p0 + w
    box = np.ones((8,3))*z_min
    box[:4, :2] = np.array([p0,p1,p2,p3])
    box[4:, :2] = box[:4, :2]
    box[4:, 2] = z_max
    
    return box   

def rotation(theta, point):
	v = np.sin(theta)
	u = np.cos(theta)
	out = np.copy(point)
	out[0] = u*point[0] + v*point[1]
	out[1] = -v*point[0] + u*point[1]
	return out

def flip_rotation(theta, point):
    v = np.sin(theta)
    u = np.cos(theta)
    out = np.copy(point)
    out[0] = u*point[0] + v*point[1]
    out[1] = v*point[0] - u*point[1]
    return out

def rotation_cluster(theta, lidar, flip = 0):
	if flip == 0:
		return np.array([rotation(theta, lidar[i]) for i in range(len(lidar)) ])
	else:
		return np.array([flip_rotation(theta, lidar[i]) for i in range(len(lidar)) ])

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


def is_in_scaled_box(lidar_points, gtbox, scale = SCALE):
    '''
    points: shape N*3
    box: numpy array of shape (1,8,3)
    return: True or False
    '''
    box = gtbox[0,:4,:2]
    points = lidar_points[:,:2]
    
    if scale != 1:
        box_center = (box[0] + box[2])/2
        d0 = (box[2] - box[0])/2
        d1 = (box[3] - box[1])/2
        
        p0 = box_center - scale*d0
        p1 = box_center - scale*d1
        p2 = box_center + scale*d0
        p3 = box_center + scale*d1
        
        scaled_box = np.array([p0,p1,p2,p3])
    else:
        scaled_box = np.copy(box)
    
    nb_points = len(points)
    for i in range(4):
        v = points - scaled_box[[i],:]
        l = (i+3)%4
        r = (i+1)%4
        vl = box[l] - scaled_box[i]
        vr = box[r] - scaled_box[i]
        prodl = np.sum(v*np.expand_dims(vl,0), axis = 1)
        prodr = np.sum(v*np.expand_dims(vr,0), axis = 1)
        if np.sum([prodl < 0]) + np.sum([prodr < 0]) > 0:
            return False
        
    return True

def nearby_car(cluster, gt_box, k = 0.25):
    assert len(gt_box) == 1, "not valid gt_box"
    box = gt_box[0]
    
    min_cluster = np.min(cluster[:,:2], axis = 0)
    max_cluster = np.max(cluster[:,:2], axis = 0)

    center = (min_cluster + max_cluster)/2
    
    box_center = np.mean(box[[0,2],:2], axis = 0 )
    distance =  np.sqrt(np.sum(np.square(center - box_center)))    

    box_diagonal = np.sqrt(np.sum(np.square(box[0,:2] - box[2,:2])))
    # The k parameter is used to control the distance from cluster to box_center
    if distance <= k*box_diagonal:
        return True
    else:
        return False

def is_good_label(lidar, gt_box):
    '''
    return:
        -1 : if no cluster is found. We better to check the frame by own eyes. Remove this frame from training set
                because there is a gt_box but the training panoramic view is full of zeros.
        0 : if no cluster is near to the gt_box
        if at least one cluster is near to gt_box, the function return the number of near cluster  
    '''
    
    assert len(gt_box) > 0, "Empty groundtruth box"
    
    lidar, labels = cluster(lidar, min_d = 2, min_z = -1.35, max_z = 0.5, max_xrange = 6,
                             max_yrange = 6, min_xrange = 0.5, min_yrange = 0.5,  
                             min_zrange = 0.2, min_points = 15, z_scale = 1,eps = 0.8, min_samples = 3)
    # list of clusters
    clusters = list(set(labels))
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        return -1, [], []
    
    n_near_clusters = 0
    list_near_cluter = []
    list_far_cluster = []
    
    for i in range(n_clusters):
        cluster_lidar = lidar[labels == clusters[i]]
        if is_in_scaled_box(cluster_lidar, gt_box):
            n_near_clusters += 1
            list_near_cluter.append(cluster_lidar)
        else:
            list_far_cluster.append(cluster_lidar)
            # we can save the car cluster separately in this step
    return n_near_clusters, list_near_cluter, list_far_cluster


def create_car_training(lidar_dir, gtbox_dir, car_dir, not_car_dir):
    # create folders contains car and not_car lidar points if not exist
    if not os.path.exists(car_dir):
        os.mkdir(car_dir)
    if not os.path.exists(not_car_dir):
        os.mkdir(not_car_dir)
    # list of subfolders containing car lidar
    list_folders = os.listdir(lidar_dir)
    print('Begin converting {} lidar folders.'.format(len(list_folders)))
    start = time.time()
    
    log_cars = open('./logs/log_car.txt', 'a')
    log_not_cars = open('./logs/log_not_car.txt', 'a')
    
    #bad_car_labels = open('bad_car_labels.txt', 'a')
    for folder in list_folders:
        sub_time = time.time()
        lidar_folder = os.path.join(lidar_dir, folder, 'lidar')
        gtbox_folder = os.path.join(gtbox_dir, folder, 'gt_boxes3d')
        
        car_folder = os.path.join(car_dir, folder)
        not_car_folder = os.path.join(not_car_dir, folder)
        
        if not os.path.exists(car_folder):
            os.mkdir(car_folder)
        if not os.path.exists(not_car_folder):
            os.mkdir(not_car_folder)
        # list of all lidar frame
        list_lidar_files = os.listdir(lidar_folder)
        #list_lidar_files = list_lidar_files[:5]
        #print(list_lidar_files)
        
        print('Begin converting {0} lidar files in folder {1}'.format(len(list_lidar_files), folder))
        nb = 0
        for file in list_lidar_files:
            # load lidar frame and gt_box
            lidar_file = os.path.join(lidar_folder,file)
            gtbox_file = lidar_file.replace(lidar_dir, gtbox_dir).replace('lidar','gt_boxes3d')
            
            lidar = np.load(lidar_file)
            gt_box = np.load(gtbox_file)
            
            n_near_clusters, list_near_cluter, list_far_cluster = is_good_label(lidar, gt_box)
            
            if n_near_clusters == 1:
                nb += 1
                car_file = os.path.join(car_folder, file.replace('lidar', 'car') )
                np.save(car_file, list_near_cluter[0])
                log_cars.write(car_file + '\n')
                if len(list_far_cluster) >= 1:
                    for i in range(len(list_far_cluster)):
                        not_car_file = os.path.join(not_car_folder, file.replace('lidar', 'not_car_' + str(i)) )
                        np.save(not_car_file, list_far_cluster[i])
                        log_not_cars.write(not_car_file + '\n')
        print('End converting {0} folder {1}. Nb saved file: {4}. Time: {2}. Time per frame: {3}'.format(
                len(list_lidar_files), folder, time.time()-sub_time , (time.time()-sub_time)/len(list_lidar_files), nb))
    log_not_cars.close()
    log_cars.close()
    print('Done creating training data. Total time: ', time.time() - start)

def list_of_data(car_dir, not_car_dir, gtbox_dir):
    list_of_cars = []
    list_of_not_cars = []
    list_of_gtboxes = []

    for f in os.listdir(car_dir):
        car_path = os.path.join(car_dir, f)
        not_car_path = os.path.join(not_car_dir, f)
        gtbox_path = os.path.join(gtbox_dir, f, 'gt_boxes3d')

        for name in os.listdir(car_path):
            car_file = os.path.join(car_path, name)
            gtbox_file = os.path.join(gtbox_path, name.replace('car', 'gt_boxes3d'))

            list_of_cars.append(car_file)
            list_of_gtboxes.append(gtbox_file)

        for name in os.listdir(not_car_path):
            not_car_file = os.path.join(not_car_path, name)
            list_of_not_cars.append(not_car_file)
            
    return list_of_cars, list_of_not_cars, list_of_gtboxes



def create_good_car_training(car_dir, not_car_dir, gtbox_dir, scale = 1.2):
    phrase1 = 'car_cluster'
    phrase2 = phrase1 + '_' + str(scale).replace('.', '_')
    new_car_dir = car_dir.replace(phrase1, phrase2)

    start = time.time()    
    if not os.path.exists(new_car_dir):
        os.mkdir(new_car_dir)
    
    for f in os.listdir(car_dir):
        new_car_folder = os.path.join(new_car_dir, f)
        if not os.path.exists(new_car_folder):
            os.mkdir(new_car_folder)    


    list_of_cars, _ , list_of_gtboxes = list_of_data(car_dir, not_car_dir, gtbox_dir)
    nb_good_files = 0

    print('Begin checking {} files'.format(len(list_of_cars)))
    for i in range(len(list_of_cars)):
        car = np.load(list_of_cars[i])
        gtbox = np.load(list_of_gtboxes[i])
        
        if is_in_scaled_box(car, gtbox, scale = scale):
            car_file = list_of_cars[i].replace(phrase1, phrase2)
            np.save(car_file, car)
            nb_good_files += 1

        if i%1000 == 0:
            print('Finished {0} files. Nb good files: {1}. Time: in {2} s'.format(i, nb_good_files, int(time.time() - start) ))
    print('End - time: {0}. Nb good files: {1}. Time per file {2}'.format(int(time.time() - start), nb_good_files,  (time.time() - start)/(len(list_of_cars))))

if __name__ == '__main__':

    car_dir = './data/training_didi_data/car_cluster/'
    not_car_dir =  './data/training_didi_data/not_car_cluster/'
    gtbox_dir = './data/training_didi_data/car_train_gt_box_edited/'
    lidar_dir = './data/training_didi_data/car_train_edited/'
    
    create_car_training(lidar_dir, gtbox_dir, car_dir, not_car_dir)


