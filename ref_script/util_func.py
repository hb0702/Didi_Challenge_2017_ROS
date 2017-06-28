import numpy as np
#import matplotlib.pyplot as plt
#import mayavi.mlab
#from mpl_toolkits.mplot3d import Axes3D



def cylindrical_projection(lidar, 
                           ver_fov = (-24.4, 2.),#(-24.9, 2.), 
                           hor_fov = (-42.,42.), 
                           v_res = 0.42,
                           h_res = 0.33):
    '''
    lidar: a numpy array of shape N*D, D>=3
    ver_fov : angle range of vertical projection in degree
    hor_fov: angle range of horizantal projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution
    d_max : maximun range distance
    
    return : cylindrical projection (or panorama view) of lidar
    '''
    
    x = lidar[:,0]
    y = lidar[:,1]
    z = lidar[:,2]
    d = np.sqrt(np.square(x)+np.square(y))
    
    
    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)
    
    x_view = np.int16(np.ceil((theta*180/np.pi - hor_fov[0])/h_res))
    y_view = np.int16(np.ceil((phi*180/np.pi + ver_fov[1])/v_res))
    
    x_max = 255
    y_max = 63
    
    indices = np.logical_and( np.logical_and(x_view >= 0, x_view <= x_max), 
                           np.logical_and(y_view >= 0, y_view <= y_max)  )
    
    x_view = x_view[indices]
    y_view = y_view[indices]
    z = z[indices]
    d = d[indices]
    d_z = [[d[i],z[i]] for i in range(len(d))]
    
    view = np.zeros([y_max+1, x_max+1, 2],dtype=np.float32)
    view[y_view,x_view] = d_z
    return view


def is_in_box(point, box):
    '''
    point: tuple (x,y,z) coordinate
    box: numpy array of shape (8,3)
    return: True or False
    '''
    low = np.min(box[:,2])
    high = np.max(box[:,2])
    if (point[2] >= high) or (point[2]<=low):
        return False
    
    v = point[:2] - box[0,:2]
    v1 = box[1,:2] - box[0,:2]
    v2 = box[3,:2] - box[0,:2]
    
    det1 = v[0]*v2[1] - v[1]*v2[0]
    if det1 == 0:
        return False
    
    det2 = v[0]*v1[1] - v[1]*v1[0]
    if det2 == 0:
        return False
    
    t1 = (v1[0]*v2[1] - v1[1]*v2[0])/det1
    s1 = (v1[0]*v[1] - v1[1]*v[0])/det1
    if (t1<=1) or (s1<=0):
        return False
    
    t2 = (v2[0]*v1[1] - v2[1]*v1[0])/det2
    s2 = (v2[0]*v[1] - v2[1]*v[0])/det2
    if (t2<=1) or (s2<=0):
        return False
    
    return True

def in_which_box(point, boxes):
    '''
    return in which box the given point belongs to, return 0 if the point doesn't belong to any boxes
    '''
    for i in range(len(boxes)):
        if is_in_box(point, boxes[i]):
            return i + 1
    return 0



def cylindrical_projection_for_training(lidar,
                                       gt_box3d,
                                       ver_fov = (-24.4, 2.),#(-24.9, 2.), 
                                       hor_fov = (-42.,42.), 
                                       v_res = 0.42,
                                       h_res = 0.33):
    '''
    lidar: a numpy array of shape N*D, D>=3
    gt_box3d: Ground truth boxes of shape B*8*3 (B : number of boxes)
    ver_fov : angle range of vertical projection in degree
    hor_fov: angle range of horizantal projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution
    
    return : cylindrical projection (or panorama view) of lidar
    '''
    
    x = lidar[:,0]
    y = lidar[:,1]
    z = lidar[:,2]
    d = np.sqrt(np.square(x)+np.square(y))
    
    
    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)
    
    x_view = np.int16(np.ceil((theta*180/np.pi - hor_fov[0])/h_res))
    y_view = np.int16(np.ceil((phi*180/np.pi + ver_fov[1])/v_res))
    
    x_max = 255
    y_max = 63
    
    indices = np.logical_and( np.logical_and(x_view >= 0, x_view <= x_max), 
                           np.logical_and(y_view >= 0, y_view <= y_max)  )
    
    x_view = x_view[indices]
    y_view = y_view[indices]
    z = z[indices]
    d = d[indices]
    d_z = [[d[i],z[i]] for i in range(len(d))]
    
    view = np.zeros([y_max+1, x_max+1, 2],dtype=np.float32)
    view[y_view,x_view] = d_z
    
    encode_boxes = np.array([box_encoder(lidar[i], gt_box3d) for i in range(len(lidar))])
    encode_boxes = encode_boxes[indices]
    
    box = np.zeros([y_max+1, x_max+1, 8],dtype=np.float32)
    box[y_view,x_view] = encode_boxes
    
    return view, box

# todo: add the case where there is no ground truth boxes
# def cylindrical_projection_for_training_with_augmentation(lidar,
# 															gt_box3d,
# 															offset,
# 															flip,
# 															ver_fov = (-24.4, 2.),#(-24.9, 2.), 
# 															hor_fov = [-42.,42.],
# 															v_res = 0.42,
# 															h_res = 0.33):
#     '''
#     lidar: a numpy array of shape N*D, D>=3
#     gt_box3d: Ground truth boxes of shape B*8*3 (B : number of boxes)
#     offset: angle (in rad) of rotation
#     flip: 0 or 1 
#     ver_fov : angle range of vertical projection in degree
#     hor_fov: angle range of horizantal projection in degree
#     v_res : vertical resolusion
#     h_res : horizontal resolution
    
#     return : cylindrical projection (or panorama view) of lidar
#     '''
#     new_lidar, new_gt_box3d = augmentation(offset, flip, lidar, gt_box3d)

#     view, box = cylindrical_projection_for_training(new_lidar, new_gt_box3d)
 
#     return view, box

def cylindrical_projection_for_test(lidar,
                                       #gt_box3d,
                                       ver_fov = (-24.4, 2.),#(-24.9, 2.), 
                                       hor_fov = (-42.,42.), 
                                       v_res = 0.42,
                                       h_res = 0.33,
                                       d_max = None):
    '''
    lidar: a numpy array of shape N*D, D>=3
    ver_fov : angle range of vertical projection in degree
    hor_fov: angle range of horizantal projection in degree
    v_res : vertical resolusion
    h_res : horizontal resolution
    d_max : maximun range distance
    
    return : cylindrical projection (or panorama view) of lidar
    '''
    
    x = lidar[:,0]
    y = lidar[:,1]
    z = lidar[:,2]
    d = np.sqrt(np.square(x)+np.square(y))
        
    
    theta = np.arctan2(-y, x)
    phi = -np.arctan2(z, d)
    
    x_view = np.int16(np.ceil((theta*180/np.pi - hor_fov[0])/h_res))
    y_view = np.int16(np.ceil((phi*180/np.pi + ver_fov[1])/v_res))
    
    x_max = 255
    y_max = 63
    
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
    
    view = np.zeros([y_max+1, x_max+1, 6],dtype=np.float32)
    view[y_view,x_view] = coord
    
    return view

def rotation(theta, point):
	v = np.sin(theta)
	u = np.cos(theta)
	out = np.copy(point)
	out[0] = u*point[0] + v*point[1]
	out[1] = -v*point[0] + u*point[1]
	return out

def rotation_y(phi, point):	
    v = np.sin(phi)
    u = np.cos(phi)
    out = np.copy(point)
    out[0] = u*point[0] + v*point[2]
    out[2] = -v*point[0] + u*point[2]
    return out


def flip_rotation(theta, point):
    v = np.sin(theta)
    u = np.cos(theta)
    out = np.copy(point)
    out[0] = u*point[0] + v*point[1]
    out[1] = v*point[0] - u*point[1]
    return out



def box_encoder(point, boxes):
    '''
        
    '''
    box_num = in_which_box(point, boxes)
    #print(box_num)
    if box_num==0:
        return np.zeros(8)
        
    
    box = boxes[box_num-1]
    #print(box.shape)
    
    theta = np.arctan2(-point[1], point[0])
    #print(theta*180/np.pi)
    #phi = -np.arctan2(point[2], np.sqrt(point[0]**2 + point[1]**2) )
    u0 = point[:3] - box[0] 
    ru0 = rotation(-theta, u0)
    
    u6 = point[:3] - box[6] 
    ru6 = rotation(-theta, u6)
    
    x = np.sqrt(np.sum(np.square(box[1,:2] - box[2,:2])))
    z = np.sqrt(np.sum(np.square(box[0,:2] - box[2,:2])))
    phi = np.arcsin(x/z)

    return np.array([1, ru0[0], ru0[1], ru0[2], ru6[0], ru6[1], ru6[2], phi])




def augmentation(offset, flip, lidar, gtboxes):
	u = np.cos(offset)
	v = np.sin(offset)

	out_lidar = np.copy(lidar)
	out_gtboxes = np.copy(gtboxes)

	if flip == 1:
		
		out_lidar[:,0] = u*lidar[:,0] + v*lidar[:,1]
		out_lidar[:,1] = v*lidar[:,0] - u*lidar[:,1]

		out_gtboxes[:,:,0] = u*gtboxes[:,:,0] + v*gtboxes[:,:,1]
		out_gtboxes[:,:,1] = v*gtboxes[:,:,0] - u*gtboxes[:,:,1]

		out_gtboxes = out_gtboxes[:,[0,3,2,1,4,7,6,5],:]


	else:

		out_lidar[:,0] = u*lidar[:,0] + v*lidar[:,1]
		out_lidar[:,1] = -v*lidar[:,0] + u*lidar[:,1]

		out_gtboxes[:,:,0] = u*gtboxes[:,:,0] + v*gtboxes[:,:,1]
		out_gtboxes[:,:,1] = -v*gtboxes[:,:,0] + u*gtboxes[:,:,1]


	return out_lidar, out_gtboxes


def predict_boxes(model,lidar, cluster = True, seg_thres=0.5, cluster_dist = 0.1, min_dist = 1.5, neigbor_thres = 3):
	view =  cylindrical_projection_for_test(lidar)
	cylindrical_view = view[:,:,[5,2]].reshape(1,64,256,2)
	pred = model.predict(cylindrical_view)
	pred = pred[0]


	pred = pred.reshape(-1,8)
	view = view.reshape(-1,6)
	thres_pred = pred[pred[:,0] > seg_thres]
	thres_view = view[pred[:,0] > seg_thres]

	num_boxes = len(thres_pred)
	box_dist = np.zeros((num_boxes, num_boxes))

	boxes = np.zeros((num_boxes,8,3))
	for i in range(num_boxes):
		boxes[i,0] = thres_view[i,:3] - rotation(thres_view[i,3],thres_pred[i,1:4])
		boxes[i,6] = thres_view[i,:3] - rotation(thres_view[i,3],thres_pred[i,4:7])

		boxes[i,2,:2] = boxes[i,6,:2]
		boxes[i,2,2] = boxes[i,0,2]

		phi = thres_pred[i,-1]

		z = boxes[i,2] - boxes[i,0]
		boxes[i,1,0] = (np.cos(phi)*z[0] + np.sin(phi)*z[1])*np.cos(phi) + boxes[i,0,0]
		boxes[i,1,1] = (-np.sin(phi)*z[0] + np.cos(phi)*z[1])*np.cos(phi) + boxes[i,0,1]
		boxes[i,1,2] = boxes[i,0,2]

		boxes[i,3] = boxes[i,0] + boxes[i,2] - boxes[i,1]
		boxes[i,4] = boxes[i,0] + boxes[i,6] - boxes[i,2]
		boxes[i,5] = boxes[i,1] + boxes[i,4] - boxes[i,0]
		boxes[i,7] = boxes[i,4] + boxes[i,6] - boxes[i,5]

	if not cluster:
		return boxes

	all_boxes = np.copy(boxes)

	flatteb_boxes = boxes.reshape(-1,24)
	for i in range(num_boxes):
		box_dist[i] = np.sqrt(np.sum(np.square(flatteb_boxes[[i]] - flatteb_boxes), axis = 1))

	cluster_boxes = []
		
	thres_box_dist = box_dist < cluster_dist
	neighbor = np.sum(thres_box_dist, axis = 1)

	while len(neighbor)>0:
		
		ind = np.argmax(neighbor)
		
		if neighbor[ind] < neigbor_thres:
		    break

		cluster_boxes.append(boxes[ind])
		
		remain_indx = box_dist[ind] > min_dist
		
		box_dist = box_dist[remain_indx]
		box_dist = box_dist[:,remain_indx]

		thres_box_dist = thres_box_dist[remain_indx]
		thres_box_dist = thres_box_dist[:,remain_indx]
		        
		boxes = boxes[remain_indx]
		
		neighbor = np.sum(thres_box_dist, axis = 1)
	
	return all_boxes, np.array(cluster_boxes) 

def get_mean_std_tensor(depth_mean, height_mean, depth_var, height_var, input_shape = (64,256,2)):
    mean_tensor = np.ones(input_shape)
    std_tensor = np.ones(input_shape)
    mean_tensor[:,:,0]*= depth_mean
    mean_tensor[:,:,1]*= height_mean
    std_tensor[:,:,0]*= np.sqrt(depth_var)
    std_tensor[:,:,1]*= np.sqrt(height_var)
    return mean_tensor, std_tensor




# def viz_mayavi_with_labels(points, boxes, view_boxes = True, vals="distance"):
#     x = points[:, 0]  # x position of point
#     y = points[:, 1]  # y position of point
#     z = points[:, 2]  # z position of pointfrom mpl_toolkits.mplot3d import Axes3D
#     # r = lidar[:, 3]  # reflectance value of point
#     d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor

#     # Plot using mayavi -Much faster and smoother than matplotlib
#     #import mayavi.mlab
#     if vals == "height":
#         col = z
#     else:
#         col = d

#     fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
#     mayavi.mlab.points3d(x, y, z,
#                          col,          # Values used for Color
#                          mode="point",
#                          colormap='spectral', # 'bone', 'copper', 'gnuplot'
#                          # color=(0, 1, 0),   # Used a fixed (r,g,b) instead
#                          figure=fig,
#                          )
    
#     if view_boxes:
#         for i in range(len(boxes)):
#             car = boxes[i]
#             x = car[:,0]
#             y = car[:,1]
#             z = car[:,2]

#             mayavi.mlab.plot3d(x[:4], y[:4], z[:4], tube_radius=0.025)#, colormap='Spectral')
#             mayavi.mlab.plot3d(x[[0,3]], y[[0,3]], z[[0,3]], tube_radius=0.025)
#             mayavi.mlab.plot3d(x[[0,4]], y[[0,4]], z[[0,4]], tube_radius=0.025)
#             mayavi.mlab.plot3d(x[[1,5]], y[[1,5]], z[[1,5]], tube_radius=0.025)
#             mayavi.mlab.plot3d(x[[2,6]], y[[2,6]], z[[2,6]], tube_radius=0.025)
#             mayavi.mlab.plot3d(x[[3,7]], y[[3,7]], z[[3,7]], tube_radius=0.025)


#             mayavi.mlab.plot3d(x[-4:], y[-4:], z[-4:], tube_radius=0.025)#, colormap='Spectral')
#             mayavi.mlab.plot3d(x[[4,7]], y[[4,7]], z[[4,7]], tube_radius=0.025)
        
#     mayavi.mlab.show()


if __name__ == '__main__':

	gt_box3d = np.load('./Code_sample/didi-udacity-2017/data/one_frame/gt_boxes3d.npy')
	gt_label = np.load('./Code_sample/didi-udacity-2017/data/one_frame/gt_labels.npy')
	gt_top_box = np.load('./Code_sample/didi-udacity-2017/data/one_frame/gt_top_boxes.npy')
	lidar = np.load('./Code_sample/didi-udacity-2017/data/one_frame/lidar.npy')
	rgb = np.load('./Code_sample/didi-udacity-2017/data/one_frame/rgb.npy')
	top = np.load('./Code_sample/didi-udacity-2017/data/one_frame/top.npy')
	print('gt_box3d.shape: ', gt_box3d.shape)
	print('gt_label.shape: ', gt_label.shape)
	print('gt_top_box.shape: ', gt_top_box.shape)
	print('lidar.shape: ', lidar.shape)
	print('rgb.shape: ', rgb.shape)
	print('top.shape: ', top.shape)

	viz_mayavi_with_labels(lidar, gt_box3d)