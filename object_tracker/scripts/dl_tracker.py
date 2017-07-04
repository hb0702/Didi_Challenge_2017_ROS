#!/usr/bin/env python
import sys
import os
import time
import rospy as rp
import numpy as np
import math
import struct
import tf
import tf.transformations as transformer

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, Float64MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.cluster import DBSCAN

import tensorflow
import keras
from keras.models import load_model
from keras.optimizers import Adam

from full_view_model import fcn_model
from full_view_train import *
from convert_to_full_view_panorama import *
from dl_filter import dl_filter

from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({"my_loss": my_loss})

PI = 3.14159265358979
PI_2 = 1.570796326794895
MAX_MARKER_COUNT = 30
PUBLISH_MARKERS = True
CAR_LABEL = 1

class dl_tracker:

	def __init__(self):
		# model
		dir_path = os.path.dirname(os.path.realpath(__file__))
		#self.model = load_model(os.path.join(dir_path, '../model/fv_model_for_car_June_30_132_63.h5'))
		self.model = load_model(os.path.join(dir_path, '../model/fv_July_02_057.h5'))
		# filter
		self.filter = dl_filter()
		# graph
		self.graph = tensorflow.get_default_graph()
		# communication
		self.initialize_communication()
		rp.loginfo("dl_tracker: initialized")
		print "dl_tracker: initialized"

	def initialize_communication(self):
		self.subscriber = rp.Subscriber("/filtered_points", Float64MultiArray, self.on_points_received, queue_size=1)
		self.detected_marker_publisher = rp.Publisher("/tracker/markers/detect", MarkerArray, queue_size=1)
		self.predicted_marker_publisher = rp.Publisher("/tracker/markers/predict", MarkerArray, queue_size=1)
		self.box_publisher = rp.Publisher("/tracker/boxes", Float32MultiArray, queue_size=1)
		self.detected_markers = MarkerArray()
		self.predicted_markers = MarkerArray()
		for i in range(MAX_MARKER_COUNT):
			marker = Marker()
			marker.id = i
			marker.header.frame_id = "velodyne"
			marker.type = marker.CUBE
			marker.action = marker.ADD
			marker.pose.position.x = 0.0
			marker.pose.position.y = 0.0
			marker.pose.position.z = 0.0
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.scale.x = 1.0
			marker.scale.y = 1.0
			marker.scale.z = 1.0
			marker.color.a = 0.0
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			self.detected_markers.markers.append(marker)
		for i in range(1):
			marker = Marker()
			marker.id = i
			marker.header.frame_id = "velodyne"
			marker.type = marker.CUBE
			marker.action = marker.ADD
			marker.pose.position.x = 0.0
			marker.pose.position.y = 0.0
			marker.pose.position.z = 0.0
			marker.pose.orientation.x = 0.0
			marker.pose.orientation.y = 0.0
			marker.pose.orientation.z = 0.0
			marker.pose.orientation.w = 1.0
			marker.scale.x = 1.0
			marker.scale.y = 1.0
			marker.scale.z = 1.0
			marker.color.a = 0.0
			marker.color.r = 1.0
			marker.color.g = 1.0
			marker.color.b = 0.0
			self.predicted_markers.markers.append(marker)

	def on_points_received(self, data):
		total_start = time.time()
		ts_sec = int(data.data[0])
		ts_nsec = int(data.data[1])

		lidar = np.empty((0,3))

		if (len(data.data) > 2):
			lidar = np.array(data.data[2:]).reshape(-1, 3)
		box = np.empty((0,8))

		# predict
		with self.graph.as_default():
			box = predict_and_correct(self.model, lidar, clusterPoint=True, seg_thres=0.3, nb_d=4,
										eps=0.8, min_samples=1, leaf_size=30)

		# filter by velocity
		if len(box) == 0:
			box = self.filter.prev_box
			if len(box) != 0:
				adv = self.filter.advance(ts_sec, ts_nsec)
				box[1] += adv[0]
				box[2] += adv[1]
		if  len(box) != 0:
			vel = self.filter.calc_velocity(box, ts_sec, ts_nsec)
			box[7] = normalize_angle(box[7])

		if PUBLISH_MARKERS:
			if box != None and len(box) > 0:			
				self.publish_markers(box)

		if len(box) > 0:			
			self.publish_detected_box(box)

		print ("total time: " + str(time.time() - total_start))

	def publish_detected_box(self, box_info):
		arr = Float32MultiArray()
		flat_box_info = np.reshape(box_info, (-1))
		arr.data = flat_box_info.tolist()
		# publish
		self.box_publisher.publish(arr)
		#rp.loginfo("dl_tracker: published %d boxes", len(box_info))

	def publish_markers(self, box_info):
		# num_boxes = len(det_box_info)
		# if num_boxes > 0:
		# 	# update markers
		# 	num_markers = min(num_boxes, MAX_MARKER_COUNT)
		# 	for i in range(num_markers):
		# 		info = det_box_info[i]
		# 		marker = self.detected_markers.markers[i]
		# 		marker.pose.position.x = info[1]
		# 		marker.pose.position.y = info[2]
		# 		marker.pose.position.z = info[3]
		# 		marker.scale.x = info[4]
		# 		marker.scale.y = info[5]
		# 		marker.scale.z = info[6]
		# 		marker.pose.orientation.z = math.degrees(info[7])			
		# 		marker.color.a = 0.3
		# 	# hide markers not used
		# 	if num_boxes < MAX_MARKER_COUNT:
		# 		for i in range(num_boxes, MAX_MARKER_COUNT):
		# 			marker = self.detected_markers.markers[i]
		# 			marker.color.a = 0.0
		if len(box_info) > 0:
			marker = self.predicted_markers.markers[0]
			marker.pose.position.x = box_info[1]
			marker.pose.position.y = box_info[2]
			marker.pose.position.z = box_info[3]
			marker.scale.x = box_info[4]
			marker.scale.y = box_info[5]
			marker.scale.z = box_info[6]
			q = transformer.quaternion_from_euler(0,0,box_info[7])
			marker.pose.orientation.x = q[0]
			marker.pose.orientation.y = q[1]
			marker.pose.orientation.z = q[2]
			marker.pose.orientation.w = q[3]
			marker.color.a = 0.3
		# publish
		# self.detected_marker_publisher.publish(self.detected_markers)
		self.predicted_marker_publisher.publish(self.predicted_markers)		
		# rp.loginfo("dl_tracker: published %d markers", num_markers)

def rotation_v(theta, points):
	v = np.sin(theta)
	u = np.cos(theta)
	out = np.copy(points)
	out[:,[0]] = u*points[:,[0]] + v*points[:,[1]]
	out[:,[1]] = -v*points[:,[0]] + u*points[:,[1]]
	return out

def distance(p,q):
    return np.sqrt(np.sum(np.square(p-q)))

def length(v):
    return distance(v,0)

def rotate(angle, lidar):
    return np.array([rotation(angle, lidar[i]) for i in range(len(lidar))])

def fit_box(lidar, nb_d = 128):
    
    lidar_2d = lidar[:,:2]    
    angle = np.pi/(nb_d*2)

    center = (np.max(lidar_2d, axis = 0) + np.min(lidar_2d, axis = 0))/2
    center = np.expand_dims(center,0)
    #rotated
    lidar_2d = lidar_2d - center
    #rotated_lidar = [rotation_v(angle*i, lidar_2d) for i in range(nb_d)]
    rotated_lidar = [rotate(angle*i, lidar_2d) for i in range(nb_d)]
        
    max_lidars = np.array([np.max(rotated_lidar[i],axis = 0) for i in range(nb_d)])
    min_lidars = np.array([np.min(rotated_lidar[i],axis = 0) for i in range(nb_d)])
    #print(max_lidars.shape, min_lidars.shape)
    #max_lidars = np.max(rotated_lidar, axis = 1)
    #min_lidars = np.min(rotated_lidar, axis = 1)
    range_lidars = max_lidars - min_lidars
    areas = range_lidars[:,0]*range_lidars[:,1]
    #print(areas.shape)
    
    arg_min = np.argmin(areas)

    rp0, rp2 = min_lidars[arg_min], max_lidars[arg_min]
    rp1, rp3 = np.array([rp2[0], rp0[1]]), np.array([rp0[0], rp2[1]])
    
    #box_2d = rotation_v(-angle*arg_min, np.array([rp0,rp1,rp2,rp3])) + center
    box_2d = np.array([rotation(-angle*arg_min, p) for p in [rp0,rp1,rp2,rp3]]) + center

    return box_2d#, box

def move_box(box, side):
    '''
    box: 2d box of shape (4,2)
    side: array of shape (2,2)
    '''

    #print(box)
    #print(side)
    v_box = box[1] - box[0]
    v_side = side[1] - side[0]
    angle_offset = np.arctan2(v_side[1], v_side[0]) - np.arctan2(v_box[1], v_box[0])
    #print(np.arctan2(v_side[1], v_side[0]), np.arctan2(v_box[1], v_box[0]))
    #print(angle_offset)
    
    rot_box = np.array([rotation(-angle_offset, box[i] - box[0]) + box[0] for i in range(4)])
    #print('rot_box')
    #print(rot_box)
    
    pos_offset = side[0] - box[0]
    correct_box = rot_box + np.expand_dims(pos_offset, axis = 0)
     
    return correct_box

def to_box2d(box_info):
	center = box_info[1:3]
	w = box_info[4] * 0.5
	h = box_info[5] * 0.5
	r = box_info[7]
	box = np.array([[-w,h],[-w,-h],[w,-h],[w,h]])
	box = rotation_v(r, box) + np.expand_dims(center, axis=0)
	return box

def normalize_angle(angle):
	while angle > PI:
		angle -= PI
	while angle < 0:
		angle += PI
	if angle < 0.002:
		angle = 0
	if angle > PI - 0.002:
		angle = PI
	return angle

def move_box_info(box_info, box):
	center = np.mean(box, axis=0)
	miny_idx = np.argmin(box[:,1])
	# wv = box[(miny_idx+1)%4] - box[miny_idx]
	# hv = box[(miny_idx+2)%4] - box[(miny_idx+1)%4]
	wv = box[2] - box[1]
	hv = box[1] - box[0]
	rz = normalize_angle(np.arctan2(wv[1], wv[0]))
	width = length(wv)
	height = length(hv)
	# # find nearest new rz from the old rz
	# rz_candidate = np.array([nrz, normalize_angle(nrz + PI_2), normalize_angle(nrz + PI), normalize_angle(nrz + 3*PI_2)])	
	# orz = normalize_angle(box_info[7])
	# rz = rz_candidate[np.argmin(np.abs(rz_candidate - orz))]
	box_info[1] = center[0]
	box_info[2] = center[1]
	box_info[4] = width
	box_info[5] = height
	box_info[7] = rz
	return box_info

def correct_box_info(predbox_info, fitbox):
    '''
    box, fitbox: 2d box of shape (4,2)
    Move box to the right position based on position of fitbox 
    '''
    predbox = to_box2d(predbox_info)
    pred_sides = np.array([distance(predbox[0], predbox[1]), distance(predbox[1], predbox[2])])
    min_pred_side = np.min(pred_sides)
    min_pred_ind = np.argmin(pred_sides)
    
    fit_distances = np.array([length(fitbox[i]) for i in range(4)])  
    min_fit_ind = np.argmin(fit_distances)
    next_fit_ind = (min_fit_ind + 1)%4
    prev_fit_ind = (min_fit_ind + 3)%4
    
    next_fit_side = distance(fitbox[min_fit_ind],fitbox[next_fit_ind])
    prev_fit_side = distance(fitbox[min_fit_ind],fitbox[prev_fit_ind])
    
    nearest_fit_point = fitbox[min_fit_ind]
    
    diff_next_side = abs(next_fit_side - min_pred_side)
    diff_prev_side = abs(prev_fit_side - min_pred_side)
    if diff_next_side < diff_prev_side:
        side = fitbox[[min_fit_ind, next_fit_ind],:]
    else:
        side = fitbox[[prev_fit_ind, min_fit_ind],:]
    
    indices = [(min_pred_ind + i)%4 for i in range(4)]
    box = np.array([predbox[i] for i in indices])
    box = move_box(box, side)
    box_distances = np.array([length(box[i]) for i in range(4)])
    min_ind = np.argmin(box_distances)
    if min_ind == min_fit_ind:
    	moved_info = move_box_info(predbox_info, box)
        return moved_info
    else:
        indices = [(min_pred_ind + i + 2)%4 for i in range(4)]
        box = np.array([predbox[i] for i in indices])
        box = move_box(box, side)
        moved_info = move_box_info(predbox_info, box)
        return moved_info
  
def correct_predicted_box(box_info, lidar, labels, nb_d=128):
	nb_clusters = np.max(labels)
    if nb_clusters == 0:
        return box_info
    else:
        box_xy = box_info[1:3]
        cluster_xy = np.zeros((nb_clusters, 2))
        for n in range(nb_clusters):
        	cluster_xy[i] = np.mean(lidar[labels==n], axis=0)
    	distances = [distance(box_xy, cluster_xy[i]) for i in range(nb_clusters)]
    	ind = np.argmin(distances)
    	cluster_points = lidar_with_idx[lidar_with_idx[:,3] == ind]
    	if (len(cluster_points) == 0):
    		return box_info
    	fitbox = fit_box(cluster_points, nb_d)
    	correctbox = correct_box_info(box_info, fitbox)
    	return correctbox

def one_box_clustering(boxes, eps = 1, min_samples = 1):
    # Extract the center from predicted boxes
    box_centers = boxes[:,1:4]
    # Do clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(box_centers)
    labels = db.labels_
       
    n_clusters = len(set(labels))
    n_points = [np.sum([labels == i]) for i in range(n_clusters) ]
    max_point_cluster = np.argmax(n_points)
    
    index = (labels == max_point_cluster)
    box = np.mean(boxes[index],axis = 0)
    return box

def rotation_v(theta, points):
	v = np.sin(theta)
	u = np.cos(theta)
	out = np.copy(points)
	out[:,[0]] = u*points[:,[0]] + v*points[:,[1]]
	out[:,[1]] = -v*points[:,[0]] + u*points[:,[1]]
	return out

def predict_and_correct(model, lidar, clusterPoint=True, 
						eps=0.8, min_samples=1, leaf_size=30, seg_thres=0.5, nb_d=128):
    
	test_view, _, labels =  fv_cylindrical_projection_for_test(lidar, clustering=clusterPoint, 
															eps=eps, min_samples=min_samples, leaf_size=leaf_size)

	view = test_view[:,:,[5,2]].reshape(1,16,320,2)

	test_view_reshape = test_view.reshape(-1,6)
	pred = model.predict(view)
	pred = pred[0].reshape(-1,8)

	thres_pred = pred[pred[:,0] > seg_thres]
	thres_view = test_view_reshape[pred[:,0] > seg_thres]

	num_boxes = len(thres_pred)
	if num_boxes == 0:
		return np.array([])
	boxes = np.zeros((num_boxes,8))

	theta = thres_view[:,[3]]
	phi = thres_pred[:,[-1]]

	min = thres_view[:,:3] - rotation_v(theta, thres_pred[:,1:4]) # 0: left top
	max = thres_view[:,:3] - rotation_v(theta, thres_pred[:,4:7]) # 6: right bottom
	center = (min + max) * 0.5
	dvec = max - min
	sinphi = np.sin(phi)
	cosphi = np.cos(phi)
	normdxy = np.linalg.norm(dvec[:,:2], axis=1) # distance between 0 and 2
	normdxy = normdxy.reshape(-1, 1)
	width = normdxy * abs(sinphi)
	height = normdxy * abs(cosphi)
	depth = dvec[:,[2]]
	ax = np.arctan2(dvec[:,[1]], dvec[:,[0]]) # angle from x axis to vector 2-0
	rz = [0.5 * PI - phi[i] + ax[i] for i in range(len(ax))]

	boxes[:,[0]] = CAR_LABEL
	boxes[:,1:4] = center
	boxes[:,[4]] = width
	boxes[:,[5]] = height
	boxes[:,[6]] = depth
	boxes[:,[7]] = rz

	one_box = one_box_clustering(boxes)
	one_box_info = correct_predicted_box(one_box, lidar, labels, nb_d)
	return one_box_info

def listen():
	processor = dl_tracker()
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('dl_tracker', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()

if __name__ == '__main__':
	listen()
