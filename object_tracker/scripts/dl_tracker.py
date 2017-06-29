#!/usr/bin/env python
import sys
import rospy as rp
import numpy as np
import math

#import cv2
#from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray

import tensorflow as tf
import keras
from keras.models import load_model
from keras.optimizers import Adam

from model import fcn_model, my_loss

from keras.utils.generic_utils import get_custom_objects
#loss = SSD_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)
get_custom_objects().update({"my_loss": my_loss})

import os 

class dl_tracker:

	def __init__(self, use_cpp_node):
		# input buffer
		self.input_buf = np.zeros([self.y_max+1, self.x_max+1, 6], dtype=np.float32);
		# model
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model = load_model(dir_path + '/../model/model.h5')
		self.graph = tf.get_default_graph()
		self.seg_thres = 0.07
		# communication
		self.use_cpp_node = True if use_cpp_node == '1' else False
		self.initialize_communication()
		rp.loginfo("dl_tracker: initialized")
		print "dl_tracker: initialized"

	def initialize_communication(self):
		self.subscriber = rp.Subscriber("/velodyne_points", PointCloud2, self.on_points_received, queue_size=1)
		self.sub_count = 0
		if self.use_cpp_node:
			self.publisher = rp.Publisher("/dl_tracker/boxes", Float32MultiArray, queue_size=1)
		else:
			self.publisher = rp.Publisher("/tracker/boxes", MarkerArray, queue_size=1)
			self.marker_array = MarkerArray()
			self.max_marker_count = 1000
			for i in range(self.max_marker_count):
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
				marker.color.r = 0.0
				marker.color.g = 0.0
				marker.color.b = 0.0
				self.marker_array.markers.append(marker)

	def on_points_received(self, data):
		self.sub_count += 1		
		rp.loginfo("dl_tracker %d: point received", self.sub_count)
		if (self.process_locked):
			rp.loginfo("dl_tracker: Process locked")
			return
		# lock process
		self.process_locked = True
		#rp.loginfo("dl_tracker: process started")
		# process points
		x = []
		y = []
		z = []
		for p in pc2.read_points(data, skip_nans=True):
			x.append(p[0])
			y.append(p[1])
			z.append(p[2])
		box_info = np.empty((0,8))
		# predict
		with self.graph.as_default():
			box_info = self.predict_boxes(x, y, z)
		# publish
		if self.use_cpp_node:
			self.publish_detected_boxes(box_info)
		else:
			self.publish_markers(box_info)
		# unlock process
		self.process_locked = False

	def publish_detected_boxes(self, box_info):
		arr = Float32MultiArray()
		flat_box_info = np.reshape(box_info, (-1))
		arr.data = flat_box_info.tolist()
		# publish
		self.publisher.publish(arr)
		#rp.loginfo("dl_tracker: published %d boxes", len(box_info))

	def publish_markers(self, box_info):
		num_boxes = len(box_info)
		# update markers
		num_markers = min(num_boxes, self.max_marker_count)
		for i in range(num_markers):
			info = box_info[i]
			marker = self.marker_array.markers[i]
			marker.pose.position.x = info[4]
			marker.pose.position.y = info[5]
			marker.pose.position.z = info[6]
			marker.pose.orientation.z = info[7]
			marker.scale.x = info[1]
			marker.scale.y = info[2]
			marker.scale.z = info[3]
			marker.color.a = 0.3
			marker.color.r = 1.0 if info[0] == 0 else 0.0
			marker.color.b = 0.0 if info[0] == 0 else 1.0
		# hide markers not used
		if num_boxes < self.max_marker_count:
			for i in range(num_boxes, self.max_marker_count):
				marker = self.marker_array.markers[i]
				marker.color.a = 0.0
		# publish
		self.publisher.publish(self.marker_array)
		rp.loginfo("dl_tracker: published %d markers", num_markers)

	def rotation(self, theta, points):
		v = np.sin(theta)
		u = np.cos(theta)
		out = np.copy(points)
		out[:,0] = u*points[:,0] + v*points[:,1]
		out[:,1] = -v*points[:,0] + u*points[:,1]
		return out

	def predict_boxes(self, x, y, z):
		x = np.array(x)
		y = np.array(y)
		z = np.array(z)
		d = np.sqrt(np.square(x)+np.square(y))
		theta = np.arctan2(-y, x)
		phi = -np.arctan2(z, d)

		all_boxes = np.empty((0,8,3))

		# repeat for horizontal segments
		for ns in range(self.num_hor_seg):
			self.input_buf.fill(0)

			x_view = np.int16(np.ceil((theta*180/np.pi - self.hor_fov_arr[ns][0])/self.h_res))
			y_view = np.int16(np.ceil((phi*180/np.pi + self.ver_fov[1])/self.v_res))

			indices = np.logical_and( np.logical_and(x_view >= 0, x_view <= self.x_max), \
				np.logical_and(y_view >= 0, y_view <= self.y_max) )

			x_view = x_view[indices]
			y_view = y_view[indices]
			x_f = x[indices]
			y_f = y[indices]
			z_f = z[indices]
			d_f = d[indices]
			theta_f = theta[indices]
			phi_f = phi[indices]

			coord = [[x_f[i],y_f[i],z_f[i],theta_f[i],phi_f[i],d_f[i]] for i in range(len(x_f))]

			self.input_buf[y_view,x_view] = coord
			cylindrical_view = self.input_buf[:,:,[5,2]].reshape(1,64,256,2)

			# predict
			pred = self.model.predict(cylindrical_view)
			pred = pred[0]
			pred = pred.reshape(-1,8)
			view = self.input_buf.reshape(-1,6)
			pred_indices = pred[:,0] > self.seg_thres
			thres_pred = pred[pred_indices]
			thres_view = view[pred_indices]

			num_boxes = len(thres_pred)
			boxes = np.zeros((num_boxes,8,3))

			# compose boxes
			boxes[:,0] = thres_view[:,:3] - self.rotation(thres_view[:,3],thres_pred[:,1:4])
			boxes[:,6] = thres_view[:,:3] - self.rotation(thres_view[:,3],thres_pred[:,4:7])
			boxes[:,2,:2] = boxes[:,6,:2]
			boxes[:,2,2] = boxes[:,0,2]

			phi_pred = thres_pred[:,-1]
			cos_phi = np.cos(phi_pred)
			sin_phi = np.sin(phi_pred)
			z_pred = boxes[:,2] - boxes[:,0]

			boxes[:,1,0] = (cos_phi*z_pred[:,0] + sin_phi*z_pred[:,1])*cos_phi + boxes[:,0,0]
			boxes[:,1,1] = (-sin_phi*z_pred[:,0] + cos_phi*z_pred[:,1])*cos_phi + boxes[:,0,1]
			boxes[:,1,2] = boxes[:,0,2]

			boxes[:,3] = boxes[:,0] + boxes[:,2] - boxes[:,1]
			boxes[:,4] = boxes[:,0] + boxes[:,6] - boxes[:,2]
			boxes[:,5] = boxes[:,1] + boxes[:,4] - boxes[:,0]
			boxes[:,7] = boxes[:,4] + boxes[:,6] - boxes[:,5]

			all_boxes = np.vstack((all_boxes, boxes))

		num_boxes = len(all_boxes)
		box_info = np.zeros((num_boxes,8), dtype=np.float32)

		# compose box info - [label, l, w, h, px, py, pz, yaw]
		lv2d = all_boxes[:,3,:2] - all_boxes[:,0,:2]
		l = np.linalg.norm(lv2d, axis=1)
		w = np.linalg.norm(all_boxes[:,1,:2] - all_boxes[:,0,:2], axis=1)
		h = all_boxes[:,4,2] - all_boxes[:,0,2]
		center = (all_boxes[:,0] + all_boxes[:,6]) * 0.5
		yaw = [math.atan2(lv2d[i,1], lv2d[i,0]) for i in range(len(lv2d))]
		box_info[:,1] = l
		box_info[:,2] = w
		box_info[:,3] = h
		box_info[:,4:7] = center
		box_info[:,7] = yaw

		return box_info

def listen(use_cpp_node):
	processor = dl_tracker(use_cpp_node)
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('dl_tracker', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()

if __name__ == '__main__':
	listen(sys.argv[1])
