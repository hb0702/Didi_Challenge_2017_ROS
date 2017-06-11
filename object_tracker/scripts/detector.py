#!/usr/bin/env python
import rospy as rp
import numpy as np
import math

#import cv2
#from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

import tensorflow as tf
import keras
from keras.models import load_model
from keras.optimizers import Adam

from model import fcn_model, my_loss

from keras.utils.generic_utils import get_custom_objects
#loss = SSD_Loss(neg_pos_ratio=neg_pos_ratio, alpha=alpha)
get_custom_objects().update({"my_loss": my_loss})

import os 

class detector:

	def __init__(self):
		# lock
		self.process_locked = False
		# input params
		self.x_max = 255
		self.y_max = 63
		self.ver_fov = (-24.4, 15.)
		self.v_res = 1.
		self.num_hor_seg = 2
		# derived input params
		self.hor_fov_arr = []
		self.h_res = 0.0
		if self.num_hor_seg == 2:
			self.hor_fov_arr.append([-180.,0.])
			self.hor_fov_arr.append([0.,180.])
			self.h_res = 0.703125
		elif self.num_hor_seg == 4:
			self.hor_fov_arr.append([-180.,-90.])
			self.hor_fov_arr.append([-90.,0.])
			self.hor_fov_arr.append([0.,90.])
			self.hor_fov_arr.append([90.,180.])
			self.h_res = 0.3515625
		# input buffer
		self.input_buf = np.zeros([self.y_max+1, self.x_max+1, 6], dtype=np.float32);
		# model
		dir_path = os.path.dirname(os.path.realpath(__file__))
		self.model = load_model(dir_path + '/../model/model.h5')
		self.graph = tf.get_default_graph()
		self.seg_thres = 0.07
		# subscribers
		self.subscriber = rp.Subscriber("/velodyne_points", PointCloud2, self.on_points_received)
		self.publisher = rp.Publisher("/detector/boxes", Float32MultiArray, queue_size=10)
		rp.loginfo("Detector: initialized")
		print "Detector: initialized"
		# #test
		# lidar = np.load(dir_path + '/../../one_frame_data/lidar_270.npy')
		# x = lidar[:,0]
		# y = lidar[:,1]
		# z = lidar[:,2]
		# with self.graph.as_default():
		# 	boxes, box_info = self.predict_boxes(x, y, z)
		# 	print len(boxes)
		# 	print boxes[0]
		# 	print box_info[0]

	def on_points_received(self, data):
		rp.loginfo("Detector: point received")
		if (self.process_locked):
			rp.loginfo("Detector: Process locked")
			return
		# lock process
		self.process_locked = True
		#rp.loginfo("Detector: process started")
		# process points
		x = []
		y = []
		z = []
		for p in pc2.read_points(data, skip_nans=True):
			x.append(p[0])
			y.append(p[1])
			z.append(p[2])
		box_info = np.empty((0,8))
		with self.graph.as_default():
			box_info = self.predict_boxes(x, y, z)
		arr = Float32MultiArray()
		flat_box_info = np.reshape(box_info, (-1))
		arr.data = flat_box_info.tolist()
		self.publisher.publish(arr)
		# unlock process
		rp.loginfo("Detector: published %d boxes", len(box_info))
		self.process_locked = False
	
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

def listen():
	processor = detector()
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('detector', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()

if __name__ == '__main__':
	listen()
