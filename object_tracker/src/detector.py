#!/usr/bin/env python
import rospy as rp
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import copy
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenates

class py_processor:

	def __init__(self):
		# lock
		self.process_locked = False
		self.receiver_locked = False
		# data received flags
		self.image_received = False
		self.points_received = False
		# data to process
		self.image_data = None
		self.points_data = None
		# subscribers
		self.image_subscriber = rp.Subscriber("/image_raw", Image, self.on_image_received)
		self.point_subscriber = rp.Subscriber("/velodyne_points", PointCloud2, self.on_points_received)
		# cv bridge
		self.bridge = CvBridge()
	
	def on_image_received(self, data):
		if self.receiver_locked:
			return
		rp.loginfo(rp.get_caller_id() + " Image received")
		self.image_data = data
		self.image_received = True
		self.process_image_and_points()
	
	def on_points_received(self, data):
		if self.receiver_locked:
			return
		rp.loginfo(rp.get_caller_id() + " Point received")
		self.points_data = data
		self.points_received = True
		#self.process_image_and_points()
		self.process_points()
	
	def process_points(self):
		if ((self.process_locked) or (not self.points_received)):
			rp.loginfo("- Process locked")
			return
		# lock process
		self.process_locked = True
		rp.loginfo("- Process started")
		# lock data receivers
		self.receiver_locked = True
		points = copy.deepcopy(self.points_data)
		# unlock data receivers
		self.points_received = False
		self.receiver_locked = False
		# process points
		x_pos = []
		y_pos = []
		z_pos = []
		for p in pc2.read_points(points, skip_nans=True):
			x_pos.append(p[0])
			y_pos.append(p[1])
			z_pos.append(p[2])
		rp.loginfo("-- %d Point converted, p0: %.2f %.2f %.2f", len(x_pos), x_pos[0], y_pos[0], z_pos[0])
		# unlock process
		rp.loginfo("- Process finished")
		self.process_locked = False
		# tensorflow test
		hello = tf.constant('Hello, TensorFlow!')
		sess = tf.Session()
		rp.loginfo(sess.run(hello))

	def process_image_and_points(self):
		if ((self.process_locked) or (not self.image_received) or (not self.points_received)):
			rp.loginfo("- Process locked")
			return
		# lock process
		self.process_locked = True
		rp.loginfo("- Process started")
		# lock data receivers
		self.receiver_locked = True
		image = copy.deepcopy(self.image_data)
		points = copy.deepcopy(self.points_data)
		# unlock data receivers
		self.image_received = False
		self.points_received = False
		self.receiver_locked = False		
		# process image
		try:
			bayer_image = self.bridge.imgmsg_to_cv2(image, "bayer_grbg8")
			rgb_img = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_GR2RGB)
		except CvBridgeError as e:
			print(e)
		rp.loginfo(rp.get_caller_id() + "-- Image converted, shape: %s", str(rgb_img.shape))
		# cv2.imwrite('/home/parkjaeil0108/challenge/Didi-Release-2/Data/1/test.jpg', rgb_img)
		# process points
		x_pos = []
		y_pos = []
		z_pos = []
		for p in pc2.read_points(points, skip_nans=True):
			x_pos.append(p[0])
			y_pos.append(p[1])
			z_pos.append(p[2])
		rp.loginfo("-- %d Point converted, p0: %.2f %.2f %.2f", len(x_pos), x_pos[0], y_pos[0], z_pos[0])
		# unlock process
		rp.loginfo("- Process finished")
		self.process_locked = False

def listen():
	processor = py_processor()
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('py_processor', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()

if __name__ == '__main__':
	listen()
