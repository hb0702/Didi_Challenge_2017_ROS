#!/usr/bin/env python
import sys
import os
import rospy as rp
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2

class tracklet_writer:

	def __init__(self, input_file):
		self.point_subscriber = rp.Subscriber("/velodyne_points", PointCloud2, self.on_points_received)
		self.image_subscriber = rp.Subscriber("/image_raw", Image, self.on_image_received)
		self.output_folder = os.path.splitext(input_file)[0] + '/'
		if not os.path.exists(self.output_folder):
			os.makedirs(self.output_folder)
		self.lidar_cnt = 0
		self.image_cnt = 0
		self.first_lidar = None
		self.lidar = None
		self.first_lidar_frame = -1
	
	def on_points_received(self, data):
		#rp.loginfo(rp.get_caller_id() + " Point received, %d", self.lidar_cnt)
		points = []
		for p in pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z")):
			points.append([p[0], p[1], p[2]])
		self.lidar = np.array(points)
		#rp.loginfo("-- Shape: %s", str(points.shape))
		#rp.loginfo("-- %s", self.output_folder + 'lidar_' + str(self.output_cnt) + '.npy')
		if self.lidar_cnt == 0:
			self.first_lidar = self.lidar
		self.lidar_cnt += 1
	
	def on_image_received(self, data):
		#rp.loginfo(rp.get_caller_id() + " Frame %d", self.image_cnt)
		if self.lidar is None:
			self.first_lidar_frame = self.image_cnt
		else:
			np.save(self.output_folder + '/lidar_' + str(self.image_cnt) + '.npy', self.lidar)
		self.image_cnt += 1
	
	def extract_first_points(self):
		if self.first_lidar_frame > -1:
			for i in range(self.first_lidar_frame+1):
				np.save(self.output_folder + '/lidar_' + str(i) + '.npy', self.first_lidar)

def listen():
	extractor = tracklet_writer(input_file = sys.argv[1])
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('tracklet_writer', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()
	rp.loginfo(rp.get_caller_id() + " Finished spinning")
	extractor.extract_first_points()

if __name__ == '__main__':
	listen()
