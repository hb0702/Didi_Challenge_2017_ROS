#!/usr/bin/env python
import sys
import os
import rospy as rp
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import sensor_msgs.point_cloud2 as pc2
import tracklet as t

class tracklet_writer:

	def __init__(self, input_file, output_folder):
		print('tracklet_writer: read bag file from ' + input_file)
		print('tracklet_writer: save tracklet in ' + output_folder)
		if (not os.path.exists(output_folder)):
			os.makedirs(output_folder)
		self.output_folder = output_folder
		output_filebase = os.path.splitext(os.path.basename(input_file))[0] + '.xml'
		self.output_file = os.path.join(output_folder, output_filebase)
		self.box_subscriber = rp.Subscriber("/tracker/boxes", Float32MultiArray, self.on_box_received)
		self.collection = t.TrackletCollection()
		
	def on_box_received(self, data):
		#rp.loginfo(rp.get_caller_id() + " Point received, %d", self.lidar_cnt)
		self.boxes = []
		box_arr = np.array(data.data).reshape(-1, 9)
		for box in box_arr:
			num_frame = box[0]
			object_type = 'Pedestrian' if box[1] == 0 else 'Car'
			l = box[5]
			w = box[6]
			h = box[7]
			tracklet = t.Tracklet(object_type=object_type, l=l, w=w, h=h, first_frame=num_frame)
			pos = {'tx':box[2], 'ty':box[3], 'tz':box[4], 'rx':0.0, 'ry':0.0, 'rz':box[8]}
			tracklet.poses.append(pos)
			self.collection.tracklets.append(tracklet)

	def write_file(self):
		for tracklet in self.collection.tracklets:
			print("frameid: " + str(tracklet.first_frame))
		self.collection.write_xml(self.output_file)
		print('tracklet_writer: exported tracklet to ' + self.output_file)

def listen():
	writer = tracklet_writer(input_file=sys.argv[1], output_folder=sys.argv[2])
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('tracklet_writer', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()
	rp.loginfo(rp.get_caller_id() + " Finished spinning")
	writer.write_file()

if __name__ == '__main__':
	listen()
