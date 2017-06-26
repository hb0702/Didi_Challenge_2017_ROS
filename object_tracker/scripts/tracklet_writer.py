#!/usr/bin/env python
import sys
import os
import rospy as rp
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultyArray, MultiArrayDimension
import sensor_msgs.point_cloud2 as pc2

class tracklet_writer:

	def __init__(self, input_file):
		self.box_subscriber = rp.Subscriber("/tracker/boxes", Float32MultyArray, self.on_box_received)
		self.image_subscriber = rp.Subscriber("/image_raw", Image, self.on_image_received)

		self.output_file = os.path.splitext(input_file)[0] + '.xml'
		self.collection = TrackletCollection()
		self.image_cnt = 0
		self.box = None
	
	def on_box_received(self, data):
		#rp.loginfo(rp.get_caller_id() + " Point received, %d", self.lidar_cnt)
		box = data.data
	
	def on_image_received(self, data):
		if self.box is not None:
			# add to tracklet collection
			np.save(self.output_folder + '/lidar_' + str(self.image_cnt) + '.npy', self.lidar)
		self.image_cnt += 1
	
	def write_file(self):
		if self.first_lidar_frame > -1:
			for i in range(self.first_lidar_frame+1):
				np.save(self.output_folder + '/lidar_' + str(i) + '.npy', self.first_lidar)

	def generate_tracklet(pred_model, input_folder, output_file, 
                      fixed_size=None, no_rotation=False, # fixed_size: [l, w, h]
                      cluster=True, seg_thres=0.5, cluster_dist=0.1, min_dist=1.5, neigbor_thres=3,
                      ver_fov=(-24.4, 15.), v_res=0.42,
                      num_hor_seg=2, # only 2 or 4
                      merge=True
                     ):
	    tracklet_list = 
	    
	    for nframe in range(648):
	        lidarfile = os.path.join(input_folder, 'lidar_' + str(nframe) + '.npy')
	        points = np.load(lidarfile)
	        
	        frame_tracklets = []
	        _, boxes = predict_boxes(pred_model, points, \
	                                cluster=cluster, seg_thres=seg_thres, cluster_dist=cluster_dist, \
	                                min_dist=min_dist, neigbor_thres=neigbor_thres, \
	                                ver_fov=ver_fov, v_res=v_res, num_hor_seg=num_hor_seg)

	        print('Frame ' + str(nframe) + ': ' + str(len(boxes)) + ' boxes detected')
	        
	        for nbox in range(len(boxes)):
	            tracklet = box_to_tracklet(boxes[nbox], nframe, fixed_size=fixed_size, no_rotation=no_rotation)
	            frame_tracklets.append(tracklet)
	        if len(frame_tracklets) > 0:
	            if merge:
	                merged_tracklet = merge_frame_tracklets(frame_tracklets)
	                tracklet_list.tracklets.append(merged_tracklet)
	            else:
	                tracklet_list.tracklets = tracklet_list.tracklets + frame_tracklets
	    
	    tracklet_list.write_xml(output_file)
	    print('Exported tracklet to ' + output_file)

def listen():
	writer = tracklet_writer(input_file = sys.argv[1])
	# In ROS, nodes are uniquely named. If two nodes with the same
	# node are launched, the previous one is kicked off. The
	# anonymous=True flag means that rospy will choose a unique
	# name for our 'listener' node so that multiple listeners can
	# run simultaneously.
	rp.init_node('tracklet_writer', anonymous=True)
	# spin() simply keeps python from exiting until this node is stopped
	rp.spin()
	rp.loginfo(rp.get_caller_id() + " Finished spinning")
	extractor.write_file()

if __name__ == '__main__':
	listen()
