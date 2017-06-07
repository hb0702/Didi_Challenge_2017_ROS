#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/player.h>
#include <sensor_msgs/Image.h>
#include <velodyne_msgs/VelodyneScan.h>
#include "std_msgs/String.h"
#include <boost/foreach.hpp>
#include <ros/package.h>

int main(int argc, char **argv)
{
    ros::init(argc, argv, "publish_scene");
	
	rosbag::PlayerOptions opts;
	
	std::string fileName(argv[1]);
	opts.bags.push_back(fileName);

	std::string bagImageTopic = "/image_raw";
	std::string bagLidarTopic = "/velodyne_packets";
	
	opts.topics.push_back(bagImageTopic);
	opts.topics.push_back(bagLidarTopic);
	
	rosbag::Player player(opts);
	
	try
	{
		player.publish();
	}
	catch (std::runtime_error& e)
	{
		ROS_FATAL("%s", e.what());
		return 1;
	}
	
	return 0;
}
