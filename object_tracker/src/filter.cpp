#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;

const float _PI = 3.14159265358979f;

class Filter
{
public:
	Filter(ros::NodeHandle n)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &Filter::onPointsReceived, this);
		cloud_ = _PointCloud::Ptr(new _PointCloud());

		ROS_INFO("Filter: initialized");
	}

	~Filter()
	{

	}

private:
	void onPointsReceived(const sensor_msgs::PointCloud2::ConstPtr& msg)
	{
		pcl::fromROSMsg(*msg, *cloud_);
		_PointVector points = cloud_->points;
		size_t pointCount = points.size();
		if (pointCount == 0u)
		{
			return;
		}

		_Point front = points.front();
		ROS_INFO("Filter: got %zd points, first point: %.2f, %.2f, %.2f", pointCount, front.x, front.y, front.z);

		// do whatever you want!
	}

private:
	ros::Subscriber subscriber_;
	_PointCloud::Ptr cloud_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    Filter filter(n);

    ros::spin();

    return 0;
}



