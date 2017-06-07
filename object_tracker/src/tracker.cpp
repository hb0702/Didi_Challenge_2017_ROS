#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;

_PointCloud::Ptr cloud (new _PointCloud);

void processLidarPoints(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    pcl::fromROSMsg(*msg, *cloud);
    _PointVector points = cloud->points;
    size_t pointCount = points.size();
    if (pointCount > 0)
    {
        _Point front = points.front();
        ROS_INFO("Got %zd points, first point: %.2f, %.2f, %.2f", pointCount, front.x, front.y, front.z);

        // To do: process lidar points
        // ...
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_points_listener");
    ros::NodeHandle n;

    ros::Subscriber sub = n.subscribe("/velodyne_points", 10, processLidarPoints);

    // To do: subscribe image processing result
    // ...

    // To do: process lidar points
    // ...

    // To do: post process (publish result?)
    // ...

    ros::spin();

    return 0;
}
