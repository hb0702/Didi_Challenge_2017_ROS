#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/core/core.hpp>

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;

const float _PI = 3.14159265358979f;

class CProcessor
{
public:
	struct FOV
	{
		FOV(float min, float max)
		{
			this->min = min;
			this->max = max;
		}

		float min;
		float max;
	};

public:
	CProcessor()
	{
		n_ = ros::NodeHandle();
    	subscriber_ = n_.subscribe("/velodyne_points", 10, processLidarPoints);

		cloud_ = new _PointCloud();

		//	ver_fov : angle range of vertical projection in degree
		//	hor_fov : angle range of horizantal projection in degree
		//	v_res : vertical resolusion
		//	h_res : horizontal resolution
		//	d_max : maximun range distance
		verFov_ = FOV(-24.9f, 2.0f);
		horFov_ = FOV(-180.0f, 180.0f);
		vRes_ = 0.42f;
		hRes_ = 0.35f;
		dMax_ = -1.0f;
		
		xMax_ = (int)std::ceil((horFov_.max - horFov_.min) / hRes_);
		yMax_ = (int)std::ceil((verFov_.max - verFov_.min) / vRes_);

		dMat_ = cv::Mat(yMax_ + 1, xMax_ + 1, CV_32S);
		zMat_ = cv::Mat(yMax_ + 1, xMax_ + 1, CV_32S);

		// create a 100x100x100 8-bit array
		int sz[] = {100, 100, 100};
		Mat bigCube(3, sz, CV_8U, Scalar::all(0));
	}

	~CProcessor()
	{
		delete cloud_;
	}

public:
	//	return : cylindrical projection(or panorama view) of lidar
	void cylindericalProjection(const _PointVector& points)
	{
		float x, y, z, d, theta, phi;
		int xView, yView;

		for (_PointVector::const_iterator it = points.begin(); it != points.end(); ++it)
		{
			x = it->x;
			y = it->y;
			z = it->z;
			d = std::sqrt(x*x + y*y);

			if (dMax_ > 0.0f && d > dMax_)
			{
				d = dMax_;
			}

			theta = std::atan2(y, x);
			phi = std::atan2(z, d);

			xView = (int)std::ceil((theta * 180.0 / _PI - horFov_.min) / hRes_);
			yView = (int)std::ceil((phi * 180.0 / _PI - verFov_.min) / vRes_);

			if (xView < 0 || xView > xMax_ || yView < 0 || yView > yMax_)
			{
				continue;
			}

		    dMat_.at<float>(yView, xView) = d;
		    zMat_.at<float>(yView, xView) = z;
		}
	}

	void processLidarPoints(const sensor_msgs::PointCloud2::ConstPtr& msg)
	{
		pcl::fromROSMsg(*msg, *cloud_);
		_PointVector points = cloud_->points;
		size_t pointCount = points.size();
		if (pointCount > 0)
		{
		    _Point front = points.front();
		    ROS_INFO("Got %zd points, first point: %.2f, %.2f, %.2f", pointCount, front.x, front.y, front.z);
		    cylindericalProjection(points);
		}
	}

private:
	ros::NodeHandle n_;
	ros::Subscriber subscriber_;

	_PointCloud::Ptr cloud_;
	
	FOV verFov_;
	FOV horFov_;
	float vRes_;
	float hRes_;
	float dMax_;
	int xMax_;
	int yMax_;
	cv::Mat dMat_;
	cv::Mat zMat_;

	// To do: make lock
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidar_points_listener");
    CProcessor processor;
    ros::spin();

    return 0;
}



