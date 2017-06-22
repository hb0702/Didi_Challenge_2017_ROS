#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

const int MAX_MARKER_COUNT = 1000;
const float MAX_VALUE = 1000.0f;

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;
typedef std::vector<char> _BitVector;
typedef float value_type;

namespace TeamKR
{

class Vector3
{
public:
	Vector3()
	{
		set(0.0f, 0.0f, 0.0f);
	}

	Vector3(value_type _x, value_type _y, value_type _z)
	{
		set(_x, _y, _z);
	}

	Vector3(const Vector3& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
	}

	Vector3(Vector3& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
	}

	Vector3& operator=(const Vector3& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
		return *this;
	}

	void set(value_type _x, value_type _y, value_type _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

public:
	value_type x;
	value_type y;
	value_type z;
};

class Cluster
{
public:
	Cluster(value_type resolution)
	{
		cellSize_ = resolution;
		hitCount_ = 0;
		min_.set(MAX_VALUE, MAX_VALUE, MAX_VALUE);
		max_.set(-MAX_VALUE, -MAX_VALUE, -MAX_VALUE);
	}

	~Cluster()
	{

	}

	void add(const Vector3& point, int hitCount)
	{
		points_.push_back(point);

		if (min_.x > point.x - 0.5 * cellSize_)
		{
			min_.x = point.x - 0.5 * cellSize_;
		}
		if (min_.y > point.y - 0.5 * cellSize_)
		{
			min_.y = point.y - 0.5 * cellSize_;
		}
		if (min_.z > point.z - 0.5 * cellSize_)
		{
			min_.z = point.z - 0.5 * cellSize_;
		}
		if (max_.x < point.x + 0.5 * cellSize_)
		{
			max_.x = point.x + 0.5 * cellSize_;
		}
		if (max_.y < point.y + 0.5 * cellSize_)
		{
			max_.y = point.y + 0.5 * cellSize_;
		}
		if (max_.z < point.z + 0.5 * cellSize_)
		{
			max_.z = point.z + 0.5 * cellSize_;
		}

		hitCount_ = hitCount;
	}

	const Vector3& min() const
	{
		return min_;
	}

	const Vector3& max() const
	{
		return max_;
	}

	int hitCount() const
	{
		return hitCount_;
	}

private:
	value_type cellSize_;
	int hitCount_;
	std::list<Vector3> points_;
	Vector3 min_;
	Vector3 max_;
}

#pragma region cluster builder
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
class ClusterBuilder
{
public:
	struct Index
	{
		Index(int _x, int _y)
		{
			x = _x;
			y = _y;
		}
		int x;
		int y;
	};

public:
	ClusterBuilder(value_type centerX, value_type centerY, value_type baseZ, value_type radius, value_type resolution)
	{
		centerX_ = centerX;
		centerY_ = centerY;
		originX_ = centerX - radius;
		originY_ = centerY - radius;
		baseZ_ = baseZ;
		cellSize_ = resolution;
		iradius_ = (int)(radius / resolution + 0.5f);
		iradius2_ = iradius_ * iradius_;
		iwidth_ = (int)(2 * radius / resolution + 0.5f);

		// init hit map
		hitmap_ = new int*[iwidth_];
		for (int ix = 0; ix < iwidth_; ++i)
		{
			hitmap_[ix] = new int[iwidth_];
			for (int iy = 0; iy < h_; ++iy)
			{
				hitmap_[ix][iy] = 0;
			}
		}

		// init depth map
		depthmap_ = new value_type*[iwidth_];
		for (int ix = 0; ix < iwidth_; ++i)
		{
			depthmap_[ix] = new value_type[iwidth_];
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				depthmap_[ix][iy] = baseZ;
			}
		}

		// init bit map
		bitmap_ = new char*[iwidth_];
		for (int ix = 0; ix < iwidth_; ++i)
		{
			bitmap_[ix] = new char[iwidth_];
			for (int iy = 0; iy < h_; ++iy)
			{
				bitmap_[ix][iy] = 0;
			}
		}
	}

	~ClusterBuilder()
	{
		// delete hit map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			delete[] hitmap_[ix];
		}
		delete[] hitmap_;

		// delete depth map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			delete[] depthmap_[ix];
		}
		delete[] depthmap_;

		// delete bit map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			delete[] bitmap_[ix];
		}
		delete[] bitmap_;
	}

	void run(const _PointVector& points, const _BitVector& filterBV, std::list<VoxelCluster>& clusters)
	{
		clear();

		_PointVector::const_iterator pit = points.begin();
		_BitVector::const_iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (*bit == 0)
			{
				hit(pit->x, pit->y, pit->z);
			}
		}

		std::stack<Index> seeds;
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				if (bitmap_[ix][iy] == 1 || hitCount(ix, iy) == 0)
				{
					continue;
				}

				seeds.clear();
				Cluster cluster(cellSize_);

				seed.push(Index(ix, iy));
				cluster.add(cellPoint(ix, iy), hitCount(ix, iy));

				while (!seeds.empty())
				{
					Index seed = seeds.top();
					seeds.pop();
					for (int ax = seed.x - 1; ax <= seed.x + 1; ++ax)
					{
						for (int ay = seed.y - 1; ay <= seed.y + 1; ++ay)
						{
							if ((ax == seed.x && ay == seed.y)
							|| ax == -1 || ay == -1 || ax == iwidth_ || ay == iwidth_
							|| bitmap_[ax][ay] == 1)
							{
								continue;
							}

							if (hitCount(ax, ay) > 0)
							{
								seeds.push(Index(ax, ay));
								cluster.add(cellPoint(ax, ay), hitCount(ax, ay));
							}
						}
					}
				}

				clusters.push_back(cluster);
			}
		}
	}

private:
	void clear()
	{
		// reset hit map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				hitmap_[ix][iy] = 0;
			}
		}

		// reset depth map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				depthmap_[ix][iy] = baseZ_;
			}
		}

		// reset bit map
		for (int ix = 0; ix < iwidth_; ++i)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				bitmap_[ix][iy] = 0;
			}
		}
	}

	void hit(double px, double py, double pz)
	{
		int rx = (int)((px - centerX_) / cellSize_ + 0.5);
		int ry = (int)((py - centerY_) / cellSize_ + 0.5);

		if (rx * rx + ry * ry > iradius2_)
		{
			return;
		}

		int x = (int)((px - originx_) / cellSize_ + 0.5);
		int y = (int)((py - originy_) / cellSize_ + 0.5);

		if (x < 0 || x >= iwidth_ || y < 0 || y >= iwidth_)
		{
			return;
		}

		hitmap_[x][y] += 1;
		if (depthmap_[x][y] < pz)
		{
			depthmap_[x][y] = pz;
		}
	}

	Vector3 cellPoint(int ix, int iy) const
	{
		return Vector3(originX_ + ((value_type)ix + 0.5) * cellSize_,
					originY_ + ((value_type)iy + 0.5) * cellSize_,
					depthmap_[ix][iy]);
	}

	int hitCount(int ix, int iy) const
	{
		return hitmap_[ix][iy];
	}

private:
	value_type centerX_;
	value_type centerY_;
	value_type originX_;
	value_type originY_;
	value_type baseZ_;
	value_type radius_;
	value_type cellSize_;
	int iwidth_;
	int iradius_;
	int iradius2_;
	int** hitmap_;
	value_type** depthmap_;
	char** bitmap_;
};
#pragma endregion

#pragma region filter
/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
class Filter
{
public:
	Filter(ros::NodeHandle n, const std::string& mode)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &Filter::onPointsReceived, this);
    	publisher_ = n.advertise<visualization_msgs::MarkerArray>("/filter/boxes", 1);
		cloud_ = _PointCloud::Ptr(new _PointCloud());

		// init cluster builder
		builder_ = new ClusterBuilder(0.0, 0.0, -1.27, 22/*radius*/, 0.4/*resolution*/)
		
		// ground filtering option
		maxGround_ = -1.27 + 0.2;

		// car filtering option
		carMin_ = Vector3(-1.5, -1.0, -1.3);
		carMax_ = Vector3(2.5, 1.0, 0.2);

		// cluster filtering option
		mode_ = mode;
		int clusterMinPointCount_;
		value_type pedMinWidth_;
		value_type carMinWidth_;
		value_type clusterMaxDepth_;

		// init markers
        for (int i = 0; i < MAX_MARKER_COUNT; i++)
        {
            visualization_msgs::Marker marker;
            marker.id = i;
            marker.header.frame_id = "velodyne";
            marker.type = marker.CUBE;
            marker.action = marker.ADD;
            marker.pose.position.x = 0.0;
            marker.pose.position.y = 0.0;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;
            marker.color.a = 0.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            markerArr_.markers.push_back(marker);
        }

        ROS_INFO("Filter: initialized");
	}

	~Filter()
	{
		delete builder_;
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

		// mark bit vector - car and ground points
		_BitVector filterBV(pointCount, 0);
		markCar(filterBV);
		markGround_simple(filterBV);

		// cluster
		std::list<Cluster> clusters;
		builder_->run(points, filterBV, clusters);
		size_t clusterCount = clusters.size();
		if (clusterCount == 0u)
		{
			return;
		}
		ROS_INFO("Filter: got %zd clusters", clusterCount);

		// TODO: filter cluster
		????

		// update markers
		int markerCnt = 0;
		std::vector<visualization_msgs::Marker>::iterator mit = markerArr_.markers.begin();
		_PointVector::const_iterator pit = sampledPoints.begin();
		for (; pit != sampledPoints.end(); ++pit, ++mit, ++markerCnt)
		{
			mit->pose.position.x = pit->x;
			mit->pose.position.y = pit->y;
			mit->pose.position.z = pit->z;
			mit->color.a = 0.3;
		}
		for (; mit != markerArr_.markers.end(); ++mit)
        {
            mit->color.a = 0.0;
        }

        // publish markers
        publisher_.publish(markerArr_);
	}

	void markCar(_BitVector& filterBV) const
	{
		_PointVector points = cloud_->points;
		_PointVector::const_iterator pit = points.begin();
		_BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->x > carMin_.x && pit->y > carMin_.y
				&& pit->x < carMax_.x && pit->y < carMax_.y)
			{
				*bit = 1;
			}
		}
	}

	void markGround_RANSAC(_BitVector& filterBV) const
	{
		pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients (true);
		seg.setModelType (pcl::SACMODEL_PLANE);
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setDistanceThreshold (0.02);
		seg.setInputCloud (cloud_);
		seg.segment (*inliers, *coefficients);

		if (inliers->indices.size () == 0)
		{
			return;
		}

		// for ()

		// _BitVector::iterator bit = groundBV.begin();
		// for (; pit != points.end(); ++pit, ++bit)
		// {
		// 	if (pit->z < maxBase_)
		// 	{
		// 		*bit = 1;
		// 	}
		// }
	}

	void markGround_simple(_BitVector& filterBV) const
	{
		_PointVector points = cloud_->points;
		_PointVector::const_iterator pit = points.begin();
		_BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < maxGround_)
			{
				*bit = 1;
			}
		}
	}

private:
	ros::Subscriber subscriber_;
	ros::Publisher publisher_;
	_PointCloud::Ptr cloud_;
	// cluster builder
	ClusterBuilder* builder_;	
	// ground filtering option
	value_type maxGround_;
	// car filtering option
	Vector3 carMin_;
	Vector3 carMax_;
	// cluster filtering option
	std::string mode_;
	int clusterMinPointCount_;
	value_type pedMinWidth_;
	value_type carMinWidth_;
	value_type clusterMaxDepth_;	
	// marker array
	visualization_msgs::MarkerArray markerArr_;
};
#pragma endregion

} // namespace TeamKR

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    std::string filterMode(argv[1]);
    TeamKR::Filter filter(n, filterMode);

    ros::spin();

    return 0;
}
