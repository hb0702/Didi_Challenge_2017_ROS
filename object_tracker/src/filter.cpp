#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

const int MAX_MARKER_COUNT = 100;
const float MAX_VALUE = 1000.0f;

const float GROUND_Z = -1.27f;
const float GROUND_EPS = 0.1f;

const float RESOLUTION = 0.3f;
const float ROI_RADIUS = 21.0f;

const int CLUSTER_POINT_COUNT_THRESHOLD = 95;

const float PEDESTRIAN_MAX_WIDTH = 1.0f;
const float PEDESTRIAN_MIN_DEPTH = 1.3f;
const float PEDESTRIAN_MAX_DEPTH = 2.0f;
const float CAR_MAX_WIDTH = 4.5f;
const float CAR_MIN_DEPTH = 0.8f;
const float CAR_MAX_DEPTH = 1.7f;

const bool USE_RANSAC = true;
const bool PUBLISH_GROUND = false;
const int RANSAC_MAX_ITERATIONS = 150;

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
	Cluster(value_type resolution, value_type baseZ)
	{
		cellSize_ = resolution;
		pointCount_ = 0;
		min_.set(MAX_VALUE, MAX_VALUE, baseZ);
		max_.set(-MAX_VALUE, -MAX_VALUE, baseZ);
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

		pointCount_ += hitCount;
	}

	const Vector3& min() const
	{
		return min_;
	}

	const Vector3& max() const
	{
		return max_;
	}

	Vector3 center() const
	{
		return Vector3(0.5 * (min_.x + max_.x), 0.5 * (min_.y + max_.y), 0.5 * (min_.z + max_.z));
	}

	int pointCount() const
	{
		return pointCount_;
	}

private:
	value_type cellSize_;
	int pointCount_;
	std::list<Vector3> points_;
	Vector3 min_;
	Vector3 max_;
};

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
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			hitmap_[ix] = new int[iwidth_];
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				hitmap_[ix][iy] = 0;
			}
		}

		// init depth map
		depthmap_ = new value_type*[iwidth_];
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			depthmap_[ix] = new value_type[iwidth_];
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				depthmap_[ix][iy] = baseZ;
			}
		}

		// init bit map
		bitmap_ = new char*[iwidth_];
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			bitmap_[ix] = new char[iwidth_];
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				bitmap_[ix][iy] = 0;
			}
		}
	}

	~ClusterBuilder()
	{
		// delete hit map
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			delete[] hitmap_[ix];
		}
		delete[] hitmap_;

		// delete depth map
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			delete[] depthmap_[ix];
		}
		delete[] depthmap_;

		// delete bit map
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			delete[] bitmap_[ix];
		}
		delete[] bitmap_;
	}

	void run(const _PointVector& points, const _BitVector& filterBV, std::list<Cluster>& clusters)
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

		for (int ix = 0; ix < iwidth_; ++ix)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				if (bitmap_[ix][iy] == 1 || hitCount(ix, iy) == 0)
				{
					continue;
				}

				std::stack<Index> seeds;
				Cluster cluster(cellSize_, baseZ_);

				seeds.push(Index(ix, iy));
				cluster.add(cellPoint(ix, iy), hitCount(ix, iy));
				bitmap_[ix][iy] = 1;

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
							|| bitmap_[ax][ay] == 1 || hitCount(ax, ay) == 0)
							{
								continue;
							}

							seeds.push(Index(ax, ay));
							cluster.add(cellPoint(ax, ay), hitCount(ax, ay));
							bitmap_[ax][ay] = 1;
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
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				hitmap_[ix][iy] = 0;
			}
		}

		// reset depth map
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				depthmap_[ix][iy] = baseZ_;
			}
		}

		// reset bit map
		for (int ix = 0; ix < iwidth_; ++ix)
		{
			for (int iy = 0; iy < iwidth_; ++iy)
			{
				bitmap_[ix][iy] = 0;
			}
		}
	}

	void hit(double px, double py, double pz)
	{
		int rx = (int)(((value_type)px - centerX_) / cellSize_ + 0.5);
		int ry = (int)(((value_type)py - centerY_) / cellSize_ + 0.5);

		if (rx * rx + ry * ry > iradius2_)
		{
			return;
		}

		int x = (int)(((value_type)px - originX_) / cellSize_ + 0.5);
		int y = (int)(((value_type)py - originY_) / cellSize_ + 0.5);

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
		builder_ = new ClusterBuilder(0.0, 0.0, GROUND_Z, ROI_RADIUS, RESOLUTION);
		
		// ground filtering option
		maxGround_ = GROUND_Z + GROUND_EPS;

		// car filtering option
		carMin_ = Vector3(-1.5, -1.0, -1.3);
		carMax_ = Vector3(2.5, 1.0, 0.2);

		// cluster filtering option
		mode_ = mode;
		clusterMinPointCount_ = CLUSTER_POINT_COUNT_THRESHOLD;
		pedMaxWidth_ = PEDESTRIAN_MAX_WIDTH;
		pedMinZ_ = GROUND_Z + PEDESTRIAN_MIN_DEPTH;
		pedMaxZ_ = GROUND_Z + PEDESTRIAN_MAX_DEPTH;
		carMaxWidth_ = CAR_MAX_WIDTH;
		carMinZ_ = GROUND_Z + CAR_MIN_DEPTH;
		carMaxZ_ = GROUND_Z + CAR_MAX_DEPTH;

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

        if (PUBLISH_GROUND)
    	{
    		publisherGround_ = n.advertise<sensor_msgs::PointCloud2>("/filter/ground", 1);
    		cloudGround_ = _PointCloud::Ptr(new _PointCloud());
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
		_BitVector pointFilterBV(pointCount, 0);
		markCar(points, pointFilterBV);

		if (USE_RANSAC)
		{
			markGround_RANSAC(pointFilterBV);
		}
		else
		{
			markGround_simple(points, pointFilterBV);
		}

		// cluster
		std::list<Cluster> clusters;
		builder_->run(points, pointFilterBV, clusters);
		size_t clusterCount = clusters.size();
		if (clusterCount == 0u)
		{
			return;
		}
		
		// mark bit vector - bad  clusters
		_BitVector clusterFilterBV(clusterCount, 0);
		markBadCluster(clusters, clusterFilterBV);

		// publish clusters
		publishMarkers(clusters, clusterFilterBV);
	}

	void markCar(const _PointVector& points, _BitVector& filterBV) const
	{
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
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setInputCloud(cloud_);		
		// variables
		seg.setDistanceThreshold(GROUND_EPS);
		seg.setAxis(Eigen::Vector3f(0.0f,0.0f,1.0f));
		seg.setMaxIterations(RANSAC_MAX_ITERATIONS);
		seg.segment(*inliers, *coefficients);

		if (inliers->indices.size () == 0)
		{
			return;
		}

		std::vector<int>::const_iterator it = inliers->indices.begin();
		for (; it != inliers->indices.end(); ++it)
		{
			filterBV[*it] = 1;
		}

		if (PUBLISH_GROUND)
		{
			pcl::ExtractIndices<_Point> extract;
			extract.setInputCloud(cloud_);
			extract.setIndices(inliers);
			extract.setNegative(false);
			extract.filter(*cloudGround_);
			sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
			pcl::toROSMsg(*cloudGround_, *msg);
			publisherGround_.publish(msg);			
		}
	}

	void markGround_simple(const _PointVector& points, _BitVector& filterBV) const
	{
		if (PUBLISH_GROUND)
		{
			cloudGround_->points.clear();
		}

		_PointVector::const_iterator pit = points.begin();
		_BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < maxGround_)
			{
				*bit = 1;

				if (PUBLISH_GROUND)
				{
					cloudGround_->points.push_back(*pit);
				}
			}
		}

		if (PUBLISH_GROUND)
		{
			sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
			pcl::toROSMsg(*cloudGround_, *msg);
			msg->header.frame_id = "velodyne";
			publisherGround_.publish(msg);
		}
	}

	void markBadCluster(const std::list<Cluster>& clusters, _BitVector& filterBV) const
	{
		std::list<Cluster>::const_iterator cit = clusters.begin();
		_BitVector::iterator bit = filterBV.begin();
		for (; cit != clusters.end(); ++cit, ++bit)
		{
			// min point count
			if (cit->pointCount() < clusterMinPointCount_)
			{
				*bit = 1;
				continue;
			}

			// cluster size
			value_type top = cit->max().z;			
			value_type maxWidth = std::max(cit->max().x - cit->min().x, cit->max().y - cit->min().y);
			if (mode_ == "car")
			{
				if (maxWidth < pedMaxWidth_ || maxWidth > carMaxWidth_
					|| top < carMinZ_ || top > carMaxZ_)
				{
					*bit = 1;
					continue;
				}
			}
			else if (mode_ == "ped")
			{
				if (maxWidth > pedMaxWidth_ || top < pedMinZ_ || top > pedMaxZ_)
				{
					*bit = 1;
					continue;
				}
			}
			else if (mode_ == "car_ped")
			{
				if (maxWidth > carMaxWidth_ || top < carMinZ_ || top > pedMaxZ_)
				{
					*bit = 1;
					continue;
				}
			}
		}
	}

	void publishMarkers(const std::list<Cluster>& clusters, const _BitVector& filterBV)
	{
		// update markers
		int markerCnt = 0;
		std::list<Cluster>::const_iterator cit = clusters.begin();
		_BitVector::const_iterator bit = filterBV.begin();
		std::vector<visualization_msgs::Marker>::iterator mit = markerArr_.markers.begin();
		for (; cit != clusters.end(); ++cit, ++bit)
		{
			if (*bit == 0)
			{
				ROS_INFO("Filter: points %d, depth %f, width %f, center %f %f %f", cit->pointCount(), cit->max().z, 
					std::max(cit->max().x - cit->min().x, cit->max().y - cit->min().y),
					cit->center().x, cit->center().y, cit->center().z);
				Vector3 center = cit->center();
				mit->pose.position.x = center.x;
				mit->pose.position.y = center.y;
				mit->pose.position.z = center.z;
				mit->scale.x = cit->max().x - cit->min().x;
				mit->scale.y = cit->max().y - cit->min().y;
				mit->scale.z = cit->max().z - cit->min().z;
				mit->color.a = 0.3;

				++mit;
				++markerCnt;
			}
		}
		for (; mit != markerArr_.markers.end(); ++mit)
        {
            mit->color.a = 0.0;
        }

        // publish markers
        publisher_.publish(markerArr_);
        ROS_INFO("Filter: published %d markers", markerCnt);
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
	value_type pedMaxWidth_;
	value_type pedMinZ_;
	value_type pedMaxZ_;
	value_type carMaxWidth_;
	value_type carMinZ_;
	value_type carMaxZ_;
	// marker array
	visualization_msgs::MarkerArray markerArr_;
	// publish ground
	ros::Publisher publisherGround_;
	_PointCloud::Ptr cloudGround_;
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
