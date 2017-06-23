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

const int MAX_MARKER_COUNT = 5000;
const float _PI = 3.14159265358979f;

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;

class V3d
{
public:
	V3d()
	{
		set(0.0, 0.0, 0.0);
	}

	V3d(double _x, double _y, double _z)
	{
		set(_x, _y, _z);
	}

	V3d(const V3d& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
	}

	V3d(V3d& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
	}

	V3d& operator=(const V3d& rhs)
	{
		set(rhs.x, rhs.y, rhs.z);
		return *this;
	}

	void set(double _x, double _y, double _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}

public:
	double x;
	double y;
	double z;
};

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
class PointSampler
{
public:
	PointSampler(double originX, double originY, double originZ,
			double width, double height, double depth,
			double resolution)
	{
		ox_ = originX;
		oy_ = originY;
		oz_ = originZ;
		width_ = width;
		height_ = height;
		depth_ = depth;
		r_ = resolution;
		w_ = (int)(width / resolution + 0.5);
		h_ = (int)(height / resolution + 0.5);
		d_ = (int)(depth / resolution + 0.5);

		// init grid
		grid_ = new int**[w_];
		for (int i = 0; i < w_; ++i)
		{
			grid_[i] = new int*[h_];
			for (int j = 0; j < h_; ++j)
			{
				grid_[i][j] = new int[d_];

				for (int k = 0; k < d_; ++k)
				{
					grid_[i][j][k] = 0;
				}
			}
		}

		// create hit thres map - c + d / r^2
		int c = 1;
		double d = (0.5 * (double)w_) * (0.5 * (double)h_) * 0.3;
		double exp = 1.8;
		int cx = w_ / 2;
		int cy = h_ / 2;
		int cz = d_ / 2;
		double dx, dy, dz;
		hitThres_ = new int**[w_];
		for (int i = 0; i < w_; ++i)
		{	
			hitThres_[i] = new int*[h_];
			for (int j = 0; j < h_; ++j)
			{
				hitThres_[i][j] = new int[d_];
				for (int k = 0; k < d_; ++k)
				{
					dx = (double)std::abs(i - cx);
					dy = (double)std::abs(j - cy);
					dz = (double)std::abs(k - cz);
					if (dx > 1 && dy > 1 && dz > 1)
					{
						hitThres_[i][j][k] = c 
										+ (int)(d 
											/ ((double)std::pow(dx, exp) 
												+ (double)std::pow(dy, exp) 
												+ (double)std::pow(dz, exp)
												+ 0.0001 // to prevent divide by zero
												)
											);
										printf("%d\n", hitThres_[i][j][k]);
					}
					else
					{

					}
				}
			}
		}
	}

	~PointSampler()
	{
		// delete grid
		for (int i = 0; i < w_; ++i)
		{
			for (int j = 0; j < h_; ++j)
			{
				delete[] grid_[i][j];
			}
			delete[] grid_[i];
		}
		delete[] grid_;

		// delete hit thres map
		for (int i = 0; i < w_; ++i)
		{
			for (int j = 0; j < h_; ++j)
			{
				delete[] hitThres_[i][j];
			}
			delete[] hitThres_[i];
		}
		delete[] hitThres_;
	}

private:
	void clear()
	{
		// reset grid values to 0
		for (int i = 0; i < w_; ++i)
		{
			for (int j = 0; j < h_; ++j)
			{
				for (int k = 0; k < d_; ++k)
				{
					grid_[i][j][k] = 0;
				}
			}
		}

		output_.clear();
	}

	void hit(double px, double py, double pz)
	{
		int x = (int)((px - ox_) / r_ + 0.5);
		int y = (int)((py - oy_) / r_ + 0.5);
		int z = (int)((pz - oz_) / r_ + 0.5);

		if (x < 0 || x >= w_
			|| y < 0 || y >= h_
			|| z < 0 || z >= d_)
		{
			return;
		}

		grid_[x][y][z] += 1;
	}

	_Point centroid(int ix, int iy, int iz) const
	{
		return _Point(ox_ + ((double)ix + 0.5) * r_, oy_ + ((double)iy + 0.5) * r_, oz_ + ((double)iz + 0.5) * r_);
	}

public:
	void run(const _PointVector& points, const std::vector<char> filterBV)
	{
		// clear
		clear();
		
		// mark grid
		_PointVector::const_iterator pit = points.begin();
		std::vector<char>::const_iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (*bit != 1)
			{
				hit(pit->x, pit->y, pit->z);
			}
		}

		// fill output
		for (int i = 0; i < w_; ++i)
		{
			for (int j = 0; j < h_; ++j)
			{
				for (int k = 0; k < d_; ++k)
				{
					if (grid_[i][j][k] > hitThres_[i][j][k])
					{
						output_.push_back(centroid(i, j, k));
					}
				}
			}
		}
		ROS_INFO("Filter: output points: %zd", output_.size());
	}

public:
	const _PointVector& output()
	{
		return output_;
	}

private:
	double ox_;
	double oy_;
	double oz_;
	double width_;
	double height_;
	double depth_;
	double r_;
	int w_;
	int h_;
	int d_;
	int*** grid_;
	int*** hitThres_;
	// output
	_PointVector output_;
};

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
class Filter
{
public:
	Filter(ros::NodeHandle n)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &Filter::onPointsReceived, this);
    	publisher_ = n.advertise<visualization_msgs::MarkerArray>("/filter/boxes", 1);
		cloud_ = _PointCloud::Ptr(new _PointCloud());
		maxBase_ = -1.27 + 0.2;
		sampler_ = new PointSampler(-40.0, -40.0, -1.27, 80.0, 80.0, 4, 0.4);

		carMin_ = V3d(-1.5, -1.0, -1.3);
		carMax_ = V3d(2.5, 1.0, 0.2);

		ROS_INFO("Filter: initialized");

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
	}

	~Filter()
	{
		delete sampler_;
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
		std::vector<char> filterBV(pointCount, 0);
		markCar(filterBV);
		markGround_simple(filterBV);

		// sample points
		sampler_->run(points, filterBV);
		_PointVector sampledPoints = sampler_->output();

		// // cluster
		// std::list<VoxelCluster> clusters;
		// voxelMap_->makeClusters(clusterHitThres_, clusters);
		// size_t clusterCount = clusters.size();
		// if (clusterCount == 0u)
		// {
		// 	return;
		// }
		// ROS_INFO("Filter: got %zd clusters", clusterCount);

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

	void markCar(std::vector<char>& filterBV) const
	{
		_PointVector points = cloud_->points;
		_PointVector::const_iterator pit = points.begin();
		std::vector<char>::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->x > carMin_.x && pit->y > carMin_.y
				&& pit->x < carMax_.x && pit->y < carMax_.y)
			{
				*bit = 1;
			}
		}
	}

	void markGround_RANSAC(std::vector<char>& filterBV) const
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

		// std::vector<char>::iterator bit = groundBV.begin();
		// for (; pit != points.end(); ++pit, ++bit)
		// {
		// 	if (pit->z < maxBase_)
		// 	{
		// 		*bit = 1;
		// 	}
		// }
	}

	void markGround_simple(std::vector<char>& filterBV) const
	{
		_PointVector points = cloud_->points;
		_PointVector::const_iterator pit = points.begin();
		std::vector<char>::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < maxBase_)
			{
				*bit = 1;
			}
		}
	}

private:
	ros::Subscriber subscriber_;
	ros::Publisher publisher_;
	_PointCloud::Ptr cloud_;
	double maxBase_;
	V3d carMin_;
	V3d carMax_;
	PointSampler* sampler_;
	visualization_msgs::MarkerArray markerArr_;
};

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    Filter filter(n);

    ros::spin();

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
// class VoxelCluster
// {
// public:
// 	VoxelCluster(double resolution, double maxBase, const V3d& minSize, const V3d& maxSize)
// 	{
// 		r_ = resolution;
// 		maxBase_ = maxBase;
// 		minSize_ = minSize;
// 		maxSize_ = maxSize;
// 		min_.set(1000, 1000, 1000);
// 		max_.set(-1000, -1000, -1000);
// 	}

// 	~VoxelCluster()
// 	{

// 	}

// 	void add(const V3d& center)
// 	{
// 		points_.push_back(center);

// 		if (min_.x > center.x - 0.5 * r_)
// 		{
// 			min_.x = center.x - 0.5 * r_;
// 		}
// 		if (min_.y > center.y - 0.5 * r_)
// 		{
// 			min_.y = center.y - 0.5 * r_;
// 		}
// 		if (min_.z > center.z - 0.5 * r_)
// 		{
// 			min_.z = center.z - 0.5 * r_;
// 		}
// 		if (max_.x < center.x + 0.5 * r_)
// 		{
// 			max_.x = center.x + 0.5 * r_;
// 		}
// 		if (max_.y < center.y + 0.5 * r_)
// 		{
// 			max_.y = center.y + 0.5 * r_;
// 		}
// 		if (max_.z < center.z + 0.5 * r_)
// 		{
// 			max_.z = center.z + 0.5 * r_;
// 		}
// 	}

// 	const std::list<V3d>& points() const
// 	{
// 		return points_;
// 	}

// 	bool valid() const
// 	{

// 		double w = max_.x - min_.x;
// 		double h = max_.y - min_.y;
// 		double d = max_.z - min_.z;

// 		return min_.z < maxBase_ 
// 			&& w < maxSize_.x && h < maxSize_.y && d < maxSize_.z
// 			&& (w > minSize_.x || h > minSize_.y) && d > minSize_.z;
// 	}

// private:
// 	std::list<V3d> points_;
// 	V3d min_;
// 	V3d max_;
// 	double r_;
// 	double maxBase_;
// 	V3d minSize_;
// 	V3d maxSize_;
// };

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
// class VoxelMap
// {
// public:
// 	struct Index
// 	{
// 		Index(int _x, int _y, int _z)
// 		{
// 			x = _x;
// 			y = _y;
// 			z = _z;
// 		}

// 		int x;
// 		int y;
// 		int z;
// 	};

// public:
// 	VoxelMap(double originX, double originY, double originZ,
// 			double width, double height, double depth,
// 			double resolution)
// 	{
// 		ox_ = originX;
// 		oy_ = originY;
// 		oz_ = originZ;
// 		width_ = width;
// 		height_ = height;
// 		depth_ = depth;
// 		r_ = resolution;
// 		w_ = (int)(width / resolution + 0.5);
// 		h_ = (int)(height / resolution + 0.5);
// 		d_ = (int)(depth / resolution + 0.5);

// 		clusterMaxBase_ = originZ + 0.6;
// 		clusterMinSize_ = V3d(3 * r_ - 0.5, 3 * r_ - 0.5, 2 * r_ - 0.5);
// 		clusterMaxSize_ = V3d(4.1, 4.1, 2.1);

// 		// init grid
// 		grid_ = new int**[w_];
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			grid_[i] = new int*[h_];
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				grid_[i][j] = new int[d_];

// 				for (int k = 0; k < d_; ++k)
// 				{
// 					grid_[i][j][k] = 0;
// 				}
// 			}
// 		}

// 		// create hitmap
// 		hitmap_ = new char**[w_];
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			hitmap_[i] = new char*[h_];
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				hitmap_[i][j] = new char[d_];
// 				for (int k = 0; k < d_; ++k)
// 				{
// 					hitmap_[i][j][k] = 0;
// 				}
// 			}
// 		}

// 		// create hit thres map - c + d / r^2
// 		int c = 1;
// 		double d = (0.5 * width) * (0.5 * height) * 0.5;
// 		int icenterx = w_ / 2;
// 		int icentery = h_ / 2;
// 		int icenterz = d_ / 2;
// 		hitThres_ = new int**[w_];
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			hitThres_[i] = new int*[h_];
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				hitThres_[i][j] = new int[d_];
// 				for (int k = 0; k < d_; ++k)
// 				{
// 					hitThres_[i][j][k] = c + (int)(d / ());
// 				}
// 			}
// 		}
// 	}

// 	~VoxelMap()
// 	{
// 		// delete grid
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				delete[] grid_[i][j];
// 			}
// 			delete[] grid_[i];
// 		}
// 		delete[] grid_;

// 		// delete hitmap
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				delete[] hitmap_[i][j];
// 			}
// 			delete[] hitmap_[i];
// 		}
// 		delete[] hitmap_;

// 		// delete hit thres map
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				delete[] hitThres_[i][j];
// 			}
// 			delete[] hitThres_[i];
// 		}
// 		delete[] hitThres_;
// 	}

// private:
// 	void clear()
// 	{
// 		// reset grid values to 0
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				for (int k = 0; k < d_; ++k)
// 				{
// 					grid_[i][j][k] = 0;
// 				}
// 			}
// 		}

// 		// reset hitmap
// 		for (int i = 0; i < w_; ++i)
// 		{
// 			for (int j = 0; j < h_; ++j)
// 			{
// 				for (int k = 0; k < d_; ++k)
// 				{
// 					hitmap_[i][j][k] = 0;
// 				}
// 			}
// 		}
// 	}

// 	void hit(double px, double py, double pz)
// 	{
// 		int x = (int)((px - ox_) / r_ + 0.5);
// 		int y = (int)((py - oy_) / r_ + 0.5);
// 		int z = (int)((pz - oz_) / r_ + 0.5);

// 		if (x < 0 || x >= w_
// 			|| y < 0 || y >= h_
// 			|| z < 0 || z >= d_)
// 		{
// 			return;
// 		}

// 		grid_[x][y][z] += 1;
// 	}

// 	void makeClusters(int hitThres, std::list<VoxelCluster>& clusters) const
// 	{
// 		// start from the top layer
// 		for (int i = 1; i < w_ - 1; ++i)
// 		{
// 			for (int j = 1; j < h_ - 1; ++j)
// 			{
// 				for (int k = d_ - 2; k > 0; --k)
// 				{
// 					if (hitmap_[i][j][k] == 1)
// 					{						
// 						continue;
// 					}

// 					std::stack<Index> seeds;
// 					VoxelCluster cluster(r_, clusterMaxBase_, clusterMinSize_, clusterMaxSize_);
					
// 					seeds.push(Index(i, j, k));
// 					cluster.add(centroid(i, j, k));
// 					hitmap_[i][j][k] = 1;

// 					while (!seeds.empty())
// 					{
// 						Index seed = seeds.top();
// 						seeds.pop();
// 						for (int x = seed.x - 1; x <= seed.x + 1; ++x)
// 						{
// 							for (int y = seed.y - 1; y <= seed.y + 1; ++y)
// 							{
// 								for (int z = seed.z - 1; z <= seed.z + 1; ++z)
// 								{
// 									if ((x == seed.x && y == seed.y && z == seed.z)
// 										|| x == -1 || y == -1 || z == -1
// 										|| x == w_ || y == h_ || z == d_
// 										|| hitmap_[x][y][z] == 1)
// 									{
// 										continue;
// 									}
// 									if (test(x, y, z, hitThres))
// 									{
// 										if (z > 0)
// 										{
// 											seeds.push(Index(x, y, z));
// 										}
// 										cluster.add(centroid(x, y, z));
// 									}
// 									hitmap_[x][y][z] = 1;
// 								}
// 							}
// 						}
// 					}

// 					if (cluster.valid())
// 					{
// 						clusters.push_back(cluster);
// 					}
// 				}
// 			}
// 		}
// 	}


// private:
// 	int value(int ix, int iy, int iz) const
// 	{
// 		return grid_[ix][iy][iz];
// 	}

// 	bool test(int ix, int iy, int iz, int thres) const
// 	{
// 		return grid_[ix][iy][iz] >= thres;
// 	}

// 	V3d centroid(int ix, int iy, int iz) const
// 	{
// 		return V3d(ox_ + ((double)ix + 0.5) * r_,
// 					oy_ + ((double)iy + 0.5) * r_,
// 					oz_ + ((double)iz + 0.5) * r_);
// 	}

// private:
// 	double ox_;
// 	double oy_;
// 	double oz_;
// 	double width_;
// 	double height_;
// 	double depth_;
// 	double r_;
// 	int w_;
// 	int h_;
// 	int d_;
// 	double clusterMaxBase_;
// 	V3d clusterMinSize_;
// 	V3d clusterMaxSize_;
// 	int*** grid_;
// 	char*** hitmap_;
// 	int*** hitThres_;
// };
