#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

const float _PI = 3.14159265358979f;

typedef pcl::PointXYZ _Point;
typedef pcl::PointCloud<pcl::PointXYZ> _PointCloud;
typedef _PointCloud::VectorType _PointVector;

template<_Value>
struct _V3
{
public:
	_V3(_Value _x, _Value _y, _Value _z)
	{
		set(_x, _y, _z);
	}

	void set(_Value _x, _Value _y, _Value _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}


	_Value x;
	_Value y;
	_Value z;
};

typedef _V3<int> V3i;
typedef _V3<double> V3d;

class Cluster
{
public:
	Cluster()
	{

	}

	~Cluster()
	{

	}

private:
	std::vector<V3d> points_;
}

class VoxelMap
{
public:
	VoxelMap(double originX, double originY, double originZ,
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
	}

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
	}

	~VoxelMap()
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
	}

public:	
	void hit(double px, double py, double pz)
	{
		int x = (int)((px - ox_) / r_ + 0.5);
		int y = (int)((py - oy_) / r_ + 0.5);
		int z = (int)((pz - oz_) / r_ + 0.5);

		if (x < 0 || x > w_ - 1
			|| y < 0 || y > h_ - 1
			|| z < 0 || z > d_ - 1)
		{
			return;
		}

		grid_[x][y][z] += 1;
	}

	void makeClusters(int hitThres, std::vector<Cluster>& clusters)
	{
		// create hitmap
		char*** hitmap;
		for (int i = 0; i < w_; ++i)
		{
			hitMap[i] = new char*[h_];

			for (int j = 0; j < h_; ++j)
			{
				hitMap[i][j] = new char[d_];

				for (int k = 0; k < d_; ++k)
				{
					hitMap[i][j][k] = 0;
				}
			}
		}

		std::stack<V3i> seeds;
		V3i vec;
		// start from the top layer
		for (int i = 1; i < w_ - 1; ++i)
		{
			for (int j = 1; j < h_ - 1; ++j)
			{
				for (int k = d_ - 2; k > 0; --k)
				{
					if (hitmap[i][j][k] == 1)
					{
						continue;
					}

					seeds.clear();
					Cluster cluster;

					vec.set(i, j, k);
					seeds.push(vec);
					clusters.push(vec);
					hitmap[i][j][k] = 1;

					while (!seeds.empty())
					{
						V3i seed = seeds.top();
						seeds.pop();
						for (int x = i - 1; x <= i + 1; ++x)
						{
							for (int y = j - 1; y <= j + 1; ++y)
							{
								for (int z = k - 1; z <= k + 1; ++z)
								{
									if ((x == i && y == j && z == k) || hitmap[x][y][z] == 1)
									{
										continue;
									}
									if (test(x, y, z, hitThres))
									{
										vec.set(x, y, z);
										seeds.push(vec);
										clusters.push(vec);
									}
									hitmap[x][y][z] = 1;
								}
							}
						}

						clusters.push_back(cluster);
					}
				}
			}
		}

		// delete hitmap
		for (int i = 0; i < w_; ++i)
		{
			for (int j = 0; j < h_; ++j)
			{
				delete[] hitmap[i][j];
			}

			delete[] hitmap[i];
		}

		delete[] hitmap;
	}


private:
	int value(int ix, int iy, int iz) const
	{
		return grid_[x][y][z];
	}

	bool test(int ix, int iy, int iz, int thres) const
	{
		return grid_[ix][iy][iz] >= thres;
	}

	V3d toCentroid(int ix, int iy, int iz) const
	{
		return V3d(ox_ + (ix + 0.5) * r_,
			oy_ + (iy + 0.5) * r_,
			oz_ + (iz + 0.5) * r_);
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
};

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

		// get ground bit vector
		std::vector<char>& groundBV(pointCount, 0);
		markGroundBV(points, groundBV);


	}

	void markGroundBV(const _PointVector& points, std::vector<char>& groundBV) const
	{
		_PointVector::iterator pit = points.begin();
		std::vector<char>::iterator bit = groundBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < -1.0)
			{
				*pit = 1;
			}
		}
	}

private:
	ros::Subscriber subscriber_;
	_PointCloud::Ptr cloud_;
	VoxelMap voxelMap_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    Filter filter(n);

    ros::spin();

    return 0;
}



