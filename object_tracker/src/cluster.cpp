#include <object_tracker/cluster.h>
#include <pcl/point_types.h>
#include <stack>

const float MAX_VALUE = 1000.0f;

namespace TeamKR
{

Cluster::Cluster(value_type resolution, value_type baseZ)
{
	cellSize_ = resolution;
	pointCount_ = 0;
	min_.set(MAX_VALUE, MAX_VALUE, baseZ);
	max_.set(-MAX_VALUE, -MAX_VALUE, baseZ);
}

Cluster::~Cluster()
{

}

void Cluster::add(const Vector3& point, int hitCount)
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

const Vector3& Cluster::min() const
{
	return min_;
}

const Vector3& Cluster::max() const
{
	return max_;
}

Vector3 Cluster::center() const
{
	return Vector3(0.5 * (min_.x + max_.x), 0.5 * (min_.y + max_.y), 0.5 * (min_.z + max_.z));
}

int Cluster::pointCount() const
{
	return pointCount_;
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ClusterBuilder::ClusterBuilder(value_type centerX, value_type centerY, value_type baseZ, value_type radius, value_type resolution)
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

ClusterBuilder::~ClusterBuilder()
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

void ClusterBuilder::run(const PCLPointVector& points, const BitVector& filterBV, std::list<Cluster>& clusters)
{
	clear();

	PCLPointVector::const_iterator pit = points.begin();
	BitVector::const_iterator bit = filterBV.begin();
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

void ClusterBuilder::clear()
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

void ClusterBuilder::hit(double px, double py, double pz)
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

Vector3 ClusterBuilder::cellPoint(int ix, int iy) const
{
	return Vector3(originX_ + ((value_type)ix + 0.5) * cellSize_,
				originY_ + ((value_type)iy + 0.5) * cellSize_,
				depthmap_[ix][iy]);
}

int ClusterBuilder::hitCount(int ix, int iy) const
{
	return hitmap_[ix][iy];
}

}
