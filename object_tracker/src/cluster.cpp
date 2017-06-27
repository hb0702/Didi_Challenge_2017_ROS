#include <object_tracker/cluster.h>
#include <stack>

const float MAX_VALUE = 1000.0f;

namespace TeamKR
{

Cluster::Cluster()
{
}

Cluster::Cluster(value_type resolution, value_type baseZ)
{
	cellSize_ = resolution;
	pointCount_ = 0;
	baseZ_ = baseZ;
	min_ << MAX_VALUE, MAX_VALUE, MAX_VALUE;
	max_ << -MAX_VALUE, -MAX_VALUE, -MAX_VALUE;
}

Cluster::Cluster(const Cluster& rhs)
{
	cellSize_ = rhs.cellSize_;
	pointCount_ = rhs.pointCount_;
	points_ = rhs.points_;
	baseZ_ = rhs.baseZ_;
	min_ = rhs.min_;
	max_ = rhs.max_;
	maxIntensity_ = rhs.maxIntensity_;
}

Cluster::~Cluster()
{

}

void Cluster::add(const Vector3& point, int hitCount, value_type intensity, value_type minZ)
{
	points_.push_back(point);

	if (min_(0) > point(0) - 0.5 * cellSize_)
	{
		min_(0) = point(0) - 0.5 * cellSize_;
	}
	if (min_(1) > point(1) - 0.5 * cellSize_)
	{
		min_(1) = point(1) - 0.5 * cellSize_;
	}
	if (min_(2) > point(2) - 0.5 * cellSize_)
	{
		min_(2) = point(2) - 0.5 * cellSize_;
	}
	if (max_(0) < point(0) + 0.5 * cellSize_)
	{
		max_(0) = point(0) + 0.5 * cellSize_;
	}
	if (max_(1) < point(1) + 0.5 * cellSize_)
	{
		max_(1) = point(1) + 0.5 * cellSize_;
	}
	if (max_(2) < point(2) + 0.5 * cellSize_)
	{
		max_(2) = point(2) + 0.5 * cellSize_;
	}

	pointCount_ += hitCount;

	if (maxIntensity_ < intensity)
	{
		maxIntensity_ = intensity;
	}

	if (min_(2) > point(2) - 0.5 * cellSize_)
	{
		min_(2) = point(2) - 0.5 * cellSize_;
	}
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
	return Vector3(0.5 * (min_(0) + max_(0)), 0.5 * (min_(1) + max_(1)), 0.5 * (min_(2) + max_(2)));
}

int Cluster::pointCount() const
{
	return pointCount_;
}

value_type Cluster::maxIntensity() const
{
	return maxIntensity_;
}

value_type Cluster::area() const
{
	value_type r = cellSize_;

	int w = (max_(0) - min_(0)) / r + 1;
	int h = (max_(1) - min_(1)) / r + 1;

	// init hit map
	char** hitmap = new char*[w];
	for (int ix = 0; ix < w; ++ix)
	{
		hitmap[ix] = new char[h];
		for (int iy = 0; iy < h; ++iy)
		{
			hitmap[ix][iy] = 0;
		}
	}

	// mark hit map
	for (std::list<Vector3>::const_iterator it = points_.begin(); it != points_.end(); ++it)
	{
		int ix = (int)(((*it)(0) - min_(0)) / r);
		int iy = (int)(((*it)(1) - min_(1)) / r);
		hitmap[ix][iy] = 1;
	}

	value_type ar = 0.0;

	for (int ix = 0; ix < w; ++ix)
	{
		for (int iy = 0; iy < h; ++iy)
		{
			if (hitmap[ix][iy] == 1)
			{
				ar += 1.0;
			}
		}
	}

	ar *= r * r;

	// delete hit map
	for (int ix = 0; ix < w; ++ix)
	{
		delete[] hitmap[ix];
	}
	delete[] hitmap;

	return ar;
}


/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

ClusterBuilder::ClusterBuilder(value_type centerX, value_type centerY)
{
	centerX_ = centerX;
	centerY_ = centerY;
	originX_ = centerX - ROI_RADIUS;
	originY_ = centerY - ROI_RADIUS;
	baseZ_ = GROUND_Z;
	cellSize_ = RESOLUTION;
	iradius_ = (int)(ROI_RADIUS / resolution + 0.5f);
	iradius2_ = iradius_ * iradius_;
	iwidth_ = (int)(2 * ROI_RADIUS / resolution + 0.5f);

	// init value map
	valuemap_ = new Value*[iwidth_];
	for (int ix = 0; ix < iwidth_; ++ix)
	{
		valuemap_[ix] = new Value[iwidth_];
		for (int iy = 0; iy < iwidth_; ++iy)
		{
			valuemap_[ix][iy].clear();
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
	// delete value map
	for (int ix = 0; ix < iwidth_; ++ix)
	{
		delete[] valuemap_[ix];
	}
	delete[] valuemap_;

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
			hit(*pit);
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
			addPoint(ix, iy, cluster);
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
						addPoint(ax, ay, cluster);
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
	// reset value map
	for (int ix = 0; ix < iwidth_; ++ix)
	{
		for (int iy = 0; iy < iwidth_; ++iy)
		{
			valuemap_[ix][iy].clear();
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

void ClusterBuilder::hit(const PCLPoint& point)
{
	int rx = (int)(((value_type)point.x - centerX_) / cellSize_ + 0.5);
	int ry = (int)(((value_type)point.y - centerY_) / cellSize_ + 0.5);

	if (rx * rx + ry * ry > iradius2_)
	{
		return;
	}

	int x = (int)(((value_type)point.x - originX_) / cellSize_ + 0.5);
	int y = (int)(((value_type)point.y - originY_) / cellSize_ + 0.5);

	if (x < 0 || x >= iwidth_ || y < 0 || y >= iwidth_)
	{
		return;
	}

	Value& value = valuemap_[x][y];

	value.hit += 1;

	if (value.top < point.z)
	{
		value.top = point.z;
	}

	if (value.base > point.z)
	{
		value.base = point.z;
	}

	if (value.intensity < point.intensity)
	{
		value.intensity = point.intensity;
	}
}

int ClusterBuilder::hitCount(int ix, int iy) const
{
	return valuemap_[ix][iy].hit;
}

void ClusterBuilder::addPoint(int ix, int iy, Cluster& cluster)
{
	const Value& value = valuemap_[ix][iy];
	Vector3 cellPoint = Vector3(originX_ + ((value_type)ix + 0.5) * cellSize_,
								originY_ + ((value_type)iy + 0.5) * cellSize_,
								value.top);
	int hitCount = value.hit;
	value_type intensity = value.intensity;
	value_type base = value.base;
	cluster.add(cellPoint, hitCount, intensity, base);
}

}
