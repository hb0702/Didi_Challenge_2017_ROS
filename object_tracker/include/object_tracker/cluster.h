#pragma once

#include <object_tracker/define.h>
#include <object_tracker/vector3.h>
#include <list>

namespace TeamKR
{

class Cluster
{
public:
	Cluster(value_type resolution, value_type baseZ);

	~Cluster();

	void add(const Vector3& point, int hitCount);

	const Vector3& min() const;

	const Vector3& max() const;

	Vector3 center() const;

	int pointCount() const;

private:
	value_type cellSize_;
	int pointCount_;
	std::list<Vector3> points_;
	Vector3 min_;
	Vector3 max_;
};

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
	ClusterBuilder(value_type centerX, value_type centerY, value_type baseZ, value_type radius, value_type resolution);

	~ClusterBuilder();

	void run(const PCLPointVector& points, const BitVector& filterBV, std::list<Cluster>& clusters);

private:
	void clear();

	void hit(double px, double py, double pz);

	Vector3 cellPoint(int ix, int iy) const;

	int hitCount(int ix, int iy) const;

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

}
