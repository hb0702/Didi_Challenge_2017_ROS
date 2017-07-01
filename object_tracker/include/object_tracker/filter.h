#pragma once

#include <object_tracker/define.h>
#include <object_tracker/cluster.h>

namespace TeamKR
{

class Box
{
public:
	Box()
	{
		px = py = pz = 0;
		width = height = depth = 0;
		rx = ry = rz = 0;
		topx = topy = topz = 0;
	}

	Box(const Box& rhs)
	{
		px = rhs.px; py = rhs.py; pz = rhs.pz;
		width = rhs.width; height = rhs.height; depth = rhs.depth;
		rx = rhs.rx; ry = rhs.ry; rz = rhs.rz;
		topx = rhs.topx; topy = rhs.topy; topz = rhs.topz;
	}

	~Box()
	{

	}

public:
	value_type px;
	value_type py;
	value_type pz;
	value_type width;
	value_type height;
	value_type depth;
	value_type rx;
	value_type ry;
	value_type rz;
	value_type topx;
	value_type topy;
	value_type topz;
};

class Filter
{
public:
	Filter(const std::string& mode);

	~Filter();

	void filterBySize(const std::list<Cluster*>& input, std::list<Cluster*>& output) const;

	void filterByVelocity(const std::list<Cluster*>& input, int tsSec, int tsNsec, std::list<Box*>& output);

private:
	double toTime(int tsSec, int tsNsec) const;

	Vector2 velocity(const Vector2& pos, double time) const;

	Box* toBox(Cluster* cluster) const;

	Cluster* selectCluster(const std::list<Cluster*>& input);

private:
	std::string mode_;
	value_type maxSpeedGrad_;
	double initTime_;
	double resetTime_;
	// status
	double initStartTime_;
	double resetStartTime_;	
	bool initialized_;
	bool valid_;
	// saved info
	double prevTime_;
	Vector2 prevVel_;
	Box prevBox_;
};

}
