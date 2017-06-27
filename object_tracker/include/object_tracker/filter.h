#pragma once

#include <object_tracker/define.h>

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
	}

	Box(const Box& rhs)
	{
		px = rhs.px; py = rhs.py; pz = rhs.pz;
		width = rhs.width; height = rhs.height; depth = rhs.depth;
		rx = rhs.rx; ry = rhs.ry; rz = rhs.rz;
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
}

class Filter
{
public:
	Filter(const std::string& mode);

	~Filter();

	void filterBySize(const std::vector<Cluster>& input, std::vector<Cluster>& output) const;

	void filterByVelocity(const std::vector<Cluster>& input, int tsSec, int tsNsec, std::vector<Box>& output);

private:
	double toTime(int tsSec, int tsNsec) const;

	Vector2 velocity(const Vector2& pos, double time) const;

	Box toBox(const Cluster& cluster) const;

	const Cluster& selectCluster(const std::vector<Cluster>& input);

private:
	std::string mode_;
	value_type speedLimit_;
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
