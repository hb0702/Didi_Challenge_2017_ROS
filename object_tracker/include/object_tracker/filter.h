#pragma once

#include <object_tracker/define.h>

namespace TeamKR
{

class Filter
{
public:
	Filter(value_type speedLimit, double initTime, double resetTime);

	~Filter();

	bool initialized() const;

	Vector2 run(const std::vector<Vector2>& points, int tsSec, int tsNsec);

private:
	double toTime(int tsSec, int tsNsec) const;

	Vector2 velocity(const Vector2& point, double time) const;

private:
	value_type speedLimit_;
	double initTime_;
	double resetTime_;
	double initStartTime_;
	double resetStartTime_;
	bool initialized_;
	// saved info
	double prevTime_;
	Vector2 prevVel_;
	Vector2 prevPos_;
};

}
