#include <object_tracker/filter.h>

namespace TeamKR
{

Filter::Filter(value_type speedLimit, double initTime, double resetTime)
{
	initialized_ = false;

	speedLimit_ = speedLimit;
	initTime_ = initTime;
	resetTime_ = resetTime;
	initStartTime_ = -1;
	resetStartTime_ = -1;

	prevTime_ = 0.0;
	prevVel_ << 0, 0;
	prevPos_ << 0, 0;
}

Filter::~Filter()
{

}

bool Filter::initialized() const
{
	return initialized_;
}

Vector2 Filter::run(const std::vector<Vector2>& points, int tsSec, int tsNsec) ???
{
	Vector2 filtered = prevPos_;
	double time = toTime(tsSec, tsNsec);

	if (!initialized_)
	{
		if (initStartTime_ < 0)
		{
			if (points.size() > 1)
			{
				return filtered;
			}

			prevPos_ = points.front();
			initStartTime_ = time;
		}
		else
		{
			bool found = false;

			for (std::vector<Vector2>::const_iterator it = points.begin(); it != points.end(); ++it)
			{
				Vector2 vel = velocity(*it, time);
				if (vel.norm() < speedLimit_)
				{
					prevTime_ = time;
					prevPos_ = *it;
					prevVel_ = vel;
					found = true;
				}
			}

			if (!found)
			{
				initStartTime_ = -1;
				return filtered;
			}

			if (time - initStartTime_ > initTime_)
			{
				initialized_ = true;
				initStartTime_ = -1;
			}
		}
	}
	else
	{
		bool found = false;

		for (std::vector<Vector2>::const_iterator it = points.begin(); it != points.end(); ++it)
		{
			Vector2 vel = velocity(*it, time);
			if (vel.norm() < speedLimit_)
			{
				prevTime_ = time;
				prevPos_ = *it;
				prevVel_ = vel;
				found = true;
			}
		}

		if (found)
		{
			filtered = prevPos_;
			resetStartTime_ = -1;
		}
		else
		{
			filtered = prevPos_ + prevVel_ * (time - prevTime_);
			if (resetStartTime_ < 0)
			{
				resetStartTime_ = time;
			}
			else if (time - resetStartTime_ > resetTime_)
			{
				initialized_ = false;
				resetStartTime_ = -1;
				// start initialization
				run(points, tsSec, tsNsec);
			}
		}
	}

	return filtered;
}

double Filter::toTime(int tsSec, int tsNsec) const
{
	return 1E-9 * (double)tsNsec + (double)tsSec;
}

Vector2 Filter::velocity(const Vector2& point, double time) const
{
	return (point - prevPos_) / (time - prevTime_);
}

}
