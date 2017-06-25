#include <object_tracker/tracker.h>

using namespace std;
using namespace Eigen;

namespace TeamKR
{

Tracker::Tracker()
{
	initialized_ = false;

	prevTimestamp_ = 0;

	x_ = VectorXd(4);
	P_ = MatrixXd(4, 4);
	P_ << 1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1000, 0,
		0, 0, 0, 1000;

	R_ = MatrixXd(2, 2);
	R_ << 0.0225, 0,
		0, 0.0225;

	H_ = MatrixXd(2, 4);
	H_ << 1, 0, 0, 0,
		0, 1, 0, 0;

	F_ = MatrixXd(4, 4);
	F_ << 1, 0, 1, 0,
		0, 1, 0, 1,
		0, 0, 1, 0,
		0, 0, 0, 1;

	noiseax_ = 5;
	noiseay_ = 5;
}

Tracker::~Tracker()
{

}

void Tracker::run(value_type px, value_type py, long timestamp)
{
	if (!initialized_)
	{
		x_ << px, py, 0.0f, 0.0f;

		prevTimestamp_ = timestamp;

		initialized_ = true;
	}
	else
	{
		value_type dt = (timestamp - prevTimestamp_) / 1000000.0;	//dt - expressed in seconds
		prevTimestamp_ = timestamp;

		value_type dt_2 = dt * dt;
		value_type dt_3 = dt_2 * dt;
		value_type dt_4 = dt_3 * dt;

		//Modify the F matrix so that the time is integrated
		F_(0, 2) = dt;
		F_(1, 3) = dt;

		Q_ = MatrixXd(4, 4);
		Q_ <<  dt_4/4*noiseax_, 0, dt_3/2*noiseax_, 0,
				0, dt_4/4*noiseay_, 0, dt_3/2*noiseay_,
				dt_3/2*noiseax_, 0, dt_2*noiseax_, 0,
				0, dt_3/2*noiseay_, 0, dt_2*noiseay_;

		predict();

		update(px, py);
	}
}

void Tracker::predict()
{
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void Tracker::update(value_type px, value_type py)
{
	VectorXd z(2);
	z << px, py;
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}

}
