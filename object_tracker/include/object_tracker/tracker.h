#pragma once

#include <object_tracker/define.h>
#include <Eigen/Dense>

namespace TeamKR
{

class Tracker
{
public:
	Tracker();

	~Tracker();

	void run(value_type px, value_type py, long timestamp);

private:
	void predict();

	void update(value_type px, value_type py);

private:
	bool initialized_;
	long prevTimestamp_;
	value_type noiseax_;
	value_type noiseay_;

	///* state vector
	Eigen::VectorXd x_;
	///* state covariance matrix
	Eigen::MatrixXd P_;
	///* state transistion matrix
	Eigen::MatrixXd F_;
	///* process covariance matrix
	Eigen::MatrixXd Q_;
	///* measurement matrix
	Eigen::MatrixXd H_;
	///* measurement covariance matrix
	Eigen::MatrixXd R_;
};

}
