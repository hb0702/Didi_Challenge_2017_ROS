#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

namespace TeamKR
{

typedef float value_type;

typedef pcl::PointXYZ PCLPoint;
typedef pcl::PointCloud<PCLPoint> PCLPointCloud;
typedef PCLPointCloud::VectorType PCLPointVector;
typedef std::vector<char> BitVector;

typedef Eigen::Vector3f Vector3;
typedef Eigen::Vector2f Vector2;

// parameters
const int MAX_MARKER_COUNT = 30;

const float GROUND_Z = -1.3f;
const float GROUND_EPS = 0.1f;

const int PEDESTRIAN_LABEL = 0;
const float PEDESTRIAN_ROI_RADIUS = 21.0f;
const float PEDESTRIAN_RESOLUTION = 0.15f;
const float PEDESTRIAN_ACTUAL_DEPTH = 1.708f;
const int PEDESTRIAN_MIN_POINT_COUNT = 30;
const float PEDESTRIAN_MAX_WIDTH = 1.2f;
const float PEDESTRIAN_MIN_DEPTH = 1.2f;
const float PEDESTRIAN_MAX_DEPTH = 2.0f;
const float PEDESTRIAN_MAX_BASE = GROUND_Z + 0.9f;
const float PEDESTRIAN_MAX_AREA = 0.45f;
const float PEDESTRIAN_MAX_SPEED_GRAD = 2.78f; // 10 km/h in m/s
const float PEDESTRIAN_FILTER_INIT_TIME = 0.5f; // sec
const float PEDESTRIAN_FILTER_RESET_TIME = 0.5f; // sec

const int CAR_LABEL = 1;
const float CAR_ROI_RADIUS = 35.0f;
const float CAR_RESOLUTION = 0.8f;
const int CAR_MIN_POINT_COUNT = 15;
const float CAR_MIN_WIDTH = 0.2f;
const float CAR_MAX_WIDTH = 6.0f;
const float CAR_MIN_DEPTH = 0.2f;
const float CAR_MIN_MEANZ = -0.5f;//0.15f;
const float CAR_MAX_MEANZ = 1.55f;
const float CAR_MAX_DEPTH = 1.9f;
const float CAR_MAX_BASE = GROUND_Z + 1.0f;
const float CAR_MAX_AREA = 4.8f * 2.0f;
const float CAR_MAX_SPEED_GRAD = 5.56f; // 20 km/h in m/s
const float CAR_FILTER_INIT_TIME = 0.5f; // sec
const float CAR_FILTER_RESET_TIME = 0.5f; // sec

const bool USE_RANSAC_GROUND_FITTING = true;
const bool USE_GOOD_CLUSTER_ONLY = true;
const int RANSAC_MAX_ITERATIONS = 150;
const bool PUBLISH_GROUND = false;
const bool PUBLISH_MARKERS = true;

}
