#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace TeamKR
{

typedef float value_type;

typedef pcl::PointXYZI PCLPoint;
typedef pcl::PointCloud<PCLPoint> PCLPointCloud;
typedef PCLPointCloud::VectorType PCLPointVector;
typedef std::vector<char> BitVector;

}
