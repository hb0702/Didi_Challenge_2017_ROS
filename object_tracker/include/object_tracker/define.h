#pragma once

#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace TeamKR
{

typedef float value_type;

typedef pcl::PointXYZ PCLPoint;
typedef pcl::PointCloud<pcl::PointXYZ> PCLPointCloud;
typedef PCLPointCloud::VectorType PCLPointVector;
typedef std::vector<char> BitVector;

}
