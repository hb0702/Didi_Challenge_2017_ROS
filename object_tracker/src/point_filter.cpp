#include <object_tracker/define.h>
#include <object_tracker/cluster.h>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>

namespace TeamKR
{

const float MAX_VALUE = 1000.0f;

class PointFilter
{
public:
	PointFilter(ros::NodeHandle n, const std::string& mode)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &PointFilter::onPointsReceived, this);
    	publisher_ = n.advertise<std_msgs::Float64MultiArray>("/filtered_points", 1);
    	
    	cloud_ = PCLPointCloud::Ptr(new PCLPointCloud());
    	filtered_ = PCLPointCloud::Ptr(new PCLPointCloud());

    	// init cluster builder
		value_type resolution = 0.0;
		value_type roiRad = 0.0;
		if (mode == "car")
		{
			resolution = CAR_RESOLUTION;
			roiRad = CAR_ROI_RADIUS;
		}
		else if (mode == "ped")
		{
			resolution = PEDESTRIAN_RESOLUTION;
			roiRad = PEDESTRIAN_ROI_RADIUS;
		}
		builder_ = new ClusterBuilder(0.0, 0.0, resolution, roiRad);

		// ground filtering option
		maxGround_ = GROUND_Z + GROUND_EPS;

		// car filtering option
		carMin_ = Vector3(-1.5, -1.0, -1.3);
		carMax_ = Vector3(2.5, 1.0, 0.2);

		// cluster filtering option
		mode_ = mode;

        ROS_INFO("PointFilter: initialized");
	}

	~PointFilter()
	{
		delete builder_;		
	}

private:
	void onPointsReceived(const sensor_msgs::PointCloud2::ConstPtr& msg)
	{
		std_msgs::Float64MultiArray output;

		// add timestamp
		int tsSec = msg->header.stamp.sec;
		int tsNsec = msg->header.stamp.nsec;
		output.data.push_back(tsSec);
		output.data.push_back(tsNsec);

		sensor_msgs::PointCloud2 response = *msg;
		pcl::fromROSMsg(response, *cloud_);
		PCLPointVector points = cloud_->points;
		size_t pointCount = points.size();
		if (pointCount == 0u)
		{
			// publish empty points
			publisher_.publish(output);
			return;
		}

		// mark bit vector - car and ground points
		BitVector badPointBV(pointCount, 0);
		markCar(points, badPointBV);

		markGround_RANSAC(badPointBV);
		markGround_simple(points, badPointBV);

		// get filtered cloud
		filtered_->points.clear();
		{
			PCLPointVector::const_iterator pit = points.begin();
			BitVector::const_iterator bit = badPointBV.begin();
			for (; pit != points.end(); ++pit, ++bit)
			{
				if (*bit == 0)
				{
					filtered_->points.push_back(*pit);
				}
			}
		}

		// add point count
		if (filtered_->points.size() != 0)
		{
			// euclidian clustering
			pcl::search::KdTree<PCLPoint>::Ptr tree (new pcl::search::KdTree<PCLPoint>);
			tree->setInputCloud(filtered_);

			std::vector<pcl::PointIndices> cluster_indices;

			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance(CAR_RESOLUTION);
			ec.setMinClusterSize(CAR_MIN_POINT_COUNT);
			ec.setMaxClusterSize(10000);
			ec.setSearchMethod(tree);
			ec.setInputCloud(filtered_);
			ec.extract(cluster_indices);

			// get point coordinate with cluster index
			int numClusters = 0;
			std::list<value_type> clusterX, clusterY;
			std::list<std::vector<value_type>> pointInfoList;
			for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
			{
				std::vector<value_type> pointInfo;
				value_type cx = 0, cy = 0;
				int np = 0;

				// filter clusters
				value_type minx = MAX_VALUE, maxx = -MAX_VALUE, miny = MAX_VALUE, maxy = -MAX_VALUE, minz = MAX_VALUE, maxz = -MAX_VALUE;
				value_type width = 0, height = 0, depth = 0;				
				for (std::vector<int>::const_iterator iit = it->indices.begin(); iit != it->indices.end(); ++iit)
				{
					PCLPoint point = filtered_->points[*iit];
					if (minx > point.x) { minx = point.x; }
					if (maxx < point.x) { maxx = point.x; }
					if (miny > point.y) { miny = point.y; }
					if (maxy < point.y) { maxy = point.y; }
					if (minz > point.z) { minz = point.z; }
					if (maxz < point.z) { maxz = point.z; }

					pointInfo.push_back(point.x);
					pointInfo.push_back(point.y);
					pointInfo.push_back(point.z);
					pointInfo.push_back(numClusters);
					cx += point.x;
					cy += point.y;
					np++;
				}

				width = maxx - minx;
				height = maxy - miny;
				depth = maxz - minz;
				if (width < CAR_MIN_WIDTH || width > CAR_MAX_WIDTH
					|| height < CAR_MIN_WIDTH || height > CAR_MAX_WIDTH
					|| depth < CAR_MIN_DEPTH || depth > CAR_MAX_DEPTH
					|| it->indices.size() < CAR_MIN_POINT_COUNT)
				{
					continue;
				}

				// save info
				numClusters++;
				cx /= np;
				cy /= np;
				clusterX.push_back(cx);
				clusterY.push_back(cy);
				pointInfoList.push_back(pointInfo);
			}

			// add num cluster
			output.data.push_back(numClusters);

			// add cluster x,y
			{
				std::list<value_type>::const_iterator xit = clusterX.begin();
				std::list<value_type>::const_iterator yit = clusterY.begin();
				for (; xit != clusterX.end(); ++xit, ++yit)
				{
					output.data.push_back(*xit);
					output.data.push_back(*yit);
				}
			}

			// add points with cluster index
			for (std::list<std::vector<value_type>>::const_iterator it = pointInfoList.begin(); it != pointInfoList.end(); ++it)
			{
				std::copy(pointInfo.begin(), pointInfo.end(), std::back_inserter(output.data));	
			}
		}

		publisher_.publish(output);		
	}

	void markCar(const PCLPointVector& points, BitVector& filterBV) const
	{
		PCLPointVector::const_iterator pit = points.begin();
		BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->x > carMin_(0) && pit->y > carMin_(1)
				&& pit->x < carMax_(0) && pit->y < carMax_(1))
			{
				*bit = 1;
			}
		}
	}

	void markGround_RANSAC(BitVector& filterBV) const
	{
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::SACSegmentation<PCLPoint> seg;
		seg.setOptimizeCoefficients(true);
		seg.setModelType(pcl::SACMODEL_PLANE);
		seg.setMethodType(pcl::SAC_RANSAC);
		seg.setInputCloud(cloud_);		
		// variables
		seg.setDistanceThreshold(GROUND_EPS);
		seg.setAxis(Eigen::Vector3f(0.0f,0.0f,1.0f));
		seg.setMaxIterations(RANSAC_MAX_ITERATIONS);
		seg.segment(*inliers, *coefficients);

		if (inliers->indices.size () == 0)
		{
			return;
		}

		std::vector<int>::const_iterator it = inliers->indices.begin();
		for (; it != inliers->indices.end(); ++it)
		{
			filterBV[*it] = 1;
		}
	}

	void markGround_simple(const PCLPointVector& points, BitVector& filterBV) const
	{
		PCLPointVector::const_iterator pit = points.begin();
		BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < maxGround_)
			{
				*bit = 1;
			}
		}
	}

	void filterClusters(const std::list<Cluster*>& input, std::list<Cluster*>& output, bool getGood) const
	{
		std::list<Cluster*>::const_iterator cit = input.begin();
		for (; cit != input.end(); ++cit)
		{
			value_type top = (*cit)->max()(2);
			value_type base = (*cit)->min()(2);
			value_type depth = top - base;
			value_type meanZ = (*cit)->meanZ();
			value_type maxWidth = std::max((*cit)->max()(0) - (*cit)->min()(0), (*cit)->max()(1) - (*cit)->min()(1));
			if (mode_ == "car")
			{
				if (maxWidth < CAR_MAX_WIDTH
					&& maxWidth > CAR_MIN_WIDTH
					&& top - base > CAR_MIN_DEPTH
					//&& meanZ > GROUND_Z + CAR_MIN_MEANZ
					//&& meanZ < GROUND_Z + CAR_MAX_MEANZ
					&& top < GROUND_Z + CAR_MAX_DEPTH
					//&& base < CAR_MAX_BASE
					&& (*cit)->pointCount() > CAR_MIN_POINT_COUNT
					//&& (*cit)->area() < CAR_MAX_AREA
					)
				{
					if (getGood)
					{
						output.push_back(*cit);
						// printf("meanz: %f, %f %f\n", meanZ, (*cit)->min()[0], (*cit)->min()[1]);
					}
					else
					{
						continue;
					}
				}
			}
			else if (mode_ == "ped")
			{
				if (maxWidth < PEDESTRIAN_MAX_WIDTH 
					&& top > GROUND_Z + PEDESTRIAN_MIN_DEPTH && top < GROUND_Z + PEDESTRIAN_MAX_DEPTH
					&& base < PEDESTRIAN_MAX_BASE
					&& (*cit)->pointCount() > PEDESTRIAN_MIN_POINT_COUNT
					//&& (*cit)->area() < PEDESTRIAN_MAX_AREA
					)
				{
					if (getGood)
					{
						output.push_back(*cit);
					}
					else
					{
						continue;
					}
				}
			}

			if (!getGood)
			{
				output.push_back(*cit);
			}
		}
	}

	void markGoodClusters(const PCLPointVector& points, const std::list<Cluster*> filter, const BitVector& badPointBV, std::vector<int>& indices) const
	{
		PCLPointVector::const_iterator pit = points.begin();
		BitVector::const_iterator bit = badPointBV.begin();
		std::vector<int>::iterator iit = indices.begin();		
		for (; pit != points.end(); ++pit, ++bit, ++iit)
		{
			if (*bit != 0)
			{
				continue;
			}

			int ci = 0;
			for (std::list<Cluster*>::const_iterator cit = filter.begin(); cit != filter.end(); ++cit, ++ci)
			{
				if (pit->x > (*cit)->min()(0) && pit->x < (*cit)->max()(0)
					&& pit->y > (*cit)->min()(1) && pit->y < (*cit)->max()(1)
					&& pit->z > (*cit)->min()(2) && pit->z < (*cit)->max()(2))
				{
					*iit = ci;
					break;
				}
			}
		}
	}

	void markBadClusters(const PCLPointVector& points, const std::list<Cluster*> filter, BitVector& filterBV) const
	{
		PCLPointVector::const_iterator pit = points.begin();
		BitVector::iterator bit = filterBV.begin();		
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (*bit == 0)
			{
				for (std::list<Cluster*>::const_iterator cit = filter.begin(); cit != filter.end(); ++cit)
				{
					if (pit->x > (*cit)->min()(0) && pit->x < (*cit)->max()(0)
						&& pit->y > (*cit)->min()(1) && pit->y < (*cit)->max()(1)
						&& pit->z > (*cit)->min()(2) && pit->z < (*cit)->max()(2))
					{
						*bit = 1;
						break;
					}
				}
			}
		}
	}

private:
	ros::Subscriber subscriber_;
	ros::Publisher publisher_;
	PCLPointCloud::Ptr cloud_;
	PCLPointCloud::Ptr filtered_;
	// cluster builder
	ClusterBuilder* builder_;
	// ground filtering option
	value_type maxGround_;
	// car filtering option
	Vector3 carMin_;
	Vector3 carMax_;
	// cluster filtering option
	std::string mode_;
};

}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    std::string filterMode(argv[1]);
    TeamKR::PointFilter filter(n, filterMode);

    ros::spin();

    return 0;
}
