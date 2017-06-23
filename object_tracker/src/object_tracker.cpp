#include <object_tracker/define.h>
#include <object_tracker/cluster.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

const int MAX_MARKER_COUNT = 30;

const float GROUND_Z = -1.27f;
const float GROUND_EPS = 0.1f;

const float RESOLUTION = 0.4f;
const float ROI_RADIUS = 21.0f;

const int CAR_POINT_COUNT_THRESHOLD = 88;
const int PEDESTRIAN_POINT_COUNT_THRESHOLD = 48;

const float PEDESTRIAN_MAX_WIDTH = 1.3f;
const float PEDESTRIAN_MIN_DEPTH = 1.3f;
const float PEDESTRIAN_MAX_DEPTH = 2.0f;
const float CAR_MAX_WIDTH = 5.7f;
const float CAR_MIN_DEPTH = 0.8f;
const float CAR_MAX_DEPTH = 1.7f;

const bool USE_RANSAC = true;
const bool PUBLISH_GROUND = false;
const int RANSAC_MAX_ITERATIONS = 150;

namespace TeamKR
{

class ObjectTracker
{
public:
	ObjectTracker(ros::NodeHandle n, const std::string& mode)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &ObjectTracker::onPointsReceived, this);
    	publisher_ = n.advertise<visualization_msgs::MarkerArray>("/filter/boxes", 1);    	
		cloud_ = PCLPointCloud::Ptr(new PCLPointCloud());

		// init cluster builder
		builder_ = new ClusterBuilder(0.0, 0.0, GROUND_Z, ROI_RADIUS, RESOLUTION);
		
		// ground filtering option
		maxGround_ = GROUND_Z + GROUND_EPS;

		// car filtering option
		carMin_ = Vector3(-1.5, -1.0, -1.3);
		carMax_ = Vector3(2.5, 1.0, 0.2);

		// cluster filtering option
		mode_ = mode;
		carMinPointCount_ = CAR_POINT_COUNT_THRESHOLD;
		pedMinPointCount_ = PEDESTRIAN_POINT_COUNT_THRESHOLD;
		pedMaxWidth_ = PEDESTRIAN_MAX_WIDTH;
		pedMinZ_ = GROUND_Z + PEDESTRIAN_MIN_DEPTH;
		pedMaxZ_ = GROUND_Z + PEDESTRIAN_MAX_DEPTH;
		carMaxWidth_ = CAR_MAX_WIDTH;
		carMinZ_ = GROUND_Z + CAR_MIN_DEPTH;
		carMaxZ_ = GROUND_Z + CAR_MAX_DEPTH;

		// init markers
        for (int i = 0; i < MAX_MARKER_COUNT; i++)
        {
            visualization_msgs::Marker marker;
            marker.id = i;
            marker.header.frame_id = "velodyne";
            marker.type = marker.CUBE;
            marker.action = marker.ADD;
            marker.pose.position.x = 0.0;
            marker.pose.position.y = 0.0;
            marker.pose.position.z = 0.0;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;
            marker.color.a = 0.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            markerArr_.markers.push_back(marker);
        }

        if (PUBLISH_GROUND)
    	{
    		publisherGround_ = n.advertise<sensor_msgs::PointCloud2>("/filter/ground", 1);
    		cloudGround_ = PCLPointCloud::Ptr(new PCLPointCloud());
    	}

        ROS_INFO("ObjectTracker: initialized");
	}

	~ObjectTracker()
	{
		delete builder_;
	}

private:
	void onPointsReceived(const sensor_msgs::PointCloud2::ConstPtr& msg)
	{
		ROS_INFO("ObjectTracker: start");

		pcl::fromROSMsg(*msg, *cloud_);
		PCLPointVector points = cloud_->points;
		size_t pointCount = points.size();
		if (pointCount == 0u)
		{
			return;
		}

		ROS_INFO("ObjectTracker: point converted");

		// mark bit vector - car and ground points
		BitVector pointFilterBV(pointCount, 0);
		markCar(points, pointFilterBV);

		if (USE_RANSAC)
		{
			markGround_RANSAC(pointFilterBV);
		}
		else
		{
			markGround_simple(points, pointFilterBV);
		}

		ROS_INFO("ObjectTracker: marked ground");

		// cluster
		std::list<Cluster> clusters;
		builder_->run(points, pointFilterBV, clusters);
		size_t clusterCount = clusters.size();
		if (clusterCount == 0u)
		{
			return;
		}

		ROS_INFO("ObjectTracker: clustered");
		
		// mark bit vector - bad  clusters
		BitVector clusterFilterBV(clusterCount, 0);
		markBadCluster(clusters, clusterFilterBV);

		ROS_INFO("ObjectTracker: marked bad clusters");

		// publish clusters
		publishMarkers(clusters, clusterFilterBV);
	}

	void markCar(const PCLPointVector& points, BitVector& filterBV) const
	{
		PCLPointVector::const_iterator pit = points.begin();
		BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->x > carMin_.x && pit->y > carMin_.y
				&& pit->x < carMax_.x && pit->y < carMax_.y)
			{
				*bit = 1;
			}
		}
	}

	void markGround_RANSAC(BitVector& filterBV) const
	{
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::SACSegmentation<pcl::PointXYZ> seg;
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

		if (PUBLISH_GROUND)
		{
			pcl::ExtractIndices<PCLPoint> extract;
			extract.setInputCloud(cloud_);
			extract.setIndices(inliers);
			extract.setNegative(false);
			extract.filter(*cloudGround_);
			sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
			pcl::toROSMsg(*cloudGround_, *msg);
			publisherGround_.publish(msg);			
		}
	}

	void markGround_simple(const PCLPointVector& points, BitVector& filterBV) const
	{
		if (PUBLISH_GROUND)
		{
			cloudGround_->points.clear();
		}

		PCLPointVector::const_iterator pit = points.begin();
		BitVector::iterator bit = filterBV.begin();
		for (; pit != points.end(); ++pit, ++bit)
		{
			if (pit->z < maxGround_)
			{
				*bit = 1;

				if (PUBLISH_GROUND)
				{
					cloudGround_->points.push_back(*pit);
				}
			}
		}

		if (PUBLISH_GROUND)
		{
			sensor_msgs::PointCloud2::Ptr msg(new sensor_msgs::PointCloud2());
			pcl::toROSMsg(*cloudGround_, *msg);
			msg->header.frame_id = "velodyne";
			publisherGround_.publish(msg);
		}
	}

	void markBadCluster(const std::list<Cluster>& clusters, BitVector& filterBV) const
	{
		std::list<Cluster>::const_iterator cit = clusters.begin();
		BitVector::iterator bit = filterBV.begin();
		for (; cit != clusters.end(); ++cit, ++bit)
		{
			value_type top = cit->max().z;			
			value_type maxWidth = std::max(cit->max().x - cit->min().x, cit->max().y - cit->min().y);
			if (mode_ == "car")
			{
				if (maxWidth < pedMaxWidth_ || maxWidth > carMaxWidth_
					|| top < carMinZ_ || top > carMaxZ_)
				{
					*bit = 1;
				}
				else if (cit->pointCount() < carMinPointCount_)
				{
					*bit = 1;
				}
			}
			else if (mode_ == "ped")
			{
				if (maxWidth > pedMaxWidth_ || top < pedMinZ_ || top > pedMaxZ_)
				{
					*bit = 1;
				}
				else if (cit->pointCount() < pedMinPointCount_)
				{
					*bit = 1;
				}
			}
			else if (mode_ == "car_ped")
			{
				if (maxWidth < pedMaxWidth_)
				{
					if (top < pedMinZ_ || top > pedMaxZ_)
					{
						*bit = 1;
					}
					else if (cit->pointCount() < pedMinPointCount_)
					{
						*bit = 1;
					}
				}
				else if (maxWidth < carMaxWidth_)
				{
					if (top < carMinZ_ || top > carMaxZ_)
					{
						*bit = 1;
					}
					else if (cit->pointCount() < carMinPointCount_)
					{
						*bit = 1;
					}
				}
				else
				{
					*bit = 1;
				}
			}
		}
	}

	void publishMarkers(const std::list<Cluster>& clusters, const BitVector& filterBV)
	{
		// update markers
		int markerCnt = 0;
		std::list<Cluster>::const_iterator cit = clusters.begin();
		BitVector::const_iterator bit = filterBV.begin();
		std::vector<visualization_msgs::Marker>::iterator mit = markerArr_.markers.begin();
		for (; cit != clusters.end(); ++cit, ++bit)
		{
			if (*bit == 0)
			{
				ROS_INFO("ObjectTracker: points %d, depth %f, width %f, center %f %f %f", cit->pointCount(), cit->max().z, 
					std::max(cit->max().x - cit->min().x, cit->max().y - cit->min().y),
					cit->center().x, cit->center().y, cit->center().z);
				Vector3 center = cit->center();
				mit->pose.position.x = center.x;
				mit->pose.position.y = center.y;
				mit->pose.position.z = center.z;
				mit->scale.x = cit->max().x - cit->min().x;
				mit->scale.y = cit->max().y - cit->min().y;
				mit->scale.z = cit->max().z - cit->min().z;
				mit->color.a = 0.3;

				++mit;
				++markerCnt;
			}
		}
		for (; mit != markerArr_.markers.end(); ++mit)
        {
            mit->color.a = 0.0;
        }

        // publish markers
        publisher_.publish(markerArr_);
        ROS_INFO("ObjectTracker: published %d markers", markerCnt);
	}

private:
	ros::Subscriber subscriber_;
	ros::Publisher publisher_;	
	PCLPointCloud::Ptr cloud_;
	// cluster builder
	ClusterBuilder* builder_;	
	// ground filtering option
	value_type maxGround_;
	// car filtering option
	Vector3 carMin_;
	Vector3 carMax_;
	// cluster filtering option
	std::string mode_;
	int carMinPointCount_;
	int pedMinPointCount_;
	value_type pedMaxWidth_;
	value_type pedMinZ_;
	value_type pedMaxZ_;
	value_type carMaxWidth_;
	value_type carMinZ_;
	value_type carMaxZ_;
	// marker array
	visualization_msgs::MarkerArray markerArr_;
	// publish ground
	ros::Publisher publisherGround_;
	PCLPointCloud::Ptr cloudGround_;
};

}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    ros::init(argc, argv, "filter");

    ros::NodeHandle n;
    std::string filterMode(argv[1]);
    TeamKR::ObjectTracker filter(n, filterMode);

    ros::spin();

    return 0;
}
