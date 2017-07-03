#include <object_tracker/define.h>
#include <object_tracker/cluster.h>
#include <object_tracker/filter.h>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace TeamKR
{

class ClusterTracker
{
public:
	ClusterTracker(ros::NodeHandle n, const std::string& mode)
	{
    	subscriber_ = n.subscribe("/velodyne_points", 1, &ClusterTracker::onPointsReceived, this);
    	boxPublisher_ = n.advertise<std_msgs::Float32MultiArray>("/tracker/boxes", 1);
    	detectedMarkerPublisher_ = n.advertise<visualization_msgs::MarkerArray>("/tracker/markers/detect", 1);
    	predictedMarkerPublisher_ = n.advertise<visualization_msgs::MarkerArray>("/tracker/markers/predict", 1);
		cloud_ = PCLPointCloud::Ptr(new PCLPointCloud());

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

		// init point filter
		float maxSpeedGrad = -1.0f;
		float initTime = -1.0f;
		float resetTime = -1.0f;
			
		filter_ = new Filter(mode);
		
		// ground filtering option
		maxGround_ = GROUND_Z + GROUND_EPS;

		// car filtering option
		carMin_ = Vector3(-1.5, -1.0, -1.3);
		carMax_ = Vector3(2.5, 1.0, 0.2);

		// cluster filtering option
		mode_ = mode;

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
            detectedMarkers_.markers.push_back(marker);
        }
        for (int i = 0; i < 1; i++)
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
            marker.color.g = 1.0;
            marker.color.b = 0.0;
            predictedMarkers_.markers.push_back(marker);
        }

        if (PUBLISH_GROUND)
    	{
    		publisherGround_ = n.advertise<sensor_msgs::PointCloud2>("/tracker/ground", 1);
    		cloudGround_ = PCLPointCloud::Ptr(new PCLPointCloud());
    	}

        ROS_INFO("ClusterTracker: initialized");
	}

	~ClusterTracker()
	{
		delete builder_;

		delete filter_;
	}

private:
	void onPointsReceived(const sensor_msgs::PointCloud2::ConstPtr& msg)
	{
		// for (uint j=0; j < msg->height * msg->width; j++){
  //           float x = msg->data[j * msg->point_step + msg->fields[0].offset];
  //           float y = msg->data[j * msg->point_step + msg->fields[1].offset];
  //           float z = msg->data[j * msg->point_step + msg->fields[2].offset];
  //           float a = msg->data[j * msg->point_step + msg->fields[3].offset];
  //           float b = msg->data[j * msg->point_step + msg->fields[4].offset];
  //           float c = msg->data[j * msg->point_step + msg->fields[5].offset];
  //           // Some other operations
  //           ROS_INFO("point: %f %f %f %f %f %f", x, y, z, a, b, c);
  //      	}

		sensor_msgs::PointCloud2 response = *msg;
		pcl::fromROSMsg(response, *cloud_);
		PCLPointVector points = cloud_->points;
		size_t pointCount = points.size();
		if (pointCount == 0u)
		{
			return;
		}

		// mark bit vector - car and ground points
		BitVector pointFilterBV(pointCount, 0);
		markCar(points, pointFilterBV);

		if (USE_RANSAC_GROUND_FITTING)
		{
			markGround_RANSAC(pointFilterBV);
		}
		else
		{
			markGround_simple(points, pointFilterBV);
		}

		// cluster
		std::list<Cluster*> clusters;
		builder_->run(points, pointFilterBV, clusters);
		size_t clusterCount = clusters.size();
		if (clusterCount == 0u)
		{
			return;
		}

		// filter by size
		std::list<Cluster*> sizeFiltered;
		filter_->filterBySize(clusters, sizeFiltered);

		// filter by velocity
		int tsSec = msg->header.stamp.sec;
		int tsNsec = msg->header.stamp.nsec;

		std::list<Box*> boxes;
		filter_->filterByVelocity(sizeFiltered, tsSec, tsNsec, boxes);

		// compensate z val
		if (mode_ == "ped")
		{
			for (std::list<Box*>::iterator it = boxes.begin(); it != boxes.end(); ++it)
			{
				value_type top = (*it)->pz + 0.5 * (*it)->depth;
				(*it)->pz = top - 0.5 * PEDESTRIAN_ACTUAL_DEPTH;
				(*it)->depth = PEDESTRIAN_ACTUAL_DEPTH;
			}
		}

		if (PUBLISH_MARKERS)
		{
			// publish detected clusters
			publishMarkers(sizeFiltered, boxes);
		}

		if (!boxes.empty())
		{
			publishBoxes(boxes);
		}

		// release memory
		for (std::list<Cluster*>::iterator it = clusters.begin(); it != clusters.end(); ++it)
		{
			delete *it;
		}
		for (std::list<Box*>::iterator it = boxes.begin(); it != boxes.end(); ++it)
		{
			delete *it;
		}
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

	void publishMarkers(const std::list<Cluster*>& clusters, const std::list<Box*>& boxes)
	{
		// update markers
		int markerCnt = 0;
		std::list<Cluster*>::const_iterator cit = clusters.begin();
		std::vector<visualization_msgs::Marker>::iterator mit = detectedMarkers_.markers.begin();
		for (; cit != clusters.end(); ++cit)
		{
			// ROS_INFO("ClusterTracker: detected: points %d, depth %f, width %f, center %f %f %f, intensity %f, top %f, base %f, area %f",
			// 		(*cit)->pointCount(), (*cit)->max()(2) - (*cit)->min()(2), 
			// 		std::max((*cit)->max()(0) - (*cit)->min()(0), (*cit)->max()(1) - (*cit)->min()(1)),
			// 		(*cit)->center()(0), (*cit)->center()(1), (*cit)->center()(2),
			// 		(*cit)->maxIntensity(),
			// 		(*cit)->max()(2), (*cit)->min()(2),
			// 		(*cit)->area());
			Vector3 center = (*cit)->center();
			mit->pose.position.x = center(0);
			mit->pose.position.y = center(1);
			mit->pose.position.z = center(2);
			mit->scale.x = (*cit)->max()(0) - (*cit)->min()(0);
			mit->scale.y = (*cit)->max()(1) - (*cit)->min()(1);
			mit->scale.z = (*cit)->max()(2) - (*cit)->min()(2);
			mit->color.a = 0.1;

			++mit;
			++markerCnt;
		}
		for (; mit != detectedMarkers_.markers.end(); ++mit)
		{
			mit->color.a = 0.0;
		}

		std::list<Box*>::const_iterator bit = boxes.begin();
		mit = predictedMarkers_.markers.begin();
		for (; bit != boxes.end(); ++bit)
		{
			// ROS_INFO("ClusterTracker: predicted: size %f %f %f, center %f %f %f",
			// 		(*bit)->width, (*bit)->height, (*bit)->depth,
			// 		(*bit)->px, (*bit)->py, (*bit)->pz);
			mit->pose.position.x = (*bit)->px;
			mit->pose.position.y = (*bit)->py;
			mit->pose.position.z = (*bit)->pz;
			mit->pose.orientation.x = (*bit)->rz;
			mit->pose.orientation.y = (*bit)->rz;
			mit->pose.orientation.z = (*bit)->rz;
			mit->scale.x = (*bit)->width;
			mit->scale.y = (*bit)->height;
			mit->scale.z = (*bit)->depth;
			mit->color.a = 0.3;

			++mit;
		}
		for (; mit != predictedMarkers_.markers.end(); ++mit)
		{
			mit->color.a = 0.0;
		}

        // publish markers
        detectedMarkerPublisher_.publish(detectedMarkers_);
        predictedMarkerPublisher_.publish(predictedMarkers_);
        // ROS_INFO("ClusterTracker: published %d markers", markerCnt);
	}

	void publishBoxes(const std::list<Box*>& boxes)
	{
		float label = -1.0f;
		if (mode_ == "car")
		{
			label = (float)CAR_LABEL;
		}
		else if (mode_ == "ped")
		{
			label = (float)PEDESTRIAN_LABEL;
		}

		int boxCnt = 0;
		boxData_.data.clear();
		std::list<Box*>::const_iterator bit = boxes.begin();
		for (; bit != boxes.end(); ++bit)
		{
			boxData_.data.push_back(label); // label 				0
			// if (mode_ == "car")
			{
				boxData_.data.push_back((*bit)->px); // center		1
				boxData_.data.push_back((*bit)->py); // center		2
				boxData_.data.push_back((*bit)->pz); // center		3
			}
			// else if (mode_ == "ped")
			// {
			// 	boxData_.data.push_back((*bit)->topx); // center	1
			// 	boxData_.data.push_back((*bit)->topy); // center	2
			// 	boxData_.data.push_back((*bit)->topz); // center	3
			// }
			boxData_.data.push_back((*bit)->width); // size 		4
			boxData_.data.push_back((*bit)->height); // size 		5
			boxData_.data.push_back((*bit)->depth); // size 		6
			boxData_.data.push_back(0.0); // rotation 				7

			++boxCnt;
		}

        // publish boxes
        boxPublisher_.publish(boxData_);
        ROS_INFO("ClusterTracker: published %d boxes", boxCnt);
	}

private:
	ros::Subscriber subscriber_;
	ros::Publisher boxPublisher_;
	ros::Publisher detectedMarkerPublisher_;
	ros::Publisher predictedMarkerPublisher_;
	PCLPointCloud::Ptr cloud_;
	// cluster builder
	ClusterBuilder* builder_;
	// velocity fiilter
	Filter* filter_;	
	// ground filtering option
	value_type maxGround_;
	// car filtering option
	Vector3 carMin_;
	Vector3 carMax_;
	// cluster filtering option
	std::string mode_;
	// box data
	std_msgs::Float32MultiArray boxData_;
	// marker array
	visualization_msgs::MarkerArray detectedMarkers_;
	visualization_msgs::MarkerArray predictedMarkers_;
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
    TeamKR::ClusterTracker filter(n, filterMode);

    ros::spin();

    return 0;
}
