#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

class Tracker
{
public:
    Tracker(ros::NodeHandle n)
    {   
        sub_ = n.subscribe("/detector/boxes", 10, &Tracker::onBoxesReceived, this);
        pub_ = n.advertise<visualization_msgs::MarkerArray>("/tracker/boxes", 1);
        processLocked_ = false;
        ROS_INFO("Tracker: initialized");
    }

    ~Tracker()
    {

    }

    void onBoxesReceived(const std_msgs::Float32MultiArray::ConstPtr& msg)
    {
        ROS_INFO("Tracker: received %d bounding boxes", (int)msg->data.size()/8);

        if (processLocked_)
        {
            ROS_INFO("Tracker: process locked");
            return;
        }

        processLocked_ = true;

        visualization_msgs::MarkerArray arr;

        for (std::vector<float>::const_iterator it = msg->data.begin(); it != msg->data.end(); it+=8)
        {
            int label = int((*it) + 0.1);
            float l = *(it+1);
            float w = *(it+2);
            float h = *(it+3);
            float px = *(it+4);
            float py = *(it+5);
            float pz = *(it+6);
            float yaw = *(it+7);

            visualization_msgs::Marker marker;
            marker.header.frame_id = "velodyne";
            marker.type = visualization_msgs::Marker::CUBE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = px;
            marker.pose.position.y = py;
            marker.pose.position.z = pz;
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = yaw;
            marker.pose.orientation.w = 1.0;
            marker.scale.x = l;
            marker.scale.y = w;
            marker.scale.z = h;
            marker.color.a = 0.5;
            marker.color.r = label == 0 ? 1.0 : 0.0;
            marker.color.g = 0.0;
            marker.color.b = label == 0 ? 0.0 : 1.0;
            arr.markers.push_back(marker);
        }

        pub_.publish(arr);
        ROS_INFO("Tracker: published %d bounding boxes", (int)arr.markers.size());
        processLocked_ = false;
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    bool processLocked_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tracker");
    ros::NodeHandle n;

    Tracker tracker(n);

    ros::spin();

    return 0;
}
