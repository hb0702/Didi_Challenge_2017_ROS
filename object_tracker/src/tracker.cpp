#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

const int MAX_MARKER_COUNT = 1000;

class Tracker
{
public:
    Tracker(ros::NodeHandle n)
    {   
        sub_ = n.subscribe("/detector/boxes", 10, &Tracker::onBoxesReceived, this);
        pub_ = n.advertise<visualization_msgs::MarkerArray>("/tracker/boxes", 1);

        processLocked_ = false;

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
            marker.scale.x = 1.0;
            marker.scale.y = 1.0;
            marker.scale.z = 1.0;
            marker.color.a = 0.0;
            marker.color.r = 0.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            markerArr_.markers.push_back(marker);
        }

        ROS_INFO("Tracker: initialized");
    }

    ~Tracker()
    {

    }

    void onBoxesReceived(const std_msgs::Float32MultiArray::ConstPtr& msg)
    {
        // ROS_INFO("Tracker: received %d bounding boxes", (int)msg->data.size()/8);

        if (processLocked_)
        {
            ROS_INFO("Tracker: process locked");
            return;
        }

        // lock process
        processLocked_ = true;

        int numBoxes = (int)(msg->data.size()/8);
        int numMarkers = std::min(numBoxes, MAX_MARKER_COUNT);

        // update markers
        std::vector<visualization_msgs::Marker>::iterator mit = markerArr_.markers.begin();
        std::vector<float>::const_iterator dit = msg->data.begin();
        for (int mi = 0; mi < numMarkers; mi++, mit++, dit+=8)
        {
            int label = int((*dit) + 0.1);
            float l = *(dit+1);
            float w = *(dit+2);
            float h = *(dit+3);
            float px = *(dit+4);
            float py = *(dit+5);
            float pz = *(dit+6);
            float yaw = *(dit+7);

            mit->pose.position.x = px;
            mit->pose.position.y = py;
            mit->pose.position.z = pz;
            mit->pose.orientation.z = yaw;
            mit->scale.x = l;
            mit->scale.y = w;
            mit->scale.z = h;
            mit->color.a = 0.3;
            mit->color.r = label == 0 ? 1.0 : 0.0;
            mit->color.b = label == 0 ? 0.0 : 1.0;
        }

        // hide unused markers
        for (; mit != markerArr_.markers.end(); mit++)
        {
            mit->color.a = 0.0;
        }

        // publish markers
        pub_.publish(markerArr_);
        ROS_INFO("Tracker: published %d markers", numBoxes);

        // unlock process
        processLocked_ = false;
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    bool processLocked_;
    visualization_msgs::MarkerArray markerArr_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tracker");
    ros::NodeHandle n;

    Tracker tracker(n);

    ros::spin();

    return 0;
}
