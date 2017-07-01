cmake_minimum_required(VERSION 2.8.3)
project(object_tracker)

## Add support for C++11, supported in ROS Kinetic and newer
# add_definitions(-std=c++11)

find_package(catkin
REQUIRED
  cmake_modules 
COMPONENTS
  roscpp
  rospy
  std_msgs
  rosbag
  pcl_ros
  velodyne_msgs
)

find_package(Eigen REQUIRED)

# Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

catkin_package(
  INCLUDE_DIRS
    include
#  LIBRARIES object_tracker
  CATKIN_DEPENDS
    roscpp
    rospy
    std_msgs
    rosbag
    pcl_ros
  DEPENDS
    system_lib
    Eigen
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/object_tracker.cpp
# )

# This will generate message header file before building the target
#add_dependencies(image_processor ???message_file_header_cpp??)
#add_dependencies(kf_localizer ???message_file_header_cpp??)

# This will create executables of the nodes
add_executable(bag_player
  src/bag_player.cpp
)
add_executable(point_filter
  src/cluster.cpp
  src/point_filter.cpp
)
add_executable(cluster_tracker
  src/cluster.cpp
  src/filter.cpp
  src/cluster_tracker.cpp
)

# This will link executables to the appropriate libraries
target_link_libraries(bag_player ${catkin_LIBRARIES})
target_link_libraries(point_filter ${catkin_LIBRARIES})
target_link_libraries(cluster_tracker ${catkin_LIBRARIES})

# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

# Uncomment if the package has a setup.py
# modules and global scripts declared therein get installed
catkin_python_setup()

# install headers
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})

# Install python scripts
catkin_install_python(
  PROGRAMS
    scripts/convert_to_full_view_panorama.py
    scripts/dl_filter.py
    scripts/dl_tracker.py
    scripts/full_view_model.py
    scripts/tracklet.py
    scripts/tracklet_writer.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install model
install(FILES model/fv_model_for_car_June_30_132_63.h5 DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
