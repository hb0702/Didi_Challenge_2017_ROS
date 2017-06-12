# ROS scripts

### How to setup environment
* Install python-2.7 or Anaconda-2 (recommended)
* Install ROS

        $ sudo apt-get install ros-indigo-desktop-full

* Setup ROS environment - add following to ~/.bashrc

        source /opt/ros/indigo/setup.bash
		source ~/catkin_ws/devel/setup.bash
		export ROS_MASTER_URI=http://localhost:11311 export ROS_HOSTNAME=localhost

* Install opencv

        $ pip install opencv-python

* Install CUDA and CuDNN
* Install tensorflow-gpu

        $ pip install tensorflow-gpu

* Install keras

        $ pip install keras

### How to setup development environment with Qt
* Go to catkin workspace

        $ cd ~/catkin_ws/src

* Download kor_didi_pkg source
* Move the original file of catkin_ws/src/CMakeLists.txt (should be /opt/ros/indigo/share/catkin/cmake/toplevel.cmake) to catkin_ws/src/, and change the file name to CMakeLists.txt
* Setup Qt Creator permission - add following to ~/.bashrc

        sudo -s chmod o+w /home/parkjaeil0108/.config/QtProject/qtcreator/*.xml
        sudo chown -R $USER:$USER /home/parkjaeil0108/.config/QtProject/
        sudo -s chmod o+w /home/parkjaeil0108/catkin_ws/src/kor_didi_pkg/*.*

* Open Qt Creator

        $ qtcreator

* Open catkin_ws/src/CMakeLists.txt on Qt Creator, set build directory to ~/catkin_ws/build

### How to run object_tracker
* Go to catkin workspace

        $ cd ~/catkin_ws/src

* Download object_tracker source

* Go to catkin workspacee

        $ cd ~/catkin_ws
* Build

        $ catkin_make

* Run object_tracker in ros_script

        $ object_tracker

        Add --use_cpp_node option if you want to use cpp node

        $ object_tracker -- use_cpp_node

* Run rviz if you want to visualize point

        $ rviz

        Fixed frame: velodyne
        Topics: PointCloud2(/velodyne_points), MarkerArray(/tracker/boxes)

* Run bag_player in ros_script with bag file path

        $ bag_player --file_path '/home/parkjaeil0108/challenge/Didi-Training-Release-1/approach_1.bag'
