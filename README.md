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

### How to run kor_didi_pkg
* Go to catkin workspace

        $ cd ~/catkin_ws/src

* Download kor_didi_pkg source

* Go to catkin workspacee

        $ cd ~/catkin_ws
* Build

        $ catkin_make

* Run kor_didi_run.sh with .bag file path

        $ sh '/home/parkjaeil0108/catkin_ws/src/kor_didi_run.sh' '/home/parkjaeil0108/challenge/Didi-Training-Release-1/approach_1.bag'

### How to extract points(.npy) from .bag file
* Go to catkin workspace source

        $ cd ~/catkin_ws/src

* Download or git clone kor_didi_pkg source

* Go to catkin workspacee

        $ cd ~/catkin_ws
        
* Build

        $ catkin_make

* Run extract_points.sh with .bag file path

        $ sh '/home/parkjaeil0108/catkin_ws/src/extract_points.sh' '/home/parkjaeil0108/challenge/Didi-Training-Release-1/approach_1.bag'

* In the directory .bag file located, (bag file name)/lidar folder will be created, and .npy files will be saved in it.
