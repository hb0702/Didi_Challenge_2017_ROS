#! /bin/bash
# full file path
filepath=$1

scriptpath=$(readlink -f "$0")
scriptdir=$(dirname "$SCRIPT")

sudo chmod 755 $scriptdir/../object_tracker/src/detector.py $scriptdir/../object_tracker/src/detector.py

/bin/echo -e "\e[92mRunning object_tracker with $filepath\e[0m"
roslaunch object_tracker.launch bag_file_path:=$filepath
