#! /bin/bash
# full file path
filepath=$1

scriptpath=$(readlink -f "$0")
scriptdir=$(dirname "$SCRIPT")

sudo chmod 755 $scriptdir/../kor_didi_pkg/src/py_processor.py $scriptdir/../kor_didi_pkg/src/py_processor.py

/bin/echo -e "\e[92mRunning kor_didi_pkg with $filepath\e[0m"
#run kor_didi.launch ver1
#roslaunch kor_didi.launch bag_file_path:=$filepath
#run kor_didi.launch ver2
roslaunch kor_didi2.launch bag_file_path:=$filepath
