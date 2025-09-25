#! /bin/bash

LOGPATH=${ROOTPATH}/logs/server_monitor.log

cd /data/yangwennuo/code/MNL/modelServer
source /data/yangwennuo/miniconda3/bin/activate vllm
ulimit -c unlimited
pkill -f trans_server.py
python -u ./trans_server.py > $LOGPATH 2>&1 &

exit 0
