#! /bin/bash

cd /data/yangwennuo/code/MNL/modelServer
source /data/yangwennuo/miniconda3/bin/activate vllm
ulimit -c unlimited
CUDA_VISIBLE_DEVICES=3,4 python -m debugpy --listen 9501 --wait-for-client modelServer.py 
