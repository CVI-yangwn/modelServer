#! /bin/bash
ROOTPATH=/data/yangwennuo/code/MNL/modelServer
LOGPATH=${ROOTPATH}/logs/server_monitor.log

#check run_data_httpserver process
runcount=`ps -ef | grep "trans_server.py"|grep -v grep |wc -l`
if [ $runcount -lt 1 ] ; then
	echo `date` +":Warning httpserver is not runnning ,restart..........." >> $LOGPATH
	cd ${ROOTPATH}
    	source /data/yangwennuo/miniconda3/bin/activate vllm
	python -u ./trans_server.py > $LOGPATH 2>&1 &
else
	echo `date` +":ok" >> $LOGPATH
fi

exit 1
