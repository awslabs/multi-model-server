#!/bin/bash

multi-model-server --start >> mms.log 2>&1
sleep 10

#cd taurus
cd benchmarks/monitoring

curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
./run_perfomance_suite.py --artifacts-dir=/tmp/mms-performance-regression/ --jmeter-path=/opt/apache-jmeter-5.3/bin/jmeter

multi-model-server --stop >> mms.log 2>&1