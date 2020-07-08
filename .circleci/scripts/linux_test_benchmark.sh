#!/bin/bash

# Hack needed to make it work with existing benchmark.py
# benchmark.py expects jmeter to be present at a very specific location
mkdir -p /home/ubuntu/.linuxbrew/Cellar/jmeter/5.3/libexec/bin/
ln -s /opt/apache-jmeter-5.3/bin/jmeter /home/ubuntu/.linuxbrew/Cellar/jmeter/5.3/libexec/bin/jmeter

multi-model-server --start >> mms.log 2>&1
sleep 30

cd benchmarks
python benchmark.py latency

multi-model-server --stop