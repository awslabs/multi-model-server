#!/bin/bash

ARTIFACTS_DIR='/tmp/mms-performance-regression/'
RESULT_DIR=$ARTIFACTS_DIR'/report/performance/'
JMETER_PATH='/opt/apache-jmeter-5.3/bin/jmeter'

# Start MMS server
multi-model-server --start >> mms.log 2>&1
sleep 10

#cd test suite
cd tests/performance

# Install dependencies
pip install -r requirements.txt

# Execute performance test suite and store exit code
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
./run_perfomance_suite.py --artifacts-dir=$ARTIFACTS_DIR --jmeter-path=$JMETER_PATH
EXIT_CODE=$?

# Stop server
multi-model-server --stop >> mms.log 2>&1

# Collect and store test results in result directory to be picked up by CircleCI
mkdir -p $RESULT_DIR
cp $ARTIFACTS_DIR'/junit.xml' $RESULT_DIR

# Exit with the same error code as that of test execution
exit EXIT_CODE
