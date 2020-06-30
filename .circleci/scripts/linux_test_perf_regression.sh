#!/bin/bash

ARTIFACTS_DIR='run_artifacts'
RESULT_DIR=$ARTIFACTS_DIR'/report/performance/'
JMETER_PATH='/opt/apache-jmeter-5.3/bin/jmeter'

# Start MMS server
#multi-model-server --start >> mms.log 2>&1
#sleep 10

cd tests/performance

# Only on a python 2 environment -
PY_MAJOR_VER=$(python -c 'import sys; major = sys.version_info.major; print(major);')
if [ $PY_MAJOR_VER -eq 2 ]; then
  # Hack to use python 3.6.5 for bzt installation and execution
  export PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
  pyenv local 3.6.5
fi

# Install dependencies
pip install -r requirements.txt
pip install bzt

# Execute performance test suite and store exit code
./run_performance_suite.py -j $JMETER_PATH -e xlarge --no-compare-local
EXIT_CODE=$?

# Stop server
#multi-model-server --stop >> mms.log 2>&1

# Collect and store test results in result directory to be picked up by CircleCI
mkdir -p $RESULT_DIR
cp $ARTIFACTS_DIR/**/performance_results.xml $RESULT_DIR

# Exit with the same error code as that of test execution
exit EXIT_CODE
