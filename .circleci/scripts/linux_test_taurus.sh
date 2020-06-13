#!/bin/bash

multi-model-server --start \
                   --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar \
                   >> mms.log 2>&1
sleep 90

cd performance_regression

# Only on a python 2 environment -
PY_MAJOR_VER=$(python -c 'import sys; major = sys.version_info.major; print(major);')
if [ $PY_MAJOR_VER -eq 2 ]; then
  # Hack to use python 3.6.5 for bzt installation and execution
  export PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
  pyenv local 3.6.5
fi

# Install dependencies
pip install bzt

curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
bzt -o modules.jmeter.path=/opt/apache-jmeter-5.3/bin/jmeter \
    -o settings.artifacts-dir=/tmp/mms-performance-regression/ \
    -o modules.console.disable=true \
    imageInputModelPlan.jmx.yaml \
    -report

multi-model-server --stop