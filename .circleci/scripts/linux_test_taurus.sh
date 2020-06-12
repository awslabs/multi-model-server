#!/bin/bash

multi-model-server --start \
                   --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar \
                   >> mms.log 2>&1
sleep 90

cd performance_regression

# Hack to use python 3.6.5 for bzt installation and execution on a Python 2.7 environment
# ToDo: Update condition to be more specific - check if python 2 is installed by default
if hash pyenv 2>/dev/null; then
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