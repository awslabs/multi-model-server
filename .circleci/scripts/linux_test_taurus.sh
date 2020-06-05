#!/bin/bash

multi-model-server --start \
                   --models squeezenet=https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar \
                   >> mms.log 2>&1
sleep 90

cd taurus

curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
bzt -o modules.jmeter.path=/opt/apache-jmeter-5.3/bin/jmeter \
    -o settings.artifacts-dir=/tmp/cci-taurus/ \
    -o modules.console.disable=true \
    imageInputModelPlan.jmx.yaml \
    -report

multi-model-server --stop