#!/bin/bash

# 1. This script downloads a model(squeezenet_v1.1.mar) from model zoo
# 2. Extracts(unzip) the model
# 3. Modifies the handle function to have a 20 second sleep at start
# 4. Repackages the model

BASE_DIR=$(pwd)
STORE_DIR="tmp_store"
MODEL_NAME="squeezenet_v1.1"
MODEL_URL="https://s3.amazonaws.com/model-server/model_archive_1.0/$MODEL_NAME.mar"
HANDLER_FILE_NAME="mxnet_vision_service"

build(){
  # Create a temporary model store and a workspace inside it
  mkdir -p $STORE_DIR/$MODEL_NAME

  # Move to workspace
  cd $STORE_DIR/$MODEL_NAME

  # Download the model from model zoo
  curl -s -O $MODEL_URL

  # Extract the model, Once extracted remove the downloaded .mar
  unzip $MODEL_NAME.mar
  rm $MODEL_NAME.mar

  #### Adds a 20 sec wait in the handle function ####
  # DO NOT - change any spacing and\or try to format the below code
  sed -i'' -e '/import logging/a\
import time' $HANDLER_FILE_NAME.py

  sed -i'' -e '/def handle(data, context):/a\
\ \ \ \ time.sleep(20)
  ' $HANDLER_FILE_NAME.py
  ##################################################

  # Install model-archiver
  pip -q install model-archiver > /dev/null 2>&1

  # Move to model store
  cd ../

  # Create a new model using model archiver
  model-archiver --model-name $MODEL_NAME --model-path ./$MODEL_NAME --handler $HANDLER_FILE_NAME:handle

  # Remove workspace
  rm -rf $MODEL_NAME

  # Move back to base dir
  cd $BASE_DIR
}

clean(){
  rm -rf $STORE_DIR
}

case $1 in
   'build')
      build
      exit $?
      ;;
   'clean')
      clean
      exit $?
      ;;
   *)
     echo $1 'Invalid'
     echo 'Please specify any one of - build | clean'
     exit 1
     ;;
esac
