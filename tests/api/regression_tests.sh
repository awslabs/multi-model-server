#!/bin/bash

set -x
set -e

MMS_REPO="https://github.com/awslabs/multi-model-server.git"
BRANCH=${1:-master}
ROOT_DIR="/workspace/"
CODEBUILD_WD=$(pwd)
MODEL_STORE=$ROOT_DIR"/model_store"
TEST_EXECUTION_LOG_FILE="/tmp/test_exec.log"
ARTIFACTS_DIR="tests/api/artifacts"
OUTPUT_DIR=/tmp/MMS_regression

install_mms_from_source() {
  echo "Cloning & Building Multi Model Server Repo from " $1
  sudo apt-get -y install nodejs-dev node-gyp libssl1.0-dev
  sudo apt-get -y install npm
  sudo npm install -g n
  sudo n latest
  export PATH="$PATH"
  sudo npm install -g newman newman-reporter-html
  pip install mxnet-mkl
  # Clone & Build MMS
  echo "Installing MMS from source"
  git clone -b $2 $1
  cd multi-model-server
  pip install .
  echo "MMS Branch : " "$(git rev-parse --abbrev-ref HEAD)" >> $3
  echo "MMS Branch Commit Id : " "$(git rev-parse HEAD)" >> $3
  echo "Build date : " "$(date)" >> $3
  echo "MMS Succesfully installed"
}

sudo rm -rf $ROOT_DIR $OUTPUT_DIR && sudo mkdir $ROOT_DIR
sudo chown -R $USER:$USER $ROOT_DIR
cd $ROOT_DIR
mkdir $MODEL_STORE

sudo rm -f $TEST_EXECUTION_LOG_FILE

echo "** Execuing MMS Regression Test Suite executon for " $MMS_REPO " **"

install_mms_from_source $MMS_REPO $BRANCH $TEST_EXECUTION_LOG_FILE
ci/scripts/linux_test_api.sh ALL >> $TEST_EXECUTION_LOG_FILE
mv $TEST_EXECUTION_LOG_FILE $ARTIFACTS_DIR
mv $ARTIFACTS_DIR $OUTPUT_DIR
echo "** Tests Complete ** "
exit 0
