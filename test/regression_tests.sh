#!/bin/bash

set -x
set -e

MMS_REPO="https://github.com/awslabs/multi-model-server.git"
BRANCH=${1:-master}
ROOT_DIR="/workspace/"
CODEBUILD_WD=$(pwd)
MODEL_STORE=$ROOT_DIR"/model_store"
MMS_LOG_FILE="/tmp/mms.log"
TEST_EXECUTION_LOG_FILE="/tmp/test_exec.log"

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
  cd -
  echo "MMS Succesfully installed"
  
}


start_mms() {

  # Start MMS with Model Store
  multi-model-server --start --model-store $1  &>> $2
  sleep 10
  curl http://127.0.0.1:8081/models
  
}

stop_mms_serve() {
  multi-model-server --stop
}

start_secure_mms() {

  # Start MMS with Model Store
  multi-model-server --start --mms-config test/resources/config.properties --model-store $1  &>> $2
  sleep 10
  curl --insecure -X GET https://127.0.0.1:8444/models
}


run_postman_test() {
  # Run Postman Scripts
  mkdir $ROOT_DIR/report/
  cd $CODEBUILD_WD/
  set +e
  # Run Management API Tests
  stop_mms_serve
  start_mms $MODEL_STORE $MMS_LOG_FILE
  newman run -e test/postman/environment.json --bail --verbose test/postman/management_api_test_collection.json \
	  -r cli,html --reporter-html-export $ROOT_DIR/report/management_report.html >>$1 2>&1

  # Run Inference API Tests after Restart
  stop_mms_serve
  start_mms $MODEL_STORE $MMS_LOG_FILE
  newman run -e test/postman/environment.json --bail --verbose test/postman/inference_api_test_collection.json \
	  -d test/postman/inference_data.json -r cli,html --reporter-html-export $ROOT_DIR/report/inference_report.html >>$1 2>&1


  # Run Https test cases
  stop_mms_serve
  start_secure_mms $MODEL_STORE $MMS_LOG_FILE
  newman run --insecure -e test/postman/environment.json --bail --verbose test/postman/https_test_collection.json \
	  -r cli,html --reporter-html-export $ROOT_DIR/report/MMS_https_test_report.html >>$1 2>&1

  stop_mms_serve
  set -e
  cd -
}


sudo rm -rf $ROOT_DIR && sudo mkdir $ROOT_DIR
sudo chown -R $USER:$USER $ROOT_DIR
cd $ROOT_DIR
mkdir $MODEL_STORE

sudo rm -f $TEST_EXECUTION_LOG_FILE $MMS_LOG_FILE

echo "** Execuing MMS Regression Test Suite executon for " $MMS_REPO " **"

install_mms_from_source $MMS_REPO $BRANCH
run_postman_test $TEST_EXECUTION_LOG_FILE

echo "** Tests Complete ** "
exit 0
