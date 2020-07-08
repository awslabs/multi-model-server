#!/bin/bash

MODEL_STORE_DIR='test/model_store'

MMS_LOG_FILE_MANAGEMENT='mms_management.log'
MMS_LOG_FILE_INFERENCE='mms_inference.log'
MMS_LOG_FILE_HTTPS='mms_https.log'
MMS_CONFIG_FILE_HTTPS='test/resources/config.properties'

POSTMAN_ENV_FILE='test/postman/environment.json'
POSTMAN_COLLECTION_MANAGEMENT='test/postman/management_api_test_collection.json'
POSTMAN_COLLECTION_INFERENCE='test/postman/inference_api_test_collection.json'
POSTMAN_COLLECTION_HTTPS='test/postman/https_test_collection.json'
POSTMAN_DATA_FILE_INFERENCE='test/postman/inference_data.json'

REPORT_FILE_MANAGEMENT='test/management-api-report.html'
REPORT_FILE_INFERENCE='test/inference-api-report.html'
REPORT_FILE_HTTPS='test/https-api-report.html'

start_mms_server() {
  multi-model-server --start --model-store $1 >> $2 2>&1
  sleep 10
}

start_mms_secure_server() {
  multi-model-server --start --mms-config $MMS_CONFIG_FILE_HTTPS --model-store $1 >> $2 2>&1
  sleep 10
}

stop_mms_server() {
  multi-model-server --stop
}

trigger_management_tests(){
  start_mms_server $MODEL_STORE_DIR $MMS_LOG_FILE_MANAGEMENT
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_MANAGEMENT \
             -r cli,html --reporter-html-export $REPORT_FILE_MANAGEMENT --verbose
  stop_mms_server
}

trigger_inference_tests(){
  start_mms_server $MODEL_STORE_DIR $MMS_LOG_FILE_INFERENCE
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_INFERENCE -d $POSTMAN_DATA_FILE_INFERENCE \
             -r cli,html --reporter-html-export $REPORT_FILE_INFERENCE --verbose
  stop_mms_server
}

trigger_https_tests(){
  start_mms_secure_server $MODEL_STORE_DIR $MMS_LOG_FILE_HTTPS
  newman run --insecure -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_HTTPS \
             -r cli,html --reporter-html-export $REPORT_FILE_HTTPS --verbose
  stop_mms_server
}

mkdir -p $MODEL_STORE_DIR

case $1 in
   'management')
      trigger_management_tests
      ;;
   'inference')
      trigger_inference_tests
      ;;
   'https')
      trigger_https_tests
      ;;
   'ALL')
      trigger_management_tests
      trigger_inference_tests
      trigger_https_tests
      ;;
   *)
     echo $1 'Invalid'
     echo 'Please specify any one of - management | inference | https | ALL'
     exit 1
     ;;
esac