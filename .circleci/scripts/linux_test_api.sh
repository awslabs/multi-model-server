#!/bin/bash

BASE_DIR="tests/api"
MODEL_STORE_DIR="$BASE_DIR/model_store"

ARTIFACTS_BASE_DIR="$BASE_DIR/artifacts"
ARTIFACTS_MANAGEMENT_DIR="$ARTIFACTS_BASE_DIR/management"
ARTIFACTS_INFERENCE_DIR="$ARTIFACTS_BASE_DIR/inference"
ARTIFACTS_HTTPS_DIR="$ARTIFACTS_BASE_DIR/https"

MMS_CONSOLE_LOG_FILE="mms_console.log"
MMS_CONFIG_FILE_HTTPS="$BASE_DIR/resources/config.properties"

POSTMAN_ENV_FILE="$BASE_DIR/postman/environment.json"
POSTMAN_COLLECTION_MANAGEMENT="$BASE_DIR/postman/management_api_test_collection.json"
POSTMAN_COLLECTION_INFERENCE="$BASE_DIR/postman/inference_api_test_collection.json"
POSTMAN_COLLECTION_HTTPS="$BASE_DIR/postman/https_test_collection.json"
POSTMAN_DATA_FILE_INFERENCE="$BASE_DIR/postman/inference_data.json"

REPORT_FILE="report.html"

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

move_logs(){
  mv $1 logs/
  mv logs/ $2
}

trigger_management_tests(){
  start_mms_server $MODEL_STORE_DIR $MMS_CONSOLE_LOG_FILE
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_MANAGEMENT \
             -r cli,html --reporter-html-export $ARTIFACTS_MANAGEMENT_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_mms_server
  move_logs $MMS_CONSOLE_LOG_FILE $ARTIFACTS_MANAGEMENT_DIR
  return $EXIT_CODE
}

trigger_inference_tests(){
  start_mms_server $MODEL_STORE_DIR $MMS_CONSOLE_LOG_FILE
  newman run -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_INFERENCE -d $POSTMAN_DATA_FILE_INFERENCE \
             -r cli,html --reporter-html-export $ARTIFACTS_INFERENCE_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_mms_server
  move_logs $MMS_CONSOLE_LOG_FILE $ARTIFACTS_INFERENCE_DIR
  return $EXIT_CODE
}

trigger_https_tests(){
  start_mms_secure_server $MODEL_STORE_DIR $MMS_CONSOLE_LOG_FILE
  newman run --insecure -e $POSTMAN_ENV_FILE $POSTMAN_COLLECTION_HTTPS \
             -r cli,html --reporter-html-export $ARTIFACTS_HTTPS_DIR/$REPORT_FILE --verbose
  local EXIT_CODE=$?
  stop_mms_server
  move_logs $MMS_CONSOLE_LOG_FILE $ARTIFACTS_HTTPS_DIR
  return $EXIT_CODE
}

mkdir -p $MODEL_STORE_DIR $ARTIFACTS_MANAGEMENT_DIR $ARTIFACTS_INFERENCE_DIR $ARTIFACTS_HTTPS_DIR

case $1 in
   'management')
      trigger_management_tests
      exit $?
      ;;
   'inference')
      trigger_inference_tests
      exit $?
      ;;
   'https')
      trigger_https_tests
      exit $?
      ;;
   'ALL')
      trigger_management_tests
      MGMT_EXIT_CODE=$?
      trigger_inference_tests
      INFR_EXIT_CODE=$?
      trigger_https_tests
      HTTPS_EXIT_CODE=$?
      # If any one of the tests fail, exit with error
      if [ "$MGMT_EXIT_CODE" -ne 0 ] || [ "$INFR_EXIT_CODE" -ne 0 ] || [ "$HTTPS_EXIT_CODE" -ne 0 ]
      then exit 1
      fi
      ;;
   *)
     echo $1 'Invalid'
     echo 'Please specify any one of - management | inference | https | ALL'
     exit 1
     ;;
esac