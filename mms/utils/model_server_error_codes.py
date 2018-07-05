# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


class ModelServerErrorCodes(object):
    # Success
    SUCCESS = 200

    # General errors
    UNKNOWN_EXCEPTION = 5000
    SOCKET_ERROR = 5001
    INVALID_COMMAND = 5002
    RECEIVE_ERROR = 5003
    INVALID_MESSAGE = 5004
    INVALID_SIGNATURE_FILE = 5005
    SOCKET_BIND_ERROR = 5006
    UNKNOWN_COMMAND = 5007

    # Load errors
    INVALID_LOAD_MESSAGE = 6000
    MODEL_ARTIFACTS_WRONG_FORMAT = 6001
    MISSING_MODEL_ARTIFACTS = 6002
    UNKNOWN_EXCEPTION_WHILE_LOADING = 6003
    VALUE_ERROR_WHILE_LOADING = 6004

    # Predict message errors
    INVALID_PREDICT_MESSAGE = 7000
    INVALID_PREDICT_INPUT = 7001
    MODEL_SERVICE_NOT_LOADED = 7002
    UNSUPPORTED_PREDICT_OPERATION = 7003

    # unload msg errors
    INVALID_UNLOAD_MESSAGE = 8000
    MODEL_CURRENTLY_NOT_LOADED = 8001

    # send errors
    SEND_MSG_FAIL = 9000
    SEND_FAILS_EXCEEDS_LIMITS = 9001

    # codec errors
    ENCODE_FAILED = 10000
    DECODE_FAILED = 10001
    CODEC_FAIL = 10002

    # Custom service error code
    CUSTOM_SERVICE_ERROR = 12000

