# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Helper utils to validate messages and inputs to backend worker
"""

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err


class ModelWorkerMessageValidators(object):
    """
    Class lists all the validators for backend-worker messages
    """
    @staticmethod
    def validate_load_message(msg):
        """
        Validate the load messages
        {
            "command" : "load", string
            "modelPath" : "/path/to/model/file", string
            "modelName" : "name", string
            "gpu" : None if CPU else gpu_id, int
            "handler" : service handler entry point if provided, string
            "batchSize" : batch size, int
        }
        :param msg:
        :return:
        """
        if u'modelPath' not in msg:
            raise MMSError(err.INVALID_LOAD_MESSAGE, "Load command missing \"modelPath\" key")

        if u'modelName' not in msg:
            raise MMSError(err.INVALID_LOAD_MESSAGE, "Load command missing \"modelName\" key")

        if u'handler' not in msg:
            raise MMSError(err.INVALID_LOAD_MESSAGE, "Load command missing \"handler\" key")

    @staticmethod
    def validate_predict_data(msg):
        """
       REQUESTS = [ {
            "requestId" : "111-222-3333",
            "encoding" : "None | base64 | utf-8",
            "modelInputs" : [ MODEL_INPUTS ]
        } ]

        MODEL_INPUTS = {
                "encoding": "base64/utf-8", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }

        :param msg:
        :return:
        """
        if u'requestId' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input missing \"request-id\" field")

    @staticmethod
    def validate_predict_inputs(msg):
        """
        MODEL_INPUTS = [{
                "encoding": "base64", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param msg:
        :return:
        """

        if u'value' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input data missing \"value\" field")

        if u'name' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input data missing \"name\" field")

    @staticmethod
    def validate_predict_msg(msg):
        """
       PREDICT COMMAND = {
            "command": "predict",
            "modelName": "model-to-run-inference-against",
            "contentType": "http-content-types",
            "requestBatch": [ REQUESTS ]
        }
        REQUESTS = {
            "requestId" : "111-222-3333",
            "encoding" : "None|base64|utf-8",
            "modelInputs" : [ MODEL_INPUTS ]
        }
        MODEL_INPUTS = {
                "encoding": "base64/utf-8", (# This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (# This is defined in the symbol file and the signature file)
        }
        """

        if u'modelName' not in msg:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "Predict command input missing \"modelName\" field.")

        if u'requestBatch' not in msg:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "Predict command input missing \"requestBatch\" field.")

        req_batch = msg[u'requestBatch']
        for req in req_batch:
            if u'modelInputs' not in req:
                raise MMSError(err.INVALID_PREDICT_MESSAGE, "Predict command input's requestBatch missing "
                                                            "\"modelInputs\" field.")

    @staticmethod
    def validate_unload_msg(msg):
        """
        {
            "command" : "unload",
            "model-name": "name"
        }

        :param msg:
        :return:
        """

        if u'model-name' not in msg:
            raise MMSError(err.INVALID_UNLOAD_MESSAGE, "Unload command input missing \"model-name\" field")
