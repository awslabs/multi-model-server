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
        :param msg:
        :return:
        """
        if u'modelPath' not in msg:
            raise MMSError(err.INVALID_LOAD_MESSAGE, "Load command missing \"model-path\" key")

        if u'modelName' not in msg:
            raise MMSError(err.INVALID_LOAD_MESSAGE, "Load command missing \"model-name\" key")

    @staticmethod
    def validate_predict_data(msg):
        """
        {
                "request-id" : "111-222-3333",
                "input1" : {}
                "input2" : {}
        }

        :param msg:
        :return:
        """
        if u'requestId' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input missing \"request-id\" field")

    @staticmethod
    def validate_predict_inputs(msg):
        """
        "input1" : {
            "encoding": "base64/utf-8",
            "value": "val1"
        }...

        :param msg:
        :return:
        """

        if u'encoding' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input data missing \"encoding\" field")

        if u'value' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input data missing \"value\" field")

        if u'name' not in msg:
            raise MMSError(err.INVALID_PREDICT_INPUT, "Predict command input data missing \"name\" field")

    @staticmethod
    def validate_predict_msg(msg):
        """
        {
           "command" : "predict",
           "model-name" : "name",
           "data": [ {}, {} ]
        }
        """
        if u'modelName' not in msg:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "Predict command input missing \"modelName\" field.")

        if u'requestBatch' not in msg:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "Predict command input missing \"requestBatch\" field.")

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
