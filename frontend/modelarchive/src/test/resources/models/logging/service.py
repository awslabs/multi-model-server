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
LoggingService defines a no operational model handler.
"""
import logging
import time
import os

class LoggingService(object):
    """
    Logging Model handler implementation.

    Extend from BaseModelHandler is optional
    """

    def __init__(self):
        logging.info("LoggingService init")
        self._context = None
        self.initialized = False

    def __del__(self):
        #print("LoggingService exit")
        logging.info("LoggingService exit")

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: model server context
        :return:
        """
        self.initialized = True
        self._context = context

    @staticmethod
    def inference(model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: inference results
        """
        time.sleep(0.01)
        logging.info("LoggingService inference [PID]: %d", os.getpid())
        return ["{} OK\n".format(os.getpid())] * len(model_input)

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param context: model server context
        :param data: list of objects, raw input from request
        :return: list of outputs to be send back to client
        """
        # Add your initialization code here
        properties = context.system_properties
        try:
            start_time = time.time()
            data = self.inference(data)
            end_time = time.time()

            context.set_response_content_type(0, "text/plain")
            content_type = context.request_processor[0].get_request_property("Content-Type")
            return data
        except Exception as e:
            logging.error(e, exc_info=True)
            context.request_processor[0].report_status(500, "Unknown inference error.")
            return ["Error {}".format(str(e))] * len(data)


_service = LoggingService()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
