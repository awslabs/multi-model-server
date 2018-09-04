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
NoopService defines a no operational model handler.
"""
import time


class NoopService(object):
    """
    Noop Model handler implementation.

    Extend from BaseModelHandler is optional
    """

    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self):
        """
        Initialize model. This will be called during model loading time

        :param:
        :return:
        """
        self.initialized = True

    @staticmethod
    def preprocess(data):
        """
        Transform raw input into model input data.

        :param data: list of objects, raw input from request
        :return: list of model input data
        """
        return data

    @staticmethod
    def inference(model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: inference results
        """
        return model_input

    @staticmethod
    def postprocess(model_output):
        return model_output[0]

    def handle(self, context, data):
        """
        Custom service entry point function.
        :param context: model server context
        :param data: list of objects, raw input from request
        :return: list of outputs to be send back to client
        """
        # Add your initialization code here
        properties = context.system_properties
        server_name = properties.get("server_name")
        server_version = properties.get("server_version")
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        batch_size = properties.get("batch_size")

        logger = context.logger
        logger.debug("server_name: {}".format(server_name))
        logger.debug("server_version: {}".format(server_version))
        logger.debug("model_dir: {}".format(model_dir))
        logger.debug("gpu_id: {}".format(gpu_id))
        logger.debug("batch_size: {}".format(batch_size))
        request_processor = context.request_processor
        try:
            start_time = time.time()

            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)

            request_processor.add_response_property("Content-Type", "text/plain")
            content_type = request_processor.get_request_property("Content-Type")
            end_time = time.time()

            metrics = context.metrics
            metrics.add_time("InferenceTime", start_time - end_time)
            return data
        except Exception as e:
            logger.error(e, exc_info=True)
            request_processor.report_status(500, "Unknown inference error.")
            return ["Error {}".format(str(e))] * len(data)


_service = NoopService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize()

    if data is None:
        return None

    return _service.handle(data, context)
