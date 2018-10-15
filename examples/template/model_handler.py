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
ModelHandler defines a base model handler.
"""
import logging


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]
        self.initialized = True

    def preprocess(self, batch):
        """
        Transform raw input into model input data.

        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        assert self._batch_size == len(batch), "Invalid input batch size: {}".format(len(batch))
        return None

    def inference(self, model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        return None

    def postprocess(self, inference_output):
        """
        Return predict result in batch.

        :param inference_output: list of inference output
        :return: list of predict results
        """
        return ["OK"] * self._batch_size

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param data: list of objects, raw input from request
        :param context: model server context
        :return: list of outputs to be send back to client
        """

        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)

            return data
        except Exception as e:
            logging.error(e, exc_info=True)
            request_processor = context.request_processor
            request_processor.report_status(500, "Unknown inference error: " + str(e))
            return ["Inference error"] * self._batch_size
