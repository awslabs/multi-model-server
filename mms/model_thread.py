# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Model thread implementation
"""

import traceback

from flask import abort

import mms.batching as batching
from mms.log import get_logger
from mms.metrics_manager import MetricsManager


logger = get_logger()


class ModelThread(object):
    """
    ModelThread is a wrapper on top of ModelService that runs the main loop
    """

    def __init__(self, service_name, model_service, data_store, batching_strategy, batching_config):
        """
        Initialize ModelThread
        """
        self.service_name = service_name
        self.model_service = model_service
        self.data_store = data_store

        batching_config['input_type'] = model_service.signature['input_type']
        batching_config['data_store'] = self.data_store
        batching_config['service_name'] = service_name
        self.batching_strategy = batching.get_batching_strategy(batching_strategy, batching_config)

    def start(self):
        """
        Main ModelThread loop
        Get data batch from queue, run inference, send results back
        """
        while True:
            ids, data = self.batching_strategy.wait_for_batch()

            try:
                output = self.model_service.inference(data)
            except Exception:  # pylint: disable=broad-except
                if self.service_name + '_Prediction5XX' in MetricsManager.metrics:
                    MetricsManager.metrics[self.service_name + '_Prediction5XX'].update(metric=1)
                logger.error(str(traceback.format_exc()))
                abort(500, "Error occurs while inference was executed on server.")

            output_type = self.model_service.signature['output_type']
            self.data_store.set_batch(ids, output, output_type)
