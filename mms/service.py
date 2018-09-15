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
CustomService class definitions
"""
import ast
import time
from collections import OrderedDict

import mms
from mms.context import Context
from mms.log import log_msg
from mms.metrics.metrics_store import MetricsStore
from mms.mxnet_model_service_error import MMSError
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators

INFERENCE_METRIC = 'InferenceTime'


class Service(object):
    """
    Wrapper for custom entry_point
    """

    def __init__(self, model_name, model_dir, manifest, entry_point, gpu, batch_size):
        self._context = Context(model_name, model_dir, manifest, batch_size, gpu, mms.__version__)
        self._entry_point = entry_point

    @property
    def context(self):
        return self._context

    @staticmethod
    def retrieve_model_input(model_inputs, req_bat_content_type=None):
        """

        MODEL_INPUTS : [{
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param req_bat_content_type: Content-type of the request-batch, the outer scope of model-inputs
        :param model_inputs: list of model_input elements each containing "encoding", "value" and "name"
        :return:
        """
        model_in = OrderedDict()
        for _, ip in enumerate(model_inputs):
            ModelWorkerMessageValidators.validate_predict_inputs(ip)
            ip_name = ip.get('name')
            content_type = ip.get('contentType')

            if content_type is None or content_type == b'':
                content_type = req_bat_content_type

            if content_type is not None and content_type != b'' and "text" in content_type.decode():
                decoded_val = ip.get(u'value').decode()
            elif content_type is not None and content_type != b'' and "json" in content_type.decode():
                decoded_val = ast.literal_eval(ip.get(u'value').decode())
            else:
                decoded_val = ip.get(u'value')
            model_in.update({ip_name.decode(): decoded_val})

        return model_in

    @staticmethod
    def retrieve_data_for_inference(requests=None):
        """
        REQUESTS = [ {
            "requestId" : "111-222-3333",
            "modelInputs" : [ MODEL_INPUTS ]
        } ]

        MODEL_INPUTS = {
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }

        inputs: List of requests
        Returns a list(dict(inputs))
        """

        req_to_id_map = {}
        invalid_reqs = {}

        if requests is None:
            raise ValueError("Received invalid inputs")

        input_batch = []
        for batch_idx, request_batch in enumerate(requests):
            ModelWorkerMessageValidators.validate_predict_data(request_batch)
            req_id = request_batch.get('requestId').decode()

            model_inputs = request_batch['modelInputs']
            req_batch_content_type = request_batch.get('contentType')
            try:
                input_data = Service.retrieve_model_input(model_inputs, req_batch_content_type)
                input_batch.append(input_data)
                req_to_id_map[batch_idx] = req_id
            except MMSError as m:
                invalid_reqs.update({req_id: m.get_code()})

        return input_batch, req_to_id_map, invalid_reqs

    def predict(self, data, codec):
        """
        PREDICT COMMAND = {
            "command": "predict",
            "modelName": "model-to-run-inference-against",
            "contentType": "http-content-types",
            "requestBatch": [ REQUESTS ]
        }

        REQUESTS = {
            "requestId" : "111-222-3333",
            "modelInputs" : [ MODEL_INPUTS ]
        }

        MODEL_INPUTS = {
                "encoding": "base64/utf-8", (# This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (# This is defined in the symbol file and the signature file)
        }

        :param data:
        :param codec:
        :return:

        """
        model_name = data[u'modelName'].decode()
        req_batch = data[u'requestBatch']
        input_batch, req_id_map, invalid_reqs = Service.retrieve_data_for_inference(req_batch)

        self.context.request_ids = req_id_map
        metrics = MetricsStore(req_id_map, model_name)
        self.context.metrics = metrics

        start_time = time.time()

        ret = self._entry_point(input_batch, self.context)

        duration = int((time.time() - start_time) * 1000)
        metrics.add_time(INFERENCE_METRIC, duration)
        emit_metrics(metrics.store)

        predictions = codec.create_response(cmd=2, resp=ret, req_id_map=req_id_map, invalid_reqs=invalid_reqs)
        return predictions, "Prediction success", 200


def emit_metrics(metrics):
    """
    Emit the metrics in the provided Dictionary

    Parameters
    ----------
    metrics: Dictionary
    A dictionary of all metrics, when key is metric_name
    value is a metric object
    """
    if metrics:
        for met in metrics:
            log_msg("[METRICS]", str(met))
