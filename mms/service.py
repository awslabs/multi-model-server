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

import json
import copy
import mms
from mms.context import Context
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators
from mms.mxnet_model_service_error import MMSError
from mms.utils.codec_helpers.codec import ModelWorkerCodecHelper
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mms.metrics.metric_encoder import MetricEncoder
from mms.log import log_msg
from mms.metrics.metrics_store import MetricsStore


class Service(object):
    """
    Wrapper for custom entry_point
    """
    def __init__(self, model_name, model_dir, manifest, entry_point, gpu, batch_size):

        self._context = Context(model_name, model_dir, manifest, batch_size, gpu, mms.__version__)
        self._entry_point = entry_point
        self._legacy = False

    @property
    def context(self):
        return self._context

    @property
    def legacy(self):
        return self._legacy

    @legacy.setter
    def legacy(self, legacy):
        self._legacy = legacy


    @staticmethod
    def retrieve_model_input(model_inputs):
        """
        MODEL_INPUTS = [{
                "encoding": "base64", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param model_inputs: list of model_input elements each containing "encoding", "value" and "name"
        :return:
        """

        model_in = dict()
        for _, ip in enumerate(model_inputs):
            ModelWorkerMessageValidators.validate_predict_inputs(ip)
            ip_name = ip.get(u'name')
            encoding = ip.get('encoding')
            decoded_val = ModelWorkerCodecHelper.decode_msg(encoding, ip['value'])

            model_in.update({ip_name: decoded_val})

        return model_in

    @staticmethod
    def retrieve_data_for_inference(requests=None):
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
            req_id = request_batch['requestId']
            # TODO: If encoding present in "REQUEST" we shouldn't look for input-names and just pass it to the
            # custom service code.

            model_inputs = request_batch['modelInputs']
            try:
                input_data = Service.retrieve_model_input(model_inputs)
                input_batch.append(input_data)
                req_to_id_map[batch_idx] = req_id
            except MMSError as m:
                invalid_reqs.update({req_id: m.get_code()})

        return input_batch, req_to_id_map, invalid_reqs

    @staticmethod
    def create_predict_response(ret, req_id_map, invalid_reqs):
        """
        Response object is as follows :
        RESPONSE =
        {
            "code": val,
            "message": "Success"
            "predictions": [ PREDICTION_RESULTS ]
        }

        PREDICTION_RESULTS = {
            "requestId": 111-222-3333,
            "code": "Success/Fail" # TODO: Add this
            "value": Abefz23=,
            "encoding": "utf-8, base64"
        }

        :param ret:
        :param req_id_map:
        :param invalid_reqs:
        :return:
        """
        result = {}
        encoding = u'base64'
        try:
            for idx, val in enumerate(ret):
                result.update({"requestId": req_id_map[idx]})
                result.update({"code": 200})

                if isinstance(val, str):
                    value = ModelWorkerCodecHelper.encode_msg(encoding, val.encode('utf-8'))
                elif isinstance(val, bytes):
                    value = ModelWorkerCodecHelper.encode_msg(encoding, val)
                else:
                    value = ModelWorkerCodecHelper.encode_msg(encoding, json.dumps(val).encode('utf-8'))

                result.update({"value": value})
                result.update({"encoding": encoding})

            for req in invalid_reqs.keys():
                result.update({"requestId": req})
                result.update({"code": invalid_reqs.get(req)})
                result.update({"value": ModelWorkerCodecHelper.encode_msg(encoding,
                                                                          "Invalid input provided".encode('utf-8'))})
                result.update({"encoding": encoding})

            resp = [result]

        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.CODEC_FAIL, "codec failed {}".format(repr(e)))
        return resp

    def predict(self, data):
        """
        PREDICT COMMAND = {
            "command": "predict",
            "modelName": "model-to-run-inference-against",
            "contentType": "http-content-types", # TODO: Add this
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

        :param data:
        :return:

        """
        try:
            retval = []
            ModelWorkerMessageValidators.validate_predict_msg(data)
            model_name = data[u'modelName']
            req_batch = data[u'requestBatch']
            input_batch, req_id_map, invalid_reqs = Service.retrieve_data_for_inference(req_batch)
            if self.legacy:
                # Initialize metrics at service level
                self._entry_point.metrics_init(model_name, req_id_map)
                retval.append(self._entry_point.inference([input_batch[0][i] for i in input_batch[0]]))
                # Dump metrics
                emit_metrics(self._entry_point.metrics_store.store)
            else:
                self.context.request_ids = req_id_map
                self.context.metrics = MetricsStore(req_id_map, model_name)
                context_copy = copy.copy(self.context)
                retval.append(self._entry_point(context_copy, [input_batch[0][i] for i in input_batch[0]]))
                emit_metrics(context_copy.metrics.store)
            response = Service.create_predict_response(retval, req_id_map, invalid_reqs)

        except ValueError as v:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "{}".format(repr(v)))
        except MMSError as m:
            raise m
        return response, "Prediction success", 200


def emit_metrics(metrics):
    """
    Emit the metrics in the provided Dictionary

    Parameters
    ----------
    metrics: Dictionary
    A dictionary of all metrics, when key is metric_name
    value is a metric object
    """

    log_msg("[METRICS]", json.dumps(metrics, separators=(',', ':'), cls=MetricEncoder))
