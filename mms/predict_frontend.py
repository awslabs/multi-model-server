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
Predict frontend implementation
"""

import ast
import base64
import traceback
import time
import uuid

from flask import abort

from mms.metrics_manager import MetricsManager
from mms.log import get_logger


logger = get_logger()


class PredictFrontend(object):
    """
    PredictFrontend abstracts the predict callback so that batching can be turned on and off
    """

    def __init__(self, handler, batching, data_store):
        self.handler = handler
        self.batching = batching
        self.data_store = data_store

        if self.batching:
            if not self.data_store:
                raise Exception("DataStore must be defined when batching is enabled.")

    def callback(self, **kwargs):
        """
        Callback for predict endpoint

        Parameters
        ----------
        kwargs :
            model_service
            input_names
            model_name

        Returns
        ----------
        Response
            Http response for predict endpiont.
        """
        handler_start_time = time.time()
        model_service = kwargs['model_service']
        input_names = kwargs['input_names']
        model_name = kwargs['model_name']

        if model_name + '_PredictionTotal' in MetricsManager.metrics:
            MetricsManager.metrics[model_name + '_PredictionTotal'].update(metric=1)

        input_type = model_service.signature['input_type']
        output_type = model_service.signature['output_type']

        # Get data from request according to input type
        input_data = []
        if input_type == 'application/json':
            try:
                for name in input_names:
                    logger.info('Request input: %s should be json tensor.', name)
                    form_data = self.handler.get_form_data(name)
                    form_data = ast.literal_eval(form_data)
                    assert isinstance(form_data, list), "Input data for request argument: %s is not correct. " \
                                                        "%s is expected but got %s instead of list" \
                                                        % (name, input_type, type(form_data))
                    input_data.append(form_data)
            except Exception as e:  # pylint: disable=broad-except
                if model_name + '_Prediction4XX' in MetricsManager.metrics:
                    MetricsManager.metrics[model_name + '_Prediction4XX'].update(metric=1)
                logger.error(str(e))
                abort(400, str(e))
        elif input_type == 'image/jpeg':
            try:
                for name in input_names:
                    logger.info('Request input: %s should be image with jpeg format.', name)
                    input_file = self.handler.get_file_data(name)
                    if input_file:
                        mime_type = input_file.content_type
                        assert mime_type == input_type, 'Input data for request argument: %s is not correct. ' \
                                                        '%s is expected but %s is given.' % \
                                                        (name, input_type, mime_type)
                        file_data = input_file.read()
                        assert isinstance(file_data, (str, bytes)), 'Image file buffer should be type str or ' \
                                                                    'bytes, but got %s' % (type(file_data))
                    else:
                        form_data = self.handler.get_form_data(name)
                        if form_data:
                            # pylint: disable=deprecated-method
                            file_data = base64.decodestring(self.handler.get_form_data(name))
                        else:
                            raise ValueError('This end point is expecting a data_name of %s. '
                                             'End point details can be found here:http://<host>:<port>/api-description'
                                             % name)
                    input_data.append(file_data)
            except Exception as e:  # pylint: disable=broad-except
                if model_name + '_Prediction4XX' in MetricsManager.metrics:
                    MetricsManager.metrics[model_name + '_Prediction4XX'].update(metric=1)
                logger.error(str(e))
                abort(400, str(e))
        else:
            msg = '%s is not supported for input content-type' % input_type
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(msg)
            abort(500, "Service setting error. %s" % (msg))

        try:
            if self.batching:
                response = self._batch_predict(input_data, input_type, output_type, model_name)
            else:
                response = self._single_predict(input_data, model_service, model_name)
        except Exception as e:  # pylint: disable=broad-except
            if model_name + '_Prediction4XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction4XX'].update(metric=1)
            logger.error(str(e))
            abort(400, str(e))

        # Construct response according to output type
        if output_type == 'application/json':
            logger.info('Response is text.')
        elif output_type == 'image/jpeg':
            logger.info('Response is jpeg image encoded in base64 string.')
        else:
            msg = '%s is not supported for input content-type.' % output_type
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(msg)
            abort(500, "Service setting error. %s" % msg)

        logger.debug("Prediction request handling time is: %s ms",
                     ((time.time() - handler_start_time) * 1000))
        return self.handler.jsonify({'prediction': response})

    def _batch_predict(self, input_data, input_type, output_type, model_name):
        _id = str(uuid.uuid4())
        input_data = {'id': _id, 'data': input_data}
        self.data_store.push(model_name, input_data, input_type)

        response = self.data_store.get(_id, output_type)['data']

        if not response:
            abort(500, "Timed out")

        return response

    def _single_predict(self, input_data, model_service, model_name):
        try:
            response = model_service.inference([input_data])
        except Exception:  # pylint: disable=broad-except
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(str(traceback.format_exc()))
            abort(500, "Error occurs while inference was executed on server.")

        return response
