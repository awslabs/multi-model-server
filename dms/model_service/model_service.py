# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""`ModelService` defines an API for base model service.
"""

import time
import sys

from abc import ABCMeta, abstractmethod, abstractproperty
from dms import metrics_manager
from dms.log import get_logger


logger = get_logger(__name__)
URL_PREFIX = ('http://', 'https://', 's3://')

class ModelService(object):
    '''ModelService wraps up all preprocessing, inference and postprocessing
    functions used by model service. It is defined in a flexible manner to
    be easily extended to support different frameworks.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, service_name, path, gpu=None):
        self.ctx = None

    @abstractmethod
    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pass

    @abstractmethod
    def ping(self):
        '''Ping to get system's health.

        Returns
        -------
        String
            A message, "health": "healthy!", to show system is healthy.
        '''
        pass

    @abstractproperty
    def signature(self):
        '''Signiture for model service.

        Returns
        -------
        Dict
            Model service signiture.
        '''
        pass


class SingleNodeService(ModelService):
    '''SingleNodeModel defines abstraction for model service which loads a
    single model.
    '''
    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()

        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        metrics_manager.pre_latency_metric.update(pre_time_in_ms)

        data = self._inference(data)
        data = self._postprocess(data)

        # Update inference latency metric
        infer_time_in_ms = (time.time() - infer_start_time) * 1000
        metrics_manager.inference_latency_metric.update(infer_time_in_ms)

        # Update overall latency metric
        metrics_manager.overall_latency_metric.update(pre_time_in_ms + infer_time_in_ms)

        return data

    @abstractmethod
    def _inference(self, data):
        '''
        Internal inference methods. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        '''
        return data

    def _preprocess(self, data):
        '''
        Internal preprocess methods. Do transformation on raw
        inputs and convert them to NDArray.

        Parameters
        ----------
        data : list of object
            Raw inputs from request.

        Returns
        -------
        list of NDArray
            Processed inputs in NDArray format.
        '''
        return data

    def _postprocess(self, data):
        '''
        Internal postprocess methods. Do transformation on inference output
        and convert them to MIME type objects.

        Parameters
        ----------
        data : list of NDArray
            Inference output.

        Returns
        -------
        list of object
            list of outputs to be sent back.
        '''
        return data


class MultiNodesService(ModelService):
    pass

