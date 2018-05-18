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
# pylint: disable=W0223

import os
import sys
import time
from abc import ABCMeta, abstractmethod, abstractproperty

from mms.log import get_logger
from mms.metrics_manager import MetricsManager

logger = get_logger()
URL_PREFIX = ('http://', 'https://', 's3://')


class ModelService(object):
    """
    ModelService wraps up all preprocessing, inference and postprocessing
    functions used by model service. It is defined in a flexible manner to
    be easily extended to support different frameworks.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model_name, model_dir, manifest, gpu=None):  # pylint: disable=unused-argument
        self.ctx = None

    @abstractmethod
    def inference(self, data):
        """
        Wrapper function to run pre-process, inference and post-process functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        pass

    @abstractmethod
    def ping(self):
        """
        Ping to get system's health.

        Returns
        -------
        String
            A message, "health": "healthy!", to show system is healthy.
        """
        pass

    @abstractproperty
    def signature(self):
        """
        Signiture for model service.

        Returns
        -------
        Dict
            Model service signature.
        """
        pass


class SingleNodeService(ModelService):
    """
    SingleNodeModel defines abstraction for model service which loads a
    single model.
    """

    def inference(self, data):
        """
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : list of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        """
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        data = self._inference(data)
        post_start_ms = time.time()
        data = self._postprocess(data)
        post_end_ms = time.time()

        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        infer_time_in_ms = (post_start_ms - infer_start_time) * 1000
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(infer_time_in_ms)

        post_time_in_ms = (post_end_ms - post_start_ms) * 1000
        if self.model_name + '_LatencyPostprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPostprocess'].update(post_time_in_ms)

        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + infer_time_in_ms +
                                                                               post_time_in_ms)
        return data

    @abstractmethod
    def _inference(self, data):
        """
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
        """
        return data

    def _preprocess(self, data):
        """
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
        """
        return data

    def _postprocess(self, data):
        """
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
        """
        return data


class MultiNodesService(ModelService):
    pass


def load_service(path, name=None):
    """
    Load the model-service into memory and associate it with each flask app worker
    :param path:
    :param name:
    :return:
    """
    try:
        if not name:
            name = os.path.splitext(os.path.basename(path))[0]

        module = None
        if sys.version_info[0] > 2:
            import importlib
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        else:
            import imp
            module = imp.load_source(name, path)

        return module
    except Exception as e:
        exc_tb = sys.exc_info()[2]
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise Exception('Error when loading service file: {} \n {}:{}:{}'.format(path, fname, exc_tb.tb_lineno, e))
