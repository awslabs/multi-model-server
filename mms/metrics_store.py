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
Metrics collection module
"""
from collections import OrderedDict
from mms.metric import Metric


class MetricsStore(object):
    """
    Class for creating, modifying different metrics. And keep them in a dictionary
    """

    def __init__(self, request_ids, model_name):
        """
        Initialize metrics map,model name and request map
        """
        self.metrics = OrderedDict()
        self.request_ids = request_ids
        self.metrics[model_name] = OrderedDict()
        if request_ids is not None:
            for req_id in request_ids.values():
                self.metrics[model_name][req_id] = OrderedDict()
        # When request id is not given it goes to ALL dimension
        self.metrics[model_name]['ALL'] = OrderedDict()
        self.metrics[model_name]['ERROR'] = OrderedDict()
        self.model_name = model_name

    def _add(self, name, value, req_id, unit, metrics_method=None):
        """
        Add a metric key value pair

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        req_id: str
            request id
        unit: str
            unit of metric
        metrics_method: str, optional
            indicates type of metric operation if it is defined
        """
        # Create a metric object
        self.metrics[self.model_name][req_id][name] = Metric(value, unit, metrics_method)

    def _get_req(self, idx):
        """
        Provide the request id dimension

        Parameters
        ----------

        idx : int
            request_id index in batch
        """
        # check if request id for the metric is given, if so use it else have 'ALL'
        req_id = 'ALL'
        if idx is not None and self.request_ids is not None and idx in self.request_ids:
            req_id = self.request_ids[idx]
        elif idx == -1:
            req_id = 'ERROR'
        return req_id

    def add_counter(self, name, value, idx=None):
        """
        Add a counter metric or increment an existing counter metric

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        """
        unit = 'count'
        req_id = self._get_req(idx)
        if name not in self.metrics[self.model_name][req_id]:
            self._add(name, value, req_id, unit, 'counter')
            return
        self.metrics[self.model_name][req_id][name].update(value)

    def add_time(self, name, value, idx=None, unit='ms'):
        """
        Add a time based metric like latency, default unit is 'ms'

        Parameters
        ----------
        name : str
            metric name
        value: int
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric,  default here is ms, s is also accepted
        """
        if unit not in ['ms', 's']:
            raise ValueError("the unit for a timed metric should be one of ['ms', 's']")
        req_id = self._get_req(idx)
        if name not in self.metrics:
            self._add(name, value, req_id, unit)
            return
        self.metrics[self.model_name][req_id][name].update(value)

    def add_size(self, name, value, idx=None, unit='MB'):
        """
        Add a size based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric, default here is 'MB', 'kB', 'GB' also supported
        """
        if unit not in ['MB', 'kB', 'GB']:
            raise ValueError("The unit for size based metric is one of ['MB','kB', 'GB']")
        req_id = self._get_req(idx)
        if name not in self.metrics:
            self._add(name, value, req_id, unit)
            return
        self.metrics[self.model_name][req_id][name].update(value)

    def add_percent(self, name, value, idx=None):
        """
        Add a percentage based metric

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        """
        unit = 'percent'
        req_id = self._get_req(idx)
        if name not in self.metrics:
            self._add(name, value, req_id, unit)
            return
        self.metrics[self.model_name][req_id][name].update(value)

    def add_error(self, name, value):
        """
        Add a Error Metric
        Parameters
        ----------
        name : str
            metric name
        value: str
            value of metric, in this case a str
        """
        unit = 'end_error'
        idx = -1
        req_id = self._get_req(idx)
        if name not in self.metrics:
            self._add(name, value, req_id, unit)
            return
        self.metrics[self.model_name][req_id][name].update(value)

    def add_metric(self, name, value, idx=None, unit=None):
        """
        Add a metric which is generic with custom metrics

        Parameters
        ----------
        name : str
            metric name
        value: int, float
            value of metric
        idx: int
            request_id index in batch
        unit: str
            unit of metric
        """
        req_id = self._get_req(idx)
        if name not in self.metrics:
            self._add(name, value, req_id, unit)
            return
        self.metrics[self.model_name][req_id][name].update(value)
