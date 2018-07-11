# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Metric class for model server
"""
from collections import OrderedDict
from json import JSONEncoder

from mms.unit import Units

MetricUnit = Units()

class Metric(object):
    """
    Class for generating metrics and printing it to stdout of the worker
    """
    def __init__(self, value,
                 unit, metric_method):
        """Constructor for Metric class

           Metric class will spawn a thread and report collected metrics to stdout of worker

        Parameters
        ----------
        value : int, float
           Can be integer or float
        unit: str
            unit can be one of ms, percent, count, MB, GB or a generic string
        metric_method: str
           useful for defining different operations, optional
        """
        self.unit = unit
        if unit  in list(MetricUnit.units.keys()):
            self.unit = MetricUnit.units[unit]
        self.metric_method = metric_method
        self.value = value

    def update(self, value, reverse=False):
        """Update function for Metric class

        Parameters
        ----------
        value : int, float
            metric to be updated
        reverse: boolean
            used for counter metrics , indicating a decrement counter
        """

        if self.metric_method == 'counter':
            if not reverse:
                self.value += value
            else:
                self.value -= value
        else:
            self.value = value
        # TODO: Add specific operations for other metric methods as required.

    def to_dict(self):
        """
        return an Ordered Dictionary
        """
        return OrderedDict({'value' : self.value, 'unit' : self.unit,})

class MetricEncoder(JSONEncoder):
    """
    Encoder class for json encoding Metric Object
    """
    def default(self, obj): #  pylint: disable=arguments-differ, method-hidden
        """
        Override only when object is of type Metric
        """
        if isinstance(obj, Metric):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)
