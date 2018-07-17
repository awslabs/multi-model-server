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
Module to collect system metrics for front-end
"""
import sys
import types
import json
from collections import OrderedDict

import psutil
from mms.metric import Metric, MetricEncoder

overall_metrics = OrderedDict()
metrics = OrderedDict()
overall_metrics['SYSTEM'] = metrics

def cpu_utilization():
    data = psutil.cpu_percent()
    metrics['CPUUtilization'] = Metric(data, 'percent')


def memory_used():
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    metrics['MemoryUsed'] = Metric(data, 'MB')

def memory_available():
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    metrics['MemoryAvailable'] = Metric(data, 'MB')


def memory_utilization():
    data = psutil.virtual_memory().percent
    metrics['MemoryUtilization'] = Metric(data, 'percent')

def disk_used():
    data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
    metrics['DiskUsage'] = Metric(data, 'GB')

def disk_utilization():
    data = psutil.disk_usage('/').percent
    metrics['DiskUtilization'] = Metric(data, 'percent')


def disk_available():
    data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
    metrics['DiskAvailable'] = Metric(data, 'GB')


def collect_all(mod):
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ != 'collect_all':
            value()
    print(json.dumps(overall_metrics, indent=4, separators=(',', ':'), cls=MetricEncoder))
    sys.stdout.flush()


if __name__ == '__main__':
    collect_all(sys.modules[__name__])
