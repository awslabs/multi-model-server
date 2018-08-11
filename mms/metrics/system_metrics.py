# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions  and limitations under the License.
"""
Module to collect system metrics for front-end
"""
import json
import sys
import types

import psutil

from mms.log import log_msg
from mms.metrics.dimension import Dimension
from mms.metrics.metric import Metric
from mms.metrics.metric_encoder import MetricEncoder

system_metrics = []
dimension = [Dimension('Level', 'Host')]


def cpu_utilization():
    data = psutil.cpu_percent()
    system_metrics.append(Metric('CPUUtilization', data, 'percent', dimension))


def memory_used():
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    system_metrics.append(Metric('MemoryUsed', data, 'MB', dimension))


def memory_available():
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    system_metrics.append(Metric('MemoryAvailable', data, 'MB', dimension))


def memory_utilization():
    data = psutil.virtual_memory().percent
    system_metrics.append(Metric('MemoryUtilization', data, 'percent', dimension))


def disk_used():
    data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric('DiskUsage', data, 'GB', dimension))


def disk_utilization():
    data = psutil.disk_usage('/').percent
    system_metrics.append(Metric('DiskUtilization', data, 'percent', dimension))


def disk_available():
    data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
    system_metrics.append(Metric('DiskAvailable', data, 'GB', dimension))


def collect_all(mod):
    members = dir(mod)
    for i in members:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ not in ('collect_all', 'log_msg'):
            value()
    log_msg(json.dumps(system_metrics, separators=(',', ':'), cls=MetricEncoder))


if __name__ == '__main__':
    collect_all(sys.modules[__name__])
