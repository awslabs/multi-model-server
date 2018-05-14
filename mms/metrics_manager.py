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
Metrics manager for publishing MMS metrics
"""
import threading

import psutil

from mms.metric import Metric, MetricUnit

# CPU and memory metric are collected every 5 seconds
intervalSec = 5


def cpu_utilization(metric_instance):
    data = psutil.cpu_percent()
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, cpu_utilization, [metric_instance])
    timer.daemon = True
    timer.start()


def memory_used(metric_instance):
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, memory_used, [metric_instance])
    timer.daemon = True
    timer.start()


def memory_available(metric_instance):
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, memory_available, [metric_instance])
    timer.daemon = True
    timer.start()


def memory_utilization(metric_instance):
    data = psutil.virtual_memory().percent
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, memory_utilization, [metric_instance])
    timer.daemon = True
    timer.start()


def disk_used(metric_instance):
    data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, disk_used, [metric_instance])
    timer.daemon = True
    timer.start()


def disk_utilization(metric_instance):
    data = psutil.disk_usage('/').percent
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, disk_utilization, [metric_instance])
    timer.daemon = True
    timer.start()


def disk_available(metric_instance):
    data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
    metric_instance.update(data)

    timer = threading.Timer(intervalSec, disk_available, [metric_instance])
    timer.daemon = True
    timer.start()


class MetricsManager(object):
    """Metrics Manager
    """
    metrics = {}

    @staticmethod
    def start(metrics_write_to, models, mutex):
        """Start service routing.

        Parameters
        ----------
        metrics_write_to : str
            Where metrics will be written to. (log, csv)
        mutex: object
            Mutex to prevent double thread writing on same resource
        """
        # pylint: disable=unused-variable
        for model_name, model_class in models.items():
            MetricsManager.metrics[model_name + '_Prediction5XX'] = Metric('Prediction5XX', mutex,
                                                                           model_name=model_name,
                                                                           unit=MetricUnit['count'],
                                                                           aggregate_method='interval_sum',
                                                                           write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_Prediction4XX'] = Metric('Prediction4XX', mutex,
                                                                           model_name=model_name,
                                                                           unit=MetricUnit['count'],
                                                                           aggregate_method='interval_sum',
                                                                           write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_PredictionTotal'] = Metric('PredictionTotal', mutex,
                                                                             model_name=model_name,
                                                                             unit=MetricUnit['count'],
                                                                             aggregate_method='interval_sum',
                                                                             write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_LatencyOverall'] = Metric('LatencyOverall', mutex,
                                                                            model_name=model_name,
                                                                            unit=MetricUnit['ms'],
                                                                            aggregate_method='interval_average',
                                                                            write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_LatencyInference'] = Metric('LatencyInference', mutex,
                                                                              model_name=model_name,
                                                                              unit=MetricUnit['ms'],
                                                                              aggregate_method='interval_average',
                                                                              write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_LatencyPreprocess'] = Metric('LatencyPreprocess', mutex,
                                                                               model_name=model_name,
                                                                               unit=MetricUnit['ms'],
                                                                               aggregate_method='interval_average',
                                                                               write_to=metrics_write_to)
            MetricsManager.metrics[model_name + '_LatencyPostprocess'] = Metric('LatencyPostprocess', mutex,
                                                                                model_name=model_name,
                                                                                unit=MetricUnit['ms'],
                                                                                aggregate_method='interval_average',
                                                                                write_to=metrics_write_to)

        MetricsManager.metrics['PingTotal'] = Metric('PingTotal', mutex,
                                                     model_name=None,
                                                     unit=MetricUnit['count'],
                                                     is_model_metric=False,
                                                     aggregate_method='interval_sum',
                                                     write_to=metrics_write_to)
        MetricsManager.metrics['APIDescriptionTotal'] = Metric('APIDescriptionTotal', mutex,
                                                               model_name=None,
                                                               unit=MetricUnit['percent'],
                                                               is_model_metric=False,
                                                               aggregate_method='interval_sum',
                                                               write_to=metrics_write_to)
        MetricsManager.metrics['CPUUtilization'] = Metric('CPUUtilization', mutex,
                                                          model_name=None,
                                                          unit=MetricUnit['percent'],
                                                          is_model_metric=False,
                                                          aggregate_method='interval_average',
                                                          write_to=metrics_write_to,
                                                          update_func=cpu_utilization)
        MetricsManager.metrics['MemoryUsed'] = Metric('MemoryUsed', mutex,
                                                      model_name=None,
                                                      unit=MetricUnit['MB'],
                                                      is_model_metric=False,
                                                      aggregate_method='interval_average',
                                                      write_to=metrics_write_to,
                                                      update_func=memory_used)
        MetricsManager.metrics['MemoryAvailable'] = Metric('MemoryAvailable', mutex,
                                                           model_name=None,
                                                           unit=MetricUnit['MB'],
                                                           is_model_metric=False,
                                                           aggregate_method='interval_average',
                                                           write_to=metrics_write_to,
                                                           update_func=memory_available)
        MetricsManager.metrics['MemoryUtilization'] = Metric('MemoryUtilization', mutex,
                                                             model_name=None,
                                                             unit=MetricUnit['percent'],
                                                             is_model_metric=False,
                                                             aggregate_method='interval_average',
                                                             write_to=metrics_write_to,
                                                             update_func=memory_utilization)
        MetricsManager.metrics['DiskUsed'] = Metric('DiskUsed', mutex,
                                                    model_name=None,
                                                    unit=MetricUnit['GB'],
                                                    is_model_metric=False,
                                                    aggregate_method='interval_average',
                                                    write_to=metrics_write_to,
                                                    update_func=disk_used)
        MetricsManager.metrics['DiskAvailable'] = Metric('DiskAvailable', mutex,
                                                         model_name=None,
                                                         unit=MetricUnit['GB'],
                                                         is_model_metric=False,
                                                         aggregate_method='interval_average',
                                                         write_to=metrics_write_to,
                                                         update_func=disk_available)
        MetricsManager.metrics['DiskUtilization'] = Metric('DiskUtilization', mutex,
                                                           model_name=None,
                                                           unit=MetricUnit['percent'],
                                                           is_model_metric=False,
                                                           aggregate_method='interval_average',
                                                           write_to=metrics_write_to,
                                                           update_func=disk_utilization)
