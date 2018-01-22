# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import psutil
import threading

from mms.metric import Metric, MetricUnit

# CPU and memory metric are collected every 5 seconds
intervalSec = 5
def cpu(metric_instance):
    cpu_usage = psutil.cpu_percent() / 100.0
    metric_instance.update(cpu_usage)

    timer = threading.Timer(intervalSec, cpu, [metric_instance])
    timer.daemon = True
    timer.start()

def memory(metric_instance):
    memory_usage = psutil.virtual_memory().used / (1024 * 1024) # in MB
    metric_instance.update(memory_usage)

    timer = threading.Timer(intervalSec, memory, [metric_instance])
    timer.daemon = True
    timer.start()

def memory_percentage(metric_instance):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_percent() / 100.0
    metric_instance.update(memory_usage)

    timer = threading.Timer(intervalSec, memory_percentage, [metric_instance])
    timer.daemon = True
    timer.start()

def disk_used(metric_instance):
    disk_usage = psutil.disk_usage('/').used / (1024 * 1024) # in MB
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk_used, [metric_instance])
    timer.daemon = True
    timer.start()

def disk_percentage(metric_instance):
    disk_usage = psutil.disk_usage('/').percent / 100.0
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk_percentage, [metric_instance])
    timer.daemon = True
    timer.start()

def disk_free(metric_instance):
    disk_usage = psutil.disk_usage('/').free / (1024 * 1024) # in MB
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk_free, [metric_instance])
    timer.daemon = True
    timer.start()

def disk_free_percentage(metric_instance):
    disk_usage = psutil.disk_usage('/').free / float(psutil.disk_usage('/').total)
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk_free_percentage, [metric_instance])
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
                                                                            unit=MetricUnit['MB'],
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
        MetricsManager.metrics['CPU'] = Metric('CPU', mutex,
                                                model_name=None,
                                                unit=MetricUnit['percent'],
                                                is_model_metric=False,
                                                aggregate_method='interval_average',
                                                write_to=metrics_write_to,
                                                update_func=cpu)
        MetricsManager.metrics['MemoryUsedMB'] = Metric('MemoryUsedMB', mutex,
                                                        model_name=None,
                                                        unit=MetricUnit['MB'],
                                                        is_model_metric=False,
                                                        aggregate_method='interval_average',
                                                        write_to=metrics_write_to,
                                                        update_func=memory)
        MetricsManager.metrics['MemoryUsedPercent'] = Metric('MemoryUsedPercent', mutex,
                                                              model_name=None,
                                                              unit=MetricUnit['percent'],
                                                              is_model_metric=False,
                                                              aggregate_method='interval_average',
                                                              write_to=metrics_write_to,
                                                              update_func=memory_percentage)
        MetricsManager.metrics['DiskUsedMB'] = Metric('DiskUsedMB', mutex,
                                                      model_name=None,
                                                      unit=MetricUnit['MB'],
                                                      is_model_metric=False,
                                                      aggregate_method='interval_average',
                                                      write_to=metrics_write_to,
                                                      update_func=disk_used)
        MetricsManager.metrics['DiskUsedPercent'] = Metric('DiskUsedPercent', mutex,
                                                            model_name=None,
                                                            unit=MetricUnit['percent'],
                                                            is_model_metric=False,
                                                            aggregate_method='interval_average',
                                                            write_to=metrics_write_to,
                                                            update_func=disk_percentage)
        MetricsManager.metrics['DiskFreeMB'] = Metric('DiskFreeMB', mutex,
                                                      model_name=None,
                                                      unit=MetricUnit['MB'],
                                                      is_model_metric=False,
                                                      aggregate_method='interval_average',
                                                      write_to=metrics_write_to,
                                                      update_func=disk_free)
        MetricsManager.metrics['DiskFreePercent'] = Metric('DiskFreePercent', mutex,
                                                            model_name=None,
                                                            unit=MetricUnit['percent'],
                                                            is_model_metric=False,
                                                            aggregate_method='interval_average',
                                                            write_to=metrics_write_to,
                                                            update_func=disk_free_percentage)
