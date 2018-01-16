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

from mms.metric import Metric

# CPU and memory metric are collected every 5 seconds
intervalSec = 5
def cpu(metric_instance):
    cpu_usage = psutil.cpu_percent() / 100.0
    metric_instance.update(cpu_usage)

    timer = threading.Timer(intervalSec, cpu, [metric_instance])
    timer.daemon = True
    timer.start()

def memory(metric_instance):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_percent() / 100.0
    metric_instance.update(memory_usage)

    timer = threading.Timer(intervalSec, memory, [metric_instance])
    timer.daemon = True
    timer.start()

def disk_percentage(metric_instance):
    disk_usage = psutil.disk_usage('/').percent / 100.0
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk_percentage, [metric_instance])
    timer.daemon = True
    timer.start()

def disk(metric_instance):
    disk_usage = psutil.disk_usage('/').used / (1024 * 1024) # in MB
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk, [metric_instance])
    timer.daemon = True
    timer.start()

MetricUnit = {
  'ms': "Milliseconds",
  'percent': 'Percent',
  'count': 'Count',
  'MB': 'Megabytes'
}

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
            MetricsManager.metrics['5XX_errors'] = Metric('5XX_errors', mutex,
                                                                model_name=model_name,
                                                                unit=MetricUnit['count'],
                                                                aggregate_method='interval_sum',
                                                                write_to=metrics_write_to)
            MetricsManager.metrics['4XX_errors'] = Metric('4XX_errors', mutex,
                                                                model_name=model_name,
                                                                unit=MetricUnit['count'],
                                                                aggregate_method='interval_sum',
                                                                write_to=metrics_write_to)
            MetricsManager.metrics['requests'] = Metric('requests', mutex,
                                                              model_name=model_name,
                                                              unit=MetricUnit['count'],
                                                              aggregate_method='interval_sum',
                                                              write_to=metrics_write_to)
            MetricsManager.metrics['overall_latency'] = Metric('overall_latency', mutex,
                                                                      model_name=model_name,
                                                                      unit=MetricUnit['MB'],
                                                                      aggregate_method='interval_average',
                                                                      write_to=metrics_write_to)
            MetricsManager.metrics['inference_latency'] = Metric('inference_latency', mutex,
                                                                        model_name=model_name,
                                                                        unit=MetricUnit['ms'],
                                                                        aggregate_method='interval_average',
                                                                        write_to=metrics_write_to)
            MetricsManager.metrics['pre_latency'] = Metric('preprocess_latency', mutex,
                                                                  model_name=model_name,
                                                                  unit=MetricUnit['ms'],
                                                                  aggregate_method='interval_average',
                                                                  write_to=metrics_write_to)

        MetricsManager.metrics['host_cpu'] = Metric('host_cpu', mutex,
                                                      model_name=None,
                                                      unit=MetricUnit['percent'],
                                                      is_resource=True,
                                                      aggregate_method='interval_average',
                                                      write_to=metrics_write_to,
                                                      update_func=cpu)
        MetricsManager.metrics['host_memory'] = Metric('host_memory', mutex,
                                                         model_name=None,
                                                         unit=MetricUnit['percent'],
                                                         is_resource=True,
                                                         aggregate_method='interval_average',
                                                         write_to=metrics_write_to,
                                                         update_func=memory)
        MetricsManager.metrics['host_disk_percentage'] = Metric('host_disk_percentage', mutex,
                                                                  model_name=None,
                                                                  unit=MetricUnit['percent'],
                                                                  is_resource=True,
                                                                  aggregate_method='interval_average',
                                                                  write_to=metrics_write_to,
                                                                  update_func=disk_percentage)
        MetricsManager.metrics['host_disk'] = Metric('host_disk', mutex,
                                                        model_name=None,
                                                        unit=MetricUnit['MB'],
                                                        is_resource=True,
                                                        aggregate_method='interval_average',
                                                        write_to=metrics_write_to,
                                                        update_func=disk)

