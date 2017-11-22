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

def disk(metric_instance):
    disk_usage = psutil.disk_usage('/').percent / 100.0
    metric_instance.update(disk_usage)

    timer = threading.Timer(intervalSec, disk, [metric_instance])
    timer.daemon = True
    timer.start()

class MetricsManager(object):
    """Metrics Manager
    """
    metrics = {}

    @staticmethod
    def start(metrics_write_to, mutex):
        """Start service routing.

        Parameters
        ----------
        metrics_write_to : str
            Where metrics will be written to. (log, csv)
        mutex: object
            Mutex to prevent double thread writing on same resource
        """
        MetricsManager.metrics['error_metric'] = Metric('errors', mutex,
                                                        aggregate_method='interval_sum', 
                                                        write_to=metrics_write_to)
        MetricsManager.metrics['request_metric'] = Metric('requests', mutex,
                                                          aggregate_method='interval_sum', 
                                                          write_to=metrics_write_to)
        MetricsManager.metrics['cpu_metric'] = Metric('cpu', mutex,
                                                      aggregate_method='interval_average', 
                                                      write_to=metrics_write_to,
                                                      update_func=cpu)
        MetricsManager.metrics['memory_metric'] = Metric('memory', mutex,
                                                         aggregate_method='interval_average', 
                                                         write_to=metrics_write_to,
                                                         update_func=memory)
        MetricsManager.metrics['disk_metric'] = Metric('disk', mutex,
                                                        aggregate_method='interval_average', 
                                                        write_to=metrics_write_to,
                                                        update_func=disk)
        MetricsManager.metrics['overall_latency_metric'] = Metric('overall_latency', mutex,
                                                                    aggregate_method='interval_average', 
                                                                    write_to=metrics_write_to)
        MetricsManager.metrics['inference_latency_metric'] = Metric('inference_latency', mutex,
                                                                    aggregate_method='interval_average', 
                                                                    write_to=metrics_write_to)
        MetricsManager.metrics['pre_latency_metric'] = Metric('preprocess_latency', mutex,
                                                                aggregate_method='interval_average', 
                                                                write_to=metrics_write_to)

