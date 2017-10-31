# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from metric import Metric
import os
import psutil
import threading


ErrorMetric = Metric('error_number')
RequestsMetric = Metric('requests_number')

OverallLatencyMetric = Metric('overall_latency')
InferenceLatencyMetric = Metric('inference_latency')
PreLatencyMetric = Metric('preprocess_latency')
PostLatencyMetric = Metric('postprocess_latency')

# CPU and memory metric are collected every 5 seconds
interval = 5
def cpu(metric_instance):
    cpu_usage = psutil.cpu_percent() / 100.0
    metric_instance.update(cpu_usage)

    t = threading.Timer(interval, cpu, [metric_instance])
    t.daemon = True
    t.start()

def memory(metric_instance):
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_percent() / 100.0
    metric_instance.update(memory_usage)

    t = threading.Timer(interval, memory, [metric_instance])
    t.daemon = True
    t.start()

CPUMetric = Metric('cpu', update_func=cpu)
MemoryMetric = Metric('memory', update_func=memory)

