#!/usr/bin/env python3

# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import psutil
from statistics import mean
from enum import Enum

"""
Custom metrics
"""
# pylint: disable=redefined-builtin

class ProcessType(Enum):
    FRONTEND = 1
    WORKER = 2
    ALL  = 3


operators = {
    'sum': sum,
    'avg': mean,
    'min': min,
    'max': max
}

process_metrics = {
    'cpu_percent': lambda p: p.get('cpu_percent', 0),

    'memory_percent': lambda p: p.get('memory_percent', 0),

    'cpu_user_time': lambda p: getattr(p.get('cpu_times', {}), 'user', 0),
    'cpu_system_time': lambda p: getattr(p.get('cpu_times', {}), 'system', 0),
    'cpu_iowait_time': lambda p: getattr(p.get('cpu_times', {}), 'iowait', 0),

    'memory_rss': lambda p: getattr(p.get('memory_info', {}), 'rss', 0),
    'memory_vms': lambda p: getattr(p.get('memory_info', {}), 'vms', 0),

    'io_read_count': lambda p: getattr(p.get('io_counters', {}), 'read_count', 0),
    'io_write_count': lambda p: getattr(p.get('io_counters', {}), 'write_count', 0),
    'io_read_bytes': lambda p: getattr(p.get('io_counters', {}), 'read_bytes', 0),
    'io_write_bytes': lambda p: getattr(p.get('io_counters', {}), 'write_bytes', 0),

    'file_descriptors': lambda p: p.get('num_fds', 0),

    'threads': lambda p: p.get('num_threads', 0)
}

system_metrics = {
    'total_processes': None,
    'total_workers': None,
    'system_disk_used': None,
    'system_memory_percent': None,
    'system_read_count': None,
    'system_write_count': None,
    'system_read_bytes': None,
    'system_write_bytes': None,
}

AVAILABLE_METRICS = list(system_metrics)
for metric in list(process_metrics):
    for ptype in list(ProcessType):
        if ptype == ProcessType.WORKER:
            for op in list(operators):
                type = 'workers'
                AVAILABLE_METRICS.append('{}_{}_{}'.format(op,type,metric))
        elif ptype == ProcessType.FRONTEND:
            type = 'frontend'
            AVAILABLE_METRICS.append('{}_{}'.format(type, metric))
        else:
            type = 'all'
            for op in list(operators):
                type = 'all'
                AVAILABLE_METRICS.append('{}_{}_{}'.format(op,type,metric))


def get_metrics(server_process, child_processes):
    """ Get Server processes specific metrics
    """

    # TODO - make this modular may be a diff function for each metric
    # TODO - allow users to add new metrics easily
    # TODO - make sure available metric list is maintained

    result = {}

    def update_metric(metric_name, type, stats):
        stats = stats if stats else [0]

        if type == ProcessType.WORKER:
            type = 'workers'
        elif type == ProcessType.FRONTEND:
            result['frontend_' + metric_name] = stats[0]
            return
        else:
            type='all'

        for op_name in operators:
            result['{}_{}_{}'.format(op_name, type, metric_name)] = operators[op_name](stats)

    # as_dict() gets all stats in one shot
    processes_stats = []
    processes_stats.append({'type': ProcessType.FRONTEND, 'stats': server_process.as_dict()})
    for process in child_processes:
        processes_stats.append({'type': ProcessType.WORKER, 'stats' : process.as_dict()})

    ### PROCESS METRICS ###
    for k in process_metrics:

        worker_stats = list(map(lambda x: x['stats'], filter(lambda x: x['type'] == ProcessType.WORKER, processes_stats)))
        all_stats = list(map(lambda x: x['stats'], processes_stats))
        server_stats = list(map(lambda x: x['stats'], filter(lambda x: x['type'] == ProcessType.FRONTEND, processes_stats)))

        update_metric(k, ProcessType.WORKER, list(map(process_metrics[k], worker_stats)))
        update_metric(k, ProcessType.ALL, list(map(process_metrics[k], all_stats)))
        update_metric(k, ProcessType.FRONTEND, list(map(process_metrics[k], server_stats)))


    # Total processes
    result['total_processes'] = len([server_process] + child_processes)
    result['total_workers'] = len(child_processes) - 1

    ### SYSTEM METRICS ###
    result['system_disk_used'] = psutil.disk_usage('/').used

    result['system_memory_percent'] = psutil.virtual_memory().percent

    system_disk_io_counters = psutil.disk_io_counters()
    result['system_read_count'] = system_disk_io_counters.read_count
    result['system_write_count'] = system_disk_io_counters.write_count
    result['system_read_bytes'] = system_disk_io_counters.read_bytes
    result['system_write_bytes'] = system_disk_io_counters.write_bytes

    return result
