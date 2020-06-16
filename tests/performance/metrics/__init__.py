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



result = {
    # 'sum_cpu_percent': None,
    # 'avg_cpu_percent': None,
    # 'min_cpu_percent': None,
    # 'max_cpu_percent': None,
    #
    # 'sum_memory_percent': None,
    # 'avg_memory_percent': None,
    # 'min_memory_percent': None,
    # 'max_memory_percent': None,
    #
    # 'sum_cpu_user_time': None,
    # 'avg_cpu_user_time': None,
    # 'min_cpu_user_time': None,
    # 'max_cpu_user_time': None,
    #
    # 'sum_cpu_system_time': None,
    # 'avg_cpu_system_time': None,
    # 'min_cpu_system_time': None,
    # 'max_cpu_system_time': None,
    #
    # 'sum_cpu_iowait_time': None,
    # 'avg_cpu_iowait_time': None,
    # 'min_cpu_iowait_time': None,
    # 'max_cpu_iowait_time': None,
    #
    # 'sum_memory_rss': None,
    # 'avg_memory_rss': None,
    # 'min_memory_rss': None,
    # 'max_memory_rss': None,
    #
    # 'sum_memory_vms': None,
    # 'avg_memory_vms': None,
    # 'min_memory_vms': None,
    # 'max_memory_vms': None,
    #
    # 'sum_io_read_count': None,
    # 'avg_io_read_count': None,
    # 'min_io_read_count': None,
    # 'max_io_read_count': None,
    #
    # 'sum_io_write_count': None,
    # 'avg_io_write_count': None,
    # 'min_io_write_count': None,
    # 'max_io_write_count': None,
    #
    # 'sum_io_read_bytes': None,
    # 'avg_io_read_bytes': None,
    # 'min_io_read_bytes': None,
    # 'max_io_read_bytes': None,
    #
    # 'sum_io_write_bytes': None,
    # 'avg_io_write_bytes': None,
    # 'min_io_write_bytes': None,
    # 'max_io_write_bytes': None,
    #
    # 'sum_file_descriptors': None,
    # 'avg_file_descriptors': None,
    # 'min_file_descriptors': None,
    # 'max_file_descriptors': None,
    #
    # 'sum_threads': None,
    # 'avg_threads': None,
    # 'min_threads': None,
    # 'max_threads': None,
    #
    # 'sum_processes': None,
    #
    #
    # 'system_disk_used': None,
    # 'system_memory_used': None,
    # 'system_read_count': None,
    # 'system_read_bytes': None,
    # 'system_write_bytes': None,

    'frontend_memory_rss': None,
    'sum_workers_memory_rss': None
}
AVAILABLE_METRICS = list(result)

def get_metrics(server_process, child_processes):
    """ Get Server processes specific metrics
    """

    # TODO - make this modular may be a diff function for each metric
    # TODO - allow users to add new metrics easily
    # TODO - make sure available metric list is maintained

    def update_metric(name, type, stats):
        if type == ProcessType.WORKER:
            type = "workers"
        elif type == ProcessType.FRONTEND :
            result['frontend_' + name] = stats[0]
            return
        else:
            type="all"

        result['sum_'+type+'_'+ name] = sum(stats)
        result['avg_'+type+'_'+ name] = mean(stats)
        result['min_'+type+'_'+ name] = min(stats)
        result['max_'+type+'_'+ name] = max(stats)

    processes_stats = []
    processes_stats.append({'type': ProcessType.FRONTEND, 'stats': server_process.as_dict()})
    for process in child_processes:
        # Get all stats in one shots
        processes_stats.append({'type': ProcessType.WORKER, 'stats' : process.as_dict()})

    ### PROCESS METRICS ###

    metrics = {
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

    for k in metrics:

        worker_stats = list(map(lambda x: x['stats'], filter(lambda x: x['type'] == ProcessType.WORKER, processes_stats)))
        all_stats = list(map(lambda x: x['stats'], processes_stats))
        server_stats = list(map(lambda x: x['stats'], filter(lambda x: x['type'] == ProcessType.FRONTEND, processes_stats)))

        if len(worker_stats): update_metric(k, ProcessType.WORKER, list(map(metrics[k], worker_stats)))
        if len(all_stats): update_metric(k, ProcessType.ALL, list(map(metrics[k], all_stats)))
        if len(server_stats): update_metric(k, ProcessType.FRONTEND, list(map(metrics[k], server_stats)))


    # Total processes
    result['sum_processes'] = len(processes_stats)

    ### SYSTEM METRICS ###
    result["system_disk_used"] = psutil.disk_usage('/').used

    result["system_memory_percent"] = psutil.virtual_memory().percent

    system_disk_io_counters = psutil.disk_io_counters()
    result["system_read_count"] = system_disk_io_counters.read_count
    result["system_write_count"] = system_disk_io_counters.write_count
    result["system_read_bytes"] = system_disk_io_counters.read_bytes
    result["system_write_bytes"] = system_disk_io_counters.write_bytes

    print('frontend_memory_rss    :', result["frontend_memory_rss"])
    print('sum_workers_memory_rss :', result["sum_workers_memory_rss"])
    print()

    return result