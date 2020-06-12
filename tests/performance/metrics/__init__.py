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

"""
Custom metrics
"""
# pylint: disable=redefined-builtin

AVAILABLE_METRICS = ["sum_cpu_percent",
                     "sum_memory_percent",
                     "sum_num_handles",
                     "server_workers"]


def get_metrics(server_process, child_processes):
    """ Get Server processes specific metrics
    """

    # TODO - make this modular may be a diff function for each metric
    # TODO - allow users to add new metrics easily
    # TODO - make sure available metric list is maintained

    sum_cpu_percent = 0
    sum_memory_percent = 0
    sum_num_handles = 0
    server_workers = 0
    metrics = {}
    for process in [server_process] + child_processes:
        try:
            process.cpu_percent()  # to warm-up
        except:
            pass
        else:
            cpu_percent = process.cpu_percent()
            memory_percent = process.memory_percent()
            sum_cpu_percent += cpu_percent
            sum_memory_percent += memory_percent
            sum_num_handles += process.num_fds()
            server_workers += 1

    metrics["sum_cpu_percent"] = sum_cpu_percent
    metrics["sum_memory_percent"] = sum_memory_percent
    metrics["sum_num_handles"] = sum_num_handles
    metrics["server_workers"] = server_workers

    return metrics