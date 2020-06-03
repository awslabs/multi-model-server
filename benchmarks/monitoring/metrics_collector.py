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
MMS server metrics collector
"""
# pylint: disable=redefined-builtin

import os
import sys
import time
import logging
import psutil
import gevent
import argparse

TMP_DIR = "/var/folders/04/6_v1bbs55mb_hrpkphh46xcc0000gn/T"
METRICS_LOG_FILE = "{}/mms_metrics_{}.log".format(TMP_DIR, int(time.time()))
MMS_PID_FILE = "{}/.model_server.pid".format(TMP_DIR)
METRICS_COLLECTOR_PID_FILE = "{}/.metrics_collector.pid".format(TMP_DIR)
MONITOR_INTERVAL = 1
PID_LIST_INTERVAL = 2
AVAILABLE_METRICS = ["sum_cpu_percent",
                     "sum_memory_percent",
                     "sum_num_handles",
                     "mms_workers"]

child_processes = set()
mms_process = None


def find_procs_by_name(name):
    """Return a list of processes matching 'name'."""
    ls = []
    for p in psutil.process_iter(["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)

    if len(ls) > 1:
        raise Exception("Multiple processes found with name {}.".format(name))

    return ls[0]


def get_process_pid_from_file(file_path):
    """Get the process pid from pid file.
    """
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            pid = int(f.readline())
    else:
        raise Exception("Invalid file {}".format(file_path))

    return pid


def store_metrics_collector_pid():
    metrics_collector_process = psutil.Process(METRICS_COLLECTOR_PID_FILE)
    pid_file = os.path.join()
    with open(pid_file, "w") as pf:
        pf.write(str(metrics_collector_process.pid))


def stop_metrics_collector_process():
    """This will stop already running metrics collector process.
       Note at a time only one pid file will be available.
    """

    pid = get_process_pid_from_file(METRICS_COLLECTOR_PID_FILE)
    if pid:
        try:
            process = psutil.Process(pid)
            if process.is_running():
                logging.info("Process with pid {} is running. Killing it.".format(process.pid))
                process.kill()
        except Exception as e:
            pass
        else:
            logging.info("Dead process with pid {} found in '{}'.".format(process.pid, METRICS_COLLECTOR_PID_FILE))

        logging.info("Removing pid file '{}'.".format(METRICS_COLLECTOR_PID_FILE))
        os.remove(METRICS_COLLECTOR_PID_FILE)


def get_child_processes(process):
    """Get all running child processes recursively"""
    child_processes = []
    for p in process.children(recursive=True):
        if p.status() == 'running':
            child_processes.append(p)
    return child_processes


def get_metrics():
    """ Get MMS processes specific metrics
    """

    # TODO - make this modular may be a diff function for each metric
    # TODO - allow users to add new metrics easliy
    # TODO - make sure available metric list is maintained

    sum_cpu_percent = 0
    sum_memory_percent = 0
    sum_num_handles = 0
    mms_workers = 0
    metrics = {}
    for process in [mms_process] + list(get_child_processes(mms_process)):
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
            mms_workers += 1

    metrics["sum_cpu_percent"] = sum_cpu_percent
    metrics["sum_memory_percent"] = sum_memory_percent
    metrics["sum_num_handles"] = sum_num_handles
    metrics["mms_workers"] = mms_workers


    return metrics


def monitor_processes(socket, interval, metrics):
    while True:
        message = []
        collected_metrics = get_metrics()
        for metric in metrics:
           message.append(str(collected_metrics.get(metric, 0)))
        message = "\t".join(message)+"\n"

        if socket:
            socket.send(message.encode("latin-1"))
        gevent.sleep(interval)


def get_mms_processes():
    """ It caches the MMS and child processes at module level.
    Ensure that you call this process so that MMS process
    """

    global mms_process
    global child_processes
    mms_pid = get_process_pid_from_file()
    if not mms_pid:
        print("MMS process not found11.")
        exit()

    try:
        mms_process = psutil.Process(mms_pid)
    except Exception as e:
        print("MMS process not found. Error: {}".format(str(e)))
        raise

    child_processes = set(get_child_processes(mms_process))


def start_metric_collector_process():
    metric_collector_pid = get_process_pid_from_file(METRICS_COLLECTOR_PID_FILE)
    if metric_collector_pid:
        try:
            perf_mon_process = psutil.Process(metric_collector_pid)
        except Exception as e:
            stop_metrics_collector_process()
        else:
            if perf_mon_process.is_running():
                raise Exception("Performance monitoring script already running. "
                                "Stop it using stop option.")
    store_metrics_collector_pid()


def start_metric_collection(socket, interval, metrics):
    bad_metrics = set(metrics) - set(AVAILABLE_METRICS)
    if bad_metrics:
        raise Exception("Metrics not available for monitoring {}.".format(bad_metrics))

    get_mms_processes()
    print("Started metric collection for MMS processes.....")
    thread2 = gevent.spawn(monitor_processes, socket, interval, metrics)
    gevent.joinall([thread2])


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(prog='perf-mon-script', description='System Performance Monitoring')
    sub_parse = parser.add_mutually_exclusive_group(required=True)
    sub_parse.add_argument('--start', action='store_true', help='Start the perf-mon-script')
    sub_parse.add_argument('--stop', action='store_true', help='Stop the perf-mon-script')

    args = parser.parse_args()

    if args.start:
        start_perf_mon()
    elif args.stop:
        stop_perf_mon_process()
