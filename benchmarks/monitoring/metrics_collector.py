#!/usr/bin/env python3

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Server metrics collector
"""
# pylint: disable=redefined-builtin

import os
import sys
import time
import logging
import psutil
import gevent
import argparse
import tempfile

logger = logging.getLogger(__name__)

from process import find_procs_by_name, get_process_pid_from_file, get_child_processes, \
    get_server_processes, get_server_pidfile
from metrics import AVAILABLE_METRICS, get_metrics

TMP_DIR = tempfile.gettempdir()
METRICS_LOG_FILE = "{}/server_metrics_{}.log".format(TMP_DIR, int(time.time()))
METRICS_COLLECTOR_PID_FILE = "{}/.metrics_collector.pid".format(TMP_DIR)

MONITOR_INTERVAL = 1


def store_metrics_collector_pid():
    """ Stores the process id of metrics collector process"""
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
                logging.info(logging.INFO, "Process with pid {} is running. Killing it.".format(process.pid))
                process.kill()
        except Exception as e:
            pass
        else:
            logging.info(logging.INFO, "Dead process with pid {} found in '{}'.".format(process.pid, METRICS_COLLECTOR_PID_FILE))

        logging.info(logging.INFO, "Removing pid file '{}'.".format(METRICS_COLLECTOR_PID_FILE))
        os.remove(METRICS_COLLECTOR_PID_FILE)


def monitor_processes(server_process, metrics, interval, socket):
    """ Monitor the metrics of server_process and its child processes
    """
    while True:
        message = []
        collected_metrics = get_metrics(server_process, get_child_processes(server_process))
        for metric in metrics:
           message.append(str(collected_metrics.get(metric, 0)))
        message = "\t".join(message)+"\n"

        if socket:
            socket.send(message.encode("latin-1"))

        # TODO - log metrics to a file METRICS_LOG_FILE if METRICS_LOG_FILE is provided
        gevent.sleep(interval)


def start_metric_collection(server_process, metrics, interval, socket):
    bad_metrics = set(metrics) - set(AVAILABLE_METRICS)
    if bad_metrics:
        raise Exception("Metrics not available for monitoring {}.".format(bad_metrics))

    logging.log(logging.INFO, "Started metric collection for target server processes.....")
    thread = gevent.spawn(monitor_processes, server_process, metrics, interval, socket)
    gevent.joinall([thread])


def start_metric_collector_process():
    """Spawn a metric collection process and keep on monitoring """
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
    server_pid = get_process_pid_from_file(get_server_pidfile())
    server_process = get_server_processes(server_pid)
    start_metric_collection(server_process, AVAILABLE_METRICS, MONITOR_INTERVAL, None)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(prog='perf-mon-script', description='System Performance Monitoring')
    sub_parse = parser.add_mutually_exclusive_group(required=True)
    sub_parse.add_argument('--start', action='store_true', help='Start the perf-mon-script')
    sub_parse.add_argument('--stop', action='store_true', help='Stop the perf-mon-script')

    args = parser.parse_args()

    if args.start:
        start_metric_collector_process()
    elif args.stop:
        stop_metrics_collector_process()
