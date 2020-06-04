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
Taurus Local plugin for server monitoring.
Should be used when server and Taurus are running on same machine.
This file should be placed in Python Path along with monitoring package.
"""
# pylint: disable=redefined-builtin

import csv
from bzt.modules import monitoring
from bzt import  TaurusConfigError
from bzt.utils import dehumanize_time
from bzt.six import PY3

from metrics import get_metrics, AVAILABLE_METRICS as AVAILABLE_SERVER_METRICS
from process import get_process_pid_from_file, get_server_processes, \
    get_child_processes, get_server_pidfile

class Monitor(monitoring.Monitoring):
    def __init__(self):
        super(Monitor, self).__init__()
        self.client_classes.update({'ServerLocalClient': ServerLocalClient})


class ServerLocalClient(monitoring.LocalClient):
    AVAILABLE_METRICS = monitoring.LocalClient.AVAILABLE_METRICS + \
                        AVAILABLE_SERVER_METRICS

    def __init__(self, parent_log, label, config, engine=None):

        super(ServerLocalClient, self).__init__(parent_log, label, config, engine=engine)
        self.label = 'ServerLocalClient'

    def connect(self):
        exc = TaurusConfigError('Metric is required in Local monitoring client')
        metric_names = self.config.get('metrics', exc)

        bad_list = set(metric_names) - set(self.AVAILABLE_METRICS)
        if bad_list:
            self.log.warning('Wrong metrics found: %s', bad_list)

        good_list = set(metric_names) & set(self.AVAILABLE_METRICS)
        if not good_list:
            raise exc

        self.metrics = list(set(good_list))

        self.monitor = ServerLocalMonitor(self.log, self.metrics, self.engine)
        self.interval = dehumanize_time(self.config.get("interval", self.engine.check_interval))

        if self.config.get("logging", False):
            if not PY3:
                self.log.warning("Logging option doesn't work on python2.")
            else:
                self.logs_file = self.engine.create_artifact("local_monitoring_logs", ".csv")
                with open(self.logs_file, "a", newline='') as mon_logs:
                    logs_writer = csv.writer(mon_logs, delimiter=',')
                    metrics = ['ts'] + sorted([metric for metric in good_list])
                    logs_writer.writerow(metrics)


class ServerLocalMonitor(monitoring.LocalMonitor):

     def _calc_resource_stats(self, interval):
         result = super()._calc_resource_stats(interval)
         server_pid = get_process_pid_from_file(get_server_pidfile())
         server_process = get_server_processes(server_pid)
         result.update(get_metrics(server_process, get_child_processes(server_process)))
         return result
