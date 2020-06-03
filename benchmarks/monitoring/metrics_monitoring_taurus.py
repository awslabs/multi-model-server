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
Taurus local plugin for MMS monitoring.
This file should be placed in Python Path along with metrics_collector.py file
"""
# pylint: disable=redefined-builtin

import csv
from bzt.modules import monitoring
from bzt import  TaurusConfigError
from bzt.utils import dehumanize_time
from bzt.six import PY3

from metrics_collector import get_metrics, get_mms_processes, AVAILABLE_METRICS as MMS_AVAILABLE_METRICS


class Monitor(monitoring.Monitoring):
    def __init__(self):
        super(Monitor, self).__init__()
        self.client_classes.update({'MMS_local': MMSLocalClient})


class MMSLocalClient(monitoring.LocalClient):
    AVAILABLE_METRICS = monitoring.LocalClient.AVAILABLE_METRICS + \
                        MMS_AVAILABLE_METRICS

    def __init__(self, parent_log, label, config, engine=None):

        super(MMSLocalClient, self).__init__(parent_log, label, config, engine=engine)
        self.label = 'MMSLocalClient'

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

        self.monitor = MMSLocalMonitor(self.log, self.metrics, self.engine)
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


class MMSLocalMonitor(monitoring.LocalMonitor):

     def _calc_resource_stats(self, interval):
         result =super()._calc_resource_stats(interval)
         get_mms_processes()
         result.update(get_metrics())
         return result





