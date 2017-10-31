# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import csv
import datetime
import threading
from log import get_logger


logger = get_logger(__name__)


class Metric(object):
    """Metric class for model server
    """
    def __init__(self, name, interval=5*60, update_func=None):
        """Constructor for Metric class

        Parameters
        ----------
        name : str
            Name of the metric
        write_to : str
            Where to write metrics
        interval : int, optional
            Interval between each data points, default 5 min
        update_func : func, optional
            Update function for metrics
        """
        self.name = name
        self.interval = interval

        # Metrics within interval
        self.interval_datapoints = 0
        self.interval_metric = 0.0

        # Metrics for the whole session
        self.total_datapoints = 0
        self.total_metric = 0.0

        if update_func is not None:
            update_func(self)


    def update(self, metric):
        """Update function for Metric class

        Parameters
        ----------
        metric : float
            metric to be updated
        """
        self.interval_metric += metric
        # Increment data points
        self.interval_datapoints += 1

    def start_recording(self, write_to, method='interval_average'):
        """Periodically record metric

        Parameters
        ----------
        write_to : float
            Where the metrics will be recorded to
        method : str
            Method to aggregate metrics
            interval_average, interval_sum, total_average, total_sum
        """
        t = threading.Timer(self.interval, self.start_recording, [write_to, method])
        t.daemon = True
        t.start()

        # Update total metrics
        self.total_metric += self.interval_metric
        self.total_datapoints += self.interval_datapoints

        # Get metric data
        metric = 0.0
        if method == 'interval_average':
            if self.interval_datapoints != 0:
                metric = self.interval_metric / self.interval_datapoints
        elif method == 'interval_sum':
            metric = self.interval_metric
        elif method == 'total_average':
            if self.total_datapoints != 0:
                metric = self.total_metric / self.total_datapoints
        elif method == 'total_sum':
            metric = self.total_metric
        else:
            raise RuntimeError(method + ' for metric: ' + self.name + ' cannot be recognized.')

        utcnow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')

        # Start recording metrics
        if write_to == 'csv':
            with open('dms_' + self.name + '.csv', 'a') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow([utcnow, metric])
        else:
            logger.info('Metric %s for last %s seconds is %f' % 
                (self.name, self.interval, metric))

        # Clear interval metrics
        self.interval_metric = 0.0
        self.interval_datapoints = 0 

