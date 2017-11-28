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
import os

from mms.log import get_logger


logger = get_logger()


class Metric(object):
    """Metric class for model server
    """
    def __init__(self, name, mutex,
                 interval_sec=30,
                 update_func=None,
                 aggregate_method='interval_average',
                 write_to='log'):
        """Constructor for Metric class

        Parameters
        ----------
        name : str
            Name of the metric
        interval_sec : int, optional
            Interval seconds between each data points, default 5 min
        update_func : func, optional
            Update function for metrics
        aggregate_method: str, optional
            The way to aggregate metrics. (interval_average, interval_sum, total_average, total_sum)
        write_to : str, optional
            Where the metrics will be recorded to. (log, csv)
        """
        self.name = name
        self.interval_sec = interval_sec

        # Metrics within interval
        self.interval_datapoints_count = 0
        self.interval_metric_aggregate = 0.0

        # Metrics for the whole session
        self.total_datapoints_count = 0
        self.total_metric_aggregate = 0.0

        self.aggregate_method = aggregate_method
        self.mutex = mutex
        self.write_to = write_to

        if update_func is not None:
            update_func(self)

        self.start_recording()

    def update(self, metric):
        """Update function for Metric class

        Parameters
        ----------
        metric : float
            metric to be updated
        """
        self.interval_metric_aggregate += metric
        # Increment data points
        self.interval_datapoints_count += 1

    def start_recording(self):
        """Periodically record metric

        Parameters
        """
        timer = threading.Timer(self.interval_sec, self.start_recording)
        timer.daemon = True
        timer.start()

        # Update total metrics
        self.total_metric_aggregate += self.interval_metric_aggregate
        self.total_datapoints_count += self.interval_datapoints_count

        # Get metric data
        metric = 0.0
        if self.aggregate_method == 'interval_average':
            if self.interval_datapoints_count != 0:
                metric = self.interval_metric_aggregate / self.interval_datapoints_count
        elif self.aggregate_method == 'interval_sum':
            metric = self.interval_metric_aggregate
        elif self.aggregate_method == 'total_average':
            if self.total_datapoints_count != 0:
                metric = self.total_metric_aggregate / self.total_datapoints_count
        elif self.aggregate_method == 'total_sum':
            metric = self.total_metric_aggregate
        else:
            raise RuntimeError(self.aggregate_method + ' for metric: ' + self.name + ' cannot be recognized.')

        utcnow = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')

        # Start recording metrics
        with self.mutex:
            if self.write_to == 'csv':
                filename = os.path.join('metrics', 'mms_' + self.name + '.csv')
                if not os.path.exists(os.path.dirname(filename)):
                     os.makedirs(os.path.dirname(filename))
                with open(filename, 'a') as csvfile:
                    csvwriter = csv.writer(csvfile, delimiter=',')
                    csvwriter.writerow([utcnow, metric])
            else:
                logger.info('Metric %s for last %s seconds is %f' % 
                    (self.name, self.interval_sec, metric))

        # Clear interval metrics
        self.interval_metric_aggregate = 0.0
        self.interval_datapoints_count = 0 

