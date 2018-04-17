# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Metric class for model server
"""

import csv
import datetime
import os
import socket
import threading
import warnings

from mms.log import get_logger

try:
    import boto3 as boto
except ImportError:
    boto = None

logger = get_logger()

MetricUnit = {
    'ms': "Milliseconds",
    'percent': 'Percent',
    'count': 'Count',
    'MB': 'Megabytes',
    'GB': 'Gigabytes'
}


class Metric(object):
    """
    Class for generating metrics and publishing it to cloudwatch/log/csvfiles
    """
    def __init__(self, name, mutex,
                 model_name,
                 unit,
                 is_model_metric=True,
                 interval_sec=60,
                 update_func=None,
                 aggregate_method='interval_average',
                 write_to='log'):
        """Constructor for Metric class

           Metric class will spawn a thread and report collected metrics to local disk
           or AWS cloudwatch between given intervals.

        Parameters
        ----------
        name : str
            Name of the metric
        mutex: object
            Lock between metric threads when they are writing local disk
        model_name: str
            Model name
        unit: str
            CloudWatch compatible unit
        is_model_metric: boolean
            Whether it is a metric for model eg. inference latency, requests count
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
        self.model_name = model_name

        # Metrics within interval
        self.interval_datapoints_count = 0
        self.interval_metric_aggregate = 0.0
        self.min_value = None
        self.max_value = None

        # Metrics for the whole session
        self.total_datapoints_count = 0
        self.total_metric_aggregate = 0.0

        self.aggregate_method = aggregate_method
        self.mutex = mutex
        self.write_to = write_to
        self.unit = unit
        self.is_model_metric = is_model_metric

        # Setup cloudwatch handle
        if self.write_to == 'cloudwatch':
            try:
                self.client = boto.client('cloudwatch')
            except Exception as e: # pylint: disable=broad-except
                warnings.warn('Failed to connect to AWS CloudWatch, \
                    metrics will be written to log.\n \
                    Failure reason ' + str(e))
                self.write_to = 'log'

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
        if self.min_value is None:
            self.min_value = metric
        else:
            self.min_value = min(self.min_value, metric)

        if self.max_value is None:
            self.max_value = metric
        else:
            self.max_value = min(self.max_value, metric)

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
            # metric starts being reported
            if self.interval_datapoints_count != 0:
                if self.write_to == 'csv':
                    filename = os.path.join('metrics', 'mms_' + self.name + '.csv')
                    if not os.path.exists(os.path.dirname(filename)):
                        os.makedirs(os.path.dirname(filename))
                    with open(filename, 'a') as csvfile:
                        csvwriter = csv.writer(csvfile, delimiter=',')
                        csvwriter.writerow([utcnow, metric])
                elif self.write_to == 'cloudwatch':
                    logger.info('Metric %s for last %s seconds is %f, writing to AWS CloudWatch...',
                                self.name, self.interval_sec, metric)
                    try:
                        update_entry = {'Value': metric}
                        if self.unit == MetricUnit['MB']:
                            update_entry = {
                                'StatisticValues': {
                                    'SampleCount': self.interval_datapoints_count,
                                    'Sum': self.interval_metric_aggregate,
                                    'Minimum': self.min_value,
                                    'Maximum': self.max_value
                                }
                            }

                        metric_data = {
                            'MetricName': self.name,
                            'Timestamp': utcnow,
                            'Unit': self.unit,
                            'Dimensions': [
                                {
                                    'Name': 'host',
                                    'Value': socket.gethostname()
                                }
                            ]
                        }
                        metric_data.update(update_entry)
                        if self.is_model_metric:
                            metric_data['Dimensions'].append({
                                'Name': 'model_name',
                                'Value': self.model_name
                            })
                        self.client.put_metric_data(
                            Namespace='MXNetModelServer',
                            MetricData=[
                                metric_data
                            ]
                        )
                    except Exception as e: # pylint: disable=broad-except
                        raise Exception("Failed to write metrics to cloudwatch " + str(e))
                else:
                    logger.info("Metric %s for last %s seconds is %f", self.name, self.interval_sec, metric)

        # Clear interval metrics
        self.interval_metric_aggregate = 0.0
        self.interval_datapoints_count = 0
        self.min_value = None
        self.max_value = None
