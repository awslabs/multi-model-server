import sys
import types
import psutil
import json

from mms.metric import Metric, MetricEncoder
from collections import OrderedDict
overall_metrics = OrderedDict()
metrics = OrderedDict()
overall_metrics['SYSTEM'] = metrics

def cpu_utilization():
    data = psutil.cpu_percent()
    metrics['CPUUtilization'] = Metric(data, 'percent')


def memory_used():
    data = psutil.virtual_memory().used / (1024 * 1024)  # in MB
    metrics['MemoryUsed'] = Metric(data, 'MB')

def memory_available():
    data = psutil.virtual_memory().available / (1024 * 1024)  # in MB
    metrics['MemoryAvailable'] = Metric(data, 'MB')


def memory_utilization():
    data = psutil.virtual_memory().percent
    metrics['MemoryUtilization'] = Metric(data, 'percent')

def disk_used():
    data = psutil.disk_usage('/').used / (1024 * 1024 * 1024)  # in GB
    metrics['DiskUsage'] = Metric(data, 'GB')

def disk_utilization():
    data = psutil.disk_usage('/').percent
    metrics['DiskUtilization'] = Metric(data, 'percent')


def disk_available():
    data = psutil.disk_usage('/').free / (1024 * 1024 * 1024)  # in GB
    metrics['DiskAvailable'] = Metric(data, 'GB')


def collect_all(mod):
    all = dir(mod)
    for i in all:
        value = getattr(mod, i)
        if isinstance(value, types.FunctionType) and value.__name__ != 'collect_all':
            value()
    print(json.dumps(overall_metrics, indent=4, separators=(',', ':'), cls=MetricEncoder))
if __name__ == '__main__':
    collect_all(sys.modules[__name__])
