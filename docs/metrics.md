# Metrics on Model Server

## Basic Logging
There are four arguments for MMS that facilitate logging of the model serving and inference activity.

1. **log-file**: optional, log file name. By default it is "mms_app.log". You may also specify a path and a custom file name such as `logs/squeezenet_inference`. This is the root file name that is used in file rotation.

1. **log-rotation-time**: optional, log rotation time. By default it is "1 H", which means one Hour. Valid format is "interval when", where _when_ can be "S", "M", "H", or "D". For a particular weekday use only "W0" - "W6". For midnight use only "midnight". When a file is rotated a timestamp is appended, for example, `squeezenet_inference` would look like `squeezenet_inference.2017-11-27_17-26` after log rotation. Check the [Python docs on logging handlers](https://docs.python.org/2/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler) for detailed information on values.

1. **log-level**: optional, log level. By default it is INFO. Possible values are NOTEST, DEBUG, INFO, ERROR and CRITICAL. Check the [Python docs for logging levels](https://docs.python.org/2/library/logging.html#logging-levels) for more information.

1. **metrics-write-to**: optional, metrics output destination. By default, various metrics are collected and written to the default log file.

  If the `csv` value is passed to this argument, the metrics are recorded every minute in separate CSV files in a metrics folder in the current directory as follows.

      a) **mms_cpu.csv** - CPU load
      b) **mms_errors.csv** - number of errors
      c) **mms_memory.csv** - memory utilization
      d) **mms_preprocess_latency.csv** - any custom pre-processing latency
      e) **mms_disk.csv** - disk utilization
      f) **mms_inference_latency.csv** - any inference latency
      g) **mms_overall_latency.csv** - collective latency
      h) **mms_requests.csv** - number of inference requests

  If the `cloudwatch` value is passed, the above metrics will write to [AWS CloudWatch Service](https://aws.amazon.com/cloudwatch/) every minute with namespace 'mxnet-model-server'. After [configuring AWS crediential](https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html), you will see the metrics are pushed to AWS CloudWatch Service.
