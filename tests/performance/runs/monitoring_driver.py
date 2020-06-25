#!/usr/bin/env python

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
Start and stop monitoring server
"""
# pylint: disable=redefined-builtin

import sys
import time
from utils import run_process


class Monitoring(object):
    """
    Context Manager class to start and stop the Monitoring Agent metrics_monitoring_server.py
    """
    def __init__(self, path, use=True, check_mms_server_status=False):
        self.path = "{}/{}".format(path, "agents/metrics_monitoring_server.py")
        self.use = use
        self.check_mms_server_status = check_mms_server_status

    def __enter__(self):
        if self.use:
            start_monitoring_server = "{} {} --start".format(sys.executable, self.path)
            run_process(start_monitoring_server, wait=False)
            time.sleep(2)
        return self

    def __exit__(self, type, value, traceback):
        if self.use:
            stop_monitoring_server = "{} {} --stop".format(sys.executable, self.path)
            run_process(stop_monitoring_server)
