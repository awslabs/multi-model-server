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
Run shell command utilities
"""
# pylint: disable=redefined-builtin, logging-format-interpolation

import logging
import sys
import os
import subprocess

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


def run_process(cmd, wait=True):
    """Utility method to run the shell commands"""
    logger.info("running command : %s", cmd)

    if wait:
        os.environ["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=True)
        lines = []
        while True:
            line = process.stdout.readline().decode('utf-8').rstrip()
            if not line:
                break
            lines.append(line)
            logger.info(line)

        return process.returncode, '\n'.join(lines)
    else:
        process = subprocess.Popen(cmd, shell=True)
        return process.returncode, ''
