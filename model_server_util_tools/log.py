# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Logging utilities.
"""
import sys


def log_msg(*args):
    msg = " ".join(a for a in args)
    sys.stdout.write(msg)
    sys.stdout.write('\n')
    sys.stdout.flush()


def log_error(*args):
    msg = " ".join(a for a in args)
    sys.stderr.write(msg)
    sys.stderr.write('\n')
    sys.stderr.flush()
