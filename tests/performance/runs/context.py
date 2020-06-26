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

import logging
import os
import sys
import time
import webbrowser

from junitparser import JUnitXml
from runs.compare import compare
from runs.junit import generate_junit_report, junit2tabulate
from runs.storage import LocalStorage, S3Storage

from utils import run_process

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)


class ExecutionEnv(object):
    """
    Context Manager class to run the performance regression suites
    """

    def __init__(self, agent, artifacts_dir, env, local_run, use=True, check_mms_server_status=False):
        self.monitoring_agent = agent
        self.artifacts_dir = artifacts_dir
        self.use = use
        self.env = env
        self.storage = LocalStorage if local_run else S3Storage
        self.check_mms_server_status = check_mms_server_status
        self.reporter = JUnitXml()

    def __enter__(self):
        if self.use:
            start_monitoring_server = "{} {} --start".format(sys.executable, self.monitoring_agent)
            run_process(start_monitoring_server, wait=False)
            time.sleep(2)
        return self

    @staticmethod
    def open_report(file_path):
        if os.path.exists(file_path):
            return webbrowser.open_new_tab('file://' + os.path.realpath(file_path))
        return False

    @staticmethod
    def report_summary(junit_xml, compare_result, artifacts_dir):
        compare_status, exit_code = ('failed', 4) if not compare_result else ('passed', 0)
        suite_status, exit_code = ('failed', 3) if junit_xml.errors or junit_xml.failures \
                                                   or junit_xml.skipped else ('passed', 0)

        logger.info("\n\nResult Summary:")
        comparison_result_html = "{}/comparison_results.html".format(artifacts_dir)
        comparison_result_xml = "{}/comparison_results.xml".format(artifacts_dir)
        suite_result_html = "{}/performance_results.html".format(artifacts_dir)
        if os.path.exists(comparison_result_html):
            logger.info("Comparison with monitoring metrics of previous run has %s.", compare_status)
            logger.info("Comparison test suite report - %s", comparison_result_html)
            logger.info("Comparison test suite summary:")
            print(junit2tabulate(comparison_result_xml))

        logger.info("Performance Regression Test suite run has %s.", suite_status)
        logger.info("Performance Regression Test suite report - %s", suite_result_html)
        logger.info("Comparison test suite summary:")
        print(junit2tabulate(junit_xml))
        ExecutionEnv.open_report(suite_result_html)

        return exit_code

    def __exit__(self, type, value, traceback):
        if self.use:
            stop_monitoring_server = "{} {} --stop".format(sys.executable, self.monitoring_agent)
            run_process(stop_monitoring_server)

        generate_junit_report(self.reporter, self.artifacts_dir, 'performance_results')
        compare_result = compare(self.storage(self.artifacts_dir, self.env))

        exit_code = ExecutionEnv.report_summary(self.reporter, compare_result, self.artifacts_dir)
        sys.exit(exit_code)
