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
from termcolor import colored

from junitparser import JUnitXml
from runs.compare import CompareReportGenerator
from runs.junit import JunitConverter, junit2tabulate

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
        self.local_run = local_run
        self.check_mms_server_status = check_mms_server_status
        self.reporter = JUnitXml()
        self.compare_reporter_generator = CompareReportGenerator(self.artifacts_dir, self.env, self.local_run)

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
    def report_summary(reporter, suite_name):
        if reporter and os.path.exists(reporter.junit_html_path):
            status = reporter.junit_xml.errors or reporter.junit_xml.failures or reporter.junit_xml.skipped
            status, code, color = ("failed", 3, "red") if status else ("passed", 0, "green")

            msg = "{} run has {}.".format(suite_name, status)
            logger.info(colored(msg, color, attrs=['reverse', 'blink']))
            logger.info("%s report - %s", suite_name, reporter.junit_html_path)
            logger.info("%s summary:", suite_name)
            print(junit2tabulate(reporter.junit_xml))
            ExecutionEnv.open_report(reporter.junit_html_path)
            return code

        else:
            msg = "{} run report is not generated.".format(suite_name)
            logger.info(colored(msg, "yellow", attrs=['reverse', 'blink']))
            return 0

    def __exit__(self, type, value, traceback):
        if self.use:
            stop_monitoring_server = "{} {} --stop".format(sys.executable, self.monitoring_agent)
            run_process(stop_monitoring_server)

        junit_reporter = JunitConverter(self.reporter, self.artifacts_dir, 'performance_results')
        junit_reporter.generate_junit_report()
        junit_compare = self.compare_reporter_generator.gen()
        junit_compare_reporter = None
        if junit_compare:
            junit_compare_reporter = JunitConverter(junit_compare, self.artifacts_dir, 'comparison_results')
            junit_compare_reporter.generate_junit_report()

        compare_exit_code = ExecutionEnv.report_summary(junit_compare_reporter, "Comparison Test suite")
        exit_code = ExecutionEnv.report_summary(junit_reporter, "Performance Regression Test suite")

        sys.exit(0 if 0 == exit_code == compare_exit_code else 3)
