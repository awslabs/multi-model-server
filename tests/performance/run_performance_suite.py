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
Run Tarus test cases and generate the Junit XML report
"""
# pylint: disable=redefined-builtin, no-value-for-parameter

import logging
import os
import pathlib
import subprocess
import sys
import time
import webbrowser

import click
from junitparser import JUnitXml
from tqdm import tqdm

from utils import run_process, Timer, get_sub_dirs
from runs.storage import LocalStorage, S3Storage
from runs.monitoring_driver import Monitoring
from runs.junit import genrate_junit_report
from runs.taurus import get_taurus_options, x2junit, update_metric_log_header
from runs.compare import compare

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

base_file_path = pathlib.Path(__file__).parent.absolute()
run_artifacts_path = "{}/run_artifacts/".format(base_file_path)
GLOBAL_CONFIG_PATH = "{}/tests/global_config.yaml".format(base_file_path)


def open_report(file_path):
    if os.path.exists(file_path):
        return webbrowser.open_new_tab('file://' + os.path.realpath(file_path))
    return False


@click.command()
@click.option('-a', '--artifacts-dir', help='Directory to store artifacts.', type=click.Path(writable=True))
@click.option('-t', '--test-dir', help='Directory containing tests.', type=click.Path(exists=True), default=None)
@click.option('-p', '--pattern', help='Test case folder name glob pattern', default="*")
@click.option('-x', '--exclude-pattern', help='Test case folder name glob pattern to exclude', default=None)
@click.option('-j', '--jmeter-path', help='JMeter executable path.')
@click.option('--monit/--no-monit', help='Start Monitoring server', default=True)
@click.option('-e', '--env-name', help='environment name which defines threshold values used by the test cases. '
                                       'This is the name of a file found in the tests environment folder without '
                                       'the extension', required=True)
@click.option('--compare-local/--no-compare-local', help='Compare with previous run with files stored'
                                                         ' in artifacts directory', default=True)
def run_test_suite(artifacts_dir, test_dir, pattern, exclude_pattern,
                   jmeter_path, monit, env_name, compare_local):
    """Collect test suites, run them and generate reports"""

    if test_dir is None:
        test_dir = "{}/tests".format(base_file_path)

    if '__' in env_name:
        raise Exception("Environment name should not have double underscores in it.")

    commit_id = subprocess.check_output('git rev-parse --short HEAD'.split()).decode("utf-8")[:-1]
    artifacts_folder_name = "{}__{}__{}".format(env_name, commit_id, int(time.time()))
    if artifacts_dir is None:
        artifacts_dir = "{}/{}".format(run_artifacts_path, artifacts_folder_name)
    else:
        artifacts_dir = os.path.abspath(artifacts_dir)
        artifacts_dir = "{}/{}".format(artifacts_dir, artifacts_folder_name)
    logger.info("Artifacts will be stored in directory %s", artifacts_dir)

    with Monitoring(base_file_path, monit):
        junit_xml = JUnitXml()
        pre_command = 'export PYTHONPATH={}/agents:$PYTHONPATH; '.format(str(base_file_path))
        test_dirs = get_sub_dirs(test_dir, exclude_list=[], include_pattern=pattern,
                                 exclude_pattern=exclude_pattern)
        logger.info("Collected tests %s", test_dirs)
        for suite_name in tqdm(test_dirs, desc="Test Suites"):
            with Timer("Test suite {} execution time".format(suite_name)) as t:
                suite_artifacts_dir = "{}/{}".format(artifacts_dir, suite_name)
                options_str = get_taurus_options(suite_artifacts_dir, jmeter_path)
                env_yaml_path = "{}/{}/environments/{}.yaml".format(test_dir, suite_name, env_name)
                env_yaml_path = "" if not os.path.exists(env_yaml_path) else env_yaml_path
                test_file = "{0}/{1}/{1}.yaml".format(test_dir, suite_name)
                with x2junit.X2Junit(suite_name, suite_artifacts_dir, junit_xml, t, env_name) as s:
                    s.code, s.err = run_process("{} bzt {} {} {} {}".format(pre_command, options_str,
                                                                            test_file, env_yaml_path,
                                                                            GLOBAL_CONFIG_PATH))

                    update_metric_log_header(suite_artifacts_dir, test_file)

        genrate_junit_report(junit_xml, artifacts_dir, 'performance_results')

    storage_class = LocalStorage if compare_local else S3Storage
    compare_result = compare(storage_class(artifacts_dir, artifacts_folder_name, env_name))

    compare_status, exit_code = ('failed', 4) if not compare_result else ('passed', 0)
    suite_status, exit_code = ('failed', 3) if junit_xml.errors or junit_xml.failures \
                                               or junit_xml.skipped else ('passed', 0)

    logger.info("\n\nResult Summary:")
    comparison_result_html = "{}/comparison_results.html".format(artifacts_dir)
    suite_result_html = "{}/performance_results.html".format(artifacts_dir)
    if os.path.exists(comparison_result_html):
        logger.info("Comparison with monitoring metrics of previous run has %s.", compare_status)
        logger.info("Comparison test suite report - %s", comparison_result_html)

    logger.info("Performance Regression Test suite run has %s.", suite_status)
    logger.info("Performance Regression Test suite report - %s", suite_result_html)

    open_report(suite_result_html)
    sys.exit(exit_code)


if __name__ == "__main__":
    run_test_suite()
