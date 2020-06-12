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
# pylint: disable=redefined-builtin

import os
import sys
import time
import logging
import socket
import argparse
import glob
import pathlib
import subprocess
from subprocess import PIPE, STDOUT
from tqdm import tqdm
from junitparser import TestCase, TestSuite, JUnitXml, Skipped, Error, Failure

logger = logging.getLogger(__name__)
code = 0


class Timer(object):
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = int(time.time())
        return self

    def __exit__(self, type, value, traceback):
        logger.info(f"{self.description}: {self.diff()}s")

    def diff(self):
        return int(time.time()) - self.start


def run_process(cmd, wait=True):
    print("running command : {}".format(cmd))

    if wait:
        os.environ["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT,
                              shell=True)
        lines =[]
        while True:
            line = p.stdout.readline().decode('utf-8').rstrip()
            if not line: break
            lines.append(line)
            print(line)

        return p.returncode, ''.join(lines)


    else:
        p = subprocess.Popen(cmd, shell=True)
        return 1, ''


def get_test_yamls(dir_path=None, pattern="*.yaml"):
    if not dir_path:
        path = pathlib.Path(__file__).parent.absolute()
        dir_path = str(path) + "/tests"

    path_pattern = "{}/{}".format(dir_path, pattern)
    return glob.glob(path_pattern)


def get_options(artifacts_dir, jmeter_path=None):
    options=[]
    if jmeter_path:
        options.append('-o modules.jmeter.path={}'.format(jmeter_path))
    options.append('-o settings.artifacts-dir={}'.format(artifacts_dir))
    options.append('-o modules.console.disable=true')
    options.append('-o settings.env.BASEDIR={}'.format(artifacts_dir))
    options_str = ' '.join(options)

    return options_str


def run_test_suite(artifacts_dir, test_dir, pattern, jmeter_path):
    if os.path.exists(artifacts_dir):
        artifacts_dir = "{}_{}".format(artifacts_dir, str(int(time.time())))
    path = pathlib.Path(__file__).parent.absolute()
    start_monitoring_server = "python3 {}/metrics_monitoring_server.py --start".format(path)
    run_process(start_monitoring_server, wait=False)

    junit_xml = JUnitXml()
    pre_command = 'export PYTHONPATH={}:$PYTHONPATH; '.format(str(path))

    test_yamls = get_test_yamls(test_dir, pattern)
    for test_file in tqdm(test_yamls, desc="Test Suites"):
        suite_name = os.path.basename(test_file).rsplit('.', 1)[0]
        with Timer("Test suite {} execution time".format(suite_name)) as t:
            suit_artifacts_dir = "{}/{}".format(artifacts_dir, suite_name)
            options_str = get_options(suit_artifacts_dir, jmeter_path)
            code, err = run_process("{} bzt {} {} ".format(pre_command, options_str, test_file))
            suite_time = t.diff()
            suite_start = t.start

        # Assumes default file name
        xunit_file = "{}/xunit.xml".format(suit_artifacts_dir)
        tests, failures, skipped, errors = 0, 0, 0, 0
        err_txt = ""
        ts = TestSuite(suite_name)
        if os.path.exists(xunit_file):
            xml = JUnitXml.fromfile(xunit_file)
            for i, suite in enumerate(xml): #tqqdm
                for case in suite:
                    name = "scenario_{}: {}".format(i, case.name)
                    result = case.result
                    if isinstance(result, Error):
                        errors += 1
                        err_txt = err
                    elif isinstance(result, Failure):
                        failures += 1
                        err_txt = err
                    elif isinstance(result, Skipped):
                        skipped += 1
                    else:
                        tests +=1

                    tc = TestCase(name)
                    tc.result = result
                    tc.system_err = err_txt[:-4]
                    ts.add_testcase(tc)
        else:
            tc = TestCase(suite_name)
            if code:
                tc.result = Error("Suite run failed", "Error")
                tc.system_err = err[:-4]
            else:
                tc.result = Skipped()
                tc.system_out = err[:-4]
            ts.add_testcase(tc)

        ts.hostname = socket.gethostname()
        ts.timestamp = suite_start
        ts.time = suite_time
        ts.tests = tests
        ts.failures = failures
        ts.skipped = skipped
        ts.errors = errors
        ts.update_statistics()
        junit_xml.add_testsuite(ts)

    junit_xml.update_statistics()
    junit_xml_path = '{}/junit.xml'.format(artifacts_dir)
    junit_html_path = '{}/junit.html'.format(artifacts_dir)
    junit_xml.write(junit_xml_path)
    run_process("vjunit -f {} -o {}".format(junit_xml_path, junit_html_path))

    stop_monitoring_server = "python3 {}/metrics_monitoring_server.py --stop".format(path)
    run_process(stop_monitoring_server)

    if junit_xml.errors or junit_xml.failures or junit_xml.skipped:
        sys.exit(3)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser(prog='run_perfomance_suite.py', description='Performance Test Suite Runner')
    parser.add_argument('-a', '--artifacts-dir', nargs=1, type=str, dest='artifacts', required=True,
                           help='A artifacts directory')

    parser.add_argument('-d', '--test-dir', nargs=1, type=str, dest='test_dir', default=[None],
                           help='A test dir')

    parser.add_argument('-p', '--pattern', nargs=1, type=str, dest='pattern', default=["*.yaml"],
                           help='Test case file name pattern. example *.yaml')

    parser.add_argument('-j', '--jmeter-path', nargs=1, type=str, dest='jmeter_path', default=[None],
                        help='JMeter executable bin path')

    parser.add_argument('-t', '--taurus-console', nargs=1, type=bool, dest='taurus_console', default=[None],
                        help='JMeter executable bin path')

    args = parser.parse_args()
    run_test_suite(args.artifacts[0], args.test_dir[0], args.pattern[0], args.jmeter_path[0])
