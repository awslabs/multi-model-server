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

import glob
import logging
import os
import pathlib
import socket
import subprocess
import sys
import time
import click
from subprocess import PIPE, STDOUT

import requests
import yaml
from junitparser import TestCase, TestSuite, JUnitXml, Skipped, Error, Failure
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

PATH = pathlib.Path(__file__).parent.absolute()
GLOBAL_CONFIG_PATH = "{}/tests/common/global_config.yaml".format(PATH)

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

class Monitoring(object):
    def __init__(self, path, use = True):
        self.path = "{}/{}".format(path, "agents/metrics_monitoring_server.py")
        self.use = use

    def __enter__(self):
        with open(GLOBAL_CONFIG_PATH) as conf_file:
            global_config = yaml.safe_load(conf_file)
        server_props = global_config["modules"]["jmeter"]["properties"]
        server_ping_url = "{}://{}:{}/ping".format(server_props["protocol"], server_props["hostname"],
                                                   server_props["port"])
        try:
            requests.get(server_ping_url)
        except requests.exceptions.ConnectionError:
            raise Exception("Server is not running. Pinged url {}. Exiting...".format(server_ping_url))

        if self.use:
            start_monitoring_server = "python {} --start".format(self.path)
            run_process(start_monitoring_server, wait=False)
            time.sleep(2)
        return self

    def __exit__(self, type, value, traceback):
        if self.use:
            stop_monitoring_server = "python {} --stop".format(self.path)
            run_process(stop_monitoring_server)

class Suite(object):
    def __init__(self, name, artifacts_dir, junit_xml, timer):
        self.ts = TestSuite(name)
        self.name = name
        self.junit_xml = junit_xml
        self.timer = timer
        self.artifacts_dir = artifacts_dir

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        xunit_file = "{}/xunit.xml".format(self.artifacts_dir)
        tests, failures, skipped, errors = 0, 0, 0, 0
        err_txt = ""
        if os.path.exists(xunit_file):
            xml = JUnitXml.fromfile(xunit_file)
            for i, suite in enumerate(xml):  # tqqdm
                for case in suite:
                    name = "scenario_{}: {}".format(i, case.name)
                    result = case.result
                    if isinstance(result, Error):
                        errors += 1
                        err_txt = self.err
                    elif isinstance(result, Failure):
                        failures += 1
                        err_txt = self.err
                    elif isinstance(result, Skipped):
                        skipped += 1
                    else:
                        tests += 1

                    tc = TestCase(name)
                    tc.result = result
                    self.ts.add_testcase(tc)
        else:
            tc = TestCase(self.name)
            if code:
                tc.result = Error("Suite run failed", "Error")
            else:
                tc.result = Skipped()
            self.ts.add_testcase(tc)


        self.ts.hostname = socket.gethostname()
        self.ts.timestamp = self.timer.start
        self.ts.time = self.timer.diff()
        self.ts.tests = tests
        self.ts.failures = failures
        self.ts.skipped = skipped
        self.ts.errors = errors
        self.ts.update_statistics()
        self.junit_xml.add_testsuite(self.ts)


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

        return p.returncode, '\n'.join(lines)
    else:
        p = subprocess.Popen(cmd, shell=True)
        return p.returncode, ''


def get_test_yamls(dir_path=None, pattern="*.yaml"):
    if not dir_path:
        dir_path = str(PATH) + "/tests"

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

@click.command()
@click.option('-a', '--artifacts-dir', help='Directory to store artifacts.', type=click.Path(writable=True), required=True)
@click.option('-t', '--test-dir', help='Directory containing tests.', type=click.Path(exists=True), default=None)
@click.option('-p', '--pattern', help='Test case file name pattern. Example --> *.yaml', default="*.yaml")
@click.option('-j', '--jmeter-path', help='JMeter executable path.')
@click.option('--monit/--no-monit', help='Start Monitoring server', default=True)
def run_test_suite(artifacts_dir, test_dir, pattern, jmeter_path, monit):
    if os.path.exists(artifacts_dir):
        msg = "Artifacts dir '{}' already exists... Please provide a new directory.".format(artifacts_dir)
        raise Exception(msg)

    with Monitoring(PATH, monit):
        junit_xml = JUnitXml()
        pre_command = 'export PYTHONPATH={}/agents:$PYTHONPATH; '.format(str(PATH))
        test_yamls = get_test_yamls(test_dir, pattern)
        for test_file in tqdm(test_yamls, desc="Test Suites"):
            suite_name = os.path.basename(test_file).rsplit('.', 1)[0]
            with Timer("Test suite {} execution time".format(suite_name)) as t:
                suite_artifacts_dir = "{}/{}".format(artifacts_dir, suite_name)
                options_str = get_options(suite_artifacts_dir, jmeter_path)
                with Suite(suite_name, suite_artifacts_dir, junit_xml, t) as s:
                    s.code, s.err = run_process("{} bzt {} {} {}".format(pre_command, options_str,
                                                                     test_file, GLOBAL_CONFIG_PATH))

        junit_xml.update_statistics()
        junit_xml_path = '{}/junit.xml'.format(artifacts_dir)
        junit_html_path = '{}/junit.html'.format(artifacts_dir)
        junit_xml.write(junit_xml_path)
        run_process("vjunit -f {} -o {}".format(junit_xml_path, junit_html_path))

    if junit_xml.errors or junit_xml.failures or junit_xml.skipped:
        sys.exit(3)

if __name__ == "__main__":
    run_test_suite()