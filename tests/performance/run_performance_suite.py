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
import yaml
import requests
from agents import configuration
import csv
import pandas as pd
import boto3
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


code = 0
metrics_monitoring_server = "agents/metrics_monitoring_server.py"
base_file_path = pathlib.Path(__file__).parent.absolute()
run_artifacts_path = "{}/run_artifacts/".format(base_file_path)
GLOBAL_CONFIG_PATH = "{}/tests/common/global_config.yaml".format(base_file_path)

S3_BUCKET = configuration.get('suite', 's3_bucket')


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


def get_folder_names(dir1):
    return [di for di in os.listdir(dir1) if os.path.isdir(os.path.join(dir1, di))]


def get_latest(names, env_id):
    max_ts = 0
    latest_run = ''
    for run_name in names:
        run_name_list = run_name.split('_')
        if env_id == run_name_list[0]:
            if int(run_name_list[2]) > max_ts:
                max_ts = int(run_name_list[2])
                latest_run = run_name

    return latest_run


def get_latest_dir(dir1, env_id):
    names= get_folder_names(dir1)
    latest_run= get_latest(names, env_id)
    return os.path.join(dir1, latest_run), latest_run


def download_s3_files(env_id, tgt_path, bucket_name=S3_BUCKET):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    result = bucket.meta.client.list_objects(Bucket=bucket.name,
                                             Delimiter='/')

    run_names = []
    for o in result.get('CommonPrefixes'):
        run_names.append(o.get('Prefix')[:-1])

    latest_run = get_latest(run_names, env_id)
    if not latest_run:
        logger.info("No run found for env_id {}".format(env_id))
        return '', ''

    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)

    tgt_path = "{}/{}".format(tgt_path, latest_run)
    run_process("aws s3 cp  s3://{}/{} {} --recursive".format(bucket.name, latest_run, tgt_path))

    return tgt_path, latest_run


def get_sub_dirs(dir, exclude_list=['comp_data']):
    dir = dir.strip()
    if not os.path.exists(dir):
        msg = "The path {} does not exit".format(dir)
        logger.error("The path {} does not exit".format(dir))
        raise Exception(msg)
    return list([x for x in os.listdir(dir) if os.path.isdir(dir + "/" + x) and x not in exclude_list])


def get_compare_metric_list(dir, sub_dir):
    diff_percents = []
    metrics = []
    test_yaml = "{}/{}/{}.yaml".format(dir, sub_dir, sub_dir)
    with open(test_yaml) as test_yaml:
        test_yaml = yaml.safe_load(test_yaml)
        for rep_section in test_yaml.get('reporting', []):
            if rep_section.get('module', None) == 'passfail':
                for criterion in rep_section.get('criteria', []):
                    if isinstance(criterion, dict) and 'monitoring' in criterion.get('class', ''):
                        subject = criterion["subject"]
                        metric = subject.rsplit('/', 1)
                        metric = metric[1] if len(metric) == 2 else metric[0]
                        diff_percent = criterion.get("diff_percent", None)

                        if diff_percent:
                            metrics.append(metric)
                            diff_percents.append(diff_percent)

    return metrics, diff_percents


def compare_artifacts(dir1, dir2, out_dir, run_name1, run_name2):
    logger.info("Comparing artifacts from {} with {}".format(dir1, dir2))
    sub_dirs_1 = get_sub_dirs(dir1)
    sub_dirs_2 = get_sub_dirs(dir2)

    over_all_pass = True

    aggregates = ["mean", "max", "min"]
    header = ["run_name1", "run_name2", "test_suite", "metric", "run1", "run2", "percentage_diff", "result"]
    rows = [header]
    for sub_dir1 in sub_dirs_1:
        if sub_dir1 in sub_dirs_2:
            metrics_file1 = glob.glob("{}/{}/SAlogs_*".format(dir1, sub_dir1))
            metrics_file2 = glob.glob("{}/{}/SAlogs_*".format(dir2, sub_dir1))
            if not (metrics_file1 and metrics_file2):
                metrics_file1 = glob.glob("{}/{}/local_*".format(dir1, sub_dir1))
                metrics_file2 = glob.glob("{}/{}/local_*".format(dir2, sub_dir1))
                if not (metrics_file1 and metrics_file2):
                    logger.info("Metrics monitoring logs are not captured for {} in either of the runs.".format(sub_dir1))
                    rows.append([run_name1, run_name2, sub_dir1, "log_file", sub_dir1, sub_dir1, "metrics are not captured for either of the runs", "pass"])
                    continue
            metrics_from_file1 = pd.read_csv(metrics_file1[0])
            metrics_from_file2 = pd.read_csv(metrics_file2[0])

            metrics, diff_percents = get_compare_metric_list(dir1, sub_dir1)
            for col, diff_percent in zip(metrics, diff_percents):
                for agg_func in aggregates:
                    name = "{}_{}".format(agg_func, str(col))
                    try:
                        val1 = getattr(metrics_from_file1[str(col)], agg_func)()
                    except TypeError:
                        val1 = "NULL"

                    if str(col) in metrics_from_file2:
                        try:
                            val2 = getattr(metrics_from_file2[str(col)], agg_func)()
                        except TypeError:
                            val2 = "NULL"
                    else:
                        val2 = "NA"

                    if val1 == val2 and val1 == "NULL":
                         diff = "NULL"
                         pass_fail = "pass"
                    elif val1 == "NULL":
                        diff = val2
                        pass_fail = "fail"
                    elif val2 == "NULL":
                        diff = val1
                        pass_fail = "fail"
                    elif val2 == "NA":
                        diff = val2
                        pass_fail = "pass"
                    else:
                        try:
                            if val2 != val1:
                                diff = (abs(val2 - val1) / ((val2 + val1)/2)) * 100
                                pass_fail = "pass" if diff < diff_percent else "fail"
                            else: # special case of 0
                                pass_fail = "pass"
                                diff = 0

                        except Exception as e:
                            logger.info("error while calculating the diff {}".format(str(e)))
                            diff = str(e)
                            pass_fail = "fail"

                    if over_all_pass:
                        over_all_pass = pass_fail == "pass"
                    rows.append([run_name1, run_name2, sub_dir1, name, val1, val2, diff, pass_fail])
        else:
            rows.append([run_name1, run_name2, sub_dir1, "log_file", "log file available", "log file not available", "NA", "pass"])

    out_path = "{}/comparison.csv".format(out_dir)
    logger.info("Writing comparison report to log file {}".format(out_path))
    with open(out_path, 'w') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(rows)
    return over_all_pass


def get_test_yamls(dir_path=None, pattern="*.yaml"):
    if not dir_path:
        dir_path = str(base_file_path) + "/tests"

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
@click.option('-a', '--artifacts-dir', help='Directory to store artifacts.', type=click.Path(writable=True))
@click.option('-t', '--test-dir', help='Directory containing tests.', type=click.Path(exists=True), default=None)
@click.option('-p', '--pattern', help='Test case file name pattern. Example --> *.yaml', default="*.yaml")
@click.option('-j', '--jmeter-path', help='JMeter executable path.')
@click.option('--monit/--no-monit', help='Start Monitoring server', default=True)
@click.option('--env-name', help='Unique machine id on which MMS server is running', default=socket.gethostname())
@click.option('--compare-local/--no-compare-local', help='Compare with previous run with files stored'
                                                         ' in artifacts directory', default=True)
def run_test_suite(artifacts_dir, test_dir, pattern, jmeter_path, monit, env_name, compare_local):
    commit_id = subprocess.check_output('git rev-parse --short HEAD'.split()).decode("utf-8")[:-1]
    artifacts_folder_name = "{}_{}_{}".format(env_name, commit_id, int(time.time()))
    if artifacts_dir is None:
        artifacts_dir = "{}/{}".format(run_artifacts_path, artifacts_folder_name)
    else:
        artifacts_dir = os.path.abspath(artifacts_dir)
        artifacts_dir = "{}/{}".format(artifacts_dir, artifacts_folder_name)
    logger.info("Artifacts will be stored in directory {}".format(artifacts_dir))

    with Monitoring(base_file_path, monit):
        junit_xml = JUnitXml()
        pre_command = 'export PYTHONPATH={}/agents:$PYTHONPATH; '.format(str(base_file_path))
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

    if compare_local:
        compare_dir, run_name = get_latest_dir(pathlib.Path(artifacts_dir).parent, env_name)
    else:
        compare_dir, run_name = download_s3_files(env_name, "{}/comp_data".format(artifacts_dir))

    compare_result = True
    if compare_dir:
        compare_result = compare_artifacts(artifacts_dir, compare_dir, artifacts_dir, artifacts_folder_name, run_name)

    if not compare_local:
        run_process("aws s3 cp {} s3://{}/{}  --recursive".format(artifacts_dir, S3_BUCKET, artifacts_folder_name))

    if junit_xml.errors or junit_xml.failures or junit_xml.skipped:
        sys.exit(3)

    if not compare_result:
        sys.exit(4)


if __name__ == "__main__":
    run_test_suite()

