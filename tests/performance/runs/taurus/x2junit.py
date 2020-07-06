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
Convert the Taurus Test suite XML to Junit XML
"""
# pylint: disable=redefined-builtin


import os
import pandas as pd
from .reader import get_compare_metric_list

from junitparser import TestCase, TestSuite, JUnitXml, Skipped, Error, Failure


class X2Junit(object):
    """
       Context Manager class to do convert Taurus Test suite XML report which is in Xunit specifications
       to JUnit XML report.
    """
    def __init__(self, name, artifacts_dir, junit_xml, timer, env_name):
        self.ts = TestSuite(name)
        self.name = name
        self.junit_xml = junit_xml
        self.timer = timer
        self.artifacts_dir = artifacts_dir
        self.env_name = env_name

        self.ts.tests, self.ts.failures, self.ts.skipped, self.ts.errors = 0, 0, 0, 0

    def __enter__(self):
        return self

    def add_compare_tests(self):
        metrics_file = os.path.join(self.artifacts_dir, "metrics.csv")
        metrics = pd.read_csv(metrics_file)
        compare_list = get_compare_metric_list(self.artifacts_dir, "")

        for metric_values in compare_list:
            col = metric_values[0]
            diff_percent = metric_values[2]
            tc = TestCase("{}_diff_run > {}".format(col, diff_percent))
            if diff_percent is None:
                tc.result = Skipped("diff_percent_run value is not mentioned")
                self.ts.skipped += 1
            else:
                col_metric_values = getattr(metrics, col, None)
                if col_metric_values is None:
                    tc.result = Error("Metric is not captured")
                    self.ts.errors += 1
                elif len(col_metric_values) < 2:
                    tc.result = Skipped("Enough values are not captured")
                    self.ts.errors += 1
                else:
                    first_value = col_metric_values.iloc[0]
                    last_value = col_metric_values.iloc[-1]

                    try:
                        if last_value == first_value == 0:
                            diff_actual = 0
                        else:
                            diff_actual = (abs(last_value - first_value) / ((last_value + first_value) / 2)) * 100

                        if float(diff_actual) <= float(diff_percent):
                            self.ts.tests += 1
                        else:
                            tc.result = Failure("The first value and last value of run are {}, {} "
                                                "with percent diff {}".format(first_value, last_value, diff_actual))

                    except Exception as e:
                        tc.result = Error("Error while comparing values {}".format(str(e)))
                        self.ts.errors += 1

            self.ts.add_testcase(tc)

    def __exit__(self, type, value, traceback):
        xunit_file = os.path.join(self.artifacts_dir, "xunit.xml")
        if os.path.exists(xunit_file):
            xml = JUnitXml.fromfile(xunit_file)
            for i, suite in enumerate(xml):
                for case in suite:
                    name = "scenario_{}: {}".format(i, case.name)
                    result = case.result
                    if isinstance(result, Error):
                        self.ts.failures += 1
                        result = Failure(result.message, result.type)
                    elif isinstance(result, Failure):
                        self.ts.errors += 1
                        result = Error(result.message, result.type)
                    elif isinstance(result, Skipped):
                        self.ts.skipped += 1
                    else:
                        self.ts.tests += 1

                    tc = TestCase(name)
                    tc.result = result
                    self.ts.add_testcase(tc)
        else:
            tc = TestCase(self.name)
            tc.result = Skipped()
            self.ts.add_testcase(tc)

        self.add_compare_tests()

        self.ts.hostname = self.env_name
        self.ts.timestamp = self.timer.start
        self.ts.time = self.timer.diff()
        self.ts.update_statistics()
        self.junit_xml.add_testsuite(self.ts)

        return False


if __name__ == "__main__":
    from utils import timer
    j = JUnitXml()
    with timer.Timer("asd") as t:
        with X2Junit("a",
                "/home/mahesh/multi-model-server/tests/performance/run_artifacts/xlarge__197a706__1594039175/api_description",
                j, t, "xlarge") as a:

            print("a")


