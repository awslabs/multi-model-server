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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        xunit_file = os.path.join(self.artifacts_dir, "xunit.xml")
        tests, failures, skipped, errors = 0, 0, 0, 0
        if os.path.exists(xunit_file):
            xml = JUnitXml.fromfile(xunit_file)
            for i, suite in enumerate(xml):
                for case in suite:
                    name = "scenario_{}: {}".format(i, case.name)
                    result = case.result
                    if isinstance(result, Error):
                        failures += 1
                        result = Failure(result.message, result.type)
                    elif isinstance(result, Failure):
                        errors += 1
                        result = Error(result.message, result.type)
                    elif isinstance(result, Skipped):
                        skipped += 1
                    else:
                        tests += 1

                    tc = TestCase(name)
                    tc.result = result
                    self.ts.add_testcase(tc)
        else:
            tc = TestCase(self.name)
            tc.result = Skipped()
            self.ts.add_testcase(tc)

        self.ts.hostname = self.env_name
        self.ts.timestamp = self.timer.start
        self.ts.time = self.timer.diff()
        self.ts.tests = tests
        self.ts.failures = failures
        self.ts.skipped = skipped
        self.ts.errors = errors
        self.ts.update_statistics()
        self.junit_xml.add_testsuite(self.ts)
