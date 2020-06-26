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

import os
import tabulate
from utils import run_process
from junitparser import JUnitXml
header = ["suite_name", "test_case", "result", "message"]


def generate_junit_report(junit_xml, out_dir, report_name):
    junit_xml.update_statistics()
    junit_xml_path = os.path.join(out_dir, '{}.xml'.format(report_name))
    junit_html_path = os.path.join(out_dir, '{}.html'.format(report_name))
    junit_xml.write(junit_xml_path)

    # vjunit pip package is used here
    run_process("vjunit -f {} -o {}".format(junit_xml_path, junit_html_path))


def junit2array(junit_xml):
    """convert junit xml junitparser.JUnitXml object to 2d array """
    rows = [header]
    for i, suite in enumerate(junit_xml):
        for case in suite:
            result = case.result
            tag, msg = (result._tag, result.message) if result else ("pass", "")
            rows.append([suite.name, case.name, tag, msg])

    return rows


def junit2tabulate(junit_xml):
    """convert junit xml junitparser.JUnitXml object or a Junit xml to tabulate string """
    if not isinstance(junit_xml, JUnitXml):
        if os.path.exists(junit_xml):
            junit_xml = JUnitXml.fromfile(junit_xml)
        else:
            return tabulate.tabulate([[header]], headers='firstrow')
    data = junit2array(junit_xml)
    return tabulate.tabulate(data, headers='firstrow', tablefmt="grid")
