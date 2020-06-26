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


def genrate_junit_report(junit_xml, out_dir, report_name):
    """This generates xml and hml report"""
    junit_xml.update_statistics()
    junit_xml_path = '{}/{}.xml'.format(out_dir, report_name)
    junit_html_path = '{}/{}.html'.format(out_dir, report_name)
    junit_xml.write(junit_xml_path)

    #vjunit pip package is used here
    run_process("vjunit -f {} -o {}".format(junit_xml_path, junit_html_path))


def junit2array(junit_xml):
    rows = [header]
    for i, suite in enumerate(junit_xml):
        for case in suite:
            name = "scenario_{}: {}".format(i, case.name)
            result = case.result
            rows.append(suite, name, result._tag, result.message)

    return rows


def junit2tabulate(junit_xml):
    if not isinstance(junit_xml, JUnitXml):
        if os.path.exists(junit_xml):
            junit_xml = JUnitXml.fromfile(junit_xml)
        else:
            return tabulate.tabulate([[header]], headers='firstrow')
    data = junit2array(junit_xml)
    return tabulate.tabulate(data, headers='firstrow')

