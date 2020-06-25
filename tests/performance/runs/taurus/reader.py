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
Run shell command utilities
"""
# pylint: disable=redefined-builtin

import yaml


def get_mon_metrics_list(test_yaml_path):
    """Utility method to get list of server-agent metrics which are being monitored from a test yaml file"""
    metrics = []
    with open(test_yaml_path) as test_yaml:
        test_yaml = yaml.safe_load(test_yaml)
        for rep_section in test_yaml.get('services', []):
            if rep_section.get('module', None) == 'monitoring' and "server-agent" in rep_section:
                for mon_section in rep_section.get('server-agent', []):
                    if isinstance(mon_section, dict):
                        metrics.extend(mon_section.get('metrics', []))

    return metrics


def get_compare_metric_list(dir, sub_dir):
    """Utility method to get list of compare monitoring metrics identified by diff_percent property"""
    diff_percents = []
    metrics = []
    test_yaml = "{}/{}/{}".format(dir, sub_dir, "effective.yml")
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
