# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from collections import namedtuple

import mms.client_sdk_generator
import pytest
from mms.client_sdk_generator import ClientSDKGenerator
from mock import patch, mock_open

sample_openapi_endpoints = dict()
sample_sdk_language = 'python'


@pytest.fixture()
def patches(mocker):
    Patches = namedtuple('Patches',
                         ['path_exists', 'dirname', 'abspath', 'makedirs', 'json_dump', 'subprocess_call', 'open',
                          'logger_info'])
    patches = Patches(
        mocker.patch('os.path.exists'),
        mocker.patch('os.path.dirname'),
        mocker.patch('os.path.abspath'),
        mocker.patch('os.makedirs'),
        mocker.patch('json.dump'),
        mocker.patch('subprocess.call'),
        mocker.patch('mms.client_sdk_generator.open', mock_open()),
        mocker.patch('mms.client_sdk_generator.logger.info')
    )
    patches.path_exists.return_value = True
    patches.dirname.return_value = "testdirname"
    return patches


def test_handles_exception(patches):
    test_exception = Exception("test")
    patches.path_exists.side_effect = test_exception
    with pytest.raises(Exception) as excinfo:
        ClientSDKGenerator.generate(sample_openapi_endpoints, sample_sdk_language)
    patches.json_dump.assert_not_called()
    patches.subprocess_call.assert_not_called()
    assert excinfo.value != test_exception
    assert excinfo.value.args[0] == 'Failed to generate client sdk: test'


def test_makes_build_directory(patches):
    patches.path_exists.return_value = False
    ClientSDKGenerator.generate(sample_openapi_endpoints, sample_sdk_language)
    patches.makedirs.assert_called_once()


def test_runs_successfully(patches):
    ClientSDKGenerator.generate(sample_openapi_endpoints, sample_sdk_language)
    patches.makedirs.assert_not_called()
    patches.dirname.assert_called_once()
    patches.abspath.assert_called_once()
    patches.json_dump.assert_called_once()
    patches.subprocess_call.assert_called_once()
    patches.logger_info.assert_called_once()

    assert patches.dirname.return_value in patches.subprocess_call.call_args_list[0][0][0]
