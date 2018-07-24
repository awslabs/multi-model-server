# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest
from mms.model_service.model_service import load_service
from types import ModuleType
from mock import patch
from collections import namedtuple

path = 'mms/tests/unit_tests/test_utils/'


@pytest.fixture()
def patches(mocker):
    named_tuples = namedtuple('patches', ['path_splitext'])
    patches = named_tuples(mocker.patch('os.path.splitext'))
    patches.path_splitext.return_value = ['dummy_model_service.py']

    return patches


def test_load_service_with_nil_name(patches):
    load_service(path=path)
    patches.path_splitext.assert_called()


def test_load_service_with_some_name(patches):
    name = 'dummy_model_service.py'
    module = load_service(path, name)

    patches.path_splitext.assert_not_called()
    assert type(module) == ModuleType
    assert module.__name__ == name


def test_load_service_with_exception():
    invalid_path = 'mms/tests/unit_tests/test_utils/dummy_model_service'
    with pytest.raises(Exception):
        load_service(invalid_path)



