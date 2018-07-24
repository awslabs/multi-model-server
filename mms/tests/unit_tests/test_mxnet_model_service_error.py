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
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes


def test_get_code():
    error = MMSError(ModelServerErrorCodes.INVALID_LOAD_MESSAGE, "invalid load")
    assert error.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE


def test_get_message():
    error = MMSError(ModelServerErrorCodes.INVALID_LOAD_MESSAGE, "invalid load")
    assert error.get_message() == "invalid load"


def test_set_code():
    error = MMSError(ModelServerErrorCodes.INVALID_LOAD_MESSAGE, "invalid load")
    error.set_code(ModelServerErrorCodes.INVALID_PREDICT_MESSAGE)

    assert error.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE


def test_set_message():
    error = MMSError(ModelServerErrorCodes.INVALID_LOAD_MESSAGE, "invalid load")
    error.set_message("invalid_load_message")

    assert error.get_message() == "invalid_load_message"