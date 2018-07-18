# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators
from mms.utils.model_server_error_codes import ModelServerErrorCodes

def test_validate_load_message_missing_model_path():
    invalid_msg = '{ \"modelName\" : \"some-model-name\" }'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_load_message(invalid_msg)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE, 'error codes don\'t match'
    assert error.value.get_message() == 'Load command missing \"model-path\" key'

    valid_msg = '{ \"modelName\" : \"some-model-name\", \"modelPath\" : \"some-model-path\" }'
    ModelWorkerMessageValidators.validate_load_message(valid_msg)

def test_validate_load_message_missing_model_name():
    invalid_msg = '{ \"modelPath\" : \"some-model-path\" }'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_load_message(invalid_msg)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE, 'error codes don\'t match'
    assert error.value.get_message() == 'Load command missing \"model-name\" key'

    valid_msg = '{ \"modelName\" : \"some-model-name\", \"modelPath\" : \"some-model-path\" }'
    ModelWorkerMessageValidators.validate_load_message(valid_msg)

def test_validate_predict_data():
    invalid_msg = '{\"request-id\" : \"some-key\"}'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_data(invalid_msg)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == 'Predict command input missing \"request-id\" field'

    valid_msg = '{ \"requestId\" : \"111-222-3333\"}'
    ModelWorkerMessageValidators.validate_predict_data(valid_msg)

def test_validate_predict_inputs_missing_value():
    invalid_input = '{ \"input1\": { \"name\" : \"some-name\" } }'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_inputs(invalid_input)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == 'Predict command input data missing \"value\" field'


    valid_input = '{ \"input1\": {\"value\" : \"some-name\", \"name\" : \"some-name\"}}'
    ModelWorkerMessageValidators.validate_predict_inputs(valid_input)

def test_validate_predict_inputs_missing_name():
    invalid_input = '{ \"input1\": { \"value\" : \"some-value\" } }'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_inputs(invalid_input)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == 'Predict command input data missing \"name\" field'

    valid_input = '{ \"input1\": {\"value\" : \"some-name\", \"name\" : \"some-name\"}}'
    ModelWorkerMessageValidators.validate_predict_inputs(valid_input)


def test_validate_predict_msg_missing_model_name():
    invalid_input = '{ \"command\" : \"some-predict-command\"}'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_msg(invalid_input)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE
    assert error.value.get_message() == 'Predict command input missing \"modelName\" field.'

    valid_input = '{ \"command\" : \"some-predict-command\", \"modelName\" : \"name\", \"requestBatch\":\"batch\"}'
    ModelWorkerMessageValidators.validate_predict_msg(valid_input)


def test_validate_predict_msg_missing_request_batch():
    invalid_input = '{ \"modelName\" : \"some-predict-command\"}'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_msg(invalid_input)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE
    assert error.value.get_message() == 'Predict command input missing \"requestBatch\" field.'

    valid_input = '{ \"command\" : \"some-predict-command\", \"modelName\" : \"name\", \"requestBatch\":\"batch\"}'
    ModelWorkerMessageValidators.validate_predict_msg(valid_input)


def test_validate_unload_msg():
    invalid_msg = '{ \"command\" : \"unload\" }'
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_unload_msg(invalid_msg)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_UNLOAD_MESSAGE
    assert error.value.get_message() == 'Unload command input missing \"model-name\" field'

    valid_msg = '{ \"command\" : \"some-predict-command\", \"model-name\" : \"name\"}'
    ModelWorkerMessageValidators.validate_unload_msg(valid_msg)