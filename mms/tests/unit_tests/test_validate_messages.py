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
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators


def test_validate_load_message_missing_model_path():
    invalid_object = {'command': 'some-command', 'modelName': 'some-model-name', 'handler': 'some handler string'}
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_load_message(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE, 'error codes don\'t match'
    assert error.value.get_message() == "Load command missing \"modelPath\" key"


def test_validate_load_message_missing_model_name():
    invalid_object = {'command': 'load', 'modelPath': 'some-model-path', 'handler': 'some handler string'}
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_load_message(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE, 'error codes don\'t match'
    assert error.value.get_message() == "Load command missing \"modelName\" key"


def test_validate_load_messages_missing_handler():
    invalid_object = {'command': 'load', 'modelPath': 'some-model-path', 'modelName': 'some model name'}
    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_load_message(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_LOAD_MESSAGE, 'error codes don\'t match'
    assert error.value.get_message() == "Load command missing \"handler\" key"


def test_valudate_load_message_with_valid_msg():
    valid_object = {'command': 'load', 'modelPath': 'some-model-path', 'modelName': 'some-model-name',
                    'handler': 'some handler string'}
    ModelWorkerMessageValidators.validate_load_message(valid_object)


def test_validate_predict_data_with_missing_request_id():
    invalid_object = {'encoding': 'None|base64|utf-8', 'modelInputs': []}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_data(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == "Predict command input missing \"request-id\" field"

    valid_msg = '{ \"requestId\" : \"111-222-3333\"}'
    ModelWorkerMessageValidators.validate_predict_data(valid_msg)


def test_validate_predict_data_with_valid_msg():
    valid_object = {'requestId': '111-222-3333', 'encoding': 'None|base64|utf-8', 'modelInputs': '[{}]'}
    ModelWorkerMessageValidators.validate_predict_data(valid_object)


def test_validate_predict_inputs_missing_value():
    invalid_object = {'encoding': 'base64', 'name': 'model_input_name'}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_inputs(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == "Predict command input data missing \"value\" field"


def test_validate_predict_inputs_missing_name():
    invalid_object = {'encoding': 'base64', 'value': 'val1'}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_inputs(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_INPUT
    assert error.value.get_message() == 'Predict command input data missing \"name\" field'


def test_validate_predict_inputs_with_valid_input():
    valid_object = {'encoding': 'base64', 'value': 'val1', 'name': 'model_input_name'}
    ModelWorkerMessageValidators.validate_predict_inputs(valid_object)


def test_validate_predict_msg_missing_model_name():
    invalid_object = {'command': 'some-prediction-command', 'requestBatch': []}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_msg(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE
    assert error.value.get_message() == "Predict command input missing \"modelName\" field."


def test_validate_predict_msg_missing_request_batch():
    invalid_object = {'command': 'some-prediction-command', 'modelName': 'name'}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_msg(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE
    assert error.value.get_message() == "Predict command input missing \"requestBatch\" field."


def test_validate_predict_msg_missing_model_inputs():
    invalid_object = {'command': 'some-pred-command', 'modelName': 'name', 'requestBatch':
                      [{'requestId': '111', 'encoding': 'None|base64|utf-8'}]}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_predict_msg(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_PREDICT_MESSAGE
    assert error.value.get_message() == "Predict command input\'s requestBatch missing \"modelInputs\" field."


def test_validate_predict_msg_valid_input():
    valid_object = {'command': 'some-pred-command', 'modelName': 'name', 'requestBatch':
                    [{'requestId': '111', 'encoding': 'None|base64|utf-8', 'modelInputs': [{}]}]}

    ModelWorkerMessageValidators.validate_predict_msg(valid_object)


def test_validate_unload_msg_with_invalid_msg():
    invalid_object = {'command': 'unload'}

    with pytest.raises(MMSError) as error:
        ModelWorkerMessageValidators.validate_unload_msg(invalid_object)

    assert error.value.get_code() == ModelServerErrorCodes.INVALID_UNLOAD_MESSAGE
    assert error.value.get_message() == "Unload command input missing \"model-name\" field"


def test_validate_unload_msg_with_valid_msg():
    valid_object = {'command': 'unload', 'model-name': 'name'}
    ModelWorkerMessageValidators.validate_unload_msg(valid_object)
