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
from mms.service_manager import ServiceManager
from mock import patch, MagicMock

model_names = ['DummyNodeService', 'vqa2', 'rnn3']

service_manager = ServiceManager()


def test_addmodelservice_to_registry():
    service_manager.add_modelservice_to_registry(model_name='resnet', model_service_class_def='resnet.py')

    # Verify that this was actually added
    response = service_manager.get_modelservices_registry(model_names=['resnet'])
    assert response['resnet'] == 'resnet.py'


def test_get_modelservices_registry_with_null_model_names():
    response = service_manager.get_modelservices_registry()

    assert response.name == 'modelservice'


def test_modelservices_registry_with_model_names():
    expected_response = dict()

    for model in model_names:
        service_manager.add_modelservice_to_registry(model, ''.join([model, '.py']))
        expected_response[model] = ''.join([model, '.py'])

    response = service_manager.get_modelservices_registry(model_names)

    assert expected_response == response


def test_get_loaded_modelservices_with_nil_model_names():
    response = service_manager.get_loaded_modelservices()

    assert response.name == "loaded_modelservices"


def test_get_loaded_modelservices_with_model_names():
    expected_response = dict()

    for model in model_names:
        service_manager.loaded_modelservices[model] = ''.join([model, '.service'])
        expected_response[model] = ''.join([model, '.service'])

    response = service_manager.get_loaded_modelservices(model_names)
    assert expected_response == response


def test_parse_modelservices_from_module_with_nil_service_file():
    with pytest.raises(Exception) as error:
        service_manager.parse_modelservices_from_module(service_file=None)

    assert error.value.args[0] == "Invalid service file given"


def test_parse_modelservices_from_module_with_incorrect_file():
    service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.p'

    with pytest.raises(Exception) as error:
        service_manager.parse_modelservices_from_module(service_file_path)

    assert "Error when loading service file" in error.value.args[0]


def test_parse_modelservices_from_module_with_correct_file():
    service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.py'

    result = service_manager.parse_modelservices_from_module(service_file_path)

    class_list = [x.__name__ for x in result]

    assert "SomeOtherClass" not in class_list
    assert "SingleNodeService" in class_list
    assert "DummyNodeService" in class_list


@patch.object(service_manager, 'get_modelservices_registry', wraps=service_manager.get_modelservices_registry)
def test_get_registered_modelservices_with_nil_modelservice_names(spy):
    service_manager.get_registered_modelservices(modelservice_names=None)
    spy.assert_called_with(None)


@patch.object(service_manager, 'get_modelservices_registry', wraps=service_manager.get_modelservices_registry)
def test_get_registered_modelservices_some_modelservice_name(spy):
    response = service_manager.get_registered_modelservices(modelservice_names=model_names[0])

    spy.assert_called_with([model_names[0]])

    assert model_names[0] in response.keys()
    assert response[model_names[0]] == ''.join([model_names[0], '.py'])


def test_unload_models():
    expected_response = dict()
    for model in model_names:
        service_manager.loaded_modelservices[model] = ''.join([model, '.service'])
        expected_response[model] = ''.join([model, '.service'])

    response = service_manager.unload_models(model_names[0])
    del (expected_response[model_names[0]])

    assert model_names[0] not in response
    assert expected_response == response


@patch.object(service_manager, 'parse_modelservices_from_module', wraps=service_manager.parse_modelservices_from_module)
def test_register_module_with_errors(parse_modelservices):
    service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.py'
    service_manager.register_module(service_file_path)
    parse_modelservices.assert_called_with(service_file_path)

    parse_modelservices.return_value = list()

    with pytest.raises(AssertionError) as error:
        service_manager.register_module(service_file_path)
        parse_modelservices.assert_called_with(service_file_path)

    assert error.value.args[0] == "No valid python class derived from Base Model Service is in module file: {}".\
        format(service_file_path)


@patch.object(service_manager, 'parse_modelservices_from_module', wraps=service_manager.parse_modelservices_from_module)
@patch.object(service_manager, 'add_modelservice_to_registry', wraps=service_manager.add_modelservice_to_registry)
def test_register_module_with_no_errors(add_modelservice, parse_modelservice):
    service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.py'
    service_manager.register_module(service_file_path)

    add_modelservice.assert_called()
    parse_modelservice.assert_called_with(service_file_path)


@patch.object(service_manager, 'register_module', wraps=service_manager.register_module)
def test_register_and_load_modules_with_assertion_error(spy):
    model_name = model_names[0]
    model_dir = "mms/tests/unit_tests/test_utils"
    manifest = {}
    module_file_path = ''.join([model_dir, '/dummy_model_service.py'])
    gpu = 0
    batch_size = 128

    spy.return_value = list()
    with pytest.raises(Exception) as error:
        service_manager.register_and_load_modules(model_name, model_dir, manifest, module_file_path, gpu, batch_size)

    assert error.value.args[0] == "Invalid service file found mms/tests/unit_tests/test_utils/dummy_model_service.py." \
                                  " Service file should contain only one service class. Found 0"
    spy.assert_called_with(module_file_path)


@patch.object(service_manager,'get_registered_modelservices',
              wraps=service_manager.get_registered_modelservices)
@patch.object(service_manager, 'load_model', wraps=service_manager.load_model)
def test_register_and_load_modules(load_model, registered_modelservices):
    model_name = model_names[0]
    model_dir = 'mms/tests/unit_tests/test_utils'
    manifest = {}
    module_file_path = ''.join([model_dir, '/dummy_model_service.py'])
    gpu = 0
    batch_size = 128

    # spy.start()
    registered_modelservices.return_value = {model_names[0]: ''.join([model_names[0], '.py'])}

    with pytest.raises(Exception):
        service_manager.register_and_load_modules(model_name, model_dir, manifest, module_file_path, gpu, batch_size)

    registered_modelservices.assert_called()
    load_model.assert_called()
