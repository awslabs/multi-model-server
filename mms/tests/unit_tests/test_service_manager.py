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
from mock import patch
model_names = ['resnet1', 'vqa2', 'rnn3']

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

    assert response.name == 'loaded_modelservices'


def test_get_loaded_modelservices_with_model_names():
    expected_response = dict()

    for model in model_names:
        service_manager.loaded_modelservices[model] = ''.join([model, '.service'])
        expected_response[model] = ''.join([model, '.service'])

    response = service_manager.get_loaded_modelservices(model_names)
    assert expected_response == response


def test_get_registered_modelservices_with_nil_modelservice_names():
    with patch.object(service_manager, 'get_modelservices_registry',
                      wraps=service_manager.get_modelservices_registry) as spy:
        service_manager.get_registered_modelservices(modelservice_names=None)
        spy.assert_called_with(None)


def test_get_registered_modelservices_some_modelservice_name():
    with patch.object(service_manager, 'get_modelservices_registry',
                      wraps=service_manager.get_modelservices_registry) as spy:
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
    del(expected_response[model_names[0]])

    assert model_names[0] not in response
    assert expected_response == response


def test_parse_modelservices_from_module_with_nil_service_file():
    with pytest.raises(Exception) as error:
        service_manager.parse_modelservices_from_module(service_file=None)

    assert error.value.args[0] == "Invalid service file given"
