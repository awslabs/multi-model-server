# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
from collections import namedtuple

import pytest
from mms.service_manager import ServiceManager

model_names = ['DummyNodeService', 'vqa2', 'rnn3']


@pytest.fixture()
def service_manager():
    return ServiceManager()


@pytest.fixture()
def spy_fixtures(service_manager, mocker):
    Patches = namedtuple('Patches', ['register_module', 'get_modelservices_registry', 'parse_modelservices_from_module',
                                     'get_registered_modelservices', 'load_model', 'add_modelservice_to_registry'])

    patches = Patches(mocker.patch.object(service_manager, 'register_module', wraps=service_manager.register_module),
                      mocker.patch.object(service_manager, 'get_modelservices_registry',
                                          wraps=service_manager.get_modelservices_registry),
                      mocker.patch.object(service_manager, 'parse_modelservices_from_module',
                                          wraps=service_manager.parse_modelservices_from_module),
                      mocker.patch.object(service_manager, 'get_registered_modelservices',
                                          wraps=service_manager.get_registered_modelservices),
                      mocker.patch.object(service_manager, 'load_model', wraps=service_manager.load_model),
                      mocker.patch.object(service_manager, 'add_modelservice_to_registry',
                                          wraps=service_manager.add_modelservice_to_registry))

    return patches


def test_addmodelservice_to_registry(service_manager):
    service_manager.add_modelservice_to_registry(model_name='resnet', model_service_class_def='resnet.py')

    # Verify that this was actually added
    response = service_manager.get_modelservices_registry(model_names=['resnet'])
    assert response['resnet'] == 'resnet.py'


def test_unload_models(service_manager):
    expected_response = dict()
    for model in model_names:
        service_manager.loaded_modelservices[model] = ''.join([model, '.service'])
        expected_response[model] = ''.join([model, '.service'])

    response = service_manager.unload_models(model_names[0])
    del (expected_response[model_names[0]])

    assert model_names[0] not in response
    assert expected_response == response


class TestGetModelServicesRegistry:

    def test_with_null_model_names(self, service_manager):
        response = service_manager.get_modelservices_registry()

        assert response.name == 'modelservice'

    def test_with_model_names(self, service_manager):
        expected_response = dict()

        for model in model_names:
            service_manager.add_modelservice_to_registry(model, ''.join([model, '.py']))
            expected_response[model] = ''.join([model, '.py'])

        response = service_manager.get_modelservices_registry(model_names)

        assert expected_response == response


class TestGetLoadedModelServices:

    def test_with_nil_model_names(self, service_manager):
        response = service_manager.get_loaded_modelservices()

        assert response.name == "loaded_modelservices"

    def test_with_model_names(self, service_manager):
        expected_response = dict()

        for model in model_names:
            service_manager.loaded_modelservices[model] = ''.join([model, '.service'])
            expected_response[model] = ''.join([model, '.service'])

        response = service_manager.get_loaded_modelservices(model_names)
        assert expected_response == response


class TestParseModelServicesFromModule:

    def test_with_nil_service_file(self, service_manager):
        with pytest.raises(Exception) as error:
            service_manager.parse_modelservices_from_module(service_file=None)

        assert error.value.args[0] == "Invalid service file given"

    def test_with_incorrect_file(self, service_manager):
        service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.p'

        with pytest.raises(Exception) as error:
            service_manager.parse_modelservices_from_module(service_file_path)

        assert "Error when loading service file" in error.value.args[0]

    def test_with_correct_file(self, service_manager):
        service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.py'

        result = service_manager.parse_modelservices_from_module(service_file_path)

        class_list = set(x.__name__ for x in result)

        assert "SomeOtherClass" not in class_list
        assert "SingleNodeService" in class_list
        assert "DummyNodeService" in class_list


class TestGetRegisteredModelServices:

    def test_with_nil_modelservice_names(self, service_manager, spy_fixtures):
        service_manager.get_registered_modelservices(modelservice_names=None)
        spy_fixtures.get_modelservices_registry.assert_called_with(None)

    def test_with_some_modelservice_name(self, service_manager, spy_fixtures):
        spy_fixtures.get_modelservices_registry.return_value = {model_names[0]: ''.join([model_names[0], '.py'])}

        response = service_manager.get_registered_modelservices(modelservice_names=model_names[0])

        spy_fixtures.get_modelservices_registry.assert_called_with([model_names[0]])

        assert model_names[0] in response.keys()
        assert response[model_names[0]] == ''.join([model_names[0], '.py'])


class TestRegisterMoule:

    service_file_path = 'mms/tests/unit_tests/test_utils/dummy_model_service.py'

    def test_with_errors(self, service_manager, spy_fixtures):
        service_manager.register_module(self.service_file_path)
        spy_fixtures.parse_modelservices_from_module.assert_called_with(self.service_file_path)

        spy_fixtures.parse_modelservices_from_module.return_value = list()

        with pytest.raises(AssertionError) as error:
            service_manager.register_module(self.service_file_path)
            spy_fixtures.parse_modelservices_from_module.assert_called_with(self.service_file_path)

        assert error.value.args[0] == "No valid python class derived from Base Model Service is in module file: {}".\
            format(self.service_file_path)

    def test_with_no_errors(self, service_manager, spy_fixtures):

        service_manager.register_module(self.service_file_path)

        spy_fixtures.parse_modelservices_from_module.assert_called_with(self.service_file_path)
        spy_fixtures.add_modelservice_to_registry.assert_called()


class TestRegisterAndLoadModules:

    model_name = model_names[0]
    model_dir = "mms/tests/unit_tests/test_utils"
    manifest = {}
    module_file_path = os.path.join(model_dir, 'dummy_model_service.py')
    gpu = 0
    batch_size = 128

    args = [model_name, model_dir, manifest, module_file_path, gpu, batch_size]

    def test_with_assertion_error(self, service_manager, spy_fixtures):

        spy_fixtures.register_module.return_value = list()
        with pytest.raises(Exception) as error:
            service_manager.register_and_load_modules(*self.args)

        assert error.value.args[0] == "Invalid service file found mms/tests/unit_tests/test_utils/dummy_model_service" \
                                      ".py. Service file should contain only one service class. Found 0"
        spy_fixtures.register_module.assert_called_with(self.module_file_path)

    def test_with_no_error(self, service_manager, spy_fixtures):
        # spy.start()
        spy_fixtures.get_registered_modelservices.return_value = {model_names[0]: ''.join([model_names[0], '.py'])}

        with pytest.raises(Exception):
            service_manager.register_and_load_modules(*self.args)

        spy_fixtures.get_registered_modelservices.assert_called()
        spy_fixtures.load_model.assert_called()
