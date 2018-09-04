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
import sys
import pytest
import importlib
import mock
import types
from collections import namedtuple
from mms.model_loader import MmsModelLoader
from mms.model_loader import ModelLoaderFactory
from mms.model_loader import LegacyModelLoader
from mms.model_service.model_service import SingleNodeService
from mms.mxnet_model_service_error import MMSError

# @pytest.mark.skip(reason="Disabling it currently until the PR #467 gets merged")
class TestModelFactory:

    def test_model_loader_factory_incorrect(self):

        with pytest.raises(ValueError) as excinfo:
            _ = ModelLoaderFactory.get_model_loader("wrong_loader")

        assert str(excinfo.value) == "Unknown model loader type: wrong_loader"

    def test_model_loader_factory_legacy(self):

        model_loader = ModelLoaderFactory.get_model_loader("legacy_mms")

        assert isinstance(model_loader, LegacyModelLoader)

    def test_model_loader_factory(self):

        model_loader = ModelLoaderFactory.get_model_loader("mms")

        assert isinstance(model_loader, MmsModelLoader)

class TestListModels:

    def test_list_models_legacy(self):

        model_loader = ModelLoaderFactory.get_model_loader("legacy_mms")
        sys.path.append(os.path.abspath('mms/tests/unit_tests/model_service/dummy_model'))
        module = importlib.import_module('dummy_model_service')
        classes = model_loader.list_model_services(module, SingleNodeService)
        assert len(classes) == 1
        assert issubclass(classes[0], SingleNodeService)

    def test_list_models(self):
        model_loader = ModelLoaderFactory.get_model_loader("mms")
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        module = importlib.import_module('dummy_class_model_service')
        classes = model_loader.list_model_services(module)
        assert len(classes) == 1
        assert classes[0].__name__ == 'CustomService'

class TestLoadModels:
    model_name = 'testmodel'
    model_dir = os.path.abspath('mms/tests/unit_tests/model_service/dummy_model')
    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['mock_open', 'json_load', 'os_path'])
        patches = Patches(
            mocker.patch('mms.model_loader.open'),
            mocker.patch('json.load'),
            mocker.patch('os.path.exists')
        )
        return patches

    def test_load_model_legacy(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]

        sys.path.append(os.path.abspath('mms/tests/unit_tests/model_service/dummy_model'))
        handler = 'dummy_model_service'
        model_loader = ModelLoaderFactory.get_model_loader("legacy_mms")
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
        patches.json_load.assert_called()
        isinstance(service._entry_point, SingleNodeService)
        assert hasattr(service._entry_point, 'inference') and isinstance(getattr(service._entry_point, 'inference'),
                                                                         types.MethodType)

    def test_load_class_model(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_class_model_service'
        model_loader = ModelLoaderFactory.get_model_loader("mms")
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
        patches.json_load.assert_called()
        isinstance(service._entry_point, types.FunctionType)
        assert service._entry_point.__name__ == 'handle'

    def test_load_func_model(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:infer'
        model_loader = ModelLoaderFactory.get_model_loader("mms")
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
        patches.json_load.assert_called()
        isinstance(service._entry_point, types.FunctionType)
        assert service._entry_point.__name__ == 'infer'

    def test_load_func_model_with_error(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:wrong'
        model_loader = ModelLoaderFactory.get_model_loader("mms")
        with pytest.raises(MMSError) as excinfo:
            _ = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
        assert str(excinfo.value.message) == "Expected only one class in custom service code or a function entry point"

    def test_load_model_with_error(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service'
        model_loader = ModelLoaderFactory.get_model_loader("mms")
        with pytest.raises(Exception) as excinfo:
            _ = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
        assert str(excinfo.value.message) == "Expected only one class in custom service code or a function entry point"
