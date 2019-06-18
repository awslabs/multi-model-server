# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import importlib
import inspect
import os
import sys
import types
from collections import namedtuple

import mock
import pytest

from mms.model_loader import LegacyModelLoader
from mms.model_loader import MmsModelLoader
from mms.model_loader import ModelLoaderFactory
from mms.model_service.model_service import SingleNodeService


# noinspection PyClassHasNoInit
# @pytest.mark.skip(reason="Disabling it currently until the PR #467 gets merged")
class TestModelFactory:

    def test_model_loader_factory_legacy(self):
        model_loader = ModelLoaderFactory.get_model_loader(
            os.path.abspath('mms/tests/unit_tests/model_service/dummy_model'))

        assert isinstance(model_loader, LegacyModelLoader)

    def test_model_loader_factory(self):
        model_loader = ModelLoaderFactory.get_model_loader(
            os.path.abspath('mms/tests/unit_tests/test_utils/'))

        assert isinstance(model_loader, MmsModelLoader)


# noinspection PyClassHasNoInit
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


# noinspection PyProtectedMember
# noinspection PyClassHasNoInit
class TestLoadModels:
    model_name = 'testmodel'
    model_dir = os.path.abspath('mms/tests/unit_tests/model_service/dummy_model')
    mock_manifest = '{"Model":{"Service":"dummy_class_model_service.py",' \
                    '"Signature":"signature.json","Model-Name":"testmodel"}}'

    @pytest.fixture()
    def patches(self, mocker):
        Patches = namedtuple('Patches', ['mock_open', 'os_path', "is_file", "open_signature"])
        patches = Patches(
            mocker.patch('mms.model_loader.open'),
            mocker.patch('os.path.exists'),
            mocker.patch('os.path.isfile'),
            mocker.patch('mms.model_service.model_service.open')
        )
        return patches

    def test_load_model_legacy(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        patches.open_signature.side_effect = [mock.mock_open(read_data='{}').return_value]
        patches.is_file.return_value = True
        patches.os_path.side_effect = [False, True]
        sys.path.append(self.model_dir)
        handler = 'dummy_model_service'
        model_loader = ModelLoaderFactory.get_model_loader(self.model_dir)
        assert isinstance(model_loader, LegacyModelLoader)
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

        assert inspect.ismethod(service._entry_point)

    def test_load_class_model(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_class_model_service'
        model_loader = ModelLoaderFactory.get_model_loader(os.path.abspath('mms/unit_tests/test_utils/'))
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

        assert inspect.ismethod(service._entry_point)

    def test_load_func_model(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:infer'
        model_loader = ModelLoaderFactory.get_model_loader(os.path.abspath('mms/unit_tests/test_utils/'))
        service = model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

        assert isinstance(service._entry_point, types.FunctionType)
        assert service._entry_point.__name__ == 'infer'

    def test_load_func_model_with_error(self, patches):
        patches.mock_open.side_effect = [mock.mock_open(read_data=self.mock_manifest).return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service:wrong'
        model_loader = ModelLoaderFactory.get_model_loader(os.path.abspath('mms/unit_tests/test_utils/'))
        with pytest.raises(ValueError, match=r"Expected only one class .*"):
            model_loader.load(self.model_name, self.model_dir, handler, 0, 1)

    def test_load_model_with_error(self, patches):
        patches.mock_open.side_effect = [
            mock.mock_open(read_data='{"test" : "h"}').return_value]
        sys.path.append(os.path.abspath('mms/tests/unit_tests/test_utils/'))
        patches.os_path.return_value = True
        handler = 'dummy_func_model_service'
        model_loader = ModelLoaderFactory.get_model_loader(os.path.abspath('mms/unit_tests/test_utils/'))
        with pytest.raises(ValueError, match=r"Expected only one class .*"):
            model_loader.load(self.model_name, self.model_dir, handler, 0, 1)
