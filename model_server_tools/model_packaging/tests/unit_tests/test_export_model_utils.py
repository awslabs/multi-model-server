# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import pytest
from mock import Mock, mock_open, patch
import os
import sys
from collections import namedtuple
from model_server_tools.model_packaging.export_model_utils import ModelExportUtils
from model_server_tools.model_packaging.manifest_components.engine import EngineType
from model_server_tools.model_packaging.manifest_components.manifest import RuntimeType


# noinspection PyClassHasNoInit
class TestExportModelUtils:

    # noinspection PyClassHasNoInit
    class TestMarExistence:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['getcwd', 'path_exists'])
            patches = Patches(mocker.patch('os.getcwd'), mocker.patch('os.path.exists'))

            patches.getcwd.return_value = '/Users/Piyush'

            return patches

        def test_export_file_is_none(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists('some-model', None, False)

            patches.path_exists.assert_called_once_with("/Users/Piyush/some-model.mar")
            assert ret_val == "/Users/Piyush/some-model.mar"

        def test_export_file_is_not_none(self, patches):
            patches.path_exists.return_value = False
            ModelExportUtils.check_mar_already_exists('some-model', '/Users/Piyush/some-model', False)

            patches.path_exists.assert_called_once_with('/Users/Piyush/some-model')

        def test_export_file_already_exists_with_override(self, patches):
            patches.path_exists.return_value = True

            ModelExportUtils.check_mar_already_exists('some-model', None, True)

            patches.path_exists.assert_called_once_with('/Users/Piyush/some-model.mar')

        def test_export_file_already_exists_with_override_false(self, patches):
            patches.path_exists.return_value = True

            with pytest.raises(SystemExit):
                ModelExportUtils.check_mar_already_exists('some-model', None, False)

            patches.path_exists.assert_called_once_with('/Users/Piyush/some-model.mar')

    # noinspection PyClassHasNoInit
    class TestCustomModelTypes:

        model_path = '/Users/Piyush'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['utils', 'listdir'])
            patch = Patches(mocker.patch('model_server_tools.model_packaging.export_model_utils.ModelExportUtils'),
                            mocker.patch('os.listdir'))

            patch.listdir.return_value = set(['a', 'b', 'c'])
            return patch

        def test_onnx_file_is_none(self, patches):
            patches.utils.find_unique.return_value = None
            ModelExportUtils.check_custom_model_types(model_path=self.model_path)

            patches.utils.find_unique.assert_called()
            patches.utils.convert_onnx_model.assert_not_called()

        def test_onnx_file_is_not_none(self, patches):
            onnx_file = 'some-file.onnx'
            patches.utils.find_unique.return_value = onnx_file
            patches.utils.convert_onnx_model.return_value = ('sym', 'param')

            temp, exclude = ModelExportUtils.check_custom_model_types(self.model_path)
            patches.utils.convert_onnx_model.assert_called_once_with(self.model_path, onnx_file)

            assert len(temp) == 2
            assert len(exclude) == 1
            assert temp[0] == os.path.join(self.model_path, 'sym')
            assert temp[1] == os.path.join(self.model_path, 'param')
            assert exclude[0] == onnx_file

    # noinspection PyClassHasNoInit
    class TestFindUnique:

        def test_with_count_zero(self):
            files = ['a.txt', 'b.txt', 'c.txt']
            suffix = '.mxnet'
            val = ModelExportUtils.find_unique(files, suffix)
            assert val is None

        def test_with_count_one(self):
            files = ['a.mxnet', 'b.txt', 'c.txt']
            suffix = '.mxnet'
            val = ModelExportUtils.find_unique(files, suffix)
            assert val == 'a.mxnet'

        def test_with_exit(self):
            files = ['a.onnx', 'b.onnx', 'c.txt']
            suffix = '.onnx'
            with pytest.raises(SystemExit):
                ModelExportUtils.find_unique(files, suffix)

    # noinspection PyClassHasNoInit
    class TestCleanTempFiles:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['remove'])
            patches = Patches(mocker.patch('os.remove'))

            patches.remove.return_value = True
            return patches

        def test_clean_call(self, patches):
            temp_files = ['a', 'b', 'c']
            ModelExportUtils.clean_temp_files(temp_files)

            patches.remove.assert_called()
            assert patches.remove.call_count == len(temp_files)

    # noinspection PyClassHasNoInit
    class TestCreateManifestFile:

        model_path = '/Users/Piyush'
        manifest = json.dumps({'some-key': 'somevalue'})

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['os_path', 'os_mkdir'])
            patches = Patches(mocker.patch('os.path'),
                              mocker.patch('os.makedirs'))

            patches.os_mkdir.return_value = True
            return patches

        def test_with_path_not_exists(self, patches):
            patches.os_path.exists.return_value = False
            patches.os_path.join.return_value = '/Users/ghaipiyu/'

            patch_open = patch('__builtin__.open', new_callable=mock_open()) if sys.version_info[0] < 3 else \
                patch('builtins.open', new_callable=mock_open())

            with patch_open:
                with patch('json.dump'):
                    ModelExportUtils.create_manifest_file(self.model_path, self.manifest)

            patches.os_mkdir.assert_called()
            patches.os_path.join.assert_called()

        def test_with_path_exists(self, patches):
            patches.os_path.exists.return_value = True
            patches.os_path.join.return_value = '/Users/ghaipiyu/'

            patch_open = patch('__builtin__.open', new_callable=mock_open()) if sys.version_info[0] < 3 else \
                patch('builtins.open', new_callable=mock_open())

            with patch_open as m:
                with patch('json.dump') as m_json:
                    ModelExportUtils.create_manifest_file(self.model_path, self.manifest)

            patches.os_mkdir.assert_not_called()
            patches.os_path.join.assert_called()
            m_json.assert_called()
            m.assert_called_with('/Users/ghaipiyu/', 'w')

    # noinspection PyClassHasNoInit
    class TestGenerateManifestProps:

        class Namespace:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        author = 'ABC'
        email = 'ABC@XYZ.com'
        engine = EngineType.MXNET.value
        model_name = 'my-model'
        handler = 'a.py::my-awesome-func'

        args = Namespace(author=author, email=email, engine=engine, model_name=model_name, handler=handler,
                         runtime=RuntimeType.PYTHON2_7.value)

        def test_publisher(self):
            pub = ModelExportUtils.generate_publisher(self.args)
            assert pub.email == self.email
            assert pub.author == self.author

        def test_engine(self):
            eng = ModelExportUtils.generate_engine(self.args)
            assert eng.engine_name == EngineType(self.engine)

        def test_model(self):
            mod = ModelExportUtils.generate_model(self.args)
            assert mod.model_name == self.model_name
            assert mod.handler == self.handler

        def test_manifest_json(self):
            manifest = ModelExportUtils.generate_manifest_json(self.args)
            manifest_json = json.loads(manifest)
            assert manifest_json['runtime'] == RuntimeType.PYTHON2_7.value
            assert 'engine' in manifest_json
            assert 'model' in manifest_json
            assert 'publisher' in manifest_json
            assert 'license' not in manifest_json
