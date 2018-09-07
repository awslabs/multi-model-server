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
from mock import Mock
import os
from collections import namedtuple
from model_server_util_tools.model_packaging.export_model_utils import ModelExportUtils
from model_server_util_tools.model_packaging.model_packaging_error import ModelPackagingError
from model_server_util_tools.model_packaging.model_packaging_error_codes import ModelPackagingErrorCodes



# noinspection PyClassHasNoInit
class TestExportModelUtils:

    # noinspection PyClassHasNoInit
    class TestMarExistence:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['getcwd', 'path_exists'])
            patches = Patches(mocker.patch('os.getcwd'),
                              mocker.patch('os.path.exists'))

            patches.getcwd.return_value = '/Users/Piyush'

            return patches

        def test_export_file_is_none(self, patches):
            patches.path_exists.return_value = False
            ret_val = ModelExportUtils.check_mar_already_exists('some-model', None)

            patches.path_exists.assert_called_once_with("/Users/Piyush/some-model.mar")
            assert ret_val == "/Users/Piyush/some-model.mar"

        def test_export_file_is_not_none(self, patches):
            patches.path_exists.return_value = False
            ModelExportUtils.check_mar_already_exists('some-model', '/Users/Piyush/some-model')

            patches.path_exists.assert_called_once_with('/Users/Piyush/some-model')

        def test_export_file_already_exists(self, patches):
            patches.path_exists.return_value = True

            with pytest.raises(ModelPackagingError) as err :
                ModelExportUtils.check_mar_already_exists('some-model', None)

            patches.path_exists.assert_called_once_with('/Users/Piyush/some-model.mar')
            assert err.value.code == ModelPackagingErrorCodes.MODEL_ARCHIVE_ALREADY_PRESENT
            assert err.value.message == "model file {} already exists.".format('/Users/Piyush/some-model.mar')

    # noinspection PyClassHasNoInit
    class TestGetAbsoluteModelPath:

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['path_expanduser'])
            patches = Patches(mocker.patch('os.path.expanduser'))

            patches.path_expanduser.return_value = '/Users/Piyush/my-model'

            return patches

        def test_path_with_tilda(self, patches):
            path = '~/my-model/'
            ret_val = ModelExportUtils.get_absolute_model_path(path)

            patches.path_expanduser.assert_called_once_with(path)
            assert ret_val == '/Users/Piyush/my-model'

        def test_path_with_no_tilda(self, patches):
            path = '/my-model'
            ret_val = ModelExportUtils.get_absolute_model_path(path)

            patches.path_expanduser.assert_not_called()
            assert ret_val == path

    # noinspection PyClassHasNoInit
    class TestCustomModelTypes:

        model_path = '/Users/Piyush'

        @pytest.fixture()
        def patches(self, mocker):
            Patches = namedtuple('Patches', ['utils', 'listdir'])
            patch = Patches(mocker.patch('model_server_util_tools.model_packaging.export_model_utils.ModelExportUtils'),
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

        def test_with_exception(self):
            files = ['a.onnx', 'b.onnx', 'c.txt']
            suffix = '.onnx'
            with pytest.raises(ModelPackagingError) as err:
                ModelExportUtils.find_unique(files, suffix)

            assert err.value.code == ModelPackagingErrorCodes.INVALID_MODEL_FILES

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