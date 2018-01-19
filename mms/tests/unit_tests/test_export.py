# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import os
import zipfile

import mxnet as mx
import pytest
from mock import patch, MagicMock

from mms.export_model import export_model, generate_manifest


def list_zip(path):
    return [f.filename for f in zipfile.ZipFile(path).infolist()]


def empty_file(path):
    open(path, 'a').close()


@pytest.fixture()
def onnx_mxnet():
    mock = MagicMock()
    modules = {
        'onnx_mxnet': mock,
    }

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    import onnx_mxnet
    yield onnx_mxnet
    patcher.stop()


@pytest.fixture
def module_dir(tmpdir):
    path = '{}/test'.format(tmpdir)
    os.mkdir(path)
    empty_file('{}/test-symbol.json'.format(path))
    empty_file('{}/test-0000.params'.format(path))
    empty_file('{}/synset.txt'.format(path))

    with open('{}/signature.json'.format(path), 'w') as sig:
        signature = {
            "input_type": "image/jpeg",
            "inputs": [
                {
                    'data_name': 'data1',
                    'data_shape': [1, 3, 64, 64]
                },
                {
                    'data_name': 'data2',
                    'data_shape': [1, 3, 32, 32]
                }
            ],
            "output_type": "application/json",
            "outputs": [
                {
                    'data_name': 'softmax',
                    'data_shape': [1, 10]
                }
            ]
        }
        json.dump(signature, sig)

    return path


def test_generate_manifest():
    manifest = generate_manifest('dir/symbol', 'dir/params', 'dir/service', 'dir/signature', 'name')
    assert 'Model-Archive-Version' in manifest
    assert manifest['Model-Archive-Description'] == 'name'
    assert 'Model-Server' in manifest
    assert 'Engine' in manifest

    assert manifest['Model'] == {
        'Symbol': 'symbol',
        'Parameters': 'params',
        'Signature': 'signature',
        'Service': 'service',
        'Description': 'name',
        'Model-Name': 'name',
        'Model-Format': 'MXNet-Symbolic'
    }


def test_export_module(tmpdir, module_dir):
    export_path = '{}/test.model'.format(tmpdir)
    export_model('test', module_dir, None, export_path)

    assert os.path.exists(export_path), 'no model created - export failed'
    zip_contents = list_zip(export_path)

    for f in ['signature.json', 'test-0000.params', 'test-symbol.json', 'synset.txt', 'MANIFEST.json']:
        assert f in zip_contents

    assert [f for f in zip_contents if f.endswith('.py')], 'missing service file'


def test_export_onnx(tmpdir, module_dir, onnx_mxnet):
    os.remove(os.path.join(module_dir, 'test-symbol.json'))
    os.remove(os.path.join(module_dir, 'test-0000.params'))
    empty_file(os.path.join(module_dir, 'test.onnx'))

    sym = mx.symbol.Variable('data')
    params = {'param_0': mx.ndarray.empty(0)}
    onnx_mxnet.import_model.return_value = (sym, params)

    export_path = '{}/test.model'.format(tmpdir)
    export_model('test', module_dir, None, export_path)

    assert os.path.exists(export_path), 'no model created - export failed'
    zip_contents = list_zip(export_path)
    for f in ['signature.json', 'test-0000.params', 'test-symbol.json', 'synset.txt', 'MANIFEST.json']:
        assert f in zip_contents

    assert [f for f in zip_contents if f.endswith('.py')], 'missing service file'


def test_export_model_no_model_files(tmpdir, module_dir):
    export_path = '{}/test.model'.format(tmpdir)
    os.remove(os.path.join(module_dir, 'test-0000.params'))
    os.remove(os.path.join(module_dir, 'test-symbol.json'))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir, None, export_path)

    assert 'models are expected as' in str(e.value)


def test_export_model_no_params(module_dir):
    os.remove(os.path.join(module_dir, 'test-0000.params'))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'Incomplete MXNet model found' in str(e.value)


def test_export_model_no_symbol(module_dir):
    os.remove(os.path.join(module_dir, 'test-symbol.json'))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'Incomplete MXNet model found' in str(e.value)


@pytest.mark.parametrize('suffix', ['-symbol.json', '-0000.params', '.onnx'])
def test_export_too_many_files(suffix, module_dir):
    empty_file('{}/a{}'.format(module_dir, suffix))
    empty_file('{}/b{}'.format(module_dir, suffix))

    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'expects only one' in str(e.value)
    assert 'expects only one' in str(e.value)


def test_export_onnx_and_module(module_dir):
    empty_file('{}/test.onnx'.format(module_dir))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'More than one model type is present' in str(e.value)


def test_export_no_epoch(module_dir):
    os.remove(os.path.join(module_dir, 'test-0000.params'))
    empty_file('{}/test.params'.format(module_dir))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'No epoch number found' in str(e.value)


def test_export_params_symbol_mismatch(module_dir):
    os.remove(os.path.join(module_dir, 'test-0000.params'))
    empty_file('{}/notest-0000.params'.format(module_dir))
    with pytest.raises(ValueError) as e:
        export_model('test', module_dir)

    assert 'prefix do not match' in str(e.value)
