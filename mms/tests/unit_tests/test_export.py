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
from mms.export_model import export_model, generate_manifest
from mock import patch, MagicMock


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
    from mxnet.contrib import onnx as onnx_mxnet
    yield onnx_mxnet
    patcher.stop()


@pytest.fixture()
def nested_module_dir(tmpdir):
    path = '{}/test'.format(tmpdir)
    os.mkdir(path)
    nested_path = '{}/nested'.format(path)
    os.mkdir(nested_path)
    empty_file('{}/test-symbol.json'.format(path))
    empty_file('{}/test-0000.params'.format(path))
    empty_file('{}/synset.txt'.format(path))
    empty_file('{}/nested-0000.params'.format(nested_path))
    empty_file('{}/nested-dummy'.format(nested_path))
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


@pytest.fixture()
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
    manifest = generate_manifest(symbol_file='dir/symbol',
                                 params_file='dir/params', service_file='dir/service',
                                 signature_file='dir/signature', model_name='name',
                                 model_type_imperative=False)
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


def test_generate_manifest_imperative_with_params():
    manifest = generate_manifest(symbol_file=None,
                                 params_file='dir/params', service_file='dir/service',
                                 signature_file='dir/signature', model_name='name',
                                 model_type_imperative=True)
    assert 'Model-Archive-Version' in manifest
    assert manifest['Model-Archive-Description'] == 'name'
    assert 'Model-Server' in manifest
    assert 'Engine' in manifest

    assert manifest['Model'] == {
        'Symbol': '',
        'Parameters': 'params',
        'Signature': 'signature',
        'Service': 'service',
        'Description': 'name',
        'Model-Name': 'name',
        'Model-Format': 'Gluon-Imperative'
    }


def test_generate_manifest_imperative_without_params():
    manifest = generate_manifest(symbol_file=None,
                                 params_file=None, service_file='dir/service',
                                 signature_file='dir/signature', model_name='name',
                                 model_type_imperative=True)
    assert 'Model-Archive-Version' in manifest
    assert manifest['Model-Archive-Description'] == 'name'
    assert 'Model-Server' in manifest
    assert 'Engine' in manifest

    assert manifest['Model'] == {
        'Symbol': '',
        'Parameters': '',
        'Signature': 'signature',
        'Service': 'service',
        'Description': 'name',
        'Model-Name': 'name',
        'Model-Format': 'Gluon-Imperative'
    }


def test_temp_files_cleanup_no_export_path(module_dir):
    if module_dir.startswith('~'):
        model_path = os.path.expanduser(module_dir)
    else:
        model_path = module_dir
    initial_user_files = set(os.listdir(model_path))
    initial_export_files = set(os.listdir(os.getcwd()))
    export_model('test', module_dir, None, None)
    export_path = '{}/test.model'.format(os.getcwd())
    assert os.path.exists(export_path), 'no model created - export failed'
    final_user_files = set(os.listdir(model_path))
    final_export_files = set(os.listdir(os.getcwd()))

    user_files_created = final_user_files-initial_user_files
    user_files_deleted = initial_user_files-final_user_files
    assert len(user_files_created) == 0, 'temporary files not deleted'
    assert len(user_files_deleted) == 0, 'user files deleted'

    export_files_created = final_export_files - initial_export_files
    assert len(export_files_created) == 1 and list(export_files_created)[0].endswith('.model'), \
        'something other than the model file got generated'
    export_files_deleted = initial_export_files - final_export_files
    assert len(export_files_deleted) == 0, 'user files deleted'

    os.remove(export_path)


def test_temp_files_cleanup_export_path(tmpdir, module_dir):
    export_path = '{}/test.model'.format(tmpdir)
    if module_dir.startswith('~'):
        model_path = os.path.expanduser(module_dir)
    else:
        model_path = module_dir
    initial_files = os.listdir(model_path)
    export_model('test', module_dir, None, export_path)
    assert os.path.exists(export_path), 'no model created - export failed'
    final_files = os.listdir(model_path)
    files_created = set(final_files)-set(initial_files)
    assert len(files_created) == 0, 'temporary files not deleted'
    os.remove(export_path)


def test_export_module(tmpdir, module_dir):
    export_path = '{}/test.model'.format(tmpdir)
    export_model('test', module_dir, None, export_path)

    assert os.path.exists(export_path), 'no model created - export failed'
    zip_contents = list_zip(export_path)

    for f in ['signature.json', 'test-0000.params', 'test-symbol.json', 'synset.txt', 'MANIFEST.json']:
        assert f in zip_contents

    assert [f for f in zip_contents if f.endswith('.py')], 'missing service file'
    os.remove(export_path)


def test_export_nested_module(tmpdir, nested_module_dir):
    export_path = '{}/test.model'.format(tmpdir)
    export_model('test', nested_module_dir, None, export_path)

    assert os.path.exists(export_path), 'no model created - export failed'
    zip_contents = list_zip(export_path)
    for f in ['signature.json', 'test-0000.params', 'test-symbol.json', 'synset.txt', 'MANIFEST.json', 'nested/',
              'nested/nested-0000.params', 'nested/nested-dummy']:
        assert f in zip_contents

    assert [f for f in zip_contents if f.endswith('.py')], 'missing service file'
    os.remove(export_path)


def test_export_module_hyphenated_basename(tmpdir, module_dir):
    os.rename(os.path.join(module_dir, 'test-symbol.json'), os.path.join(module_dir, 'test-hyphens-symbol.json'))
    os.rename(os.path.join(module_dir, 'test-0000.params'), os.path.join(module_dir, 'test-hyphens-0000.params'))
    export_path = '{}/test-hyphens.model'.format(tmpdir)
    export_model('test-hyphens', module_dir, None, export_path)
    assert os.path.exists(export_path), 'no model created - export failed'
    os.remove(export_path)


def test_export_onnx(tmpdir, module_dir, onnx_mxnet):
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
    os.remove(export_path)


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
