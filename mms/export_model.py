# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Command line interface to export model files to be used for inference by MXNet Model Server
"""

import glob
import inspect
import json
import os
import re
import shutil
import zipfile

import mxnet as mx

import mms
from mms.arg_parser import ArgParser
from mms.log import get_logger
from mms.model_service.model_service import load_service
from mms.model_service.mxnet_model_service import MXNetBaseService, GluonImperativeBaseService

logger = get_logger()

try:
    basestring
except NameError:
    # pylint: disable=redefined-builtin
    basestring = str

MMS_SERVICE_FILES = {k: os.path.splitext(v.__file__)[0] + '.py' for k, v in {
    'image/jpeg': mms.model_service.mxnet_vision_service,
    'application/json': mms.model_service.mxnet_model_service
}.items()}

SIG_REQ_ENTRY = ['inputs', 'input_type', 'outputs', 'output_types']
VALID_MIME_TYPE = ['image/jpeg', 'application/json']
SIGNATURE_FILE = 'signature.json'
MODEL_ARCHIVE_VERSION = 0.2
MODEL_SERVER_VERSION = mms.__version__
MANIFEST_FILE_NAME = 'MANIFEST.json'
MXNET_TYPE = 'mxnet'
ONNX_TYPE = 'onnx'
MXNET_ATTRS = 'attrs'
MXNET_VERSION = 'mxnet_version'

NO_MODEL_FILES_MESSAGE = '''
No model files found in the model directory {}.

mxnet-model-export supports the exporting of MXNet and ONNX models to a
mxnet-model-server .model file.

MXNet models are expected as two files in the same directory, a params file
and a symbol file, both with the same prefix and where 0000 is the param's
epoch number (any number from 0 to n).
Example: modelname-0000.params and modelname-symbol.json.

ONNX models are expected as one file.
Example: modelname.onnx.

See https://github.com/onnx/onnx for converting PyTorch, Caffe2, CNTK,
and other models to the ONNX format.

Gluon models are expected to have a
Custom Service file derived from GluonImperativeBaseService,
a Parameters file,
a Signature file.

In order to export Gluon models, the export tool expects --service-file-path pointing to the Gluon custom service
code.
'''.strip()

MIXED_MODEL_FILES_MESSAGE = 'More than one model type is present in the model directory {}.'
INCOMPLETE_MODULE_FILES_MESSAGE = '''
Incomplete MXNet model found in the model directory {}.

MXNet models require a parameter file and a symbol file.
Example: modelname-0000.params and modelname-symbol.json.
'''.strip()

NO_EPOCH_NUMBER_MESSAGE = '''
No epoch number found in the parameter filename {}.

When exporting an MXNet model, mxnet-model-export expects a parameters file
that includes the epoch number in the filename. 0000 is usually sufficient,
but if known, you can supply any epoch number in the format: modelname-1234.params.
'''.strip()

MODEL_PREFIX_MISMATCH_MESSAGE = '''
Your parameters file and symbols file naming prefix do not match.

When exporting an MXNet model, mxnet-model-export expects two files in the
same directory, a params file and a symbol file, both with the same prefix
(a common name) and where 0000 is the param's epoch number (any number from 0 to n).
Example: modelname-0000.params and modelname-symbol.json.
'''.strip()


def validate_signature(model_path):
    """
    Internal helper to check signature error when exporting model with CLI.
    """
    signature_file = os.path.join(model_path, SIGNATURE_FILE)

    assert os.path.isfile(signature_file), \
        "signature.json is not found in %s." % model_path
    with open(signature_file) as js_file:
        signature = json.load(js_file)

    assert 'input_type' in signature and 'output_type' in signature, \
        'input_type and output_type are required in signature.'
    assert isinstance(signature['input_type'], basestring) and \
           isinstance(signature['output_type'], basestring), \
        'Value of input_type and output_type should be string'
    assert signature['input_type'] in VALID_MIME_TYPE and \
           signature['output_type'] in VALID_MIME_TYPE, \
        'Valid type should be picked from %s. ' \
        'Got %s for input and %s for output' % \
        (VALID_MIME_TYPE, signature['input_type'], signature['output_type'])

    assert 'inputs' in signature and 'outputs' in signature, \
        'inputs and outputs are required in signature.'
    assert isinstance(signature['inputs'], list) and \
           isinstance(signature['outputs'], list), \
        'inputs and outputs values must be list.'
    for ip in signature['inputs']:
        assert isinstance(ip, dict), 'Each input must be a dictionary.'
        assert 'data_name' in ip, 'data_name is required for input.'
        assert isinstance(ip['data_name'], basestring), 'data_name value must be string.'
        assert 'data_shape' in ip, 'data_shape is required for input.'
        assert isinstance(ip['data_shape'], list), 'data_shape value must be list.'
    for output in signature['outputs']:
        assert isinstance(output, dict), 'Each output must be a dictionary.'
        assert 'data_name' in output, 'data_name is required for output.'
        assert isinstance(output['data_name'], basestring), 'data_name value must be string.'
        assert 'data_shape' in output, 'data_shape is required for output.'
        assert isinstance(output['data_shape'], list), 'data_shape value must be list.'

    return signature_file


def validate_service(model_path, service_file, signature_file):
    """
    Validate the service
    :param model_path:
    :param service_file:
    :param signature_file:
    :return:
    """

    is_gluon_service = False
    if service_file:

        assert os.path.isfile(service_file) or os.path.isfile(os.path.join(model_path, service_file)), \
            "Service File not found in %s or in %s." % (service_file, model_path)

        service_file = service_file if os.path.isfile(service_file) \
            else glob.glob(model_path + service_file)[0]

        module = load_service(service_file)

        classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        # Check if subclass of MXNetBaseService or GluonImperativeBaseService
        # pylint: disable=deprecated-lambda
        service_classes_mxnet = list(filter(lambda cls: issubclass(cls, MXNetBaseService), classes))
        service_classes_gluon = list(filter(lambda cls: issubclass(cls, GluonImperativeBaseService), classes))
        is_mxnet_service = len(service_classes_mxnet) > 1
        is_gluon_service = len(service_classes_gluon) > 1
        assert (len(service_classes_mxnet) > 1 or len(service_classes_gluon) > 1), \
            "The Service class should be derived from MXNetBaseService or GluonImperativeBaseService, " \
            "found %s classes" % str(service_classes_mxnet)

        if is_gluon_service and is_mxnet_service:
            raise ValueError("Service file contains both symbolic and imperative sub-classes. The service file"
                             "should contain either MXNetBaseSerivce class or GluonImperativeBaseService class")

        # remove the compiled python code
        if os.path.exists(service_file + 'c'):
            os.remove(service_file + 'c')

    else:
        input_type = None

        with open(signature_file) as js_file:
            input_type = json.load(js_file)['input_type']

        if input_type not in VALID_MIME_TYPE:
            raise ValueError("input_type should be one of %s or have your own service file handling it"
                             % str(VALID_MIME_TYPE))

        service_file = MMS_SERVICE_FILES[input_type]
        if not os.path.exists(service_file):
            raise ValueError('Service File {} is missing in mms installation'
                             .format(os.path.basename(service_file)))

    return is_gluon_service, service_file


def split_on_letter(s):
    match = re.compile(r"[^\W\d]").search(s)
    if match:
        return float(s[:match.start()])

    return float(s)

def generate_manifest(symbol_file=None, params_file=None, service_file=None, signature_file=None,
                      model_name=None, model_type_imperative=False):
    """
    Funtion to generate manifest file
    :param symbol_file:
    :param params_file:
    :param service_file:
    :param signature_file:
    :param model_name:
    :param model_type:
    :return:
    """
    manifest = {}
    conv_int = split_on_letter(MODEL_SERVER_VERSION)
    manifest["Model-Archive-Version"] = MODEL_ARCHIVE_VERSION
    manifest["Model-Archive-Description"] = model_name
    manifest["Model-Server"] = conv_int
    manifest["Model"] = {}
    manifest["Model"]["Symbol"] = os.path.split(symbol_file)[1] if symbol_file else ""
    manifest["Model"]["Parameters"] = os.path.split(params_file)[1] if params_file else ""
    manifest["Model"]["Signature"] = os.path.split(signature_file)[1]
    manifest["Model"]["Service"] = os.path.split(service_file)[1]
    manifest["Model"]["Description"] = model_name
    manifest["Model"]["Model-Name"] = model_name
    manifest["Model"]["Model-Format"] = "MXNet-Symbolic" if model_type_imperative is False else "Gluon-Imperative"

    mxnet_version = mx.__version__

    if symbol_file is not None and os.path.exists(symbol_file):
        symbol_json = json.load(open(symbol_file))
        if MXNET_ATTRS in symbol_json:
            if MXNET_VERSION in symbol_json[MXNET_ATTRS]:
                mxnet_version = symbol_json[MXNET_ATTRS][MXNET_VERSION]

    manifest["Engine"] = {"MXNet": mxnet_version}

    return manifest


def convert_onnx_model(model_path, onnx_file):
    """
    Util to convert onnx model to MXNet model
    :param model_path:
    :param onnx_file:
    :return:
    """
    from mxnet.contrib import onnx as onnx_mxnet
    import onnx
    model_name = os.path.splitext(os.path.basename(onnx_file))[0]
    symbol_file = '%s-symbol.json' % model_name
    params_file = '%s-0000.params' % model_name
    signature_file = 'signature.json'
    # Find input symbol name and shape
    model_proto = onnx.load(os.path.join(model_path, onnx_file))
    graph = model_proto.graph
    _params = set()
    for tensor_vals in graph.initializer:
        _params.add(tensor_vals.name)

    input_data = []
    for graph_input in graph.input:
        shape = []
        if graph_input.name not in _params:
            for val in graph_input.type.tensor_type.shape.dim:
                shape.append(val.dim_value)
            input_data.append((graph_input.name, tuple(shape)))

    sym, arg_params, aux_params = onnx_mxnet.import_model(os.path.join(model_path, onnx_file))
    # UNION of argument and auxillary parameters
    params = dict(arg_params, **aux_params)
    # rewrite input data_name correctly
    with open(os.path.join(model_path, signature_file), 'r') as f:
        data = json.loads(f.read())
        data['inputs'][0]['data_name'] = input_data[0][0]
    with open(os.path.join(model_path, signature_file), 'w') as f:
        f.write(json.dumps(data, indent=2))

    with open(os.path.join(model_path, symbol_file), 'w') as f:
        f.write(sym.tojson())

    save_dict = {('arg:%s' % k): v.as_in_context(mx.cpu()) for k, v in params.items()}
    mx.nd.save(os.path.join(model_path, params_file), save_dict)
    return symbol_file, params_file


def find_unique(files, suffix):
    """
    Function to find unique model params file
    :param files:
    :param suffix:
    :return:
    """
    match = [f for f in files if f.endswith(suffix)]
    count = len(match)

    if count == 0:
        return None
    elif count == 1:
        return match[0]
    else:
        params = {
            '.params': ('.params', 'parameter file'),
            '-symbol.json': ('...-symbol.json', 'symbol file'),
            '.onnx': ('.onnx', 'ONNX model file'),
        }[suffix]

        message = 'mxnet-model-export expects only one ' + params[0] + ' file. Please supply the single ' + \
                  params[1] + ' file you wish to export.'

        raise ValueError(message)


def validate_epoch_number(params_file):
    """
    Util to validate epoch number
    :param params_file:
    :return:
    """
    if re.match(r'^[\w\-\.]+-\d+\.params$', params_file) is None:
        raise ValueError(NO_EPOCH_NUMBER_MESSAGE.format(params_file))


def validate_prefix_match(symbol_file=None, params_file=None):
    """
    Validate prefix match
    :param symbol_file:
    :param params_file:
    :return:
    """
    symbol_prefix = symbol_file.replace('-symbol.json', '')
    params_prefix = re.sub(r'-\d+\.params$', '', params_file)
    if symbol_prefix != params_prefix:
        raise ValueError(MODEL_PREFIX_MISMATCH_MESSAGE.format(symbol_file, params_file))


def validate_model_files(model_path, onnx_file, params_file, symbol_file, model_type_imperative=False):
    """
    Validate the model files
    :param model_path:
    :param onnx_file:
    :param params_file:
    :param symbol_file:
    :param model_type:
    :return:
    """

    mask = 0
    if onnx_file:
        mask += 1
    if params_file:
        mask += 2
    if symbol_file:
        mask += 4
    if model_type_imperative is True:
        mask += 8

    if mask == 0:
        raise ValueError(NO_MODEL_FILES_MESSAGE.format(model_path))
    if mask in [3, 5, 7, 9, 14, 15]:
        raise ValueError(MIXED_MODEL_FILES_MESSAGE.format(model_path))
    if mask in [2, 4, 12]:
        raise ValueError(INCOMPLETE_MODULE_FILES_MESSAGE.format(model_path))
    if mask == 6:
        # an mxnet model
        validate_epoch_number(params_file)
        validate_prefix_match(symbol_file, params_file)


def export_model(model_name, model_path, service_file=None, export_file=None):
    """
    Internal helper for the exporting model command line interface.
    """
    temp_files = []
    try:
        if export_file is None:
            export_file = '{}/{}.model'.format(os.getcwd(), model_name)
        assert not os.path.exists(export_file), "model file {} already exists.".format(export_file)

        if model_path.startswith('~'):
            model_path = os.path.expanduser(model_path)

        files = os.listdir(model_path)
        onnx_file = find_unique(files, '.onnx')
        symbol_file = find_unique(files, '-symbol.json')
        params_file = find_unique(files, '.params')

        signature_file = validate_signature(model_path)
        is_imperative, service_file = validate_service(model_path, service_file, signature_file)
        validate_model_files(model_path, onnx_file, params_file, symbol_file, is_imperative)
        if os.path.basename(service_file) not in files:
            temp_files.append(os.path.basename(service_file))
            shutil.copyfile(service_file, os.path.join(model_path, os.path.basename(service_file)))
        service_file = os.path.basename(service_file)

        if onnx_file:
            symbol_file, params_file = convert_onnx_model(model_path, onnx_file)
            files.remove(onnx_file)
            temp_files.extend([symbol_file, params_file])

        manifest = generate_manifest(symbol_file, params_file, service_file, signature_file, model_name, is_imperative)
        with open(os.path.join(model_path, MANIFEST_FILE_NAME), 'w') as m:
            json.dump(manifest, m, indent=4)
        temp_files.append(MANIFEST_FILE_NAME)

        with zipfile.ZipFile(export_file, 'w') as z:
            for f in files + temp_files:
                z.write(os.path.join(model_path, f), f)
        logger.info("Successfully exported model %s to file %s", model_name, export_file)

    finally:
        for f in temp_files:
            os.remove(os.path.join(model_path, f))


def export():
    """
    Export as MXNet model
    :return:
    """
    args = ArgParser.export_parser().parse_args()
    export_model(model_name=args.model_name, model_path=args.model_path,
                 service_file=args.service_file_path)


if __name__ == '__main__':
    export()
