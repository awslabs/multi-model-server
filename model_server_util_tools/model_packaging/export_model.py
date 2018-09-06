# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import os
import zipfile

from model_server_util_tools import model_packaging
from model_server_util_tools.model_packaging.arg_parser import ArgParser
from model_server_util_tools.log import log_msg

from model_server_util_tools.model_packaging.manifest_components.publisher import Publisher
from model_server_util_tools.model_packaging.manifest_components.engine import Engine
from model_server_util_tools.model_packaging.manifest_components.manifest import Manifest
from model_server_util_tools.model_packaging.manifest_components.model import Model
from model_server_util_tools.model_packaging.model_packaging_error import ModelPackagingError
from model_server_util_tools.model_packaging.model_packaging_error_codes import ModelPackagingErrorCodes


MODEL_ARCHIVE_EXTENSION = '.mar'
MODEL_SERVER_VERSION = '1.0'
MODEL_ARCHIVE_VERSION = model_packaging.__version__
MANIFEST_FILE_NAME = 'MANIFEST.json'
MAR_INF = 'MAR-INF'
ONNX_TYPE = 'onnx'


def generate_publisher(publisher):
    pub = Publisher(publisher.author, publisher.email)
    return pub


def generate_engine(engine):
    engine = Engine(engine.engine_name, engine.engine_version)
    return engine


def generate_model(model):
    model = Model(model.model_name, model.description, model.model_version, dict(), model.handler)
    return model


def generate_manifest(args):
    """
    Function to generate manifest as a json string from the inputs provided by the user in the command line
    :param args:
    :return:
    """

    publisher = generate_publisher(args.publisher)
    engine = generate_engine(args.engine)
    model = generate_model(args.model)

    try:
        manifest = Manifest(args.runtime, engine, model, publisher, args.specification_version,
                            args.implementation_version, args.model_server_version, args.license, args.description,
                            args.user_data)

    except ModelPackagingError as err:
        raise err

    return str(manifest)


def convert_onnx_model(model_path, onnx_file):
    """
    Util to convert onnx model to MXNet model
    :param model_path:
    :param onnx_file:
    :return:
    """
    from mxnet.contrib import onnx as onnx_mxnet
    import onnx
    import mxnet as mx
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

        raise ModelPackagingError(ModelPackagingErrorCodes.INVALID_MODEL_FILES, message)


def get_files_from_model_path(model_path):
    files_set = set(os.listdir(model_path))
    os.chdir(model_path)
    # Look in the nested folders for other necessary model/resource files
    for directory_path, _, file_names in os.walk('.'):
        for f in file_names:
            if directory_path != '.':
                files_set.add(os.path.join(directory_path, f))

    return files_set


def create_manifest_file(model_path, manifest):
    mar_inf_path = os.path.join(model_path, MAR_INF)
    if not os.path.exists(mar_inf_path):
        os.makedirs(mar_inf_path)

    with open(os.path.join(mar_inf_path, MANIFEST_FILE_NAME), 'w') as m:
        json.dump(manifest, m, indent=4)


def zip_dir(path, ziph):
    for root, _, files in os.walk(path):
        for f in files:
            ziph.write(os.path.join(root, f))


def clean_temp_files(temp_files):
    for f in temp_files:
        os.remove(f)


def export_model(model_name, model_path, manifest, export_file=None):
    """
    Internal helper for the exporting model command line interface.
    """
    temp_files = []
    try:
        if export_file is None:
            export_file = '{}/{}.{}'.format(os.getcwd(), model_name, MODEL_ARCHIVE_EXTENSION)
        assert not os.path.exists(export_file), "model file {} already exists.".format(export_file)

        if model_path.startswith('~'):
            model_path = os.path.expanduser(model_path)
        # Entry point model here, this is in the main folder
        tmp = os.getcwd()
        files_set = get_files_from_model_path(model_path)

        onnx_file = find_unique(files_set, '.onnx')
        files_list = list(files_set)

        os.chdir(tmp)

        if onnx_file is not None:
            symbol_file, params_file = convert_onnx_model(model_path, onnx_file)
            files_list.remove(onnx_file)
            temp_files.append(os.path.join(model_path, symbol_file))
            temp_files.append(os.path.join(model_path, params_file))

        create_manifest_file(model_path, manifest)
        temp_files.append(os.path.join(model_path, MAR_INF, MANIFEST_FILE_NAME))

        with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as z:
            zip_dir(model_path, z)
        log_msg.info("Successfully exported model %s to file %s", model_name, export_file)

    finally:
        clean_temp_files(temp_files)


def export():
    """
    Export as MXNet model
    :return:
    """
    args = ArgParser.export_model_args_parser().parse_args()
    # TODO : Add CLI args to the parser
    manifest = generate_manifest(args)
    export_model(model_name=args.model.model_name, model_path=args.model_path, manifest=manifest)


if __name__ == '__main__':
    export()
