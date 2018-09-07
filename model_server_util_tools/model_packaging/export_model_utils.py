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
Helper utils for Model Export tool
"""

import os
import json
import zipfile
import sys
import logging
from model_server_util_tools.model_packaging.manifest_components.publisher import Publisher
from model_server_util_tools.model_packaging.manifest_components.manifest import Manifest
from model_server_util_tools.model_packaging.manifest_components.engine import Engine
from model_server_util_tools.model_packaging.manifest_components.model import Model
from model_server_util_tools import model_packaging

MODEL_ARCHIVE_EXTENSION = '.mar'
MODEL_SERVER_VERSION = '1.0'
MODEL_ARCHIVE_VERSION = model_packaging.__version__
MANIFEST_FILE_NAME = 'MANIFEST.json'
MAR_INF = 'MAR-INF'
ONNX_TYPE = '.onnx'


class ModelExportUtils(object):
    """
    Helper utils for Model Export tool.
    This class lists out all the methods such as validations for model archiving, ONNX model checking etc.
    This is to keep the code in export_model.py clean and simple.
    """

    @staticmethod
    def check_mar_already_exists(model_name, export_file, overwrite):
        """
        Function to check if .mar already exists
        :param model_name:
        :param export_file:
        :param model_archive_extension:
        :return:
        """
        if export_file is None:
            export_file = os.path.join(os.getcwd(), '{}{}'.format(model_name, MODEL_ARCHIVE_EXTENSION))

        if os.path.exists(export_file):
            if overwrite:
                logging.warn("%s already exists. It will be overwritten since --force/-f was specified", export_file)

            else:
                logging.error("%s already exists. Since no --force/-f was specified, it will not be overwritten. "
                              "Exiting the program here. Specify --force/-f flag to overwrite the %s file. "
                              "See -h/--help for more details", export_file, export_file)

                sys.exit(1)

        return export_file

    @staticmethod
    def check_custom_model_types(model_path):
        """
        This functions checks whether any special handling is required for custom model extensions such as
        .onnx, or in the future, for Tensorflow and PyTorch extensions.
        :param model_path:
        :return:
        """
        temp_files = [] # List of temp files added to handle custom models
        files_to_exclude = [] # List of files to be excluded from .mar packaging.

        files_set = set(os.listdir(model_path))
        onnx_file = ModelExportUtils.find_unique(files_set, ONNX_TYPE)
        if onnx_file is not None:
            symbol_file, params_file = ModelExportUtils.convert_onnx_model(model_path, onnx_file)
            files_to_exclude.append(onnx_file)
            temp_files.append(os.path.join(model_path, symbol_file))
            temp_files.append(os.path.join(model_path, params_file))

        # More cases will go here as an if-else block

        return temp_files, files_to_exclude

    @staticmethod
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

            message = "model-export-tool expects only one %s file. Please supply the single %s file you wish to " \
                      "export." % (params[0], params[1])

            logging.error(message)
            sys.exit(1)

    @staticmethod
    def convert_onnx_model(model_path, onnx_file):
        """
        Util to convert onnx model to MXNet model
        :param model_path:
        :param onnx_file:
        :return:
        """
        try:
            import mxnet as mx
        except ImportError:
            logging.error("MXNet package is not installed. Run command : pip install mxnet to install it. ")
            sys.exit(1)

        try:
            import onnx
        except ImportError:
            logging.error("Onnx package is not installed. Run command : pip install mxnet to install it. ")
            sys.exit(1)

        from mxnet.contrib import onnx as onnx_mxnet
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

    @staticmethod
    def generate_publisher(publisherargs):
        publisher = Publisher(author=publisherargs.author, email=publisherargs.email)
        return publisher

    @staticmethod
    def generate_engine(engineargs):
        engine = Engine(engine_name=engineargs.engine)
        return engine

    @staticmethod
    def generate_model(modelargs):
        model = Model(model_name=modelargs.model_name, handler=modelargs.handler)
        return model

    @staticmethod
    def generate_manifest_json(args):
        """
        Function to generate manifest as a json string from the inputs provided by the user in the command line
        :param args:
        :return:
        """

        publisher = ModelExportUtils.generate_publisher(args) if 'publisher' in args else None

        engine = ModelExportUtils.generate_engine(args)

        model = ModelExportUtils.generate_model(args)

        manifest = Manifest(runtime=args.runtime, engine=engine, model=model, publisher=publisher)

        return str(manifest)

    @staticmethod
    def create_manifest_file(model_path, manifest):
        """
        Function creates a file called manifest.json under model_path/MAR-INF folder

        :param model_path:
        :param manifest:
        :return:
        """
        mar_inf_path = os.path.join(model_path, MAR_INF)

        if not os.path.exists(mar_inf_path):
            os.makedirs(mar_inf_path)

        manifest_path = os.path.join(mar_inf_path, MANIFEST_FILE_NAME)

        with open(manifest_path, 'w') as m:
            json.dump(manifest, m, indent=4)

        return manifest_path

    @staticmethod
    def clean_temp_files(temp_files):
        for f in temp_files:
            os.remove(f)

    @staticmethod
    def zip(export_file, model_path, files_to_exclude):
        with zipfile.ZipFile(export_file, 'w', zipfile.ZIP_DEFLATED) as z:
            ModelExportUtils.zip_dir(model_path, z, set(files_to_exclude))

    @staticmethod
    def zip_dir(path, ziph, files_to_exclude):
        for root, _, files in os.walk(path):
            for f in files:
                if f not in files_to_exclude:
                    ziph.write(os.path.join(root, f))
