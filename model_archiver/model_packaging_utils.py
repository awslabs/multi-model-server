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
import re
from .manifest_components.publisher import Publisher
from .manifest_components.manifest import Manifest
from .manifest_components.engine import Engine
from .manifest_components.model import Model

MODEL_ARCHIVE_EXTENSION = '.mar'
MODEL_SERVER_VERSION = '1.0'
MODEL_ARCHIVE_VERSION = '1.0'
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
    def check_mar_already_exists(model_name, export_file_path, overwrite):
        """
        Function to check if .mar already exists
        :param model_name:
        :param export_file_path:
        :param overwrite:
        :return:
        """
        if export_file_path is None:
            export_file_path = os.getcwd()

        export_file = os.path.join(export_file_path, '{}{}'.format(model_name, MODEL_ARCHIVE_EXTENSION))

        if os.path.exists(export_file):
            if overwrite:
                logging.warning("%s already exists. It will be overwritten since --force/-f was specified", export_file)

            else:
                logging.error("%s already exists. Since no --force/-f was specified, it will not be overwritten. "
                              "Exiting the program here. Specify --force/-f flag to overwrite the %s file. "
                              "See -h/--help for more details", export_file, export_file)

                sys.exit(1)

        return export_file_path

    @staticmethod
    def check_custom_model_types(model_path):
        """
        This functions checks whether any special handling is required for custom model extensions such as
        .onnx, or in the future, for Tensorflow and PyTorch extensions.
        :param model_path:
        :return:
        """
        temp_files = []  # List of temp files added to handle custom models
        files_to_exclude = []  # List of files to be excluded from .mar packaging.

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
                '.onnx': ('.onnx', 'ONNX model file')
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
        arg_dict = vars(args)

        publisher = ModelExportUtils.generate_publisher(args) if 'author' in arg_dict and 'email' in arg_dict else None

        engine = ModelExportUtils.generate_engine(args) if 'engine' in arg_dict else None

        model = ModelExportUtils.generate_model(args)

        manifest = Manifest(runtime=args.runtime, model=model, engine=engine, publisher=publisher)

        return str(manifest)

    @staticmethod
    def clean_temp_files(temp_files):
        for f in temp_files:
            os.remove(f)

    @staticmethod
    def zip(export_file, model_name, model_path, files_to_exclude, manifest):
        mar_path = os.path.join(export_file, '{}{}'.format(model_name, MODEL_ARCHIVE_EXTENSION))
        with zipfile.ZipFile(mar_path, 'w', zipfile.ZIP_DEFLATED) as z:
            ModelExportUtils.zip_dir(model_path, z, set(files_to_exclude))
            # Write the manifest here now as a json
            z.writestr(os.path.join(MAR_INF, MANIFEST_FILE_NAME), bytes=bytes(json.dumps(manifest, indent=4)))

    @staticmethod
    def zip_dir(path, ziph, files_to_exclude):

        """
        This method zips the dir and filters out some files based on a expression
        :param path:
        :param ziph:
        :param files_to_exclude:
        :return:
        """
        unwanted_dirs = {'__MACOSX', '__pycache__', 'MANIFEST.json'}

        for root, directories, files in os.walk(path):
            # Filter directories
            directories[:] = [d for d in directories if ModelExportUtils.directory_filter(d, unwanted_dirs)]
            # Filter files
            files[:] = [f for f in files if ModelExportUtils.file_filter(f, files_to_exclude)]
            for f in files:
                ziph.write(os.path.join(root, f), f)

    @staticmethod
    def directory_filter(directory, unwanted_dirs):

        """
        This method weeds out unwanted hidden directories from the model archive .mar file
        :param directory:
        :param unwanted_dirs:
        :return:
        """
        if directory in unwanted_dirs:
            return False
        if directory.startswith('.'):
            return False

        return True

    @staticmethod
    def file_filter(current_file, files_to_exclude):

        """
        This method weeds out unwanted files
        :param current_file:
        :param files_to_exclude:
        :return:
        """

        if current_file in files_to_exclude:
            return False

        elif current_file.endswith(('.pyc', '.DS_Store')):
            return False

        return True

    @staticmethod
    def check_model_name_regex_or_exit(model_name):

        """
        Method checks whether model name passes regex filter.
        If the regex Filter fails, the method exits.
        :param model_name:
        :return:
        """
        pattern = re.compile(r'[A-Za-z][A-Za-z0-9_\-.]+')
        if pattern.match(model_name) is None:

            logging.error("Model name contains special characters. The allowed regular expression filter for model "
                          "name is %s ", r'[A-Za-z][A-Za-z0-9_\-.]+')
            sys.exit(1)
