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
This module parses the arguments given through the mxnet-model-server command-line. This is used by model-server
at runtime.
"""

import argparse
import os
from .manifest_components.manifest import RuntimeType


# noinspection PyTypeChecker
class ArgParser(object):

    """
    Argument parser for model-export-tool commands
    More detailed example is available at https://github.com/awslabs/mxnet-model-server/blob/master/README.md
    """

    @staticmethod
    def export_model_args_parser():

        """ Argument parser for mxnet-model-export
        """
        # TODO Add more CLI args here later
        runtime_types = ', '.join(s.value for s in RuntimeType)

        parser_export = argparse.ArgumentParser(prog='model-archiver', description='Model Archiver Tool')

        parser_export.add_argument('--model-name',
                                   required=True,
                                   type=str,
                                   default=None,
                                   help='Exported model name. Exported file will be named as '
                                        'model-name.mar and saved in current working directory if no --export-path is '
                                        'specified, else it will be saved under the export path')

        parser_export.add_argument('--model-path',
                                   required=True,
                                   type=str,
                                   default=None,
                                   help='Path to the folder containing model related files.')

        parser_export.add_argument('--handler',
                                   required=True,
                                   dest="handler",
                                   type=str,
                                   default=None,
                                   help='Handler path to handle custom MMS inference logic.')

        parser_export.add_argument('--runtime',
                                   required=False,
                                   type=str,
                                   default=RuntimeType.PYTHON.value,
                                   choices=[s.value for s in RuntimeType],
                                   help='The runtime specifies which language to run your inference code on. '
                                        'The default runtime is {}. At the present moment we support the '
                                        'following runtimes \n {}'.format(RuntimeType.PYTHON, runtime_types))

        parser_export.add_argument('--export-path',
                                   required=False,
                                   type=str,
                                   default=os.getcwd(),
                                   help='Path where the exported .mar file will be saved. This is an optional '
                                        'parameter. If --export-path is not specified, the file will be saved in the '
                                        'current working directory. ')

        parser_export.add_argument('-f', '--force',
                                   required=False,
                                   action='store_true',
                                   help='When the -f or --force flag is specified, an existing .mar file with same '
                                        'name as that provided in --model-name in the path specified by --export-path '
                                        'will overwritten')

        return parser_export
