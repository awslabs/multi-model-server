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

from model_export import export_model as export_tool
from model_export.arg_parser import ArgParser
import warnings


def export_model(model_name, model_path, service_file=None, export_file=None):
    """
    Internal helper for the exporting model command line interface.
    """
    # print ("Use model-export-tool instead of mxnet-model-export. mxnet-model-export is deprecated")
    warnings.warn("Use model-export-tool instead of mxnet-model-export. mxnet-model-export is deprecated.")
    export_tool.export_model(model_name, model_path, service_file, export_file)


def export():
    """
    Export as MXNet model
    :return:
    """
    args = ArgParser.export_model_args_parser().parse_args()
    export_model(model_name=args.model_name, model_path=args.model_path, service_file=args.service_file_path)


if __name__ == '__main__':
    export()
