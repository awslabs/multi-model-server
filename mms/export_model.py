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

import warnings
from model_server_tools.model_packaging import export_model as export_tool


def export():
    """
    Export as MXNet model
    :return:
    """
    warnings.warn("Use model-export-tool instead of mxnet-model-export. mxnet-model-export is deprecated.",
                  DeprecationWarning, stacklevel=2)
    export_tool.export()


if __name__ == '__main__':
    export()
