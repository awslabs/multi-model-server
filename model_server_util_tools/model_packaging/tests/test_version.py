# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import model_server_util_tools.model_packaging


def test_model_export_tool_version():
    with (open(os.path.join('model_server_util_tools', 'model_packaging', 'version.py'))) as f:
        exec(f.read(), globals())

    assert __version__ == str(model_server_util_tools.model_packaging.__version__), "Versions do not match"

