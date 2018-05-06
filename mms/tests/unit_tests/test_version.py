# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mms
import os
import re


def test_mms_version():
    with open(os.path.join("../../mms", "version.py")) as f:
        version = f.read()
        version_re = r"^__version__ = ([^'\"]*)(?:\n\s*)"
        version_groups = re.search(version_re, version, re.M)
        if version_groups:
            ver = version_groups.group(1)
        else:
            assert 0, "Version file not found"

        assert ver.strip() == str(mms.__version__), "Versions don't match"
