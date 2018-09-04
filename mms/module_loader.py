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
Handle loading service module file
"""

import os
import sys


def load_service(path, name=None):
    """
    Load the model-service into memory and associate it with each flask app worker
    :param path:
    :param name:
    :return:
    """
    # If requirements file is given is custom service file path
    requirements = os.path.join(path.rsplit('/', 1)[0], 'requirements.txt')
    if os.path.exists(requirements):
        from pip._internal import main as pipmain
        try:
            with open(requirements, 'r') as req_file:
                for req in req_file.readlines():
                    pipmain(['install', req])
        except Exception as e:
            exc_tb = sys.exc_info()[2]
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            raise Exception(
                'Error when installing requirements file: {} \n {}:{}:{}'.format(path, fname, exc_tb.tb_lineno, e))
    try:
        if not name:
            name = os.path.splitext(os.path.basename(path))[0]
        module = None
        if sys.version_info[0] > 2:
            import importlib
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

        else:
            import imp
            module = imp.load_source(name, path)

        return module
    except Exception as e:
        exc_tb = sys.exc_info()[2]
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        raise Exception('Error when loading service file: {} \n {}:{}:{}'.format(path, fname, exc_tb.tb_lineno, e))
