# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import platform
import os
import ctypes
from setuptools import setup, find_packages

def PyPiDescription():
    """Imports the long description for the project page"""
    with open('PyPiDescription.rst') as f:
        return f.read()


pkgs = find_packages()
pkgs.append('tools')

# To build and upload a new version, follow the steps below.
# Notes:
# - this is a "Universal Wheels" package that is pure Python and supports both Python2 and Python3
# - Twine is a secure PyPi upload package
# - Make sure you have bumped the version! at mms/version.py
# $ pip install twine
# $ pip install wheel
# $ python setup.py bdist_wheel --universal
# $ twine upload dist/*

with open(os.path.join("mms", "version.py")) as f:
    exec(f.read())

requirements = ['Flask', 'Pillow', 'requests', 'flask-cors',
                'psutil', 'jsonschema', 'onnx==1.1.1', 'boto3', 'importlib2',
                'fasteners']
# Enable Cu90 only when using linux with cuda enabled
gpu_platform = False
if platform.system().lower() == 'linux':
    try:
        # Check if CUDA is installed
        cuda = ctypes.cdll.LoadLibrary('libcudart.so')
        deviceCount = ctypes.c_int()
        # get the number of supported GpUs
        cuda.cudaGetDeviceCount(ctypes.byref(deviceCount))
        if deviceCount.value > 0:
            gpu_platform = True
    except Exception as e:
        gpu_platform = False
if gpu_platform:
    requirements = ['mxnet-cu90mkl>=1.2'] + requirements
else:
    requirements = ['mxnet-mkl>=1.2'] + requirements
setup(
    name='mxnet-model-server',
    version=__version__.strip(),
    description='Model Server for Apache MXNet is a tool for serving neural net models for inference',
    long_description=PyPiDescription(),
    url='https://github.com/awslabs/mxnet-model-server',
    keywords='MXNet Model Server Serving Deep Learning Inference AI',
    packages=pkgs,
    install_requires=requirements,
    entry_points={
        'console_scripts': ['mxnet-model-server=mms.mxnet_model_server:start_serving',
                            'mxnet-model-export=mms.export_model:export']
    },
    include_package_data=True,
    license='Apache License Version 2.0'
)
