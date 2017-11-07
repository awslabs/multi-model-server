# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from setuptools import setup, find_packages

pkgs = find_packages()
pkgs.append('tools')

setup(
    name='deep-model-server',
    version='0.1.1',
    description='Deep Model Server is a tool for deploying neural net models for inference',
    url='https://github.com/deep-learning-tools/deep-model-server',
    keywords='MXNet Model Serving Deep Learning Inference',
    packages=pkgs,
    install_requires=['mxnet>=0.11.0', 'Flask', 'Pillow', 'requests', 'flask-cors', 'psutil'],
    entry_points={
        'console_scripts':['deep-model-server=dms.deep_model_server:start_serving', 'deep-model-export=dms.export_model:export']
    },
    include_package_data=True,
    license='Apache License Version 2.0'
)