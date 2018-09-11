# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# To build and upload a new version, follow the steps below.
# Notes:
# - this is a "Universal Wheels" package that is pure Python and supports both Python2 and Python3
# - Twine is a secure PyPi upload package
# - Make sure you have bumped the version! at mms/version.py
# $ pip install twine
# $ pip install wheel
# $ python setup.py bdist_wheel --universal

# *** TEST YOUR PACKAGE WITH TEST PI ******
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# If this is successful then push it to actual pypi

# $ twine upload dist/*

"""
Setup.py for the model export tool package
"""

from datetime import date
import sys
from setuptools import setup, find_packages

pkgs = find_packages()


def pypi_description():
    """Imports the long description for the project page"""
    with open('PyPiDescription.rst') as df:
        return df.read()


def detect_packaging_tool_version():
    # pylint: disable = relative-import
    from version import __version__
    return __version__


if __name__ == '__main__':
    opt_set = set(sys.argv)
    version = detect_packaging_tool_version()

    requirements = ['future', 'enum34']

    setup(
        name='model-archiver',
        version=version.strip() + 'b' + str(date.today()).replace('-', '') + '2',
        description='Model Export Tool is used for creating archives of trained neural net models that can be consumed '
                    'by MXNet-Model-Server inference',
        long_description=pypi_description(),
        url='https://github.com/awslabs/mxnet-model-server/model_export/',
        keywords='MXNet Model Server Serving Deep Learning Inference AI',
        packages=pkgs,
        install_requires=requirements,
        extras_require={
            'mxnet-mkl': ['mxnet-mkl==1.2.0'],
            'mxnet-cu90mkl': ['mxnet-cu90mkl==1.2.0'],
            'mxnet': ['mxnet==1.2'],
            'onnx': ['onnx==1.1.1']
        },
        entry_points={
            'console_scripts': ['model-archiver=model_archiver.model_packaging:generate_model_archive']
        },
        include_package_data=True,
        license='Apache License Version 2.0'
    )
