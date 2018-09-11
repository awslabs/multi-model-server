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
Setup.py for the model server package
"""

import platform
import os
import ctypes
import errno
from datetime import date
import sys
from shutil import copyfile, rmtree
import subprocess
from setuptools import setup, find_packages, Command
import setuptools.command.build_py

pkgs = find_packages()
source_server_file = os.path.abspath('frontend/server/build/libs/server-1.0.jar')
dest_file_name = os.path.abspath('mms/frontend/model-server.jar')


def pypi_description():
    """Imports the long description for the project page"""
    with open('PyPiDescription.rst') as df:
        return df.read()


def detect_model_server_version():
    with open(os.path.abspath(os.path.join("mms", "version.py")), 'r') as vf:
        exec(vf.read(), None, globals())
        # TODO: Look to remove this exec and version coming from file
    return __version__


class BuildFrontEnd(Command):
    """
    Class defined to run custom commands. In this class we build the frontend if it hasn't been built in the last
    5 mins. This is to avoid the accidentally publishing old binaries.
    """
    description = 'Build Model Server Frontend'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """
        Actual method called to run the build command
        :return:
        """
        front_end_bin_dir = os.path.abspath('.') + '/mms/frontend'
        try:
            os.mkdir(front_end_bin_dir)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(front_end_bin_dir):
                pass
            else:
                raise

        if os.path.exists(source_server_file):
            os.remove(source_server_file)

        cwd = os.getcwd()
        # Remove build/lib directory.
        if os.path.exists(os.path.abspath('./build/lib/')):
            rmtree(os.path.abspath('./build/lib/'))
        os.chdir(os.path.abspath('./frontend/'))
        try:
            subprocess.check_call('./gradlew build', shell=True)
        except OSError:
            assert 0, "build failed"
        os.chdir(cwd)
        copyfile(source_server_file, dest_file_name)


class BuildPy(setuptools.command.build_py.build_py):
    """
    Class to invoke the custom command defined above.
    """

    def run(self):
        sys.stderr.flush()
        self.run_command('build_frontend')
        setuptools.command.build_py.build_py.run(self)


if __name__ == '__main__':
    opt_set = set(sys.argv)
    version = detect_model_server_version()

    requirements = ['Pillow', 'psutil', 'future', 'model-archiver']

    setup(
        name='mxnet-model-server',
        version=version.strip() + 'b'+ str(date.today()).replace('-', ''),
        description='Model Server for Apache MXNet is a tool for serving neural net models for inference',
        long_description=pypi_description(),
        url='https://github.com/awslabs/mxnet-model-server',
        keywords='MXNet Model Server Serving Deep Learning Inference AI',
        packages=pkgs,
        cmdclass={
            'build_frontend': BuildFrontEnd,
            'build_py': BuildPy,
        },
        install_requires=requirements,
        extras_require={
            'mxnet-mkl': ['mxnet-mkl==1.2.0'],
            'mxnet-cu90mkl': ['mxnet-cu90mkl==1.2.0'],
            'mxnet': ['mxnet==1.2'],
        },
        entry_points={
            'console_scripts': ['mxnet-model-server=mms.model_server:start',
                                'mxnet-model-export=mms.export_model:main']
        },
        include_package_data=True,
        license='Apache License Version 2.0'
    )
