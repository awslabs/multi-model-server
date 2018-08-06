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
Download and extract the model archivefiles
"""
import os
import json

from mms.log import get_logger

logger = get_logger()

MANIFEST_FILENAME = 'MANIFEST.json'


class ModelLoader(object):
    """
    Model Loader
    """
    @staticmethod
    def load(model_dir, handler):
        """
        Load models

        Parameters
        ----------
        model_dir : str
            Model path
        handler : str
            Model entry point service handler file name

        Returns
        ----------
        map
            Model manifest
        """
        manifest = None
        manifest_file = os.path.join(model_dir, MANIFEST_FILENAME)
        if os.path.isfile(manifest_file):
            try:
                manifest = json.load(open(manifest_file))
            except Exception as e:
                raise Exception('Failed to open manifest file. Stacktrace: ' + str(e))
            model = manifest['model']
            engine = manifest['engine']
            if engine['engineName'] == 'MxNet':
                # symbol, parameters are required for MxNet
                parameter_file = model.get('parametersFile')
                if not parameter_file:
                    raise Exception("parameterFile not defined in MANIFEST.json.")
                if not os.path.isfile(model_dir + '/' + parameter_file):
                    raise Exception("parameterFile not found: {}.".format(parameter_file))

                symbol_file = model.get('symbolFile')
                if not symbol_file:
                    raise Exception("symbolFile not defined in MANIFEST.json.")
                if not os.path.isfile(model_dir + '/' + symbol_file):
                    raise Exception("symbolFile not found: {}.".format(symbol_file))

        if handler is None:
            raise Exception('No handler is provided.')

        handler_file = os.path.join(model_dir, handler)
        if not os.path.isfile(handler_file):
            # TODO: search PYTHONPATH and MODELPATH for handler file
            raise Exception("handler file not not found: {}.".format(handler_file))

        return manifest, handler_file
