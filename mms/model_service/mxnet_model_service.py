# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""`MXNetBaseService` defines an API for MXNet service.
"""

import mxnet as mx
import requests
import zipfile
import json
import shutil
import os

from mxnet.io import DataBatch
from mms.log import get_logger
from mms.model_service.model_service import SingleNodeService, URL_PREFIX


logger = get_logger()


def check_input_shape(inputs, signature):
    '''Check input data shape consistency with signature.

    Parameters
    ----------
    inputs : List of NDArray
        Input data in NDArray format.
    signature : dict
        Dictionary containing model signature.
    '''
    assert isinstance(inputs, list), 'Input data must be a list.'
    assert len(inputs) == len(signature['inputs']), 'Input number mismatches with ' \
                                           'signature. %d expected but got %d.' \
                                           % (len(signature['inputs']), len(inputs))
    for input, sig_input in zip(inputs, signature['inputs']):
        assert isinstance(input, mx.nd.NDArray), 'Each input must be NDArray.'
        assert len(input.shape) == \
               len(sig_input['data_shape']), 'Shape dimension of input %s mismatches with ' \
                                'signature. %d expected but got %d.' \
                                % (sig_input['data_name'], len(sig_input['data_shape']),
                                   len(input.shape))
        for idx in range(len(input.shape)):
            if idx != 0 and sig_input['data_shape'][idx] != 0:
                assert sig_input['data_shape'][idx] == \
                       input.shape[idx], 'Input %s has different shape with ' \
                                         'signature. %s expected but got %s.' \
                                         % (sig_input['data_name'], sig_input['data_shape'],
                                            input.shape)

class MXNetBaseService(SingleNodeService):
    '''MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    '''
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        signature_file_path = os.path.join(model_dir, manifest['Model']['Signature'])
        if not os.path.isfile(signature_file_path):
            raise RuntimeError('Signature file is not found. Please put signature.json '
                               'into the model file directory...' + signature_file_path)
        try:
            signature_file = open(signature_file_path)
            self._signature = json.load(signature_file)
        except:
            raise Exception('Failed to open model signiture file: %s' % signature_file_path)

        data_names = []
        data_shapes = []
        for input in self._signature['inputs']:
            data_names.append(input['data_name'])
            # Replace 0 entry in data shape with 1 for binding executor.
            # Set batch size as 1
            data_shape = input['data_shape']
            data_shape[0] = 1
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1
            data_shapes.append((input['data_name'], tuple(data_shape)))
        
        # Load MXNet module
        epoch = 0
        try:
            param_filename = manifest['Model']['Parameters']
            epoch = int(param_filename[len(model_name) + 1: -len('.params')])
        except Exception as e:
            logger.warn('Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, manifest['Model']['Symbol'][:-12]), epoch)
        self.mx_model = mx.mod.Module(symbol=sym, context=self.ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

        # Read synset file
        # If synset is not specified, check whether model archive contains synset file.
        archive_synset = os.path.join(model_dir, 'synset.txt')

        if os.path.isfile(archive_synset):
            synset = archive_synset
            self.labels = [line.strip() for line in open(synset).readlines()]

    def _preprocess(self, data):
        return map(mx.nd.array, data)

    def _postprocess(self, data):
        return [str(d.asnumpy().tolist()) for d in data]

    def _inference(self, data):
        '''Internal inference methods for MXNet. Run forward computation and
        return output.

        Parameters
        ----------
        data : list of NDArray
            Preprocessed inputs in NDArray format.

        Returns
        -------
        list of NDArray
            Inference output.
        '''
        # Check input shape
        check_input_shape(data, self.signature)
        data = [item.as_in_context(self.ctx) for item in data]
        self.mx_model.forward(DataBatch(data))
        return self.mx_model.get_outputs()

    def ping(self):
        '''Ping to get system's health.

        Returns
        -------
        String
            MXNet version to show system is healthy.
        '''
        return mx.__version__

    @property
    def signature(self):
        '''Signiture for model service.

        Returns
        -------
        Dict
            Model service signiture.
        '''
        return self._signature

    

