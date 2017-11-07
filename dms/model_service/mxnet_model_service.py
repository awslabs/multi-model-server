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
from dms.log import get_logger
from dms.model_service.model_service import SingleNodeService, URL_PREFIX


logger = get_logger(__name__)
SIGNATURE_FILE = 'signature.json'

def download(url, path=None, overwrite=False):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    if path is None:
        fname = url.split('/')[-1]
    elif os.path.isdir(path):
        fname = os.path.join(path, url.split('/')[-1])
    else:
        fname = path

    if overwrite or not os.path.exists(fname):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open("%s.temp" % (fname), 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        os.rename("%s.temp" % (fname), fname)
    return fname

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

def _extract_zip(zip_file, destination):
    '''Extract zip to destination without keeping directory structure

        Parameters
        ----------
        zip_file : str
            Path to zip file.
        destination : str
            Destination directory.
    '''
    with zipfile.ZipFile(zip_file) as file_buf:
        for item in file_buf.namelist():
            filename = os.path.basename(item)
            # skip directories
            if not filename:
                continue

            # copy file (taken from zipfile's extract)
            source = file_buf.open(item)
            target = open(os.path.join(destination, filename), 'wb')
            with source, target:
                shutil.copyfileobj(source, target)


class MXNetBaseService(SingleNodeService):
    '''MXNetBaseService defines the fundamental loading model and inference
       operations when serving MXNet model. This is a base class and needs to be
       inherited.
    '''
    def __init__(self, service_name, path, gpu=None):
        self.ctx = mx.gpu(int(gpu)) if gpu is not None else mx.cpu()
        model_dir, model_name = self._extract_model(service_name, path)

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
            param_filename = filter(lambda file: file.endswith('.params'), os.listdir(model_dir))[0]
            epoch = int(param_filename[len(model_name) + 1: -len('.params')])
        except Exception as e:
            logger.warn('Failed to parse epoch from param file, setting epoch to 0')

        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, model_name), epoch)
        self.mx_model = mx.mod.Module(symbol=sym, context=self.ctx,
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

        # Read synset file
        # If synset is not specified, check whether model archive contains synset file.
        archive_synset = '%s/synset.txt' % (model_dir)
        if os.path.isfile(archive_synset):
            synset = archive_synset
            self.labels = [line.strip() for line in open(synset).readlines()]

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

    def _extract_model(self, service_name, path, check_multi_sym=True):
        curr_dir = os.getcwd()
        model_file = download(url=path, path='%s/%s.model' % (curr_dir, service_name), overwrite=True) \
            if path.lower().startswith(URL_PREFIX) else path

        model_file = os.path.abspath(model_file)
        model_file_prefix = os.path.splitext(os.path.basename(model_file))[0]
        model_dir = os.path.join(os.path.dirname(model_file), model_file_prefix )
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        try:
            _extract_zip(model_file, model_dir)
        except Exception as e:
            raise Exception('Failed to open model file %s for model %s. Stacktrace: %s'
                            % (model_file, model_file_prefix , e))

        symbol_file_postfix = '-symbol.json'
        symbol_file_num = 0
        model_name = ''
        for dirpath, _, filenames in os.walk(model_dir):
            for file_name in filenames:
                if file_name.endswith(symbol_file_postfix):
                    symbol_file_num += 1
                    model_name = file_name[:-len(symbol_file_postfix)]
        if check_multi_sym:
            assert symbol_file_num == 1, "Exported model file should have exactly one MXNet " \
                                         "symbol json file. Otherwise you need to override " \
                                         "__init__ method in service class."

        signature_file_path = os.path.join(model_dir, SIGNATURE_FILE)
        if not os.path.isfile(signature_file_path):
            raise RuntimeError('Signature file is not found. Please put signature.json '
                               'into the model file directory...' + signature_file_path)
        try:
            signature_file = open(signature_file_path)
            self._signature = json.load(signature_file)
        except:
            raise Exception('Failed to open model signiture file: %s' % signature_file_path)

        return model_dir, model_name


