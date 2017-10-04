import mxnet as mx
import numpy as np
import zipfile
import json
import shutil
import os

from mxnet.gluon.utils import download
from mxnet.io import DataBatch
from mms.model_service.model_service import SingleNodeService, URL_PREFIX

SIGNATURE_FILE = 'signature.json'


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

def _extrac_zip(zip_file, destination):
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
    def __init__(self, path, synset=None, ctx=mx.cpu()):
        super(MXNetBaseService, self).__init__(path, ctx)
        model_dir, model_name = self._extract_model(path)

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
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (model_dir, model_name), 0)
        self.mx_model = mx.mod.Module(symbol=sym, context=mx.cpu(),
                                      data_names=data_names, label_names=None)
        self.mx_model.bind(for_training=False, data_shapes=data_shapes)
        self.mx_model.set_params(arg_params, aux_params, allow_missing=True)

        # Read synset file
        # If synset is not specified, check whether model archive contains synset file.
        archive_synset = '%s/synset.txt' % (model_dir)
        if synset is None and os.path.isfile(archive_synset):
            synset = archive_synset
        if synset:
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

    def _extract_model(self, path):
        curr_dir = os.getcwd()
        model_file = download(url=path, path=curr_dir) \
            if path.lower().startswith(URL_PREFIX) else path

        model_file = os.path.abspath(model_file)
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        model_dir = os.path.join(os.path.dirname(model_file), model_name)
        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)
        try:
            _extrac_zip(model_file, model_dir)
        except Exception as e:
            raise Exception('Failed to open model file %s for model %s. Stacktrace: %s'
                            % (model_file, model_name, e))

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


