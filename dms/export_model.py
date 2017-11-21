# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

'''Command line interface to export model files to be used for inference by Deep Model Server
'''

import os
import logging
import json
import zipfile
import mxnet as mx
from dms.arg_parser import ArgParser

try:
    basestring
except NameError:
    basestring = str


SIG_REQ_ENTRY = ['inputs', 'input_type', 'outputs', 'output_types']
VALID_MIME_TYPE = ['image/jpeg', 'application/json']

def _check_signature(model_path):
    '''Internal helper to check signature error when exporting model with CLI.
    '''
    sig_file = '%s/signature.json' % (model_path)
    assert os.path.isfile(sig_file), \
        "signature.json is not found in %s." % (model_path)
    with open(sig_file) as js_file:
        signature = json.load(js_file)

    assert 'input_type' in signature and 'output_type' in signature, \
        'input_type and output_type are required in signature.'
    assert isinstance(signature['input_type'], basestring) and \
           isinstance(signature['output_type'], basestring), \
        'Value of input_type and output_type should be string'
    assert signature['input_type'] in VALID_MIME_TYPE and \
           signature['output_type'] in VALID_MIME_TYPE, \
        'Valid type should be picked from %s. ' \
        'Got %s for input and %s for output' % \
        (VALID_MIME_TYPE, signature['input_type'], signature['output_type'])

    assert 'inputs' in signature and 'outputs' in signature, \
        'inputs and outputs are required in signature.'
    assert isinstance(signature['inputs'], list) and \
           isinstance(signature['outputs'], list), \
        'inputs and outputs values must be list.'
    for input in signature['inputs']:
        assert isinstance(input, dict), 'Each input must be a dictionary.'
        assert 'data_name' in input, 'data_name is required for input.'
        assert isinstance(input['data_name'], basestring), 'data_name value must be string.'
        assert 'data_shape' in input, 'data_shape is required for input.'
        assert isinstance(input['data_shape'], list), 'data_shape value must be list.'
    for output in signature['outputs']:
        assert isinstance(output, dict), 'Each output must be a dictionary.'
        assert 'data_name' in output, 'data_name is required for output.'
        assert isinstance(output['data_name'], basestring), 'data_name value must be string.'
        assert 'data_shape' in output, 'data_shape is required for output.'
        assert isinstance(output['data_shape'], list), 'data_shape value must be list.'


def _export_model(args):
    '''Internal helper for the exporting model command line interface.
    '''
    model_name = args.model_name
    model_path = args.model_path
    destination = os.getcwd()
    if model_path.startswith('~'):
        model_path = os.path.expanduser(model_path)
    _check_signature(model_path)

    symbol_file_postfix = '-symbol.json'
    symbol_file_num = 0
    file_list = []
    for dirpath, _, filenames in os.walk(model_path):
        for file_name in filenames:
            if file_name.endswith(symbol_file_postfix):
                symbol_file_num += 1
            file_list.append(os.path.join(dirpath, file_name))
    if symbol_file_num == 0:
        logging.warning("No MXNet model symbol json file is found. "
                        "You may need to manually load the model in your service class.")
    if symbol_file_num > 1:
        logging.warning("More than one model symbol json file was found. "
                        "You must manually load the model in your service class.")

    export_file = os.path.join(destination,'%s.model' % model_name)
    if os.path.isfile(export_file):
        raise RuntimeError("%s.model already exists in %s directory." % (model_name, destination))
    with zipfile.ZipFile(export_file, 'w') as zip_file:
        for item in file_list:
            zip_file.write(item, os.path.basename(item))
    print('Successfully exported %s model. Model file is located in %s directory.'
          % (model_name, destination))


def export_serving(model, filename, signature, export_path=None, aux_files=None):
    """Export a module object to a .model file to be used by Deep Model Server.

    Parameters
    ----------
    model : mx.mod.Module or mx.mod.BucketModule
        Module object to be exported.
    filename : str
        Prefix of exported model file.
    signature : dict
        A dictionary containing model input and output information.
        Data names or data shapes of inputs and outputs can be automatically
        collected from the module. They are optional. You need to specify the
        MIME type of the inputs and outputs for the http request. Currently only
        'image/jpeg' and 'application/json' are supported. All inputs should
        have the same type. This also applies for outputs.
        An example signature would be:
            signature = { "input_type": "image/jpeg", "output_type": "application/json" }
        A full signature containing inputs and outputs:
            {
                "input_type": "image/jpeg",
                "inputs" : [
                    {
                        "data_name": "data",
                        "data_shape": [1, 3, 224, 224]
                    }
                ],
                "outputs" : [
                    {
                        "data_name": "softmax",
                        "data_shape": [1, 1000]
                    }
                ],
                "output_type": "application/json"
            }
    export_path : str
        Destination path for export file. By default the model file
        is saved to current working directory.
    aux_files : List
        A list of string containing other utility files for inference.
        One example is class label file for classification task.

    Examples
    --------
    >>> model1 = mx.mod.Module(...)
    >>> signature1 = { "input_type": "image/jpeg", "output_type": "application/json" }
    >>> export_serving(model1, filename='resnet-18', signature=signature1,
    >>>                aux_files=['synset.txt'])
    >>>
    >>> model2 = mx.mod.Module(...)
    >>> signature2 = {
    >>>                  "input_type": "image/jpeg",
    >>>                  "inputs" : [
    >>>                      {
    >>>                          "data_name": "data",
    >>>                          "data_shape": [1, 3, 224, 224]
    >>>                      }
    >>>                  ],
    >>>                  "outputs" : [
    >>>                      {
    >>>                          "data_name": "softmax",
    >>>                          "data_shape": [1, 1000]
    >>>                      }
    >>>                  ]
    >>>                  "output_type": "application/json"
    >>>              }
    >>> export_serving(model2, filename='resnet-152', signature=signature2,
    >>>                aux_files=['synset.txt'])
    Exported model to "resnet-18.model"
    Exported model to "resnet-152.model"
    """
    assert issubclass(type(model), mx.mod.BaseModule) or issubclass(type(model), mx.gluon.Block), \
        "Model is type %s. It must be a subclass of mx.mod.BaseModule or mx.gluon.BLock." % (type(model))

    epoch_placeholder = 0
    destination = export_path or os.getcwd()
    if destination.startswith('~'):
        destination = os.path.expanduser(destination)
    sig_file = '%s/signature.json' % (destination)

    if issubclass(type(model), mx.mod.BaseModule):
        if 'inputs' not in signature:
            signature['inputs'] = list()
            for name, shape in model.data_shapes:
                signature['inputs'].append({
                    'data_name': name,
                    'data_shape': list(shape)
                })
        if 'outputs' not in signature:
            signature['outputs'] = list()
            for name, shape in model.output_shapes:
                signature['outputs'].append({
                    'data_name': name,
                    'data_shape': list(shape)
                })
        model.save_checkpoint('%s/%s' % (destination, filename), epoch_placeholder)
    else:
        assert 'inputs' in signature and 'outputs' in signature, \
            "Inputs and outputs information is required for gluon model signature."
        model.export('%s/%s' % (destination, filename))

    with open(sig_file, 'w') as sig:
        json.dump(signature, sig)
    _check_signature(destination)

    file_list = ['%s/%s-symbol.json' % (destination, filename), '%s/%s-%04d.params' %
                    (destination, filename, epoch_placeholder), sig_file]
    if aux_files:
        file_list += aux_files

    abs_model_path = os.path.join(destination,'%s.model' % filename)
    if os.path.isfile(abs_model_path):
        raise RuntimeError("%s already exists." % (abs_model_path))
    with zipfile.ZipFile(abs_model_path, 'w') as zip_file:
        for item in file_list:
            zip_file.write(item)
            os.remove(item)
    print('Exported model to %s/%s.model' %( destination, filename))


def export():
    args = ArgParser.export_parser().parse_args()
    _export_model(args)

if __name__ =='__main__':
    export()
