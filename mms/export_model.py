'''Command line interface to export MXNet model to be used for inference by MXNet Model Server
'''

import os
import logging
import json
import zipfile
import mxnet as mx
from arg_parser import ArgParser

SIG_REQ_ENTRY = ['inputs', 'input_type', 'outputs', 'output_types']
VALID_MIME_TYPE = ['image/jpeg', 'application/json']

def _check_signature(sig_file):
    '''Internal helper to check signature error when exporting model with CLI.
    '''
    with open(sig_file) as js_file:
        signature = json.load(js_file)

    assert 'input_type' in signature and 'output_type' in signature, \
        'input_type and output_type are required in signature.'
    assert isinstance(signature['input_type'], basestring) and \
           isinstance(signature['output_type'], basestring), \
        'Value of input_type and output_type should be string'
    assert signature['input_type'] in VALID_MIME_TYPE and \
           signature['output_type'] in VALID_MIME_TYPE, \
        'Valid type should be picked from %s' % (VALID_MIME_TYPE)

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
    '''Internal helper for exporting model.
    '''
    _check_signature(args.signature)
    model_name, model_path = args.model.split('=')
    destination = args.export_path or os.getcwd()
    if model_path.startswith('~'):
        model_path = os.path.expanduser(model_path)
    if destination.startswith('~'):
        destination = os.path.expanduser(destination)
    file_list = [args.signature]
    for dirpath, _, filenames in os.walk(model_path):
        for file_name in filenames:
            if file_name.endswith('.json') or file_name.endswith('.params'):
                file_list.append(os.path.join(dirpath, file_name))
    if args.synset:
        file_list += [args.synset]
    with zipfile.ZipFile(os.path.join(destination,'%s.model' % model_name), 'w') as zip_file:
        for item in file_list:
            zip_file.write(item, os.path.basename(item))
    logging.info('Successfully exported %s model. Model file is located at %s/%s.model.',
                 model_name, destination, model_name)


def export_serving(model, filename, signature, export_path = None, util_files=None):
    """Export a MXNet module object to a zip file used by MXNet model serving.

    Parameters
    ----------
    model : mx.mod.Module or mx.mod.BucketModule
        MXNet module object to be exported.
    filename : str
        Prefix of exported model file.
    signature : dict
        A dictionary containing model input and output information.
        Data names or data shapes of inputs and outputs can be automatically
        collected from module. They are optional. User needs to specify inputs
        and outputs MIME type for http request. Currently only 'image/jpeg'
        and 'application/json' are supported. All inputs should have the same type.
        This also applies for outputs.
        An example signature would be:
            signature = { "input_type": "image/jpeg", "output_type": "application/json" }
        A full signature containing inputs and outputs:
            {
                "input_type": "image/jpeg",
                "inputs" : [
                    {
                        'data_name': 'data',
                        'data_shape': [1, 3, 224, 224]
                    }
                ],
                "outputs" : [
                    {
                        'data_name': 'softmax',
                        'data_shape': [1, 1000]
                    }
                ]
                "output_type": "application/json"
            }
    export_path : str
        Destination path for export file. By default the model file
        is saved to current working directory.
    util_files : List
        A list of string containing other utility files for inference. One example is class
        label file for classification task.

    Examples
    --------
    >>> signature1 = { "input_type": "image/jpeg", "output_type": "application/json" }
    >>> export_serving(model1, filename='resnet-18', signature=signature1,
    >>>                util_files=['synset.txt'])
    >>> signature2 = {
    >>>                    "input_type": "image/jpeg",
    >>>                    "inputs" : [
    >>>                        {
    >>>                            'data_name': 'data',
    >>>                            'data_shape': [1, 3, 224, 224]
    >>>                        }
    >>>                    ],
    >>>                    "outputs" : [
    >>>                        {
    >>>                            'data_name': 'softmax',
    >>>                            'data_shape': [1, 1000]
    >>>                        }
    >>>                    ]
    >>>                    "output_type": "application/json"
    >>>              }
    >>> export_serving(model2, filename='resnet-152', signature=signature2,
    >>>                util_files=['synset.txt'])
    Exported model to "resnet-18.model"
    Exported model to "resnet-152.model"
    """
    assert issubclass(type(model), mx.mod.BaseModule), \
        "Model is type %s. It must be a subclass of mx.mod.BaseModule." % (type(model))

    epoch_placeholder = 0
    destination = export_path or os.getcwd()
    if destination.startswith('~'):
        destination = os.path.expanduser(destination)
    sig_file = '%s/signature.json' % (destination)

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

    with open(sig_file, 'w') as sig:
        json.dump(signature, sig)
    _check_signature(sig_file)
    model.save_checkpoint('%s/%s' % (destination, filename), epoch_placeholder)

    file_list = ['%s/%s-symbol.json' % (destination, filename), '%s/%s-%04d.params' %
                    (destination, filename, epoch_placeholder), sig_file]
    if util_files:
        file_list += util_files

    with zipfile.ZipFile(os.path.join(destination,'%s.model' % filename), 'w') as zip_file:
        for item in file_list:
            zip_file.write(item)
    logging.info('Exported model to %s/%s.model', destination, filename)


def export():
    args = ArgParser.parse_export_args()
    _export_model(args)

if __name__ =='__main__':
    export()
