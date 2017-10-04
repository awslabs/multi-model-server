'''Command line interface to export MXNet model.
'''

import os
import json
import zipfile
from mms.arg_parser import ArgParser

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
    with zipfile.ZipFile('%s/%s.zip' % (destination, model_name), 'w') as zip_file:
        for item in file_list:
            zip_file.write(item, os.path.basename(item))
    print('Successfully exported %s model. Model file is located at %s/%s.zip.'
          % (model_name, destination, model_name))


def export():
    args = ArgParser.parse_export_args()
    _export_model(args)

if __name__ =='__main__':
    export()
