# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

'''Command line interface to export model files to be used for inference by MXNet Model Server
'''

import os
import glob
import json
import zipfile
import imp
import mxnet as mx
import inspect
from mms.arg_parser import ArgParser
import mms.model_service.mxnet_model_service as base_service 
import mms.model_service.mxnet_vision_service as vision_service
from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.log import get_logger

logger = get_logger()

try:
    basestring
except NameError:
    basestring = str


SIG_REQ_ENTRY = ['inputs', 'input_type', 'outputs', 'output_types']
VALID_MIME_TYPE = ['image/jpeg', 'application/json']
SIGNATURE_FILE = 'signature.json'
MODEL_ARCHIVE_VERSION = 0.1
MODEL_SERVER_VERSION = 0.1
MANIFEST_FILE_NAME = 'MANIFEST.json'

def validate_signature(model_path):
    '''Internal helper to check signature error when exporting model with CLI.
    '''
    signature_file = os.path.join(model_path, SIGNATURE_FILE)

    assert os.path.isfile(signature_file), \
        "signature.json is not found in %s." % (model_path)
    with open(signature_file) as js_file:
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

    return signature_file

def validate_symbol(model_path):
    '''
    Checks if a single Symbol file ending with '-symbol.json' file exists within the model_path
    '''
    symbol_file_postfix = '-symbol.json'
    sym_files = glob.glob(model_path + "*" + symbol_file_postfix)
    
    assert len(sym_files) == 1, \
           "should have 1 symbol file ending with -symbol.json in the filename, given:%s in model_path:%s" \
           % (str(sym_files), model_path)
    
    return sym_files[0]

def validate_params(model_path):
    '''
    Checks if a single parameter file ending with .params extension in the model_path
    and returns that file
    '''
    params_file_postfix = '.params'
    
    param_files = glob.glob(model_path + "*" + params_file_postfix)
    
    assert len(param_files) == 1, \
           "should have 1 parameter file ending with .params , given:%s in model_path:%s" \
           %(str(param_files), model_path)
    
    return param_files[0]

def validate_service(model_path, service_file, signature_file):    
    
    if service_file:

        assert os.path.isfile(service_file) or os.path.isfile(os.path.join(model_path, service_file)), \
            "Service File not found in %s or in %s." % (service_file, model_path)
            
        service_file = service_file if os.path.isfile(service_file) \
            else glob.glob(model_path + service_file)[0]
        
        module = None
        try:
            module =  imp.load_source(
                os.path.splitext(os.path.basename(service_file))[0],
                service_file)
        except Exception as e:
            raise Exception('Incorrect or missing service file: ' + service_file)
    
        classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        # Check if subclass of MXNetBaseService
        service_classes = list(filter(lambda cls: issubclass(cls, MXNetBaseService), classes))
        
        assert len(service_classes) > 1, \
                "The Service class should be derived from MXNetBaseService, found %s classes" % str(service_classes)
        
        #remove the compiled python code
        if os.path.exists(service_file + 'c'):
            os.remove(service_file + 'c')
    
    else:
        input_type = None
        with open(signature_file) as js_file:
            input_type = json.load(js_file)['input_type']
        
        if input_type == 'image/jpeg':
            service_file = vision_service.__file__
        elif input_type == 'application/json':
            service_file = base_service.__file__
        else:
            assert input_type in VALID_MIME_TYPE, \
                   "input_type should be one of %s or have your own service file handling it" % str(VALID_MIME_TYPE)

        #if service_file is a compile file, get the corresponding python code file
        if service_file.endswith('.pyc'):
            service_file = service_file[:-1]            
            assert os.path.isfile(service_file), \
                   "Vision Service File is missing in mms installation" 
        
    return service_file
    
def generate_manifest(symbol_file, params_file, service_file, signature_file, model_name):
    manifest = {}
    manifest["Model-Archive-Version"] = MODEL_ARCHIVE_VERSION
    manifest["Model-Archive-Description"] = model_name
    manifest["License"] = "Apache 2.0"
    manifest["Model-Server"] = MODEL_SERVER_VERSION
    
    manifest["Model"] = {}
    manifest["Model"]["Symbol"] = os.path.split(symbol_file)[1]
    manifest["Model"]["Parameters"] = os.path.split(params_file)[1]
    manifest["Model"]["Signature"] = os.path.split(signature_file)[1]
    manifest["Model"]["Service"] =  os.path.split(service_file)[1]
    manifest["Model"]["Description"] = model_name
    manifest["Model"]["Model-Name"] = model_name
    manifest["Model"]["Model-Format"] = "MXNet-Symbolic"
    manifest["Engine"]  = {"MXNet":0.12}
    
    return manifest
    
def export_model(model_name, model_path, service_file=None):
    '''Internal helper for the exporting model command line interface.
    '''
    destination = os.getcwd()
    
    if model_path.startswith('~'):
        model_path = os.path.expanduser(model_path)
    
    model_path = model_path + os.sep if model_path[-1] != os.sep else model_path

    signature_file = validate_signature(model_path)
    
    service_file = validate_service(model_path, service_file, signature_file)

    symbol_file = validate_symbol(model_path)
    
    params_file = validate_params(model_path)
    
    manifest = generate_manifest(symbol_file, params_file, service_file, signature_file, model_name)
    manifest_file = os.path.join(model_path, MANIFEST_FILE_NAME)
    with open(manifest_file, 'w') as m:
        json.dump(manifest, m, indent=4)

    file_list = [signature_file, service_file, symbol_file, params_file, manifest_file]
    
    #add all the auxillary files    
    for dirpath, _, filenames in os.walk(model_path):
        for filename in filenames:
            filename = os.path.join(model_path, filename)
            
            if os.path.isfile(filename) and filename not in file_list:
                file_list.append(filename)
    export_file = os.path.join(destination,'%s.model' % model_name)

    assert False == os.path.isfile(export_file), "%s.model already exists in %s directory." % (model_name, destination)

    with zipfile.ZipFile(export_file, 'w') as zip_file:
        for item in file_list:
            zip_file.write(item, os.path.basename(item))

    logger.info('Successfully exported %s model. Model file is located in %s directory.'
          % (model_name, destination))

def export():
    args = ArgParser.export_parser().parse_args()
    export_model(args.model_name, args.model_path, args.service_file_path)

if __name__ =='__main__':
    export()
