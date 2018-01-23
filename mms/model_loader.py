# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import glob
import requests
import zipfile
import shutil
import json
import jsonschema
import fasteners

from mms.log import get_logger
from jsonschema import validate


logger = get_logger()

URL_PREFIX = ('http://', 'https://', 's3://')
MANIFEST_DIR = "manifest_schema"
MANIFEST_SCHEMA_FILE = 'manifest-schema.json'
MANIFEST_FILENAME = 'MANIFEST.json'
LOCK_FILE = '/tmp/tmp_lock_file'

@fasteners.interprocess_locked(LOCK_FILE)
def _download_and_extract(model_location, path=None, overwrite=False):
    """Download an given URL

    Parameters
    ----------
    model_location : str
        Location for local model or URL to download
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
    model_file = model_location
    if model_location.lower().startswith(URL_PREFIX):
        if path is None:
            model_file = model_location.split('/')[-1]
        elif os.path.isdir(path):
            model_file = os.path.join(path, model_location.split('/')[-1])
        else:
            model_file = path

        if overwrite or not os.path.exists(model_file):
            dirname = os.path.dirname(os.path.abspath(os.path.expanduser(model_file)))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            print('Downloading %s from %s...' % (model_file, model_location))
            r = requests.get(model_location, stream=True)
            if r.status_code != 200:
                raise RuntimeError("Failed downloading url %s" % model_location)
            with open("%s.temp" % (model_file), 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            os.rename("%s.temp" % (model_file), model_file)

    model_file = os.path.abspath(model_file)
    [model_name, model_extension] = os.path.splitext(os.path.basename(model_file))
    model_file_prefix = model_name if model_extension == '.model' else model_file

    model_dir = os.path.join(os.path.dirname(model_file), model_file_prefix)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        try:
            if '.model' in model_file:
                _extract_zip(model_file, model_dir)
        except Exception as e:
            raise Exception('Failed to open model file %s for model %s. Stacktrace: %s'
                            % (model_file, model_file_prefix , e))
    return model_dir

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

def _extract_model(service_name, path):
    if path.endswith('.onnx') or path.endswith('.pb2'):
        raise ValueError('Convert ONNX model using mxnet-model-export before serving.')

    model_dir = _download_and_extract(model_location=path, overwrite=True)
        
    try:
        #manifest schema
        import mms
        mms_pkg_loc = os.path.split(mms.__file__)[0]
        manifest_schema_file = os.path.join(mms_pkg_loc, MANIFEST_DIR, MANIFEST_SCHEMA_FILE)

        assert os.path.isfile(manifest_schema_file), \
               "manifest-schema file missing mms pkg location:%s" % mms_pkg_loc

        schema = json.load(open(manifest_schema_file))
        manifest = json.load(open(os.path.join(model_dir, MANIFEST_FILENAME)))
    except Exception as e:
        raise Exception('Failed to open manifest file. Stacktrace: ' + str(e))

    validate(manifest, schema)

    assert len(glob.glob(os.path.join(model_dir, manifest['Model']['Signature']))) == 1, \
    'Signature file in model archive is inconsistent with manifest.'

    assert len(glob.glob(os.path.join(model_dir, manifest['Model']['Symbol']))) == 1, \
    'Symbol file in model archive is inconsistent with manifest.'

    assert len(glob.glob(os.path.join(model_dir, manifest['Model']['Parameters']))) == 1, \
    'Parameter file in model archive is inconsistent with manifest.'

    assert len(glob.glob(os.path.join(model_dir, manifest['Model']['Service']))) == 1, \
    'Service file in model archive is inconsistent with manifest.'

    model_name = manifest['Model']['Model-Name']
            
    return service_name, model_name, model_dir, manifest

class ModelLoader(object):
    """Model Loader
    """
    @staticmethod
    def load(models):
        """
        Load models 

        Parameters
        ----------
        models : dict
            Model name and model path pairs
            
        Returns
        ----------
        list
            (Model Name, Model Path, Model Schema) tuple list
        """
        return list(map(lambda model: _extract_model(model[0], model[1]), models.items()))
