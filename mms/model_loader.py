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

from mms.log import get_logger
from jsonschema import validate


logger = get_logger()

URL_PREFIX = ('http://', 'https://', 's3://')


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
    curr_dir = os.getcwd()
    
    model_file = download(url=path, overwrite=True) \
    if path.lower().startswith(URL_PREFIX) else path

    model_file = os.path.abspath(model_file)
    model_file_prefix = os.path.splitext(os.path.basename(model_file))[0]
    model_dir = os.path.join(os.path.dirname(model_file), model_file_prefix)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    try:
        _extract_zip(model_file, model_dir)
    except Exception as e:
        raise Exception('Failed to open model file %s for model %s. Stacktrace: %s'
                        % (model_file, model_file_prefix , e))
    try:
        schema = json.load(open(os.path.join(model_dir, 'manifest-schema.json')))
        manifest = json.load(open(os.path.join(model_dir, 'manifest.json')))
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
