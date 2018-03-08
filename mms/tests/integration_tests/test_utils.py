import json
import os
import shutil
import subprocess
import tarfile
from threading import Thread

import sys
import time
import signal
try:
    from urllib2 import urlopen, URLError, HTTPError
except:
    from urllib.request import urlopen, URLError, HTTPError

from mms import mxnet_model_server



def _download_file(download_dir, url):
    """
        Helper function to download the file from specified url
    :param url: File to download
    :return: None
    """
    try:
        f = urlopen(url)
        print("Downloading - {}".format(url))
        with open(os.path.join(download_dir, os.path.basename(url)), "wb") as local_file:
            local_file.write(f.read())
    except HTTPError as e:
        print("Failed to download {}. HTTP Error {}".format(url, e.code))
    except URLError as e:
        print("Failed to download {}. HTTP Error {}".format(url, e.reason))


def cleanup(tmpdir):
    print("Deleting all downloaded resources for SSD MXNet Model Server Integration Test")
    shutil.rmtree(str(tmpdir))




def create_model(tmpdir,URL,onnx_model, onnx_source_model_zoo = True, is_onnx_model= True):
    # Download the files required for onnx integ tests.
    download_dir = tmpdir + '/scratch'
    try:
        if is_onnx_model:
            os.mkdir(download_dir)
            if onnx_source_model_zoo:
                _download_file(download_dir, URL[onnx_model])
                _download_file(download_dir, "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/signature.json")
                _download_file(download_dir, "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/synset.txt")
                model_tar = '{}/{}.tar.gz'.format(download_dir,onnx_model) 
                tar = tarfile.open(model_tar, "r:*")
                tar.extractall(path=download_dir)
                tar.close()
                model_dir = '{}/{}'.format(download_dir, onnx_model)
                model_path = os.path.join(model_dir, 'model.onnx')
                new_path = os.path.join(model_dir, '{}.onnx'.format(onnx_model))
                os.rename(model_path, new_path)
                shutil.move(new_path, download_dir)
       
            else:
                _download_file(download_dir, URL[onnx_model])
                _download_file(download_dir, "https://s3.amazonaws.com/model-server/models/" + onnx_model + "/signature.json")
                _download_file(download_dir, "https://s3.amazonaws.com/model-server/models/" + onnx_model + "/synset.txt")
            print('model files prepared for model {} '.format(onnx_model))
            # Export the model.
            print("Exporting the mxnet model server model...")
            subprocess.check_call(['mxnet-model-export', '--model-name', onnx_model, '--model-path', download_dir], cwd=download_dir)
            shutil.move('{}/{}.model'.format(download_dir, onnx_model), tmpdir)
            for root, dirs, files in os.walk(download_dir):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))
            shutil.rmtree(download_dir)
        else:
            _download_file(tmpdir, URL[onnx_model]) 
    except Exception as e:
        print("Failed to create models. {}".format(str(e)))
        raise
  


def start_test(tmpdir, URL, onnx_model, port = '8080', onnx_source_model_zoo = False, is_onnx_model = True, test_multiple_models = False):
   model_list = []
   model_names = []
   if test_multiple_models:
        for model in URL.keys():
            create_model(tmpdir, URL, model, onnx_source_model_zoo, is_onnx_model)
            model_names.append( '{}={}/{}.model'.format(model, tmpdir, model))
            model_list.append(model)
   else:
        model_names.append('{}={}/{}.model'.format(onnx_model, tmpdir, onnx_model))
        create_model(tmpdir, URL, onnx_model, onnx_source_model_zoo, is_onnx_model) 
        model_list.append(onnx_model)    
   server_pid = subprocess.Popen(['mxnet-model-server', '--models']+ model_names+[ '--port',port]).pid
   
    #server_pid = subprocess.Popen(['mxnet-model-server', '--models', '{}={}/{}.model'.format(onnx_model, tmpdir, onnx_model),'--port',port]).pid
   try:  
        time.sleep(30)
        if is_onnx_model:
            data_name = 'input_0'
        else:
            data_name = 'data'
        for models in model_list:
            output = subprocess.check_output(['curl', '-X', 'POST', 'http://127.0.0.1:'+port+'/'+ models+ '/predict', '-F',
                                      '{}=@{}/kitten.jpg'.format(data_name,tmpdir)])
            if sys.version_info[0] >= 3:
                output = output.decode("utf-8")
            predictions = json.dumps(json.loads(output))
            # Assert objects are detected.
            assert predictions is not None
            assert len(predictions) > 0
   finally:
        os.kill(server_pid, signal.SIGQUIT)

