import json
import os
import shutil
import subprocess
import tarfile
from threading import Thread

import sys
import time

try:
    from urllib2 import urlopen, URLError, HTTPError
except:
    from urllib.request import urlopen, URLError, HTTPError

from mms import mxnet_model_server




### models from onnx-mxnet model zoo
onnx_mxnet_model_URLs = {
    'onnx-alexnet'    : 'https://s3.amazonaws.com/model-server/models/onnx-alexnet/alexnet.onnx',
    'onnx-squeezenet' : 'https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.onnx',
    'onnx-inception_v1' : 'https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.onnx',
    'onnx-vgg19' : 'https://s3.amazonaws.com/model-server/models/onnx-vgg19/vgg19.onnx'
}


### models from onnx model zoo
onnx_model_URLs = {
    'bvlc_alexnet'    : 'https://s3.amazonaws.com/download.onnx/models/bvlc_alexnet.tar.gz',
    'densenet121'     : 'https://s3.amazonaws.com/download.onnx/models/densenet121.tar.gz',
    'inception_v1'    : 'https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz',
    'inception_v2'    : 'https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz',
    'resnet50'        : 'https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz',
    'shufflenet'      : 'https://s3.amazonaws.com/download.onnx/models/shufflenet.tar.gz',
    'squeezenet'      : 'https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz',
    'vgg16'           : 'https://s3.amazonaws.com/download.onnx/models/vgg16.tar.gz',
    'vgg19'           : 'https://s3.amazonaws.com/download.onnx/models/vgg19.tar.gz'
}

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
    shutil.rmtree(tmpdir)


def setup_onnx_integ(tmpdir, URL,port):
    # Start the mxnet model server for onnx integ tests.
    print("Starting MXNet Model Server for onnx integ test..")

    # Set argv parameters
    sys.argv = ['mxnet-model-server']
    sys.argv.append('--models')
    for onnx_model in URL.keys():
        sys.argv.append("{}={}/{}.model".format(onnx_model, tmpdir, onnx_model))
    sys.argv.append('--port')
    sys.argv.append(port)
    mxnet_model_server.start_serving()


def create_model(tmpdir,URL, onnx_source_model_zoo = True):
    # Download the files required for onnx integ tests.
    download_dir = tmpdir + '/scratch'
    try:
        os.mkdir(download_dir)
        for onnx_model in URL.keys():
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
            
            # Export the model.
            print("Exporting the mxnet model server model...")
            subprocess.check_call(['mxnet-model-export', '--model-name', onnx_model, '--model-path', download_dir], cwd=download_dir)
            shutil.move('{}/{}.model'.format(download_dir, onnx_model), tmpdir)
            for root, dirs, files in os.walk(download_dir):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    shutil.rmtree(os.path.join(root, d))

        _download_file(tmpdir, "https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg")
        shutil.rmtree(download_dir)
    except Exception as e:
        print("Failed to create models. {}".format(str(e)))
        raise 

def test_onnx_integ(tmpdir):
    start_test(tmpdir,onnx_mxnet_model_URLs,port='8081', onnx_source_model_zoo= False)
    #start_test(tmpdir,onnx_model_URLs, port = '8082',onnx_source_model_zoo = True)
    #cleanup
    cleanup(tmpdir)

def start_test(tmpdir, URL, port='8081', onnx_source_model_zoo= False):
    tmpdir = str(tmpdir)
    create_model(tmpdir, URL, onnx_source_model_zoo)
    start_test_server_thread = Thread(target=setup_onnx_integ, args=(tmpdir, URL, port))
    start_test_server_thread.daemon = True
    start_test_server_thread.start()
    time.sleep(20)
    for onnx_model in URL.keys():
        output = subprocess.check_output(['curl', '-X', 'POST', 'http://127.0.0.1:'+port+'/'+ onnx_model+ '/predict', '-F',
                                      'input_0=@{}/Cute-kittens-12929201-1600-1200.jpg'.format(tmpdir)])
        if sys.version_info[0] >= 3:
           output = output.decode("utf-8")
        predictions = json.dumps(json.loads(output))
        # Assert objects are detected.
        assert predictions is not None
        assert len(predictions) > 0
        
