import json
import os
import shutil
import subprocess
import tarfile
from threading import Thread
import pytest
import sys
import time
import signal
import test_utils as utils
try:
    from urllib2 import urlopen, URLError, HTTPError
except:
    from urllib.request import urlopen, URLError, HTTPError

from mms import mxnet_model_server



### models from onnx-mxnet model zoo
onnx_model_URLs = {
    #'bvlc_alexnet'    : 'https://s3.amazonaws.com/download.onnx/models/bvlc_alexnet.tar.gz',
    'densenet121'     : 'https://s3.amazonaws.com/download.onnx/models/densenet121.tar.gz',
    'inception_v1'    : 'https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz',
    'inception_v2'    : 'https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz',
    'resnet50'        : 'https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz',
    'shufflenet'      : 'https://s3.amazonaws.com/download.onnx/models/shufflenet.tar.gz',
    'squeezenet'      : 'https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz',
    #'vgg16'           : 'https://s3.amazonaws.com/download.onnx/models/vgg16.tar.gz',
    #'vgg19'           : 'https://s3.amazonaws.com/download.onnx/models/vgg19.tar.gz'
}



#@pytest.fixture( params= onnx_model_URLs.keys())
#def test_data(request):
#    return request.param
 
def test_onnx_integ(tmpdir):
    tmpdir= str(tmpdir)
    utils._download_file(tmpdir, "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    utils.start_test(tmpdir, onnx_model_URLs, None, port='8080',onnx_source_model_zoo= True, is_onnx_model= True, test_multiple_models = True)
    utils.cleanup(tmpdir)
