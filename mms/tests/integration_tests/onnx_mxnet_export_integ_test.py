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
onnx_mxnet_model_URLs = {
    'onnx-alexnet'    : 'https://s3.amazonaws.com/model-server/models/onnx-alexnet/alexnet.onnx',
    'onnx-squeezenet' : 'https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.onnx',
    'onnx-inception_v1' : 'https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.onnx',
    'onnx-vgg19' : 'https://s3.amazonaws.com/model-server/models/onnx-vgg19/vgg19.onnx'
}



@pytest.fixture( params= onnx_mxnet_model_URLs.keys())
def test_data(request):
    return request.param
 
def test_onnx_integ(tmpdir, test_data):
    tmpdir= str(tmpdir)
    utils._download_file(tmpdir, "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    utils.start_test(tmpdir, onnx_mxnet_model_URLs, test_data, port='8080',onnx_source_model_zoo= False, is_onnx_model= True)
    utils.cleanup(tmpdir)
