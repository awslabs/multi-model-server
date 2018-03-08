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
mxnet_model_URLs = {
    ##TODO: check why these two models are failing
    'caffenet'        : 'https://s3.amazonaws.com/model-server/models/caffenet/caffenet.model',
    'Inception-BN'    : 'https://s3.amazonaws.com/model-server/models/inception-bn/Inception-BN.model',
    #'lstm_ptb'        : 'https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb.model',
    'nin'             : 'https://s3.amazonaws.com/model-server/models/nin/nin.model',
    'resnet-152'      : 'https://s3.amazonaws.com/model-server/models/resnet-152/resnet-152.model',
    'resnet-18'       : 'https://s3.amazonaws.com/model-server/models/resnet-18/resnet-18.model',
    #'resnet50_ssd_model'    : 'https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model.model',
    'resnext-101-64x4d' : 'https://s3.amazonaws.com/model-server/models/resnext-101-64x4d/resnext-101-64x4d.model',
    'squeezenet_v1.1' : 'https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model',
    'vgg19'           : 'https://s3.amazonaws.com/model-server/models/vgg19/vgg19.model',
  }


@pytest.fixture( params= mxnet_model_URLs.keys())
def test_data(request):
    return request.param
 
def test_onnx_integ(tmpdir, test_data):
    tmpdir= str(tmpdir)
    utils._download_file(tmpdir, "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    utils.start_test(tmpdir, mxnet_model_URLs, test_data, port='8080',onnx_source_model_zoo= False, is_onnx_model= False)
    utils.cleanup(tmpdir)
