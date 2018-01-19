import json
import os
import shutil
import subprocess
from threading import Thread

import sys
import time

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
    shutil.rmtree(tmpdir)


def setup_onnx_integ(tmpdir):
    # Start the mxnet model server for onnx integ tests.
    print("Starting MXNet Model Server for onnx integ test..")

    # Set argv parameters
    sys.argv = ['mxnet-model-server']
    sys.argv.append("--models")
    sys.argv.append("squeezenet={}/squeezenet.model".format(tmpdir))
    mxnet_model_server.start_serving()


def create_model(tmpdir):
    # Download the files required for onnx integ tests.
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.onnx")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/signature.json")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/onnx-squeezenet/synset.txt")
    # Download input image.
    _download_file(tmpdir, "https://upload.wikimedia.org/wikipedia/commons/8/8f/Cute-kittens-12929201-1600-1200.jpg")
    # Export the model.
    print("Exporting the mxnet model server model...")
    subprocess.check_call(['mxnet-model-export', '--model-name', 'squeezenet', '--model-path', tmpdir], cwd=tmpdir)


def test_onnx_integ(tmpdir):
    create_model(tmpdir)
    start_test_server_thread = Thread(target=setup_onnx_integ, args=(str(tmpdir),))
    start_test_server_thread.daemon = True
    start_test_server_thread.start()
    time.sleep(5)
    output = subprocess.check_output(['curl', '-X', 'POST', 'http://127.0.0.1:8080/squeezenet/predict', '-F',
                                      'input_0=@{}/Cute-kittens-12929201-1600-1200.jpg'.format(tmpdir)])
    if sys.version_info[0] >= 3:
        output = output.decode("utf-8")
    predictions = json.dumps(json.loads(output))
    # Assert objects are detected.
    assert predictions is not None
    assert len(predictions) > 0
    # Cleanup
    cleanup(str(tmpdir))
