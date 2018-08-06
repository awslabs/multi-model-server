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

from mms import model_server


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


def start_ssd_server(tmpdir):
    # Start the mxnet model server for SSD
    print("Starting SSD MXNet Model Server for test..")

    # Set argv parameters
    sys.argv = ['mxnet-model-server']
    sys.argv.append("--models")
    sys.argv.append("SSD={}/resnet50_ssd_model.model".format(tmpdir))
    # sys.argv.append("--service")
    # sys.argv.append("{}/ssd_service.py".format(tmpdir))
    model_server.start_serving()


def cleanup(tmpdir):
    print("Deleting all downloaded resources for SSD MXNet Model Server Integration Test")
    shutil.rmtree(tmpdir)


def test_ssd_extend_export_predict_service(tmpdir):
    tmpdir = str(tmpdir)
    create_model(tmpdir)
    start_test_server_thread = Thread(target=start_ssd_server, args=(tmpdir,))
    start_test_server_thread.daemon = True
    start_test_server_thread.start()
    time.sleep(5)
    output = subprocess.check_output(
        'curl -X POST http://127.0.0.1:8080/SSD/predict -F "data=@{}/street.jpg"'.format(tmpdir),
        shell=True)
    if sys.version_info[0] >= 3:
        output = output.decode("utf-8")

    predictions = json.dumps(json.loads(output))
    # Assert objects are detected.
    assert predictions is not None
    assert len(predictions) > 0
    # Cleanup
    cleanup(tmpdir)


def create_model(tmpdir):

    # Download the files required for SSD model in temp folder.
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/signature.json")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/ssd_service.py")
    # Download input image.
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/street.jpg")
    # Export the model.
    print("Exporting the mxnet model server model...")
    subprocess.check_call(
        ['mxnet-model-export', '--model-name', 'resnet50_ssd_model', '--model-path', tmpdir, '--service-file-path',
         '{}/ssd_service.py'.format(tmpdir)], cwd=tmpdir)
