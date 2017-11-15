import json
import subprocess
import time
import os
import sys
import shutil

from threading import Thread
from urllib2 import urlopen, URLError, HTTPError

from dms import export_model, deep_model_server


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
    except HTTPError, e:
        print("Failed to download {}. HTTP Error {}".format(url, e.code))
    except URLError, e:
        print("Failed to download {}. HTTP Error {}".format(url, e.reason))


def setup_ssd_server(tmpdir):
    """
        Downloads and Setup the SSD model server.
    :return: None
    """
    # Download the files required for SSD model in temp folder.
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/signature.json")
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/ssd_service.py")

    # Download input image.
    _download_file(tmpdir, "https://s3.amazonaws.com/model-server/models/resnet50_ssd/street.jpg")

    # Export the model.
    print("Exporting the deep model server model...")
    sys.argv = ['deep-model-export']
    sys.argv.append("--model-name")
    sys.argv.append("{}/resnet50_ssd_model".format(tmpdir))
    sys.argv.append("--model-path")
    sys.argv.append(tmpdir)
    export_model.export()

    # Start the deep model server for SSD
    print("Starting SSD Deep Model Server for test..")

    # Set argv parameters
    sys.argv = ['deep-model-server']
    sys.argv.append("--models")
    sys.argv.append("SSD={}/resnet50_ssd_model.model".format(tmpdir))
    sys.argv.append("--service")
    sys.argv.append("{}/ssd_service.py".format(tmpdir))
    deep_model_server.start_serving()


def cleanup(tmpdir):
    print("Deleting all downloaded resources for SSD Deep Model Server Integration Test")
    shutil.rmtree(tmpdir)


def test_ssd_extend_export_predict_service(tmpdir):
    start_test_server_thread = Thread(target = setup_ssd_server, args=(str(tmpdir),))
    start_test_server_thread.daemon = True
    start_test_server_thread.start()
    time.sleep(10)
    output = subprocess.check_output('curl -X POST http://127.0.0.1:8080/SSD/predict -F "data=@{}/street.jpg"'.format(str(tmpdir)), shell=True)
    predictions = json.dumps(json.loads(output))
    # Assert objects are detected.
    assert predictions is not None
    assert len(predictions) > 0
    # Cleanup
    cleanup(str(tmpdir))
