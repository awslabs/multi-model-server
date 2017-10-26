# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json
import subprocess
import time
import os

from urllib2 import urlopen, URLError, HTTPError
from nose.tools import with_setup
# This is used to store the process ID of the DMS server started.
# This is then used in tear_down to shutdown the server.
dms_process = None


def _download_file(url):
    """
        Helper function to download the file from specified url
    :param url: File to download
    :return: None
    """
    try:
        f = urlopen(url)
        print("Downloading - {}".format(url))
        with open(os.path.basename(url), "wb") as local_file:
            local_file.write(f.read())
    except HTTPError, e:
        print("Failed to download {}. HTTP Error {}".format(url, e.code))
    except URLError, e:
        print("Failed to download {}. HTTP Error {}".format(url, e.reason))


def setup_ssd_server():
    """
        Downloads and Setup the SSD model server.
    :return: None
    """

    # Download the model files.
    _download_file("https://s3.amazonaws.com/mms-models/examples/resnet50_ssd/resnet50_ssd_model-symbol.json")
    _download_file("https://s3.amazonaws.com/mms-models/examples/resnet50_ssd/resnet50_ssd_model-0000.params")

    # Download input image.
    _download_file("https://s3.amazonaws.com/mms-models/examples/resnet50_ssd/street.jpg")

    # List
    output = subprocess.check_output('pwd; ls -l', shell=True)
    print("Hellooo... ")
    print(output)

    # Export the model.
    subprocess.popen('cp examples/ssd/synset.txt .')
    subprocess.popen('cp examples/ssd/signature.json .')
    subprocess.popen("deep-model-export --model-name resnet50_ssd_model --model-path .")

    # Start the deep model server for SSD
    print("Starting SSD Deep Model Server for test..")
    global dms_process
    dms_process = subprocess.popen("deep-model-server --models SSD=resnet50_ssd_model.model --service "
                                   "examples/ssd/ssd_service.py &", shell=False)
    time.sleep(5)


def teardown_ssd_server():
    """
        Terminates the SSD server started for the test.
    :return: None
    """
    print("Terminating SSD Deep Model Server..")
    if dms_process is not None:
        dms_process.terminate()


@with_setup(setup_ssd_server, teardown_ssd_server)
def test_ssd_extend_export_predict_service():
    output = subprocess.check_output('curl -X POST http://127.0.0.1:8080/SSD/predict -F "input0=@street.jpg"',
                                     shell=True)
    predictions = json.dumps(json.loads(output))['prediction']
    # Assert objects are detected.
    assert predictions is not None
    assert len(predictions) > 0
