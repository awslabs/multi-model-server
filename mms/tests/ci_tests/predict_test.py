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

from retrying import retry


@retry(stop_max_attempt_number=5, wait_fixed=1000)
def predict():
    output = subprocess.check_output('curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@cat.jpg"', shell=True)
    response = json.dumps(json.loads(output)['prediction'], sort_keys=True)
    expected = '[[{"class": "n02342885 hamster", "probability": 0.1685473769903183}, {"class": "n02328150 Angora, Angora rabbit", "probability": 0.1247132420539856}, {"class": "n02109961 Eskimo dog, husky", "probability": 0.09091565012931824}, {"class": "n02113023 Pembroke, Pembroke Welsh corgi", "probability": 0.06013256683945656}, {"class": "n02395406 hog, pig, grunter, squealer, Sus scrofa", "probability": 0.04349429905414581}]]'
    assert response == expected

subprocess.check_call('deep-model-server --models resnet-18=resnet-18.model &', shell=True)

time.sleep(5)

predict()