import json
import os
import subprocess
import time

from retrying import retry


@retry(stop_max_attempt_number=5, wait_fixed=1000)
def predict():
    output = subprocess.check_output('curl -X POST http://127.0.0.1:8080/resnet-18/predict -F "input0=@cat.jpg"', shell=True)
    response = json.dumps(json.loads(output)['prediction'], sort_keys=True)
    expected = '[[{"class": "n02342885 hamster", "probability": 0.1685473769903183}, {"class": "n02328150 Angora, Angora rabbit", "probability": 0.1247132420539856}, {"class": "n02109961 Eskimo dog, husky", "probability": 0.09091565012931824}, {"class": "n02113023 Pembroke, Pembroke Welsh corgi", "probability": 0.06013256683945656}, {"class": "n02395406 hog, pig, grunter, squealer, Sus scrofa", "probability": 0.04349429905414581}]]'
    assert response == expected

subprocess.check_call('mxnet-model-server --models resnet-18=resnet-18.zip &', shell=True)

time.sleep(5)

predict()