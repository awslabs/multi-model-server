# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from mms.model_service.mxnet_vision_service import MXNetVisionService
# Test will fail here
import wrong_package

''' This class will not be loaded by the test, since error is above in the service class
    The test is testing if the failed import propogates up to the model service level.
'''
class DummyService(MXNetVisionService):

    def __init__(self, model_name, model_dir, manifest, gpu=None):
        pass
    def _preprocess(self,data):
        pass
    def _inference(self,data):
        pass
    def _postprocess(self,data):
        pass
