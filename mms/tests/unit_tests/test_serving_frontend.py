# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import sys
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../..')

import unittest
import mock

from model_service.mxnet_vision_service import MXNetVisionService
from model_service.mxnet_model_service import MXNetBaseService
from serving_frontend import ServingFrontend


class TestServingFrontend(unittest.TestCase):
    def setUp(self):
        self.test_frontend = ServingFrontend('test')

    def test_register_module(self):
        # Mock 
        ret = [MXNetVisionService]
        self.test_frontend.service_manager.parse_modelservices_from_module = mock.Mock(return_value=ret)
        self.test_frontend.service_manager.add_modelservice_to_registry = mock.Mock()

        self.assertEqual(self.test_frontend.register_module('mx_vision_service'), ret)

    def test_get_registered_modelservices(self):
        # Mock
        all_model_services = {
                                MXNetBaseService.__name__: MXNetBaseService, 
                                MXNetVisionService.__name__: MXNetVisionService
                            }

        self.test_frontend.service_manager.get_modelservices_registry = mock.Mock(return_value=all_model_services)
        self.assertEqual(self.test_frontend.get_registered_modelservices(), all_model_services)

    def runTest(self):
        self.test_register_module()
        self.test_get_registered_modelservices()