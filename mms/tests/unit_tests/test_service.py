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
import json
curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../..')

import PIL
import unittest
import numpy as np
import mxnet as mx
from io import BytesIO
from model_service.mxnet_vision_service import MXNetVisionService as mx_vision_service
from helper.pixel2pixel_service import UnetGenerator, Pixel2pixelService
#from export_model import export_serving

class TestService(unittest.TestCase):
    def _train_and_export(self, path):
        model_path = curr_path + '/' + path
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        num_class = 10
        data1 = mx.sym.Variable('data1')
        data2 = mx.sym.Variable('data2')
        conv1 = mx.sym.Convolution(data=data1, kernel=(2, 2), num_filter=2, stride=(2, 2))
        conv2 = mx.sym.Convolution(data=data2, kernel=(3, 3), num_filter=3, stride=(1, 1))
        pooling1 = mx.sym.Pooling(data=conv1, kernel=(2, 2), stride=(1, 1), pool_type="avg")
        pooling2 = mx.sym.Pooling(data=conv2, kernel=(2, 2), stride=(1, 1), pool_type="max")
        flatten1 = mx.sym.flatten(data=pooling1)
        flatten2 = mx.sym.flatten(data=pooling2)
        sum = mx.sym.sum(data=flatten1, axis=1) + mx.sym.sum(data=flatten2, axis=1)
        fc = mx.sym.FullyConnected(data=sum, num_hidden=num_class)
        sym = mx.sym.SoftmaxOutput(data=fc, name='softmax')

        dshape1 = (10, 3, 64, 64)
        dshape2 = (10, 3, 32, 32)
        lshape = (10,)

        mod = mx.mod.Module(symbol=sym, data_names=('data1', 'data2'),
                            label_names=('softmax_label',))
        mod.bind(data_shapes=[('data1', dshape1), ('data2', dshape2)],
                 label_shapes=[('softmax_label', lshape)])
        mod.init_params()
        mod.init_optimizer(optimizer_params={'learning_rate': 0.01})

        data_batch = mx.io.DataBatch(data=[mx.nd.random.uniform(0, 9, dshape1),
                                           mx.nd.random.uniform(5, 15, dshape2)],
                                     label=[mx.nd.ones(lshape)])
        mod.forward(data_batch)
        mod.backward()
        mod.update()
        signature = {'input_type': 'image/jpeg', 'output_type': 'application/json'}
        with open('%s/synset.txt' % (model_path), 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % (i))
#         export_serving(mod, 'test', signature, export_path=model_path,
#                        aux_files=['%s/synset.txt' % (model_path)])

    def _write_image(self, img_arr):
        img_arr = mx.nd.transpose(img_arr, (1, 2, 0)).astype(np.uint8).asnumpy()
        mode = 'RGB'
        image = PIL.Image.fromarray(img_arr, mode)
        output = BytesIO()
        image.save(output, format='jpeg')
        return output.getvalue()

    def test_vision_init(self):
        path = 'test'
        self._train_and_export(path)
        model_path = curr_path + '/' + path
        manifest = {
            "Model": {
                "Parameters": 'test-0000.params',
                "Signature": "signature.json"
            },
            "Assets": {
                "Synset": "synset.txt"
            }
        }
        os.system('rm -rf %s' % (model_path))

    def test_vision_inference(self):
        path = 'test'
        self._train_and_export(path)
        model_path = curr_path + '/' + path
        manifest = {
            "Model": {
                "Parameters": 'test-0000.params',
                "Signature": "signature.json"
            },
            "Assets": {
                "Synset": "synset.txt"
            }
        }
        
        os.system('rm -rf %s/test' % (curr_path))

    def test_gluon_inference(self):
        path = 'gluon'
        model_name = 'gluon1'
        model_path = curr_path + '/' + path
        os.mkdir(model_path)
        ctx = mx.cpu()
        netG = UnetGenerator(in_channels=3, num_downs=8)
        data = mx.nd.random_uniform(0, 255, shape=(1, 3, 256, 256))
        netG.initialize(mx.init.Normal(0.02), ctx=ctx)
        netG(data)
        netG.save_params('%s/%s.params' % (model_path, model_name))
        with open('%s/signature.json' % (model_path), 'w') as sig:
            signature = {
                "input_type": "image/jpeg",
                "inputs": [
                    {
                        'data_name': 'data',
                        'data_shape': [1, 3, 256, 256]
                    },
                ],
                "output_type": "image/jpeg",
                "outputs": [
                    {
                        'data_name': 'output',
                        'data_shape': [1, 3, 256, 256]
                    }
                ]
            }
            json.dump(signature, sig)

        cmd = 'python %s/../../export_model.py --model-name %s --model-path %s' \
              % (curr_path, model_name, model_path)
        os.system(cmd)
        
        os.system('rm -rf %s %s/%s.model %s/%s' % (model_path, os.getcwd(),
                                                   model_name, os.getcwd(), model_name))

    def runTest(self):
        self.test_vision_init()
        self.test_vision_inference()
        self.test_gluon_inference()
