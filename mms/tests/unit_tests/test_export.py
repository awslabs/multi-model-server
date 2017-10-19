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
import mxnet as mx
import json
from model_service.mxnet_vision_service import MXNetVisionService as mx_vision_service
from export_model import export_serving


class TestExport(unittest.TestCase):
    def _train_and_save(self, path):
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

        data_batch = mx.io.DataBatch(data=[mx.random.uniform(0, 9, dshape1),
                                           mx.random.uniform(5, 15, dshape2)],
                                     label=[mx.nd.ones(lshape)])
        mod.forward(data_batch)
        mod.backward()
        mod.update()
        with open('%s/synset.txt' % (model_path), 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % (i))
        with open('%s/signature.json' % (model_path), 'w') as sig:
            signature = {
                "input_type": "image/jpeg",
                "inputs": [
                    {
                        'data_name': 'data1',
                        'data_shape': [1, 3, 64, 64]
                    },
                    {
                        'data_name': 'data2',
                        'data_shape': [1, 3, 32, 32]
                    }
                ],
                "output_type": "application/json",
                "outputs": [
                    {
                        'data_name': 'softmax',
                        'data_shape': [1, 10]
                    }
                ]
            }
            json.dump(signature, sig)
        mod.save_checkpoint('%s/test' % (model_path), 0)

    def test_export_CLI(self):
        path = 'test'
        self._train_and_save(path)
        model_name = 'test1'
        model_path = curr_path + '/' + path
        export_file = '%s/%s.model' % (os.getcwd(), model_name)

        cmd = 'python %s/../../export_model.py --model-name %s --model-path %s ' \
              % (curr_path, model_name, model_path)
        os.system(cmd)
        assert os.path.isfile(export_file), "No model file is found. Export failed!"

        mx_vision_service(export_file)
        os.system('rm -rf %s %s %s/%s' % (export_file, model_path, os.getcwd(), model_name))

    def test_export_API(self):
        path = 'test'
        model_path = curr_path + '/' + path
        os.mkdir(model_path)
        sym = mx.sym.Variable('data')
        sym = mx.sym.FullyConnected(sym, num_hidden=100)
        sym = mx.symbol.Activation(name="act_1", data=sym, act_type='sigmoid')
        sym = mx.symbol.LinearRegressionOutput(data=sym, name='softmax', grad_scale=2)

        mod = mx.mod.Module(sym, ('data',))
        mod.bind(data_shapes=[('data', (10, 10))])
        mod.init_params()
        mod.init_optimizer(optimizer_params={'learning_rate': 0.1, 'momentum': 0.9})
        mod.update()

        signature = {'input_type': 'application/json', 'output_type': 'application/json'}
        with open('%s/synset.txt' % (model_path), 'w') as synset:
            synset.write('test label')
        export_serving(mod, 'test', signature, export_path=model_path, aux_files=['%s/synset.txt' % model_path])
        assert os.path.isfile('%s/test.model' % (model_path)), "No zip file found for export_serving."
        assert os.path.isfile('%s/signature.json' % (model_path)), "No signature file found for export_serving."
        with open('%s/signature.json' % (model_path)) as f:
            sig = json.load(f)
        assert sig['input_type'] == signature['input_type'], \
            "Input type incorrect. Expect %s but got %s" % (signature['input_type'], sig['input_type'])
        assert sig['output_type'] == signature['output_type'], \
            "Output type incorrect. Expect %s but got %s" % (signature['output_type'], sig['output_type'])
        for input, data in zip(sig['inputs'], mod.data_shapes):
            assert input['data_name'] == data[0], "Input name mistach. %s vs %s" % (input['data_name'], data[0])
            assert input['data_shape'] == list(data[1]), "Input shape mistach. %s vs %s" % (
            input['data_shape'], data[1])
        for output, data in zip(sig['outputs'], mod.output_shapes):
            assert output['data_name'] == data[0], "Output name mistach. %s vs %s" % (output['data_name'], data[0])
            assert output['data_shape'] == list(data[1]), "Output shape mistach. %s vs %s" % (
            output['data_shape'], data[1])
        os.system('rm -rf %s' % (model_path))

    def runTest(self):
        self.test_export_CLI()
        self.test_export_API()
