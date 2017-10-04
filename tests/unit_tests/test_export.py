import sys
sys.path.append('../..')

import unittest
import mock
import mxnet as mx
import os
import json
from mxnet_vision_service import MXNetVisionService as mx_vision_service


class TestExport(unittest.TestCase):
    def _train_and_export(self):
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
        with open('synset.txt', 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % (i))
        with open('signature.json', 'w') as sig:
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
        mod.save_checkpoint('test', 0)

    def test_export(self):
        self._train_and_export()
        model_name = 'test'
        model_path = '.'
        signature = 'signature.json'
        synset = 'synset.txt'
        export_path = '.'

        cmd = 'python ../../export_model.py --model %s=%s --signature %s ' \
              '--synset %s --export-path %s' % (model_name, model_path,
                                                signature, synset, export_path)
        os.system(cmd)
        assert os.path.isfile('test.zip'), "No zip file is found. Export failed!"

        mx_vision_service('test.zip')

    def runTest(self):
        self.test_export()
