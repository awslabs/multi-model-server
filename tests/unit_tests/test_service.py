import os
import sys
import json
sys.path.append('../..')

import unittest
import mock
import mxnet as mx
from mxnet_vision_service import MXNetVisionService as mx_vision_service
from utils.mxnet_utils import Image
from service.pixel2pixel_service import UnetGenerator, Pixel2pixelService

class TestService(unittest.TestCase):
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
        signature = {'input_type': 'image/jpeg', 'output_type': 'application/json'}
        with open('synset.txt', 'w') as synset:
            for i in range(10):
                synset.write('test label %d\n' % (i))
        mod.export_serving('test', 0, signature, use_synset=True)

    def test_vision_init(self):
        self._train_and_export()
        model_path = 'test.zip'
        service = mx_vision_service(path=model_path)
        assert hasattr(service, 'labels'), "Fail to load synset file from model archive."
        assert len(service.labels) > 0, "Labels attribute is empty."

    def test_vision_inference(self):
        self._train_and_export()
        model_path = 'test.zip'
        service = mx_vision_service(path=model_path)

        # Test same size image inputs
        data1 = mx.nd.random_uniform(0, 255, shape=(3, 64, 64))
        data2 = mx.nd.random_uniform(0, 255, shape=(3, 32, 32))
        img_buf1 = Image.write(data1)
        img_buf2 = Image.write(data2)

        output = service.inference([img_buf1, img_buf2])
        assert len(output[0]) == 5

        # test different size image inputs
        data1 = mx.nd.random_uniform(0, 255, shape=(3, 96, 96))
        data2 = mx.nd.random_uniform(0, 255, shape=(3, 24, 24))
        img_buf1 = Image.write(data1)
        img_buf2 = Image.write(data2)

        output = service.inference([img_buf1, img_buf2])
        assert len(output[0]) == 5

    def test_gluon_inference(self):
        ctx = mx.cpu()
        netG = UnetGenerator(in_channels=3, num_downs=8)
        data = mx.nd.random_uniform(0, 255, shape=(1, 3, 256, 256))
        netG.initialize(mx.init.Normal(0.02), ctx=ctx)
        netG(data)
        netG.save_params('gluon.params')
        if not os.path.isdir('gluon-dir'):
            os.mkdir('gluon-dir')
        if os.path.isfile('test/signature.json'):
            os.remove('test/signature.json')
        with open('gluon-dir/signature.json', 'w') as sig:
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
        model_name = 'gluon'
        model_path = '.'
        signature = 'gluon-dir/signature.json'
        export_path = '.'

        cmd = 'python ../../export_model.py --model %s=%s --signature %s ' \
              '--export-path %s' % (model_name, model_path,
                                    signature, export_path)
        #os.remove('gluon/signature.json')
        os.system(cmd)

        service = Pixel2pixelService('gluon.zip')
        data = mx.nd.random_uniform(0, 255, shape=(3, 256, 256))
        img_buf = Image.write(data)
        service.inference([img_buf])

    def runTest(self):
        self.test_vision_init()
        self.test_vision_inference()
        self.test_gluon_inference()
