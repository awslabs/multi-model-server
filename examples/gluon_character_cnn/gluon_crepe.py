# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from gluon_base_service import GluonBaseService
import numpy as np
import ast
from mxnet import nd


class GluonCrepe(HybridBlock):
    """
    Hybrid Block gluon Crepe model
    """
    def __init__(self, classes=7, **kwargs):
        super(GluonCrepe, self).__init__(**kwargs)
        self.NUM_FILTERS = 256 # number of convolutional filters per convolutional layer
        self.NUM_OUTPUTS = classes # number of classes
        self.FULLY_CONNECTED = 1024 # number of unit in the fully connected dense layer
        self.features = nn.HybridSequential()
        with self.name_scope():
            self.features.add(
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=7, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=7, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=self.NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Flatten(),
                nn.Dense(self.FULLY_CONNECTED, activation='relu'),
                nn.Dense(self.FULLY_CONNECTED, activation='relu'),
            )
            self.output = nn.Dense(self.NUM_OUTPUTS)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class CharacterCNNService(GluonBaseService):
    """
    Gluon Character-level Convolution Service
    """
    def __init__(self):
        super(CharacterCNNService, self).__init__()
        # The 69 characters as specified in the paper
        self.ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
        # Map Alphabets to index
        self.ALPHABET_INDEX = {letter: index for index, letter in enumerate(self.ALPHABET)}
        # max-length in characters for one document
        self.FEATURE_LEN = 1014

    def initialize(self, params):
        self.net = GluonCrepe()
        self.param_filename = "crepe_gluon_epoch6.params"
        super(CharacterCNNService, self).initialize(params)
        # Hybridize imperative model for best performance
        self.net.hybridize()

    def preprocess(self, data):
        """
        Pre-process text to a encode it to a form, that gives spatial information to the CNN
        """
        # build the text from the request
        if data[0].get('data') is not None:
            data = ast.literal_eval(data[0].get('data').decode('utf-8'))
        text = '{}|{}'.format(data[0].get('review_title'), data[0].get('review'))

        encoded = np.zeros([len(self.ALPHABET), self.FEATURE_LEN], dtype='float32')
        review = text.lower()[:self.FEATURE_LEN-1:-1]
        i = 0
        for letter in text:
            if i >= self.FEATURE_LEN:
                break;
            if letter in self.ALPHABET_INDEX:
                encoded[self.ALPHABET_INDEX[letter]][i] = 1
            i += 1
        return nd.array([encoded], ctx=self.ctx)

    def inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data)
        return output.softmax()

    def postprocess(self, data):
        # Post process and output the most likely category
        predicted = self.labels[np.argmax(data[0].asnumpy())]
        return [{'category': predicted}]


svc = CharacterCNNService()


def crepe_inference(data, context):
    res = ""
    if not svc.initialized:
        svc.initialize(context)

    if data is not None:
        res = svc.predict(data)

    return res

