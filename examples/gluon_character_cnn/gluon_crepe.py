# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mms.model_service.mxnet_vision_service import MXNetVisionService
import numpy as np
from mxnet import nd
import os


class GluonCrepe(HybridBlock):
    """
    Hybrid Block gluon Crepe model
    """
    def __init__(self, classes=7, **kwargs):
        super(GluonCrepe, self).__init__(**kwargs)
        NUM_FILTERS = 256 # number of convolutional filters per convolutional layer
        NUM_OUTPUTS = classes # number of classes
        FULLY_CONNECTED = 1024 # number of unit in the fully connected dense layer
        DROPOUT_RATE = 0.5 # probability of node drop out
        self.features = nn.HybridSequential()
        with self.name_scope():
            self.features.add(
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=7, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.Conv1D(channels=NUM_FILTERS, kernel_size=3, activation='relu'),
                nn.MaxPool1D(pool_size=3, strides=3),
                nn.Flatten(),
                nn.Dense(FULLY_CONNECTED, activation='relu'),
                nn.Dropout(DROPOUT_RATE),
                nn.Dense(FULLY_CONNECTED, activation='relu'),
                nn.Dropout(DROPOUT_RATE),
            )
            self.output = nn.Dense(NUM_OUTPUTS)
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class CharacterCNNService(MXNetVisionService):
    """
    Gluon Character-level Convolution Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(CharacterCNNService, self).__init__(model_name, model_dir, manifest, gpu)
        # Initialize model and load pre-trained weights
        self.net = GluonCrepe()
        if self.param_filename:
            self.net.load_params(os.path.join(model_dir, self.param_filename), ctx=self.ctx)
        self.net.hybridize()

    def _preprocess(self, data):
        # build the text from the request
        text = '{}|{}'.format(data[0][0]['review_title'], data[0][0]['review'])
        # The 69 characters as specified in the paper
        ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
        # Map Alphabets to index
        ALPHABET_INDEX = {letter: index for index, letter in enumerate(ALPHABET)}
        # max-length in characters for one document
        FEATURE_LEN = 1014
        encoded = np.zeros([len(ALPHABET), FEATURE_LEN], dtype='float32')
        review = text.lower()[:FEATURE_LEN-1:-1]
        i = 0
        for letter in text:
            if i >= FEATURE_LEN:
                break;
            if letter in ALPHABET_INDEX:
                encoded[ALPHABET_INDEX[letter]][i] = 1
            i += 1
        return nd.array([encoded], ctx=self.ctx)

    def _inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data)
        return output.softmax()

    def _postprocess(self, data):
        # Post process and output the most likely category
        predicted = self.labels[np.argmax(data[0].asnumpy())]
        return [{'category': predicted}]
