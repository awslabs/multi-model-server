# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from mxnet import gluon
from mxnet.gluon import nn
from mms.model_service.gluon_vision_service import GluonVisionService

"""
MMS examples for loading Gluon Imperative models
"""


class GluonImperativeAlexNet(gluon.Block):
    """
    Fully imperative gluon Alexnet model
    """
    def __init__(self, classes=1000, **kwargs):
        super(GluonImperativeAlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(classes)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


class MMSImperativeService(GluonVisionService):
    """
    Gluon alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSImperativeService, self).__init__(model_name, model_dir, manifest, GluonImperativeAlexNet(), gpu)

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
            float(data[0, int(i.asscalar())].asscalar())} for i in idx]
