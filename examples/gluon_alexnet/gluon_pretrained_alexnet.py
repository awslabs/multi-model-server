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
from mms.model_service.gluon_vision_service import GluonVisionService


class MMSPretrainedAlexnet(GluonVisionService):
    """
    Pretrained alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSPretrainedAlexnet, self).__init__(model_name, model_dir, manifest,
                                                   mxnet.gluon.model_zoo.vision.alexnet(pretrained=True),
                                                   gpu)

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability':
            float(data[0, int(i.asscalar())].asscalar())} for i in idx]
