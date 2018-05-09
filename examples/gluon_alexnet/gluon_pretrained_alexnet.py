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
from mms.model_service.mxnet_vision_service import MXNetVisionService
import numpy as np


class MMSPretrainedAlexnet(MXNetVisionService):
    """
    Pretrained alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSPretrainedAlexnet, self).__init__(model_name, model_dir, manifest, gpu)

        self.net = mxnet.gluon.model_zoo.vision.alexnet(pretrained=True)

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = mxnet.img.imdecode(img)
            img_arr = mxnet.image.imresize(img_arr, w, h)
            img_arr = img_arr.astype(np.float32)
            img_arr /= 255
            img_arr = mxnet.image.color_normalize(img_arr,
                                                  mean=mxnet.nd.array([0.485, 0.456, 0.406]),
                                                  std=mxnet.nd.array([0.229, 0.224, 0.225]))
            img_arr = mxnet.nd.transpose(img_arr, (2, 0, 1))
            img_arr = img_arr.expand_dims(axis=0)
            img_list.append(img_arr)
        return img_list

    def _inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data[0])
        return output.softmax()

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability': float(data[0, int(i.asscalar())].asscalar())}
                for i in idx]
