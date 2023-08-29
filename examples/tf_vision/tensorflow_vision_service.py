# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
TensorflowVisionService defines a TF based vision service
"""
import logging

from tensorflow_saved_model_service import TensorflowSavedModelService
import image


class TensorflowVisionService(TensorflowSavedModelService):
    """
    TensorflowVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to numpy and resized respect to input
    shape in signature.
    In post process, raw tensors are returned.
    """

    def preprocess(self, request):
        """
        Decode all input images into numpy array.

        Note: This implementation doesn't properly handle error cases in batch mode,
        If one of the input images is corrupted, all requests in the batch will fail.

        :param request:
        :return:
        """
        img_list = []
        param_name = self.signature['inputs'][0]['data_name']
        input_shape = self.signature['inputs'][0]['data_shape']

        for idx, data in enumerate(request):
            img = data.get(param_name)
            if img is None:
                img = data.get("body")

            if img is None:
                img = data.get("data")

            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            # We are assuming input shape is NHWC
            [h, w] = input_shape[1:3]

            try:
                img_arr = image.read(img)
            except Exception as e:
                logging.warn(e, exc_info=True)
                self.error = "Corrupted image input"
                return None

            img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)
            img_list.append(img_arr)

        #Convert to dict before returning [{name: image}]
        img_list = [{param_name: img} for img in img_list]
        return img_list

    def postprocess(self, data):
        if self.error is not None:
            return [self.error] * self._batch_size

        for key in data:
            data[key] = str(data[key])

        return [data]


_service = TensorflowVisionService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)