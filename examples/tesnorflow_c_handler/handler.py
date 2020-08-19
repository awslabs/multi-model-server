# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Defines an API for Pybind Module of Tensorflow C API.
"""
import json
import os
import logging
import pathlib
import tf_c_inference  # import pybind module


class ModelHandler(object):
    """
    A base Model handler implementation.
    """

    def __init__(self):
        self.error = None
        self._context = None
        self._batch_size = 0
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self._batch_size = context.system_properties["batch_size"]

        # Call the utility to import the graph definition into default graph.
        # The MobileNet model is downloaded from here
        # https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
        out = tf_c_inference.load_model("mobilenet_v2_1.4_224_frozen.pb",
                                        "input",
                                        "MobilenetV2/Predictions/Reshape_1")

        self.initialized = True

    def preprocess(self, request):
        """
        Decode all input images into ndarray.

        Note: This implementation doesn't properly handle error cases in batch mode,
        If one of the input images is corrupted, all requests in the batch will fail.

        :param request:
        :return:
        """

        img_list = []
        img_shape_list = []

        logging.info("Worker :{} received {} requests with batch size {}".format(os.getpid(), len(request),
                                                                                 self._batch_size))
        for idx, data in enumerate(request):
            if idx < len(request):
                data = request[idx]
                img = data.get("body")

                if img is None:
                    img = data.get("data")

                if img is None or len(img) == 0:
                    self.error = "Empty image input"
                    return None

                if isinstance(img, (bytes, bytearray, str)):
                    img = json.loads(img)

                img_shape = img["inputs"][0]["shape"]
                arr = np.array(img["inputs"][0]["data"]).flatten().tolist()
                img_list.append(arr)
                img_shape_list.append(img_shape)

        return img_list, img_shape_list

    def inference(self, input_data, input_shape):
        """
         Internal inference methods for MXNet. Run forward computation and
         return output.

         :param model_input: list of NDArray
             Preprocessed inputs in NDArray format.
         :return: list of NDArray
             Inference output.
         """
        # Assuming MMS batch size 1
        return tf_c_inference.run_model(input_data[0], input_shape[0])

    def postprocess(self, inference_output):
        """
        Return predict result in batch.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        return inference_output

    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        input_data, input_shape = self.preprocess(data)
        model_out = self.inference(input_data, input_shape)
        return self.postprocess(model_out)


_service = ModelHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None
        return _service.handle(data, context)

    except Exception as e:
        import traceback
        return [[str(e), traceback.format_exc()]]


if __name__ == "__main__":
    import mms
    import numpy as np
    from mms.context import Context

    source_path = pathlib.Path(__file__).parent.absolute()
    model_dir = '{}/saved'.format(source_path)
    context = Context('tf_model', model_dir, '', 1, 'gpu', mms.__version__)

    data = np.random.uniform(size=(3, 224, 224, 3)).astype('float32')

    json_data = {"id": "1",
                 "inputs": [{"name": "input",
                             "shape": [3, 224, 224, 3],
                             "datatype": "FP32", "parameters": {},
                             "data": data.tolist()}
                            ]
                 }
    data = [{"body": json.dumps(json_data)}]

    print(handle(data, context))

# model-archiver --model-name c_model --handler c_model:handle --export-path ~/tmp/models --model-path c_model -f
# multi-model-server --start --model-store=/users/demo/tmp/models/
# curl -s -X DELETE http://localhost:8081/models/c_model
# curl -s -X POST "http://localhost:8081/models?url=c_model.mar&batch_size=4&max_batch_delay=30"
# curl -s -X PUT  "http://localhost:8081/models/c_model?min_worker=2&synchronous=true"
# curl -X POST http://127.0.0.1:8080/predictions/c_model --data @car.json  -H "Content-Type: application/json"
# curl -X POST http://127.0.0.1:8080/predictions/c_model --data @coffe-mug.json  -H "Content-Type: application/json"
# curl -X POST http://127.0.0.1:8080/predictions/c_model --data @valture.json  -H "Content-Type: application/json"


## tar.gz format
## model-archiver --model-name c_model --handler c_model:handle --export-path ~/tmp/models --model-path c_model --archive-format tgz
