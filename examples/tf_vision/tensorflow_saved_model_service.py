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
TensorflowSavedModelService defines an API for running a tensorflow saved model
"""
import json
import os

import tensorflow as tf

from model_handler import ModelHandler


class TensorflowSavedModelService(ModelHandler):
    """
    TensorflowSavedModelService defines the fundamental loading model and inference
    operations when serving a TF saved model. This is a base class and needs to be
    inherited.
    """

    def __init__(self):
        super(TensorflowSavedModelService, self).__init__()
        self.predictor = None
        self.labels = None
        self.signature = None
        self.epoch = 0

    # noinspection PyMethodMayBeStatic
    def get_model_files_prefix(self, context):
        return context.manifest["model"]["modelName"]

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: Initial context contains model server system properties.
        :return:
        """
        super(TensorflowSavedModelService, self).initialize(context)

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")

        with open(signature_file_path) as f:
            self.signature = json.load(f)

        #Define signature.json and work here
        data_names = []
        data_shapes = []
        for input_data in self.signature["inputs"]:
            data_name = input_data["data_name"]
            data_shape = input_data["data_shape"]

            # Replace 0 entry in data shape with 1 for binding executor.
            for idx in range(len(data_shape)):
                if data_shape[idx] == 0:
                    data_shape[idx] = 1

            data_names.append(data_name)
            data_shapes.append((data_name, tuple(data_shape)))

        self.predictor = tf.contrib.predictor.from_saved_model(model_dir)

    def inference(self, model_input):
        """
        Internal inference methods for TF - saved model. Run forward computation and
        return output.

        :param model_input: list of dict of {name : numpy_array}
            Batch of preprocessed inputs in tensor dict.
        :return: list of dict of {name: numpy_array}
            Batch of inference output tensor dict
        """
        if self.error is not None:
            return None

        # Check input shape
        check_input_shape(model_input, self.signature)

        #Restricting to one request which contains the whole batch. Remove this line if adding custom batching support
        model_input = model_input[0]

        results = self.predictor(model_input)

        return results

def check_input_shape(inputs, signature):
    """
    Check input data shape consistency with signature.

    Parameters
    ----------
    inputs : List of dicts
        Input data in this format [{input_name: input_tensor, input2_name: input2_tensor}, {...}]
    signature : dict
        Dictionary containing model signature.
    """

    assert isinstance(inputs, list), 'Input data must be a list.'
    for input_dict in inputs:
        assert isinstance(input_dict, dict), 'Each request must be dict of input_name: input_tensor.'
        assert len(input_dict) == len(signature["inputs"]), \
            "Input number mismatches with " \
            "signature. %d expected but got %d." \
            % (len(signature['inputs']), len(input_dict))
        for tensor_name, sig_input in zip(input_dict, signature["inputs"]):
            assert len(input_dict[tensor_name].shape) == len(sig_input["data_shape"]), \
                'Shape dimension of input %s mismatches with ' \
                'signature. %d expected but got %d.' \
                % (sig_input['data_name'],
                   len(sig_input['data_shape']),
                   len(input_dict[tensor_name].shape))
            for idx in range(len(input_dict[tensor_name].shape)):
                if idx != 0 and sig_input['data_shape'][idx] != 0:
                    assert sig_input['data_shape'][idx] == input_dict[tensor_name].shape[idx], \
                        'Input %s has different shape with ' \
                        'signature. %s expected but got %s.' \
                        % (sig_input['data_name'], sig_input['data_shape'],
                           input_dict[tensor_name].shape)
