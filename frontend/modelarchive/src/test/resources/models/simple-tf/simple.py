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
Simple TF model example
"""
import tensorflow as tf
import numpy as np
import time
import os
import logging

class tf_service(object):
    def __init__(self):
        logging.info("tf_service init")
        self._context = None
        self.initialized = False
        self.ninp = 10
        self.nout = 5
        self.inputs = tf.placeholder(shape = [None, self.ninp],
                                     dtype=tf.float32)
        self.weights = tf.get_variable(name ="weights",
                                       dtype = tf.float32,
                                       shape = [self.ninp, self.nout],
                                       initializer=tf.contrib.layers.xavier_initializer())
        self.bias = tf.Variable(tf.zeros(shape= [self.nout]), dtype=tf.float32)
        self.pred = tf.add(tf.matmul(self.inputs, self.weights), self.bias)
        self.out = tf.argmax(self.pred, 1)
        self.sess = tf.Session()

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time

        :param context: model server context
        :return:
        """
        self.initialized = True
        self._context = context
        init = tf.global_variables_initializer()
        with self.sess.as_default():
            self.sess.run(init)

    def inference(self, model_input):
        """
        Internal inference methods

        :param model_input: transformed model input data
        :return: inference results
        """
        logging.info("tf_service inference [PID]: %d", os.getpid())
        inp = np.random.rand(1, self.ninp)
        with self.sess.as_default():
            out = self.sess.run(self.out,feed_dict={self.inputs:inp})
        return ["{} OK pred={}\n".format(os.getpid(), out[0])]

    def handle(self, data, context):
        """
        Custom service entry point function.

        :param context: model server context
        :param data: list of objects, raw input from request
        :return: list of outputs to be send back to client
        """
        # Add your initialization code here
        properties = context.system_properties
        try:
            start_time = time.time()
            data = self.inference(data)
            end_time = time.time()
            context.set_response_content_type(0, "text/plain")
            content_type = context.request_processor[0].\
                               get_request_property("Content-Type")
            return data
        except Exception as e:
            logging.error(e, exc_info=True)
            context.request_processor[0].\
                report_status(500,"Unknown inference error.")
            return ["Error {}".format(str(e))] * len(data)


_service = tf_service()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
