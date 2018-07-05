# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.


import socket
import os
import sys
import json

from mms.service_manager import ServiceManager
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators
from mms.utils.codec_helpers.codec import ModelWorkerCodecHelper
from mms.mxnet_model_service_error import MMSError
from mms.utils.validators.validate_model_artifacts import ModelArtifactValidator
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err

"""
ModelServiceWorker is the worker that is started by the MMS front-end. 

Communication message format: JSON message 
 
"""

MAX_FAILURE_THRESHOLD = 5


class MXNetModelServiceWorker(object):
    def __init__(self, sock_name=None):
        if sock_name is None:
            raise ValueError("Incomplete data provided: Model worker expects \"socket name\"")
        self.sock_name = sock_name
        self.model_services = {}
        self.service_manager = ServiceManager()
        self.send_failures = 0

        try:
            os.unlink(sock_name)
        except OSError:
            if os.path.exists(sock_name):
                raise MMSError(err.SOCKET_ERROR, "socket already in use: {}.".format(sock_name))

        try:
            msg = "Listening on port: {}".format(sock_name)
            log_msg(msg)

            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except (IOError, OSError) as e:
            raise MMSError(err.SOCKET_ERROR, "Socket error in init {}. {}".format(self.sock_name, repr(e)))
        except Exception as e:
            raise MMSError(err.UNKNOWN_EXCEPTION, "{}".format(repr(e)))

    def retrieve_input_names_from_signature(self, signature_file_inputs):
        d_names = set()
        if signature_file_inputs is None:
            raise ValueError("Invalid model service given to retrieve input names")

        for item in signature_file_inputs:
            d_names.add(item['data_name'])

        return d_names

    def create_predict_response(self, ret, req_id_map, invalid_reqs):
        """
        Response object is as follows :
        RESPONSE =
        {
            "code": val,
            "message": "Success"
            "predictions": [ PREDICTION_RESULTS ]
        }

        PREDICTION_RESULTS = {
            "requestId": 111-222-3333,
            "code": "Success/Fail" # TODO: Add this
            "value": Abefz23=,
            "encoding": "utf-8, base64"
        }

        :param ret:
        :param req_id_map:
        :param invalid_reqs:
        :return:
        """
        encoding = u'base64'  # TODO: Remove this hardcoding and encode based on output mime type
        result = {}
        try:
            for idx in range(len(ret)):
                result.update({"requestId": req_id_map[idx]})
                result.update({"code": 200})
                result.update({"value":
                               ModelWorkerCodecHelper.encode_msg(encoding, bytes(json.dumps(ret[idx]).encode()))})
                result.update({"encoding": encoding})

            for req in invalid_reqs.keys():
                result.update({"requestId": req})
                result.update({"code": invalid_reqs.get(req)})
                result.update({"value": "Invalid input provided".encode(encoding)})
                result.update({"encoding": encoding})

            resp = []
            resp.append(result)

        except Exception as e:
            raise MMSError(err.CODEC_FAIL, "codec failed {}".format(repr(e)))
        return resp

    @staticmethod
    def recv_msg(client_sock):
        data = b''
        try:
            while True:
                pkt = client_sock.recv(1024)
                data += pkt
                # Check if we received last segment
                if pkt[-2:] == '\r\n':
                    break
            in_msg = json.loads(data.decode('utf8'))
            if u'command' not in in_msg:
                raise MMSError(err.INVALID_COMMAND, "Invalid message received")
        except (IOError, OSError) as sock_err:
            raise MMSError(err.RECEIVE_ERROR, "{}".format(sock_err.message))
        except ValueError as v:
            raise MMSError(err.INVALID_MESSAGE, "JSON message format error: {}".format(v))
        except Exception as e:
            raise MMSError(err.UNKNOWN_EXCEPTION, "{}".format(e))

        return in_msg['command'], in_msg

    def retrieve_model_input(self, model_inputs, input_names=None):
        """
        MODEL_INPUTS = [{
                "encoding": "base64/utf-8", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param model_inputs: list of model_input elements each containing "encoding", "value" and "name"
        :param input_names:
        :return:
        """

        model_in = {}
        validation_set = set()  # Set to validate if all the inputs were provided
        for input_idx in range(len(model_inputs)):
            input = model_inputs[input_idx]
            ModelWorkerMessageValidators.validate_predict_inputs(input)
            input_name = input[u'name']
            encoding = input[u'encoding']
            validation_set.add(input_name)
            decoded_val = ModelWorkerCodecHelper.decode_msg(encoding, input[u'value'])

            model_in.update({input_name: decoded_val})

        if len(input_names.symmetric_difference(validation_set)) != 0:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "Missing input data "
                                                        "{}".format(input_names.symmetric_difference(validation_set)))
        return model_in

    def retrieve_data_for_inference(self, requests=None, model_service=None):
        """
        REQUESTS = [ {
            "requestId" : "111-222-3333",
            "encoding" : "None | base64 | utf-8",
            "modelInputs" : [ MODEL_INPUTS ]
        } ]

        MODEL_INPUTS = {
                "encoding": "base64/utf-8", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }

        inputs: List of requests
        Returns a list(dict(inputs))
        """

        req_to_id_map = {}
        invalid_reqs = {}

        if requests is None:
            raise ValueError("Received invalid inputs")

        if req_to_id_map is None:
            raise ValueError("Request ID map is invalid")

        if model_service is None:
            raise ValueError("Model Service metadata is invalid")

        input_batch = []
        try:
            signature = model_service.signature
            input_data_names = self.retrieve_input_names_from_signature(signature['inputs'])
        except AttributeError as e:
            msg = "Attribute error {}".format(e)
            log_msg(msg)
            input_data_names = {u'data'}  # TODO: Remove this default

        for batch_idx in range(len(requests)):
            ModelWorkerMessageValidators.validate_predict_data(requests[batch_idx])
            req_id = requests[batch_idx][u'requestId']
            # TODO: If encoding present in "REQUEST" we shouldn't look for input-names and just pass it to the
            # custom service code.

            model_inputs = requests[batch_idx][u'modelInputs']
            try:
                input_data = self.retrieve_model_input(model_inputs, input_data_names)
                input_batch.append(input_data)
                req_to_id_map[batch_idx] = req_id
            except MMSError as m:
                invalid_reqs.update({req_id: m.get_code()})

        return input_batch, req_to_id_map, invalid_reqs

    def predict(self, data):
        """
        PREDICT COMMAND = {
            "command": "predict",
            "modelName": "model-to-run-inference-against",
            "contentType": "http-content-types", # TODO: Add this
            "requestBatch": [ REQUESTS ]
        }

        REQUESTS = {
            "requestId" : "111-222-3333",
            "encoding" : "None|base64|utf-8",
            "modelInputs" : [ MODEL_INPUTS ]
        }

        MODEL_INPUTS = {
                "encoding": "base64/utf-8", (# This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (# This is defined in the symbol file and the signature file)
        }

        :param data:
        :return:

        """
        try:
            retval = []
            ModelWorkerMessageValidators.validate_predict_msg(data)
            model_name = data[u'modelName']
            loaded_services = self.service_manager.get_loaded_modelservices()
            if model_name not in loaded_services:
                raise MMSError(err.MODEL_SERVICE_NOT_LOADED, "Model {} is currently not loaded".format(model_name))
            model_service = loaded_services[model_name]
            req_batch = data[u'requestBatch']
            batch_size = len(req_batch)  # num-inputs gives the batch size
            input_batch, req_id_map, invalid_reqs = self.retrieve_data_for_inference(req_batch, model_service)
            if batch_size == 1:
                retval.append(model_service.inference([input_batch[0][i] for i in input_batch[0]]))
            else:
                raise MMSError(err.UNSUPPORTED_PREDICT_OPERATION, "Invalid batch size {}".format(batch_size))

            response = self.create_predict_response(retval, req_id_map, invalid_reqs)

        except ValueError as v:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "{}".format(repr(v)))
        except MMSError as m:
            raise m
        return response, "Prediction success", 200

    def load_model(self, data):
        """
        Expected command
        {
            "command" : "load", string
            "model-path" : "/path/to/model/file", string
            "model-name" : "name", string
            "gpu" : None if CPU else gpu_id, int
        }

        :param data:
        :return:
        """
        gpu = None
        try:
            from mms.model_loader import ModelLoader
            ModelWorkerMessageValidators.validate_load_message(data)
            path = data[u'modelPath']
            model_name = data[u'modelName']

            if u'gpu' in data:
                gpu = int(data[u'gpu'])

            model_dict = {model_name: path}

            models = ModelLoader.load(model_dict)
            ModelArtifactValidator.validate_model_metadata(models)
            manifest = models[0][3]
            service_file_path = os.path.join(models[0][2], manifest['Model']['Service'])

            self.service_manager.register_and_load_modules(service_file_path, models, gpu)
        except ValueError as v:
            raise MMSError(err.VALUE_ERROR_WHILE_LOADING, "{}".format(v))
        except MMSError as m:
            raise m
        except Exception as e:
            raise MMSError(err.UNKNOWN_EXCEPTION_WHILE_LOADING, "{}".format(repr(e)))

        return "loaded model {}".format(service_file_path), 200

    def unload_model(self, request):
        """
        Expected request
        {
            "command" : "unload",
            "model-name": "name"
        }

        :param request:
        :return:
        """
        try:
            ModelWorkerMessageValidators.validate_unload_msg(request)
            model_name = request[u'model-name']
            self.service_manager.unload_models(model_name)
        except KeyError:
            raise MMSError(err.MODEL_CURRENTLY_NOT_LOADED, "Model is not being served on model server")
        except MMSError as m:
            raise m
        except Exception as e:
            raise MMSError(err.UNKNOWN_EXCEPTION, "Unknown error {}".format(repr(e)))
        return "Unloaded model {}".format(model_name), 200

    def stop_server(self, sock):
        """
        Expected request
        {
             "command" : "stop"
        }
        :param sock:
        :return:
        """
        if sock is None:
            raise ValueError("Invalid parameter passed to stop server connection")
        try:
            resp = {}
            resp['code'] = 200
            resp['response'] = "Stopped server"
            self.send_response(sock, json.dumps(resp).encode('utf-8'))
            sock.close()
            # os.unlink(self.sock_name)
        except Exception as e:
            msg = "Error closing the socket {}. Msg: {}".format(sock, repr(e))
            log_msg(msg)

    def send_response(self, sock, msg):
        """
        Send a response back to thae client
        :param sock:
        :param msg:
        :return:
        """
        try:
            msg += '\r\n'
            sock.send(msg.encode())
        except (IOError, OSError) as e:
            # Can't send this response. So, log it.
            self.send_failures += 1
            msg = "{}: Send failed. {}.\nMsg: {}".format(err.SEND_MSG_FAIL, repr(e), msg)
            log_msg(msg)

            if self.send_failures >= MAX_FAILURE_THRESHOLD:
                exit(err.SEND_FAILS_EXCEEDS_LIMITS)

    def create_and_send_response(self, sock, c, m, p=None):
        resp = {}
        resp['code'] = c
        resp['message'] = m
        if p is not None:
            resp['predictions'] = p
        self.send_response(sock, json.dumps(resp))

    def handle_connection(self, cl_socket):
            predictions = None
            code = 200
            result = None
            cmd = None

            while True:
                try:
                    cmd, data = self.recv_msg(cl_socket)
                    resp = {}
                    if cmd.lower() == u'stop':
                        self.stop_server(cl_socket)
                        exit(1)
                    elif cmd.lower() == u'predict':
                        predictions, result, code = self.predict(data)
                    elif cmd.lower() == u'load':
                        result, code = self.load_model(data)
                    elif cmd.lower() == u'unload':
                        result, code = self.unload_model(data)
                    else:
                        result = "Received unknown command: {}".format(cmd)
                        code = err.UNKNOWN_COMMAND

                    self.create_and_send_response(cl_socket, code, result, predictions)

                except MMSError as m:
                    msg = "MMSError {} data {}".format(cmd, m.get_message())
                    log_msg(msg)
                    self.create_and_send_response(cl_socket, m.get_code(), m.get_message())
                except Exception as e:
                    msg = "Exception {} data {}".format(cmd, repr(e))
                    log_msg(msg)
                    self.create_and_send_response(cl_socket, err.UNKNOWN_EXCEPTION, repr(e))

    def run_server(self):
        try:
            self.sock.bind(self.sock_name)
            self.sock.listen(1)
            msg = "MxNet worker started.\n"
            log_msg(msg)

        except Exception as e:
            raise MMSError(err.SOCKET_BIND_ERROR,
                           "Socket {} could not be bound to. {}: {}".format(self.sock_name, e.__module__, e.message))

        # while True:
        #  TODO: In the initial release we will only support single connections to a worker. If the
        # socket fails, the backend worker will quit

        try:
            msg = "Waiting for a connections"
            log_msg(msg)

            (cl_socket, address) = self.sock.accept()
            self.handle_connection(cl_socket)
        except (OSError, IOError) as e:
            raise e
        except Exception:
            raise


def log_msg(*args):
    print(" ".join(map(str, args)))
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        assert 0, "Invalid parameters given"
    sock_name = sys.argv[1]
    worker = None
    try:
        worker = MXNetModelServiceWorker(sock_name)
        worker.run_server()
    except MMSError as m:
        msg = "{}".format(m.message())
        log_msg(msg)
        exit(1)
    except Exception as e:
        print()
        msg = "Error starting the server. {}".format(str(e))
        log_msg(msg)
        exit(1)
    exit(0)
