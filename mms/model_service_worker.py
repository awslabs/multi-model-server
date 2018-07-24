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
ModelServiceWorker is the worker that is started by the MMS front-end.
Communication message format: JSON message
"""

# pylint: disable=redefined-builtin

import socket
import os
import sys
import json
from builtins import bytes
from builtins import str

from mms.service_manager import ServiceManager
from mms.log import log_msg, log_error
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators
from mms.utils.codec_helpers.codec import ModelWorkerCodecHelper
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mms.metrics.metric_encoder import MetricEncoder

MAX_FAILURE_THRESHOLD = 5


class MXNetModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """
    def __init__(self, s_name=None):
        if s_name is None:
            raise ValueError("Incomplete data provided: Model worker expects \"socket name\"")
        self.sock_name = s_name
        self.model_services = {}
        self.service_manager = ServiceManager()
        self.send_failures = 0

        try:
            os.unlink(s_name)
        except OSError:
            if os.path.exists(s_name):
                raise MMSError(err.SOCKET_ERROR, "socket already in use: {}.".format(s_name))

        try:
            msg = "Listening on port: {}\n".format(s_name)
            log_msg(msg)

            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except (IOError, OSError) as e:
            raise MMSError(err.SOCKET_ERROR, "Socket error in init {}. {}".format(self.sock_name, repr(e)))
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.UNKNOWN_EXCEPTION, "{}".format(repr(e)))

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
        result = {}
        encoding = u'base64'
        try:
            for idx, val in enumerate(ret):
                result.update({"requestId": req_id_map[idx]})
                result.update({"code": 200})

                if isinstance(val, bytes):
                    value = ModelWorkerCodecHelper.encode_msg(encoding, val)
                elif isinstance(val, str):
                    value = ModelWorkerCodecHelper.encode_msg(encoding, val.encode('utf-8'))
                else:
                    value = ModelWorkerCodecHelper.encode_msg(encoding, json.dumps(val).encode('utf-8'))

                result.update({"value": value})
                result.update({"encoding": encoding})

            for req in invalid_reqs.keys():
                result.update({"requestId": req})
                result.update({"code": invalid_reqs.get(req)})
                result.update({"value": "Invalid input provided".encode(encoding)})
                result.update({"encoding": encoding})

            resp = [result]

        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.CODEC_FAIL, "codec failed {}".format(repr(e)))
        return resp

    @staticmethod
    def recv_msg(client_sock):
        """
        Receive a message from a given socket file descriptor
        :param client_sock:
        :return:
        """
        data = b''
        try:
            while True:
                pkt = client_sock.recv(1024)
                if not pkt:
                    exit(1)

                data += pkt
                # Check if we received last segment. Assumption is we receive only one packet at a time.
                # TODO:
                # Modify this. If we receive multiple packets, the '\r\n' be in the middle of the pkt/data. We
                # need to buffer the rest.
                if data[-2:] == b'\r\n':
                    break
            in_msg = json.loads(data.decode('utf8'))
            if u'command' not in in_msg:
                raise MMSError(err.INVALID_COMMAND, "Invalid message received")
        except (IOError, OSError) as sock_err:
            raise MMSError(err.RECEIVE_ERROR, "{}".format(sock_err.message))
        except ValueError as v:
            raise MMSError(err.INVALID_REQUEST, "JSON message format error: {}".format(v))
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.UNKNOWN_EXCEPTION, "{}".format(e))

        return in_msg['command'], in_msg

    def retrieve_model_input(self, model_inputs):
        """
        MODEL_INPUTS = [{
                "encoding": "base64", (This is how the value is encoded)
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param model_inputs: list of model_input elements each containing "encoding", "value" and "name"
        :return:
        """

        model_in = dict()
        for _, ip in enumerate(model_inputs):
            ModelWorkerMessageValidators.validate_predict_inputs(ip)
            ip_name = ip.get(u'name')
            encoding = ip.get('encoding')
            decoded_val = ModelWorkerCodecHelper.decode_msg(encoding, ip['value'])

            model_in.update({ip_name: decoded_val})

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
        for batch_idx, request_batch in enumerate(requests):
            ModelWorkerMessageValidators.validate_predict_data(request_batch)
            req_id = request_batch['requestId']
            # TODO: If encoding present in "REQUEST" we shouldn't look for input-names and just pass it to the
            # custom service code.

            model_inputs = request_batch['modelInputs']
            try:
                input_data = self.retrieve_model_input(model_inputs)
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
                # Initialize metrics at service level
                model_service.metrics_init(model_name, req_id_map)
                retval.append(model_service.inference([input_batch[0][i] for i in input_batch[0]]))
                # Dump metrics
                emit_metrics(model_service.metrics.store)

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
            "modelPath" : "/path/to/model/file", string
            "modelName" : "name", string
            "gpu" : None if CPU else gpu_id, int
            "handler" : service handler entry point if provided, string
            "batchSize" : batch size, int
        }

        :param data:
        :return:
        """
        try:
            from mms.model_loader import ModelLoader
            ModelWorkerMessageValidators.validate_load_message(data)
            model_dir = data['modelPath']
            model_name = data['modelName']
            handler = data['handler']
            batch_size = None
            if 'batchSize' in data:
                batch_size = int(data['batchSize'])

            gpu = None
            if u'gpu' in data:
                gpu = int(data[u'gpu'])

            manifest, service_file_path = ModelLoader.load(model_dir, handler)

            self.service_manager.register_and_load_modules(model_name, model_dir, manifest,
                                                           service_file_path, gpu, batch_size)
        except ValueError as v:
            raise MMSError(err.VALUE_ERROR_WHILE_LOADING, "{}".format(v))
        except MMSError as m:
            raise m
        except Exception as e:  # pylint: disable=broad-except
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
        except Exception as e:  # pylint: disable=broad-except
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
            resp = {'code': 200, 'response': "Stopped server"}
            self.send_response(sock, json.dumps(resp).encode('utf-8'))
            sock.close()
            # os.unlink(self.sock_name)
        except Exception as e:  # pylint: disable=broad-except
            log_msg("Error closing the socket {}. Msg: {}".format(sock, repr(e)))

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
            log_error("{}: Send failed. {}.\nMsg: {}".format(err.SEND_MSG_FAIL, repr(e), msg))

            if self.send_failures >= MAX_FAILURE_THRESHOLD:
                exit(err.SEND_FAILS_EXCEEDS_LIMITS)

    def create_and_send_response(self, sock, c, message, p=None):
        try:
            resp = {'code': c, 'message': message}
            if p is not None:
                resp['predictions'] = p
            self.send_response(sock, json.dumps(resp))
        except Exception as ex:
            log_error("{}".format(ex))
            raise ex

    def handle_connection(self, cl_socket):
        """
        Handle socket connection.

        :param cl_socket:
        :return:
        """
        predictions = None
        code = 200
        result = None
        cmd = None

        while True:
            try:
                cmd, data = self.recv_msg(cl_socket)
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
                log_error("MMSError {} data {}".format(cmd, m.get_message()))
                if m.get_code() == err.SEND_FAILS_EXCEEDS_LIMITS:
                    log_error("Exceeded send failures. Ending the connections. {}".format(m))
                    break
                self.create_and_send_response(cl_socket, m.get_code(), m.get_message())
            except Exception as e:  # pylint: disable=broad-except
                log_error("Exception {} data {}".format(cmd, repr(e)))
                self.create_and_send_response(cl_socket, err.UNKNOWN_EXCEPTION, repr(e))

    def run_server(self):
        """
        Run the backend worker process and listen on a socket
        :return:
        """
        try:
            self.sock.bind(self.sock_name)
            self.sock.listen(1)
            log_msg("MxNet worker started.\n")

        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.SOCKET_BIND_ERROR,
                           "Socket {} could not be bound to. {}: {}".format(self.sock_name, e.__module__, e.message))

        while True:
            #  TODO: In the initial release we will only support single connections to a worker. If the
            # socket fails, the backend worker will quit

            try:
                log_msg("Waiting for a connections")

                (cl_socket, _) = self.sock.accept()
                self.handle_connection(cl_socket)
                if debug is False:
                    exit(1)

            except Exception as ex:  # pylint: disable=broad-except
                if debug is False:
                    raise ex
                log_error("Backend worker error {}".format(ex))


def emit_metrics(metrics):
    """
    Emit the metrics in the provided Dictionary

    Parameters
    ----------
    metrics: Dictionary
    A dictionary of all metrics, when key is metric_name
    value is a metric object
    """

    print('[METRICS]')
    sys.stdout.flush()
    print(json.dumps(metrics, indent=4, separators=(',', ':'), cls=MetricEncoder))
    sys.stdout.flush()
    print('[/METRICS]')
    sys.stdout.flush()


if __name__ == "__main__":
    # TODO: Use the argprocess
    debug = False
    if len(sys.argv) != 2:
        assert 0, "Invalid parameters given"
    socket_name = sys.argv[1]
    worker = None
    try:
        worker = MXNetModelServiceWorker(socket_name)
        worker.run_server()
    except MMSError as m:
        log_error("{}".format(m.get_message()))
        exit(1)
    except Exception as e:  # pylint: disable=broad-except
        log_error("Error starting the server. {}".format(str(e)))
        exit(1)
    exit(0)
