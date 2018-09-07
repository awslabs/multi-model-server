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
import ast
import os
from builtins import str
from collections import OrderedDict

from mms.model_loader import ModelLoader
from mms.arg_parser import ArgParser
from mms.service_manager import ServiceManager
from mms.log import log_msg, log_error
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mms.protocol.otf_message_handler import OtfCodecHandler
from mms.metrics.metric_encoder import MetricEncoder

MAX_FAILURE_THRESHOLD = 5
SOCKET_ACCEPT_TIMEOUT = 30.0
debug = False
BENCHMARK = False


class MXNetModelServiceWorker(object):
    """
    Backend worker to handle Model Server's python service code
    """
    def __init__(self, s_type=None, s_name=None, host_addr=None, port_num=None):
        if os.environ.get('OMP_NUM_THREADS') is None:
            os.environ['OMP_NUM_THREADS'] = "1"
        self.sock_type = s_type
        if s_type == 'unix':
            if s_name is None:
                raise MMSError(err.INVALID_ARGUMENTS, "Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = s_name, -1
            try:
                os.unlink(s_name)
            except OSError:
                if os.path.exists(s_name):
                    raise MMSError(err.SOCKET_ERROR, "socket already in use: {}.".format(s_name))
        elif s_type == 'tcp':
            self.sock_name = host_addr if host_addr is not None else "127.0.0.1"
            if port_num is None:
                raise MMSError(err.INVALID_ARGUMENTS, "Wrong arguments passed. No socket port given.")
            self.port = port_num
        else:
            raise ValueError("Incomplete data provided")

        self.model_services = {}
        self.service_manager = ServiceManager()
        self.send_failures = 0
        self.codec = OtfCodecHandler()

        try:
            msg = "Listening on port: {}\n".format(s_name)
            log_msg(msg)
            socket_family = socket.AF_INET if s_type == 'tcp' else socket.AF_UNIX
            self.sock = socket.socket(socket_family, socket.SOCK_STREAM)

        except (IOError, OSError) as e:
            raise MMSError(err.SOCKET_ERROR, "Socket error in init {}. {}".format(self.sock_name, repr(e)))
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.UNKNOWN_EXCEPTION, "{}".format(repr(e)))

    def retrieve_model_input(self, model_inputs, req_bat_content_type=None):
        """

        MODEL_INPUTS : [{
                "value": "val1"
                "name" : model_input_name (This is defined in the symbol file and the signature file)
        }]

        :param req_bat_content_type: Content-type of the request-batch, the outer scope of model-inputs
        :param model_inputs: list of model_input elements each containing "encoding", "value" and "name"
        :return:
        """

        model_in = OrderedDict()
        for _, ip in enumerate(model_inputs):
            ModelWorkerMessageValidators.validate_predict_inputs(ip)
            ip_name = ip.get('name')
            content_type = ip.get('contentType')

            if content_type is None or content_type == b'':
                content_type = req_bat_content_type

            if content_type is not None and content_type != b'' and "text" in content_type.decode():
                decoded_val = ip.get(u'value').decode()
            elif content_type is not None and content_type != b'' and "json" in content_type.decode():
                decoded_val = ast.literal_eval(ip.get(u'value').decode())
            else:
                decoded_val = ip.get(u'value')
            model_in.update({ip_name.decode(): decoded_val})

        return model_in

    def retrieve_data_for_inference(self, requests=None):
        """
        REQUESTS = [ {
            "requestId" : "111-222-3333",
            "modelInputs" : [ MODEL_INPUTS ]
        } ]

        MODEL_INPUTS = {
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

        input_batch = []
        for batch_idx, request_batch in enumerate(requests):
            ModelWorkerMessageValidators.validate_predict_data(request_batch)
            req_id = request_batch.get('requestId').decode()

            model_inputs = request_batch['modelInputs']
            req_batch_content_type = request_batch.get('contentType')
            try:
                input_data = self.retrieve_model_input(model_inputs, req_batch_content_type)
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
            "contentType": "http-content-types",
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
            model_name = data[u'modelName'].decode()
            loaded_services = self.service_manager.get_loaded_modelservices()
            if model_name not in loaded_services:
                raise MMSError(err.MODEL_SERVICE_NOT_LOADED, "Model {} is currently not loaded".format(model_name))
            model_service = loaded_services[model_name]
            req_batch = data[u'requestBatch']
            batch_size = len(req_batch)  # num-inputs gives the batch size
            input_batch, req_id_map, invalid_reqs = self.retrieve_data_for_inference(req_batch)
            if batch_size == 1:
                # Initialize metrics at service level
                model_service.metrics_init(model_name, req_id_map)
                retval.append(model_service.inference([input_batch[0][i] for i in input_batch[0]]))
                emit_metrics(model_service.metrics_store.store)
            else:
                raise MMSError(err.UNSUPPORTED_PREDICT_OPERATION, "Invalid batch size {}".format(batch_size))

            msg = self.codec.create_response(cmd=2, resp=retval, req_id_map=req_id_map, invalid_reqs=invalid_reqs)

        except ValueError as v:
            raise MMSError(err.INVALID_PREDICT_MESSAGE, "{}".format(repr(v)))
        except MMSError as m:
            raise m
        return msg, "Prediction success", 200

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
            ModelWorkerMessageValidators.validate_load_message(data)
            model_dir = data['modelPath'].decode()
            model_name = data['modelName'].decode()
            handler = data['handler'].decode()
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
            model_name = request[u'model-name'].decode()
            self.service_manager.unload_models(model_name)
        except KeyError:
            raise MMSError(err.MODEL_CURRENTLY_NOT_LOADED, "Model is not being served on model server")
        except MMSError as m:
            raise m
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.UNKNOWN_EXCEPTION, "Unknown error {}".format(repr(e)))
        return "Unloaded model {}".format(model_name), 200

    def send_response(self, sock, msg):
        """
        Send a response back to thae client
        :param sock:
        :param msg:
        :return:
        """
        try:
            sock.send(msg)
        except (IOError, OSError) as e:
            # Can't send this response. So, log it.
            self.send_failures += 1
            log_error("{}: Send failed. {}.\nMsg: {}".format(err.SEND_MSG_FAIL, repr(e), msg))

            if self.send_failures >= MAX_FAILURE_THRESHOLD:
                exit(err.SEND_FAILS_EXCEEDS_LIMITS)

    def create_and_send_response(self, sock, c, message, p=None):
        try:
            resp = bytearray()
            resp += self.codec.create_response(cmd=3, code=c, message=message, predictions=p)
            self.send_response(sock, resp)
        except Exception as e:
            log_error("{}".format(e))
            raise e

    def handle_connection(self, cl_socket):
        """
        Handle socket connection.

        :param cl_socket:
        :return:
        """
        predictions = None
        cmd = None

        while True:
            try:
                cmd, msg = self.codec.retrieve_msg(conn=cl_socket)
                if cmd.lower() == u'predict':
                    predictions, result, code = self.predict(msg)
                elif cmd.lower() == u'load':
                    result, code = self.load_model(msg)
                elif cmd.lower() == u'unload':
                    result, code = self.unload_model(msg)
                else:
                    result = "Received unknown command: {}".format(cmd)
                    code = err.UNKNOWN_COMMAND

                self.create_and_send_response(cl_socket, code, result, predictions)
            except MMSError as m:
                log_error("MMSError {} data {}".format(cmd, m.get_message()))
                if m.get_code() == err.SEND_FAILS_EXCEEDS_LIMITS or m.get_code() == err.ENCODE_FAILED or \
                   m.get_code() == err.DECODE_FAILED:
                    log_error("Can not recover from this error. Worker shutting down. {}".format(m))
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
            self.sock.settimeout(SOCKET_ACCEPT_TIMEOUT)
            self.sock.setblocking(True)  # workaround error(35, 'Resource temporarily unavailable') on OSX

            if self.sock_type == 'unix':
                self.sock.bind(self.sock_name)
            else:
                self.sock.bind((self.sock_name, int(self.port)))
            self.sock.listen(1)
            log_msg("[PID]{}".format(os.getpid()))
            log_msg("MxNet worker started.")
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(err.SOCKET_BIND_ERROR,
                           "Socket {} could not be bound to. {}".format(self.sock_name, repr(e)))

        while True:
            #  TODO: In the initial release we will only support single connections to a worker. If the
            # socket fails, the backend worker will quit

            try:
                log_msg("Waiting for a connection")

                if BENCHMARK:
                    pr.disable()
                    pr.dump_stats('/tmp/mmsPythonProfile.prof')
                (cl_socket, _) = self.sock.accept()
                if BENCHMARK:
                    pr.enable()
                self.handle_connection(cl_socket)
                if debug is False:
                    exit(1)

            except socket.timeout:
                log_error("Backend worker did not receive connection from frontend")
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
    met_list = [str(met) for met in metrics]
    log_msg("[METRICS] [", ",".join(met_list), "]")


if __name__ == "__main__":
    args = ArgParser.model_service_worker_args().parse_args()
    debug = False
    socket_name = args.sock_name
    sock_type = args.sock_type
    host = args.host
    port = args.port
    worker = None
    try:
        if BENCHMARK:
            import cProfile
            pr = cProfile.Profile()
            pr.enable()
        worker = MXNetModelServiceWorker(sock_type, socket_name, host, port)
        worker.run_server()
        if BENCHMARK:
            pr.disable()
            pr.dump_stats('/tmp/mmsPythonProfile.prof')
    except MMSError as m:
        log_error("{}".format(m.get_message()))
        exit(1)
    except Exception as ex:  # pylint: disable=broad-except
        log_error("Error starting the server. {}".format(str(ex)))
        exit(1)
    exit(0)
