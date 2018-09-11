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

import os
import socket

from mms.arg_parser import ArgParser
from mms.log import log_msg, log_error
from mms.model_loader import ModelLoaderFactory
from mms.mxnet_model_service_error import MMSError
from mms.protocol.otf_message_handler import OtfCodecHandler
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err
from mms.utils.validators.validate_messages import ModelWorkerMessageValidators

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
                raise MMSError(Err.INVALID_ARGUMENTS, "Wrong arguments passed. No socket name given.")
            self.sock_name, self.port = s_name, -1
            try:
                os.unlink(s_name)
            except OSError:
                if os.path.exists(s_name):
                    raise MMSError(Err.SOCKET_ERROR, "socket already in use: {}.".format(s_name))
        elif s_type == 'tcp':
            self.sock_name = host_addr if host_addr is not None else "127.0.0.1"
            if port_num is None:
                raise MMSError(Err.INVALID_ARGUMENTS, "Wrong arguments passed. No socket port given.")
            self.port = port_num
        else:
            raise ValueError("Incomplete data provided")

        self.model_services = {}
        self.send_failures = 0
        self.codec = OtfCodecHandler()

        try:
            msg = "Listening on port: {}\n".format(s_name)
            log_msg(msg)
            socket_family = socket.AF_INET if s_type == 'tcp' else socket.AF_UNIX
            self.sock = socket.socket(socket_family, socket.SOCK_STREAM)

        except (IOError, OSError) as e:
            raise MMSError(Err.SOCKET_ERROR, "Socket error in init {}. {}".format(self.sock_name, repr(e)))
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(Err.UNKNOWN_EXCEPTION, "{}".format(repr(e)))

    @staticmethod
    def load_model(data):
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

            model_loader = ModelLoaderFactory.get_model_loader(model_dir)
            service = model_loader.load(model_name, model_dir, handler, gpu, batch_size)
            return service, "loaded model {}".format(model_name), 200

        except ValueError as v:
            raise MMSError(Err.VALUE_ERROR_WHILE_LOADING, "{}".format(v))
        except MMSError as m:
            raise m
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(Err.UNKNOWN_EXCEPTION_WHILE_LOADING, "{}".format(repr(e)))

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
            log_error("{}: Send failed. {}.\nMsg: {}".format(Err.SEND_MSG_FAIL, repr(e), msg))

            if self.send_failures >= MAX_FAILURE_THRESHOLD:
                exit(Err.SEND_FAILS_EXCEEDS_LIMITS)

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
        service = None
        while True:
            try:
                cmd, msg = self.codec.retrieve_msg(conn=cl_socket)
                if cmd == u"predict":
                    predictions, result, code = service.predict(msg, self.codec)
                elif cmd == u"load":
                    service, result, code = self.load_model(msg)
                else:
                    result = "Received unknown command: {}".format(cmd)
                    code = Err.UNKNOWN_COMMAND

                self.create_and_send_response(cl_socket, code, result, predictions)
            except MMSError as m:
                log_error("MMSError {} data {}".format(cmd, m.get_message()))
                if m.get_code() == Err.SEND_FAILS_EXCEEDS_LIMITS or m.get_code() == Err.ENCODE_FAILED or \
                   m.get_code() == Err.DECODE_FAILED:
                    log_error("Can not recover from this error. Worker shutting down. {}".format(m))
                    break
                self.create_and_send_response(cl_socket, m.get_code(), m.get_message())
            except Exception as e:  # pylint: disable=broad-except
                log_error("Exception {} data {}".format(cmd, repr(e)))
                self.create_and_send_response(cl_socket, Err.UNKNOWN_EXCEPTION, repr(e))

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
            raise MMSError(Err.SOCKET_BIND_ERROR,
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
    except MMSError as mms_error:
        log_error("{}".format(mms_error.get_message()))
        exit(1)
    except Exception as exc:  # pylint: disable=broad-except
        log_error("Error starting the server. {}".format(repr(exc)))
        exit(1)
    exit(0)
