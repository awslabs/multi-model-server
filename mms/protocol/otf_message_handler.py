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
OTF Codec
"""
import struct
import json

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err

int_size = 4
double_size = 8
END_OF_LIST = -2
START_OF_LIST = -1
PREDICT_MSG = 2
LOAD_MSG = 1
RESPONSE = 3


class OtfCodecHandler(object):
    """
    OTF Codec class
    """

    @staticmethod
    def _retrieve_buffer(conn, length):
        data = bytearray()

        try:
            while length > 0:
                pkt = conn.recv(length)
                if len(pkt) == 0:
                    raise MMSError(Err.DECODE_FAILED, "Remote side disconnected.. recv None")
                data += pkt
                length -= len(pkt)
        except (IOError, OSError) as sock_err:
            raise MMSError(Err.DECODE_FAILED, "{}".format(repr(sock_err)))
        except Exception as e:  # pylint: disable=broad-except
            raise MMSError(Err.DECODE_FAILED, "{}".format(e))

        return data

    def _retrieve_int(self, conn):
        data = self._retrieve_buffer(conn, int_size)
        return struct.unpack('!i', data)[0]

    def _retrieve_double(self, conn):
        data = self._retrieve_buffer(conn, double_size)
        return struct.unpack('!d', data)[0]

    def _retrieve_load_msg(self, conn):
        """
        MSG Frame Format:

        | 1.0 | int cmd_length | cmd value | int model-name length | model-name value |
        | int model-path length | model-path value |
        | int batch-size length | batch-size value | int handler length | handler value |
        | int gpu id length | gpu ID value |

        :param conn:
        :return:
        """
        msg = dict()
        length = self._retrieve_int(conn)
        msg['modelName'] = self._retrieve_buffer(conn=conn, length=length)
        length = self._retrieve_int(conn)
        msg['modelPath'] = self._retrieve_buffer(conn=conn, length=length)
        msg['batchSize'] = self._retrieve_int(conn=conn)
        length = self._retrieve_int(conn=conn)
        msg['handler'] = self._retrieve_buffer(conn, length)
        gpu_id = self._retrieve_int(conn=conn)
        if gpu_id >= 0:
            msg['gpu'] = gpu_id

        return "load", msg

    def _retrieve_model_inputs(self, conn, msg):
        end = False
        while end is False:
            model_input = dict()
            length = self._retrieve_int(conn)
            if length > 0:
                model_input['name'] = self._retrieve_buffer(conn, length)
            elif length == END_OF_LIST:
                end = True
                continue

            length = self._retrieve_int(conn)

            model_input['contentType'] = self._retrieve_buffer(conn, length)
            length = self._retrieve_int(conn)
            value = self._retrieve_buffer(conn, length)
            model_input['value'] = value
            msg.append(model_input)

    def _retrieve_request_batch(self, conn, msg):
        end = False
        while end is False:
            req_batch = dict()
            length = self._retrieve_int(conn)

            if length > 0:
                req_batch['requestId'] = self._retrieve_buffer(conn, length)
            elif length == END_OF_LIST:
                end = True
                continue

            length = self._retrieve_int(conn)
            req_batch['contentType'] = self._retrieve_buffer(conn, length)

            length = self._retrieve_int(conn)
            if length == START_OF_LIST:  # Beginning of list
                req_batch['modelInputs'] = list()
                self._retrieve_model_inputs(conn, req_batch['modelInputs'])

            msg.append(req_batch)

    def _retrieve_inference_msg(self, conn):
        msg = dict()
        length = self._retrieve_int(conn)
        msg['modelName'] = self._retrieve_buffer(conn, length)
        length = self._retrieve_int(conn)
        if length == START_OF_LIST:
            msg['requestBatch'] = list()
            self._retrieve_request_batch(conn, msg['requestBatch'])

        return "predict", msg

    def retrieve_msg(self, conn):
        """
        Retrieve a message from the socket channel.

        :param conn:
        :return:
        """
        # Validate its beginning of a message
        version = self._retrieve_int(conn)
        if version != 1:
            raise MMSError(Err.DECODE_FAILED, "Invalid message received")

        cmd = self._retrieve_int(conn)

        if cmd == LOAD_MSG:
            cmd, msg = self._retrieve_load_msg(conn)
        elif cmd == PREDICT_MSG:
            cmd, msg = self._retrieve_inference_msg(conn)
        else:
            return "unknown", "Wrong command"

        return cmd, msg

    @staticmethod
    def _encode_inference_response(kwargs):
        try:
            req_id_map = kwargs['req_id_map']
            invalid_reqs = kwargs['invalid_reqs']
            ret = kwargs['resp']
            msg = bytearray()
            msg += struct.pack('!i', -1)  # start of list

            for idx, val in enumerate(ret):
                msg += struct.pack("!i", len(req_id_map[idx]))
                msg += struct.pack('!{}s'.format(len(req_id_map[idx])), req_id_map[idx].encode('utf-8'))

                if isinstance(val, str):
                    content_type = "text"

                    ctype_encoded = content_type.encode('utf-8')
                    msg += struct.pack('!i', len(ctype_encoded))
                    msg += struct.pack('!{}s'.format(len(content_type)), ctype_encoded)

                    val_encoded = val.encode('utf-8')
                    msg += struct.pack('!i', len(val_encoded))
                    msg += struct.pack('!{}s'.format(len(val_encoded)), val_encoded)
                elif isinstance(val, (bytes, bytearray)):
                    content_type = "binary"

                    ctype_encoded = content_type.encode('utf-8')
                    msg += struct.pack('!i', len(ctype_encoded))
                    msg += struct.pack('!{}s'.format(len(ctype_encoded)), ctype_encoded)

                    msg += struct.pack('!i', len(val))
                    msg += val
                else:
                    content_type = 'json'
                    ctype_encoded = content_type.encode('utf-8')
                    json_value = json.dumps(val)
                    msg += struct.pack('!i', len(ctype_encoded))
                    msg += struct.pack('!{}s'.format(len(ctype_encoded)), ctype_encoded)

                    val_encoded = json_value.encode('utf-8')
                    msg += struct.pack('!i', len(val_encoded))
                    msg += struct.pack('!{}s'.format(len(val_encoded)), val_encoded)

            for req in invalid_reqs.keys():
                req_encoded = req.encode('utf-8')
                msg += struct.pack('!i', len(req_encoded))
                msg += struct.pack('!{}s'.format(len(req_encoded)), req_encoded)

                msg += struct.pack('!i', 4)
                msg += struct.pack('!i', invalid_reqs.get(req))

                msg_encoded = "Invalid input provided".encode('utf-8')
                msg += struct.pack('!i', len(msg_encoded))
                msg += struct.pack('!{}s'.format(msg_encoded), msg_encoded)

                content_type = "text"
                ctype_encoded = content_type.encode('utf-8')
                msg += struct.pack('!i', len(ctype_encoded))
                msg += struct.pack('!{}s'.format(len(ctype_encoded)), ctype_encoded)
            msg += struct.pack('!i', -2)  # End of list

            return msg

        except Exception:
            raise MMSError(Err.ENCODE_FAILED, "Invalid message received for encode")

    @staticmethod
    def _encode_response(kwargs):
        msg = bytearray()
        try:
            msg += struct.pack('!i', int(kwargs['code']))
            msg_len = len(kwargs['message'])
            msg += struct.pack('!i', msg_len)
            msg += struct.pack('!{}s'.format(msg_len), kwargs['message'].encode())
            if 'predictions' in kwargs and kwargs['predictions'] is not None:
                msg += kwargs['predictions']
            else:
                msg += struct.pack('!i', 0)  # no predictions
        except Exception:
            raise MMSError(Err.ENCODE_FAILED, "Failed to encode response.")
        return msg

    def create_response(self, cmd, **kwargs):
        if cmd == PREDICT_MSG:  # Predict request response
            return self._encode_inference_response(kwargs=kwargs)
        elif cmd == RESPONSE:  # All responses
            return self._encode_response(kwargs=kwargs)
        else:
            raise MMSError(Err.ENCODE_FAILED, "Unknown message received {}".format(cmd))
