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
On The Fly Codec tester
"""

from collections import namedtuple

import pytest

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err
from mock import Mock
from mms.protocol.otf_message_handler import OtfCodecHandler


@pytest.fixture()
def socket_patches(mocker):
    Patches = namedtuple('Patches', ['socket'])
    mock_patch = Patches(mocker.patch('socket.socket'))
    mock_patch.socket.recv.return_value = b'1'
    return mock_patch


@pytest.fixture()
def otf_codec_fixture():
    return OtfCodecHandler()


class TestOtfCodecHandler:
    def test_retrieve_buffer(self, socket_patches, otf_codec_fixture):
        socket_patches.socket.recv.return_value = b'1'
        data = otf_codec_fixture._retrieve_buffer(socket_patches.socket, 1)
        assert data == b'1'

    def test_retrieve_buffer_ioerror(self, socket_patches, otf_codec_fixture):
        socket_patches.socket.recv.side_effect = IOError("Error")
        with pytest.raises(MMSError) as e:
            data = otf_codec_fixture._retrieve_buffer(socket_patches.socket, 1)
        assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_buffer_oserror(self, socket_patches, otf_codec_fixture):
        socket_patches.socket.recv.side_effect = OSError("Error")
        with pytest.raises(MMSError) as e:
            data = otf_codec_fixture._retrieve_buffer(socket_patches.socket, 1)
        assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_buffer_mmserror(self, socket_patches, otf_codec_fixture):
        socket_patches.socket.recv.side_effect = MMSError(Err.UNKNOWN_CONTENT_TYPE, "Error")
        with pytest.raises(MMSError) as e:
            data = otf_codec_fixture._retrieve_buffer(socket_patches.socket, 1)
        assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_buffer_exception(self, socket_patches, otf_codec_fixture):
        socket_patches.socket.recv.side_effect = Exception("Error")
        with pytest.raises(MMSError) as e:
            data = otf_codec_fixture._retrieve_buffer(socket_patches.socket, 1)
        assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_int(self, socket_patches, otf_codec_fixture):
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_buffer.return_value = b'\x00\x00\x00\x01'
        data = otf_codec_fixture._retrieve_int(socket_patches.socket)
        assert data == 1

    def test_retrieve_double(self, socket_patches, otf_codec_fixture):
        expected = 1.0
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_buffer.return_value = b'?\xf0\x00\x00\x00\x00\x00\x00'
        data = otf_codec_fixture._retrieve_double(socket_patches.socket)
        assert data == expected

    def test_retrieve_load(self, socket_patches, otf_codec_fixture):
        expected = {'modelName': b'asdf', 'modelPath': b'/dummy',
                    'batchSize': 1, 'handler': b'fn', 'gpu': 0}

        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()

        otf_codec_fixture._retrieve_int.side_effect = [len('asdf'), len('/dummy'),
                                                            1, len('fn'), 0]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'asdf', b'/dummy', b'fn']

        cmd, ret = otf_codec_fixture._retrieve_load_msg(socket_patches.socket)

        assert cmd == 'load'
        assert ret == expected

    def test_retrieve_load_no_gpu(self, socket_patches, otf_codec_fixture):
        expected = {'modelName': b'asdf', 'modelPath': b'/dummy',
                    'batchSize': 1, 'handler': b'fn'}

        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()

        otf_codec_fixture._retrieve_int.side_effect = [len('asdf'), len('/dummy'),
                                                            1, len('fn'), -1]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'asdf', b'/dummy', b'fn']

        cmd, ret = otf_codec_fixture._retrieve_load_msg(socket_patches.socket)

        assert cmd == 'load'
        assert ret == expected

    def test_retrieve_model_inputs(self, socket_patches, otf_codec_fixture):
        data = {'name': b'asdf', 'contentType': b'json', 'value': b'val'}
        expected = list()
        expected.append(data)
        ret = list()
        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()

        otf_codec_fixture._retrieve_int.side_effect = [len(b'asdf'),
                                                           len(b'contentType'),
                                                           len(b'val'), -2]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'asdf', b'json', b'val']
        otf_codec_fixture._retrieve_model_inputs(socket_patches.socket, ret)

        assert ret == expected

    def test_retrieve_request_batch(self, socket_patches, otf_codec_fixture):
        data = {'requestId': b'111', 'contentType': b'json',
                    'modelInputs': list()}
        expected = list()
        expected.append(data)

        ret = list()
        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_model_inputs = Mock()

        otf_codec_fixture._retrieve_int.side_effect = [len('111'), len('json'), -1,
                                                           -2]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'111', b'json']
        otf_codec_fixture._retrieve_request_batch(socket_patches.socket, ret)
        assert ret == expected

    def test_retrieve_inference_msg(self, socket_patches, otf_codec_fixture):
        expected = {'modelName': b'asdf', 'requestBatch': list()}

        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_request_batch = Mock()

        otf_codec_fixture._retrieve_int.side_effect = [len(b'asdf'), -1]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'asdf']

        cmd, msg = otf_codec_fixture._retrieve_inference_msg(socket_patches.socket)

        assert cmd == 'predict'
        assert msg == expected

    def test_retrieve_msg_raise_err(self, socket_patches, otf_codec_fixture):
        otf_codec_fixture._retrieve_double = Mock()
        otf_codec_fixture._retrieve_double.side_effect = [2.0]

        with pytest.raises(MMSError) as e:
            otf_codec_fixture.retrieve_msg(socket_patches.socket)
        assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_msg_load(self, socket_patches, otf_codec_fixture):
        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_inference_msg = Mock()
        otf_codec_fixture._retrieve_load_msg = Mock()
        otf_codec_fixture._retrieve_double = Mock()
        otf_codec_fixture._retrieve_double.side_effect = [1.0]

        otf_codec_fixture._retrieve_int.side_effect = [1, 1]
        otf_codec_fixture._retrieve_load_msg.side_effect = [('load', b'asdf')]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'\r\n']

        cmd, msg = otf_codec_fixture.retrieve_msg(socket_patches.socket)

        assert cmd == 'load'
        assert msg == b'asdf'

    def test_retrieve_msg_load_error(self, socket_patches, otf_codec_fixture):
            otf_codec_fixture._retrieve_int = Mock()
            otf_codec_fixture._retrieve_buffer = Mock()
            otf_codec_fixture._retrieve_inference_msg = Mock()
            otf_codec_fixture._retrieve_load_msg = Mock()
            otf_codec_fixture._retrieve_double = Mock()
            otf_codec_fixture._retrieve_double.side_effect = [1.0]

            otf_codec_fixture._retrieve_int.side_effect = [1, 1]
            otf_codec_fixture._retrieve_load_msg.side_effect = [('load', b'asdf')]
            otf_codec_fixture._retrieve_buffer.side_effect = [b'']

            with pytest.raises(MMSError) as e:
                cmd, msg = otf_codec_fixture.retrieve_msg(socket_patches.socket)

            assert e.value.code == Err.DECODE_FAILED

    def test_retrieve_msg_load(self, socket_patches, otf_codec_fixture):
        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_inference_msg = Mock()
        otf_codec_fixture._retrieve_load_msg = Mock()
        otf_codec_fixture._retrieve_double = Mock()
        otf_codec_fixture._retrieve_double.side_effect = [1.0]

        otf_codec_fixture._retrieve_int.side_effect = [1, 2]
        otf_codec_fixture._retrieve_inference_msg.side_effect = [('predict', b'asdf')]
        otf_codec_fixture._retrieve_buffer.side_effect = [b'\r\n']

        cmd, msg = otf_codec_fixture.retrieve_msg(socket_patches.socket)

        assert cmd == 'predict'
        assert msg == b'asdf'

    def test_retrieve_msg_load_error(self, socket_patches, otf_codec_fixture):
            otf_codec_fixture._retrieve_int = Mock()
            otf_codec_fixture._retrieve_buffer = Mock()
            otf_codec_fixture._retrieve_inference_msg = Mock()
            otf_codec_fixture._retrieve_load_msg = Mock()
            otf_codec_fixture._retrieve_double = Mock()
            otf_codec_fixture._retrieve_double.side_effect = [1.0]

            otf_codec_fixture._retrieve_int.side_effect = [2]
            otf_codec_fixture._retrieve_inference_msg.side_effect = [('predict', b'asdf')]
            otf_codec_fixture._retrieve_buffer.side_effect = [b'']

            with pytest.raises(MMSError) as e:
                cmd, msg = otf_codec_fixture.retrieve_msg(socket_patches.socket)

            assert e.value.code == Err.DECODE_FAILED


    def test_retrieve_msg_load(self, socket_patches, otf_codec_fixture):
        otf_codec_fixture._retrieve_int = Mock()
        otf_codec_fixture._retrieve_buffer = Mock()
        otf_codec_fixture._retrieve_inference_msg = Mock()
        otf_codec_fixture._retrieve_load_msg = Mock()
        otf_codec_fixture._retrieve_double = Mock()
        otf_codec_fixture._retrieve_double.side_effect = [1]

        otf_codec_fixture._retrieve_int.side_effect = [1, 3]
        cmd, msg = otf_codec_fixture.retrieve_msg(socket_patches.socket)

        assert cmd == 'unknown'
        assert msg == 'Wrong command'

    def test_encode_response_predictions(self, otf_codec_fixture):
        msg = {'predictions': b'dummy', 'message': 'Success', 'code': 200}
        ret = otf_codec_fixture._encode_response(msg)
        assert b'dummy' in ret

    def test_encode_response_no_predictions(self, otf_codec_fixture):
        msg = {'message': 'Success', 'code': 200}
        ret = otf_codec_fixture._encode_response(msg)
        assert 'Success'.encode() in ret

    def test_encode_response_error(self, otf_codec_fixture):
        msg = {'message': 'Success', 'code': "hello"}
        with pytest.raises(Exception):
            ret = otf_codec_fixture._encode_response(msg)

    def test_create_response_predict(self, otf_codec_fixture):
        dummy = dict()
        otf_codec_fixture._encode_inference_response = Mock()
        otf_codec_fixture._encode_response = Mock()

        otf_codec_fixture._encode_inference_response.side_effect = [b'encoded msg']

        enc_msg = otf_codec_fixture.create_response(cmd=2, kwargs=dummy)
        assert enc_msg == b'encoded msg'

    def test_create_response_general(self, otf_codec_fixture):
        dummy = dict()
        otf_codec_fixture._encode_inference_response = Mock()
        otf_codec_fixture._encode_response = Mock()

        otf_codec_fixture._encode_response.side_effect = [b'encoded msg']

        enc_msg = otf_codec_fixture.create_response(cmd=3, kwargs=dummy)
        assert enc_msg == b'encoded msg'

    def test_create_response_general(self, otf_codec_fixture):
        dummy = dict()

        with pytest.raises(MMSError) as e:
            enc_msg = otf_codec_fixture.create_response(cmd=4, kwargs=dummy)
        assert e.value.code == Err.ENCODE_FAILED





