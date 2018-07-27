import base64
import pytest
from mock import Mock
import binascii
from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err
from mms.utils.codec_helpers.codec import ModelWorkerCodecHelper


class TestMXNetImageUtils():

    def test_model_worker_encoder(self):
        msg = b"Hello World!!"
        assert base64.b64encode(msg).decode('utf-8') == ModelWorkerCodecHelper.encode_msg(u'base64', msg), \
            "base64 encoder not correct"

    def test_invalid_encode(self):
        msg = b"Hello World!!"
        with pytest.raises(MMSError):
            ModelWorkerCodecHelper.encode_msg('dummy', msg)

    class TestDecodeMsg():
        encoded_msg = "SGVsbG8gV29ybGQh"

        def test_decode_msg(self):
            assert b"Hello World!" == ModelWorkerCodecHelper.decode_msg('base64', self.encoded_msg)

        def test_with_no_encoding(self):
            assert self.encoded_msg == ModelWorkerCodecHelper.decode_msg('garbage', self.encoded_msg)

        def test_with_error(self, mocker):
            b64_mock = mocker.patch('base64.b64decode')
            exception = binascii.Error('error')
            b64_mock.side_effect = exception

            with pytest.raises(Exception) as ex:
                ModelWorkerCodecHelper.decode_msg('base64', self.encoded_msg)

            assert isinstance(ex.value, MMSError)
            assert ex.value.get_code() == err.DECODE_FAILED
            assert ex.value.get_message() == "base64 decode error {}".format(repr(exception))

        def test_with_type_error(self, mocker):
            b64_mock = mocker.patch('base64.b64decode')
            exception = TypeError('error')
            b64_mock.side_effect = exception

            with pytest.raises(Exception) as ex:
                ModelWorkerCodecHelper.decode_msg('base64', self.encoded_msg)

            assert isinstance(ex.value, MMSError)
            assert ex.value.get_code() == err.DECODE_FAILED
            assert ex.value.get_message() == "base64 decode error {}".format(repr(exception))
