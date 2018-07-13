import unittest
import base64
import pytest
from mms.mxnet_model_service_error import MMSError

from mms.utils.codec_helpers.codec import ModelWorkerCodecHelper


class TestMXNetImageUtils(unittest.TestCase):
    def test_model_worker_encoder(self):
        msg = b"Hello World!!"
        assert base64.b64encode(msg).decode('utf-8') == ModelWorkerCodecHelper.encode_msg(u'base64', msg), \
            "base64 encoder not correct"

    def test_invalid_encode(self):
        msg = b"Hello World!!"
        with pytest.raises(MMSError):
            ModelWorkerCodecHelper.encode_msg('dummy', msg)
