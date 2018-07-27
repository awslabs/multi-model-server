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
Utils class to have all encode and decode helper functions for the backend worker
"""
import base64
import binascii

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as err


class ModelWorkerCodecHelper(object):
    """
    This class defines all the encoding and decoding utility functions used by backend model worker
    """
    @staticmethod
    def decode_msg(encoding, msg):
        try:
            if encoding == u'base64':
                return base64.b64decode(msg)

            return msg
        except (binascii.Error, TypeError) as e:
            raise MMSError(err.DECODE_FAILED, "base64 decode error {}".format(repr(e)))

    @staticmethod
    def encode_msg(encoding, msg):
        """
        encode bytes to utf-8 string
        msg is assumed to be a bytes
        :param encoding:
        :param msg:
        :return:
        """
        try:
            if encoding == u'base64':
                val = base64.b64encode(msg).decode('utf-8')
            else:
                raise TypeError("Invalid encoding type given. {}".format(encoding))

        except (binascii.Error, TypeError) as e:
            raise MMSError(err.ENCODE_FAILED, "Encoding failed {}".format(e))

        return val
