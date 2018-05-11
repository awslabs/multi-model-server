# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Codec definition and implementations for images and objects
"""
from abc import ABCMeta, abstractmethod
import base64
import json
import sys


class Codec(object):
    """
    Abstract class for serializing and deserializing data
    """
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def serialize(data):
        pass

    @staticmethod
    @abstractmethod
    def deserialize(data):
        pass


class ImageCodec(Codec):
    """
    Codec for serializing and deserializing images
    """

    @staticmethod
    def serialize(data):
        data['data'] = [base64.b64encode(x).decode('utf-8') for x in data['data']]
        return JsonCodec.serialize(data)

    @staticmethod
    def deserialize(data):
        data = JsonCodec.deserialize(data)
        _id = data['id']
        data = data['data']

        deserialized = []
        for datum in data:
            if sys.version_info.major == 3:
                datum = bytes(datum, encoding='utf-8')
            deserialized.append(base64.decodestring(datum))  # pylint: disable=deprecated-method

        return {'id': _id, 'data': deserialized}


class JsonCodec(Codec):
    """
    Codec for serializing and deserializing python objects
    """

    @staticmethod
    def serialize(data):
        return json.dumps(data)

    @staticmethod
    def deserialize(data):
        data = data.decode('utf-8')
        return json.loads(data)
