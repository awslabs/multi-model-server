# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
        return base64.b64encode(data).decode('utf-8')

    @staticmethod
    def deserialize(data):
        data = JsonCodec.deserialize(data)
        id = data['id']
        image = data['data']

        if sys.version_info.major == 3:
            image = bytes(image, encoding='utf-8')

        return {'id': id, 'data': base64.decodestring(image)}


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
