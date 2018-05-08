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
Data Store for use in back-end batching of requests
"""

from mms.data_store.codecs import JsonCodec, ImageCodec
from mms.data_store.hashmap import RedisHashMap
from mms.data_store.queue import RedisQueue


class DataStore(object):
    """
    DataStore abstracts the queue, hashmap, and associated codecs
    """

    def __init__(self, name_prefix, config):
        self.hashmap = RedisHashMap(name_prefix, config)
        self.queue = RedisQueue(name_prefix, config)

    def pop_batch(self, model_name, length, data_type):
        """
        Pops up to length items from self.queue and deserializes it according to data_type
        """
        codec = self._get_codec(data_type)
        batch = self.queue.pop(model_name, length)
        deserialized = [codec.deserialize(item) for item in batch]
        ids, data = [], []
        for item in deserialized:
            ids.append(item['id'])
            data.append(item['data'])
        return ids, data

    def push(self, model_name, data, data_type):
        """
        Serializes data according to data_type and pushes it onto the appropriate queue
        """
        codec = self._get_codec(data_type)
        self.queue.push(model_name, codec.serialize(data))

    def get(self, _id, data_type, timeout=60):
        """
        Blocking pop of key _id with timeout and then deserializes it according to data_type
        """
        codec = self._get_codec(data_type)
        return codec.deserialize(self.hashmap.get(_id, timeout))

    # performance could possibly be improved by using Redis Pipelining
    def set_batch(self, ids, data, data_type):
        """
        Sets multiple key:value pairs with id and serialized data
        """
        codec = self._get_codec(data_type)
        for _id, datum in zip(ids, data):
            self.hashmap.put(_id, codec.serialize(datum))

    @staticmethod
    def _get_codec(data_type):
        """
        Returns the codec used by data_type
        """
        if data_type == "image/jpeg":
            return ImageCodec
        elif data_type == "application/json":
            return JsonCodec
        else:
            raise Exception("%s is not a valid data type" % data_type)
