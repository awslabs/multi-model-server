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
Queue definition and Redis implementation
"""

from abc import ABCMeta, abstractmethod

import redis


class Queue(object):
    """
    Queue is an abstract class for queueing requests the server and model engine
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def push(self, name, data):
        pass

    @abstractmethod
    def pop(self, name, max_n):
        pass


class RedisQueue(Queue):
    """
    RedisQueue implements Queue with Redis as a persistent data store.
    """
    def __init__(self, name_prefix, config):
        self.name_prefix = name_prefix
        try:
            self.redis = redis.StrictRedis(**config)
        except Exception as e:
            raise Exception("Failed to connect to Redis: %s" % e)

    def push(self, name, data):
        self.redis.rpush(self.name_prefix + name, data)

    def pop(self, name, max_n):
        length = self.len(name)
        get_n = length if length < max_n else max_n
        pipe = self.redis.pipeline()
        pipe.lrange(self.name_prefix + name, 0, get_n - 1)
        pipe.ltrim(self.name_prefix + name, get_n, -1)
        data, _ = pipe.execute()

        return data

    def len(self, name):
        return self.redis.llen(self.name_prefix + name)
