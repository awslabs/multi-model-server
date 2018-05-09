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
This module parses a Redis configuration file to ensure the same parameters
used to initialize Redis are used by the Redis object to access Redis.
"""

CONFIG_MAP = {"bind": "host", "port": "port", "requjrepass": "password"}


class RedisConfParser(object):

    @staticmethod
    def parse_conf(loc):
        """
        Parameters
        ----------
        loc : str
            location of configuration file

        Returns
        -------
        config : dict
            dictionary with configuration arguments
        """
        config = {}
        for line in open(loc, 'r'):
            if line.startswith("#"):
                continue
            line = line.split()
            if line[0] in CONFIG_MAP:
                config[CONFIG_MAP[line[0]]] = line[1]

        return config
