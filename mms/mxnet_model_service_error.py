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
Class to trigger MMS Errors
"""


class MMSError(Exception):
    """
    Class defining the MMS Error. This is used by backend worker and custom service code to throw errors.
    """
    def __init__(self, code, message):
        super(MMSError, self).__init__(message)
        self.code = code
        self.message = message

    def get_code(self):
        return self.code

    def get_message(self):
        return self.message

    def set_code(self, c):
        self.code = c

    def set_message(self, m):
        self.message = m
