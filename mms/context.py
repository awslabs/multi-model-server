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
Context object of incoming request
"""


class Context(object):
    """
    Context stores model relevant worker information
    Some fixed during load times and some
    """

    def __init__(self, model_name, model_dir, manifest, batch_size, gpu, mms_version):
        self.model_name = model_name
        self.manifest = manifest
        self._system_properties = {
            "model_dir": model_dir,
            "gpu_id": gpu,
            "batch_size": batch_size,
            "server_name": "MMS",
            "server_version": mms_version
        }
        self.request_ids = None
        self.request_processor = RequestProcessor(dict())
        self._metrics = None

    @property
    def system_properties(self):
        return self._system_properties

    @property
    def request_processor(self):
        return self._request_processor

    @request_processor.setter
    def request_processor(self, request_processor):
        self._request_processor = request_processor

    @property
    def metrics(self):
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def set_response_content_type(self, request_id, value):
        self._request_processor.add_response_property(request_id, {'content-type': value})

    def get_response_content_type(self, request_id):
        response_headers = self._request_processor.get_response_header().get(request_id)
        if response_headers is not None:
            return response_headers.get('content-type')
        return None

    def __eq__(self, other):
        return isinstance(other, Context) and self.__dict__ == other.__dict__


class RequestProcessor(object):
    """
    Request processor
    """

    def __init__(self, request_header):
        self._status_code = 200
        self._reason_phrase = None
        self._response_header = {}
        self._request_header = request_header

    def get_request_property(self, key):
        return self._request_header.get(key)

    def report_status(self, code, reason_phrase=None):
        self._status_code = code
        self._reason_phrase = reason_phrase

    def add_response_property(self, key, value):
        self._response_header[key] = value

    def get_response_header(self):
        return self._response_header
