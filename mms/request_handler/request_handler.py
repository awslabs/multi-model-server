# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""HttpRequestHandler for handling http requests.
"""

from abc import ABCMeta, abstractmethod


class RequestHandler(object):
    """
    Class for defining the request handler
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, app_name):
        """
        Contructor for request handler.

        Parameters
        ----------
        app_name : string
            App name for handler.
        """
        pass

    @abstractmethod
    def start_handler(self, host, port):
        """
        Start request handler.

        Parameters
        ----------
        host : string
            Host to setup handler.
        port: int
            Port to setup handler.
        """
        pass

    @abstractmethod
    def add_endpoint(self, endpoint, api_name, callback, methods):
        """
        Add endpoint for request handler.

        Parameters
        ----------
        endpoint : string
            Endpoint for handler.
        api_name: string
            Endpoint ID for handler.

        callback: function
            Callback function for endpoint.

        methods: List
            Http request methods [POST, GET].
        """
        pass

    @abstractmethod
    def get_query_string(self, field=None):
        """
        Get query string from a request.

        Parameters
        ----------
        field : string
            Get field data from query string.

        Returns
        ----------
        Object:
            Field data from query string.
        """
        pass

    @abstractmethod
    def get_form_data(self, field=None):
        """
        Get form data from request.

        Parameters
        ----------
        field : string
            Get field data from form data

        Returns
        ----------
        Object:
            Field data from form data.
        """
        pass

    @abstractmethod
    def get_file_data(self, field=None):
        """
        Get file data from request.

        Parameters
        ----------
        field : string
            Get field data from file data.

        Returns
        ----------
        Object:
            Field data from file data.
        """
        pass


    @abstractmethod
    def jsonify(self, response):
        """
        Jsonify a response.

        Parameters
        ----------
        response : Response
            response to be jsonified.

        Returns
        ----------
        Response:
            Jsonified response.
        """
        pass

    @abstractmethod
    def send_file(self, filename, mimetype):
        """
        Send a file in Http response.

        Parameters
        ----------
        file : Buffer
            File to be sent in the response.

        mimetype: string
            Mimetype (Image/jpeg).

        Returns
        ----------
        Response:
            Response with file to be sent.
        """
        pass
