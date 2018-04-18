# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Flask HttpRequestHandler for handling requests.
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from mms.request_handler.request_handler import RequestHandler
from mms.log import get_logger


logger = get_logger()


class FlaskRequestHandler(RequestHandler):
    """
    Class defining flask request handler
    """
    def __init__(self, app_name):
        """
        Contructor for Flask request handler.

        Parameters
        ----------
        app_name : string
            App name for handler.
        """
        # pylint: disable=super-init-not-called
        self.app = Flask(app_name)
        CORS(self.app)

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
        try:
            self.app.run(host=host, port=port)
        except Exception:  # pylint: disable=broad-except
            raise

    def add_endpoint(self, api_name, endpoint, callback, methods):
        """
        Add an endpoint for Flask

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
        # pylint: disable=arguments-differ

        # Flask need to be passed with a method list
        try:
            assert isinstance(methods, list), 'methods should be a list: [GET, POST] by Flask.'
            self.app.add_url_rule(endpoint, api_name, callback, methods=methods)
        except Exception:  # pylint: disable=broad-except
            raise

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
        logger.info('Getting query string from request.')
        if field is None:
            return request.args

        return request.args[field]

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
        logger.info('Getting form data from request.')
        form = {k: v[0] for k, v in dict(request.form).items()}
        if field is None:
            return form
        if field in form:
            return form[field]
        return None

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
        logger.info('Getting file data from request.')
        files = {k: v[0] for k, v in dict(request.files).items()}
        if field is None:
            return files
        if field in files:
            return files[field]
        return None

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
        logger.info('Jsonifying the response: %s', (str(response)))
        return jsonify(response)

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
        # pylint: disable=logging-format-interpolation
        logger.info('Sending file with mimetype: {}'.format(mimetype))
        return send_file(filename, mimetype=mimetype)
