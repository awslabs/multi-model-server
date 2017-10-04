from flask import Flask, request, jsonify, send_file
from ..log import get_logger
from .request_handler import RequestHandler
from flask_cors import CORS


logger = get_logger(__name__)


class FlaskRequestHandler(RequestHandler):
    '''Flask HttpRequestHandler for handling requests.
    '''
    def __init__(self, app_name):
        '''
        Contructor for Flask request handler.
        
        Parameters
        ----------
        app_name : string 
            App name for handler.
        '''
        self.app = Flask(app_name)
        CORS(self.app)

    def start_handler(self, host, port):
        '''
        Start request handler.

        Parameters
        ----------
        host : string 
            Host to setup handler.
        port: int
            Port to setup handler.
        '''
        try:
            self.app.run(host=host, port=port)
        except Exception as e:
            raise Exception('Flask handler failed to start: ' + str(e))

    def add_endpoint(self, api_name, endpoint, callback, methods):
        '''
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
        '''

        # Flask need to be passed with a method list
        try:
            assert isinstance(methods, list), 'methods should be a list: [GET, POST] by Flask.'
            self.app.add_url_rule(endpoint, api_name, callback, methods=methods)
        except Exception as e:
            raise Exception('Flask handler failed to add endpoints: ' + str(e))
        
    def get_query_string(self, field=None):
        '''
        Get query string from a request.

        Parameters
        ----------
        field : string 
            Get field data from query string.

        Returns
        ----------
        Object: 
            Field data from query string.
        '''
        logger.info('Getting query string from request.')
        if field is None:
            return request.args

        return request.args[field]

    def get_form_data(self, field=None):
        '''
        Get form data from request.
        
        Parameters
        ----------
        field : string 
            Get field data from form data

        Returns
        ----------
        Object: 
            Field data from form data.
        '''
        logger.info('Getting from data from request.')
        form = {k: v[0] for k, v in dict(request.form).items()}
        if field is None:
            return form

        return form[field]

    def get_file_data(self, field=None):
        '''
        Get file data from request.
        
        Parameters
        ----------
        field : string 
            Get field data from file data.

        Returns
        ----------
        Object: 
            Field data from file data.
        '''
        logger.info('Getting file data from request.')
        files = {k: v[0] for k, v in dict(request.files).items()}
        if field is None:
            return files

        return files[field]


    def jsonify(self, response):
        '''
        Jsonify a response.
        
        Parameters
        ----------
        response : Response 
            response to be jsonified.

        Returns
        ----------
        Response: 
            Jsonified response.
        '''
        logger.info('Jsonifying the response: ' + str(response))
        return jsonify(response)

    def send_file(self, file, mimetype):
        '''
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
        '''
        logger.info('Sending file with mimetype: ' + mimetype)
        return send_file(file, mimetype=mimetype)
