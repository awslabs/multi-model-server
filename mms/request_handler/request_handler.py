from abc import ABCMeta, abstractmethod


class RequestHandler(object):
    '''HttpRequestHandler for handling http requests.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, app_name):
        '''
        Contructor for request handler.
        
        Parameters
        ----------
        app_name : string 
            App name for handler.
        '''
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def add_endpoint(self, endpoint, api_name, callback, methods):
        '''
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
        '''
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass


    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

