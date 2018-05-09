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
Serving frontend for MMS
"""
import ast
import traceback
import time
import base64

from functools import partial
from flask import abort
from mms.service_manager import ServiceManager
from mms.request_handler.flask_handler import FlaskRequestHandler
from mms.log import get_logger
from mms.metrics_manager import MetricsManager

logger = get_logger()


class ServingFrontend(object):
    """ServingFrontend warps up all internal services including
    model service manager, request handler. It provides all public
    apis for users to extend and use our system.
    """
    def __init__(self, app_name):
        """
        Initialize handler for FlaskHandler and ServiceManager.

        Parameters
        ----------
        app_name : str
            App name to initialize request handler.
        """
        try:
            self.service_manager = ServiceManager()
            self.handler = FlaskRequestHandler(app_name)

            logger.info('Initialized serving frontend.')
        except Exception as e:
            raise Exception('Failed to initialize serving frontend: ' + str(e))

    def start_handler(self, host, port):
        """
        Start handler using given host and port.

        Parameters
        ----------
        host : str
            Host that server will use.
        port: int
            Port that server will use.
        """
        self.handler.start_handler(host, port)

    def load_models(self, models, ModelServiceClassDef, gpu=None):
        """
        Load models by using user passed Model Service Class Definitions.

        Parameters
        ----------
        models : List of model_name, model_path pairs
            List of model_name, model_path pairs that will be initialized.
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        gpu : int
            Id of gpu device. If machine has two gpus, this number can be 0 or 1.
            If it is not set, cpu will be used.
        """
        for service_name, model_name, model_path, manifest in models:
            self.service_manager.load_model(service_name, model_name, model_path, manifest, ModelServiceClassDef, gpu)

    def register_module(self, user_defined_module_file_path):
        """
        Register a python module according to user_defined_module_name
        This module should contain a valid Model Service Class whose
        pre-process and post-process can be derived and customized.

        Parameters
        ----------
        user_defined_module_file_path : Python module file path
            A python module will be loaded according to this file path.


        Returns
        ----------
        List of model service class definitions.
            Those python class can be used to initialize model service.
        """
        model_class_definations = self.service_manager.parse_modelservices_from_module(user_defined_module_file_path)
        assert len(model_class_definations) >= 1, \
            'No valid python class derived from Base Model Service is in module file: %s' % \
            user_defined_module_file_path

        for ModelServiceClassDef in model_class_definations:
            self.service_manager.add_modelservice_to_registry(ModelServiceClassDef.__name__, ModelServiceClassDef)

        return model_class_definations

    def get_registered_modelservices(self, modelservice_names=None):
        """
        Get all registered Model Service Class Definitions into a dictionary
        according to name or list of names.
        If nothing is passed, all registered model services will be returned.

        Parameters
        ----------
        modelservice_names : str or List, optional
            Names to retrieve registered model services

        Returns
        ----------
        Dict of name, model service pairs
            Registered model services according to given names.
        """
        if not isinstance(modelservice_names, list) and modelservice_names is not None:
            modelservice_names = [modelservice_names]

        return self.service_manager.get_modelservices_registry(modelservice_names)

    def get_loaded_modelservices(self, modelservice_names=None):
        """
        Get all model services which are loaded in the system into a dictionary
        according to name or list of names.
        If nothing is passed, all loaded model services will be returned.

        Parameters
        ----------
        modelservice_names : str or List, optional
            Names to retrieve loaded model services

        Returns
        ----------
        Dict of name, model service pairs
            Loaded model services according to given names.
        """
        if not isinstance(modelservice_names, list) and modelservice_names is not None:
            modelservice_names = [modelservice_names]

        return self.service_manager.get_loaded_modelservices(modelservice_names)

    def get_query_string(self, field):
        """
        Get field data in the query string from request.

        Parameters
        ----------
        field : str
            Field in the query string from request.

        Returns
        ----------
        Object
            Field data in query string.
        """
        return self.handler.get_query_string(field)

    def add_endpoint(self, api_definition, callback, **kwargs):
        """
        Add an endpoint with OpenAPI compatible api definition and callback.

        Parameters
        ----------
        api_definition : dict(json)
            OpenAPI compatible api definition.

        callback: function
            Callback function in the endpoint.

        kwargs: dict
            Arguments for callback functions.
        """
        endpoint = list(api_definition.keys())[0]
        method = list(api_definition[endpoint].keys())[0]
        api_name = api_definition[endpoint][method]['operationId']

        logger.info('Adding endpoint: %s to Flask', api_name)
        self.handler.add_endpoint(api_name, endpoint, partial(callback, **kwargs), [method.upper()])

    def setup_openapi_endpoints(self, host, port):
        """
        Firstly, construct Openapi compatible api definition for
        1. Predict
        2. Ping
        3. API description
        4. Root API (reusing ping functionality)

        Then the api definition is used to setup web server endpoint.

        Parameters
        ----------
        host : str
            Host that server will use

        port: int
            Port that server will use
        """
        modelservices = self.service_manager.get_loaded_modelservices()
        self.openapi_endpoints = {
            'swagger': '2.0',
            'info': {
                'version': '1.0.0',
                'title': 'Model Serving Apis'
            },
            'host': host + ':' + str(port),
            'schemes': ['http'],
            'paths': {},
        }

        # 1. Predict endpoints
        for model_name, modelservice in modelservices.items():
            input_type = modelservice.signature['input_type']
            inputs = modelservice.signature['inputs']
            output_type = modelservice.signature['output_type']

            # Contruct predict openapi specs
            endpoint = '/' + model_name + '/predict'
            predict_api = {
                endpoint: {
                    'post': {
                        'operationId': model_name + '_predict',
                        'consumes': ['multipart/form-data'],
                        'produces': [output_type],
                        'parameters': [],
                        'responses': {
                            '200': {}
                        }
                    }
                }
            }
            input_names = []
            # Setup endpoint for each modelservice
            # pylint: disable=consider-using-enumerate
            for idx in range(len(inputs)):
                # Check input content type to set up proper openapi consumes field
                input_names.append(inputs[idx]['data_name'])
                if input_type == 'application/json':
                    parameter = {
                        'in': 'formData',
                        'name': inputs[idx]['data_name'],
                        'description': '%s should tensor with shape: %s' %
                                       (inputs[idx]['data_name'], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'schema': {
                            'type': 'string'
                        }
                    }
                elif input_type == 'image/jpeg':
                    parameter = {
                        'in': 'formData',
                        'name': inputs[idx]['data_name'],
                        'description': '%s should be image which will be resized to: %s' %
                                       (inputs[idx]['data_name'], inputs[idx]['data_shape'][1:]),
                        'required': 'true',
                        'type': 'file'
                    }
                else:
                    msg = '%s is not supported for input content-type' % input_type
                    logger.error(msg)
                    abort(500, "Service setting error. %s" % msg)
                predict_api[endpoint]['post']['parameters'].append(parameter)

            # Contruct openapi response schema
            if output_type == 'application/json':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'prediction': {
                                'type': 'string'
                            }
                        }
                    }
                }
            elif output_type == 'image/jpeg':
                responses = {
                    'description': 'OK',
                    'schema': {
                        'type': 'file'
                    }
                }
            else:
                msg = '%s is not supported for output content-type' % output_type
                logger.error(msg)
                abort(500, "Service setting error. %s" % msg)
            predict_api[endpoint]['post']['responses']['200'].update(responses)

            self.openapi_endpoints['paths'].update(predict_api)

            # Setup Flask endpoint for predict api
            self.add_endpoint(predict_api,
                              self.predict_callback,
                              modelservice=modelservice,
                              input_names=input_names,
                              model_name=model_name)

        # 2. Ping endpoints
        ping_api = {
            '/ping': {
                'get': {
                    'operationId': 'ping',
                    'produces': ['application/json'],
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'health': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.openapi_endpoints['paths'].update(ping_api)
        self.add_endpoint(ping_api, self.ping_callback)

        # 3. Describe apis endpoints
        api_description_api = {
            '/api-description': {
                'get': {
                    'produces': ['application/json'],
                    'operationId': 'api-description',
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'description': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.openapi_endpoints['paths'].update(api_description_api)
        self.add_endpoint(api_description_api, self.api_description)

        #4. root endpoint (A secondary way  to ping functionality)
        root_api = {
            '/': {
                'get': {
                    'operationId': 'root',
                    'produces': ['application/json'],
                    'responses': {
                        '200': {
                            'description': 'OK',
                            'schema': {
                                'type': 'object',
                                'properties': {
                                    'health': {
                                        'type': 'string'
                                    }
                                }
                            }
                        }
                    }
                }

            }
        }
        self.openapi_endpoints['paths'].update(root_api)
        self.add_endpoint(root_api, self.ping_callback)

        return self.openapi_endpoints

    def ping_callback(self, **kwargs):
        """
        Callback function for ping endpoint.

        Returns
        ----------
        Response
            Http response for ping endpoint.
        """
        # pylint: disable=unused-argument
        if 'PingTotal' in MetricsManager.metrics:
            MetricsManager.metrics['PingTotal'].update(metric=1)
        try:
            for model in self.service_manager.get_loaded_modelservices().values():
                model.ping()
        except Exception:  # pylint: disable=broad-except
            logger.error('Model serving is unhealthy.')
            return self.handler.jsonify({'health': 'unhealthy!'})

        return self.handler.jsonify({'health': 'healthy!'})

    def api_description(self, **kwargs):
        # pylint: disable=unused-argument
        """
        Callback function for api description endpoint.

        Returns
        ----------
        Response
            Http response for api description endpoint.
        """
        if 'APIDescriptionTotal' in MetricsManager.metrics:
            MetricsManager.metrics['APIDescriptionTotal'].update(metric=1)
        return self.handler.jsonify({'description': self.openapi_endpoints})

    def predict_callback(self, **kwargs):
        """
        Callback for predict endpoint

        Parameters
        ----------
        modelservice : ModelService
            ModelService handler.

        input_names: list
            Input names in request form data.

        Returns
        ----------
        Response
            Http response for predict endpiont.
        """
        handler_start_time = time.time()
        modelservice = kwargs['modelservice']
        input_names = kwargs['input_names']
        model_name = kwargs['model_name']

        if model_name + '_PredictionTotal' in MetricsManager.metrics:
            MetricsManager.metrics[model_name + '_PredictionTotal'].update(metric=1)

        input_type = modelservice.signature['input_type']
        output_type = modelservice.signature['output_type']

        # Get data from request according to input type
        input_data = []
        if input_type == 'application/json':
            try:
                for name in input_names:
                    logger.info('Request input: %s should be json tensor.', name)
                    form_data = self.handler.get_form_data(name)
                    form_data = ast.literal_eval(form_data)
                    assert isinstance(form_data, list), "Input data for request argument: %s is not correct. " \
                                                        "%s is expected but got %s instead of list" \
                                                        % (name, input_type, type(form_data))
                    input_data.append(form_data)
            except Exception as e:  # pylint: disable=broad-except
                if model_name + '_Prediction4XX' in MetricsManager.metrics:
                    MetricsManager.metrics[model_name + '_Prediction4XX'].update(metric=1)
                logger.error(str(e))
                abort(400, str(e))
        elif input_type == 'image/jpeg':
            try:
                for name in input_names:
                    logger.info('Request input: %s should be image with jpeg format.', name)
                    input_file = self.handler.get_file_data(name)
                    if input_file:
                        mime_type = input_file.content_type
                        assert mime_type == input_type, 'Input data for request argument: %s is not correct. ' \
                                                        '%s is expected but %s is given.' % \
                                                        (name, input_type, mime_type)
                        file_data = input_file.read()
                        assert isinstance(file_data, (str, bytes)), 'Image file buffer should be type str or ' \
                                                                    'bytes, but got %s' % (type(file_data))
                    else:
                        form_data = self.handler.get_form_data(name)
                        if form_data:
                            # pylint: disable=deprecated-method
                            file_data = base64.decodestring(self.handler.get_form_data(name))
                        else:
                            raise ValueError('This end point is expecting a data_name of %s. '
                                             'End point details can be found here:http://<host>:<port>/api-description'
                                             % name)
                    input_data.append(file_data)
            except Exception as e:  # pylint: disable=broad-except
                if model_name + '_Prediction4XX' in MetricsManager.metrics:
                    MetricsManager.metrics[model_name + '_Prediction4XX'].update(metric=1)
                logger.error(str(e))
                abort(400, str(e))
        else:
            msg = '%s is not supported for input content-type' % input_type
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(msg)
            abort(500, "Service setting error. %s" % (msg))

        # Doing prediction on model
        try:
            response = modelservice.inference(input_data)
        except Exception:  # pylint: disable=broad-except
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(str(traceback.format_exc()))
            abort(500, "Error occurs while inference was executed on server.")

        # Construct response according to output type
        if output_type == 'application/json':
            logger.info('Response is text.')
        elif output_type == 'image/jpeg':
            logger.info('Response is jpeg image encoded in base64 string.')
        else:
            msg = '%s is not supported for input content-type.' % output_type
            if model_name + '_Prediction5XX' in MetricsManager.metrics:
                MetricsManager.metrics[model_name + '_Prediction5XX'].update(metric=1)
            logger.error(msg)
            abort(500, "Service setting error. %s" % msg)

        logger.debug("Prediction request handling time is: %s ms",
                     ((time.time() - handler_start_time) * 1000))
        return self.handler.jsonify({'prediction': response})
