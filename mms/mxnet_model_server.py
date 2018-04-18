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
Model Server code
"""
from multiprocessing import Lock
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
from mms.arg_parser import ArgParser
from mms.client_sdk_generator import ClientSDKGenerator
from mms.log import get_logger, LOG_LEVEL_DICT, _Formatter
from mms.serving_frontend import ServingFrontend
from mms.metrics_manager import MetricsManager
from mms.model_loader import ModelLoader

if sys.version_info[0] == 3:
    import socketserver as SocketServer
else:
    import SocketServer

VALID_ROTATE_UNIT = ['S', 'M', 'H', 'D', 'midnight'] + ['W%d' % (i) for i in range(7)]
logger = get_logger()


def _set_root_logger(log_file, log_level, log_rotation_time):
    """Internal function to setup root logger
    """
    assert log_level in LOG_LEVEL_DICT, "log_level must be one of the keys in %s" % (str(LOG_LEVEL_DICT))
    rotate_time_list = log_rotation_time.split(' ')
    assert len(rotate_time_list) > 0 and len(rotate_time_list) < 3, \
        "log_rotation_time must be in format 'interval when' or 'when' for weekday and midnight."
    interval = int(rotate_time_list[0]) if len(rotate_time_list) > 1 else 1
    when = rotate_time_list[-1]
    assert isinstance(interval, int) and interval > 0, "interval must be a positive integer."
    assert when in VALID_ROTATE_UNIT, "rotate time unit must be one of the values in %s." \
                                      % (str(VALID_ROTATE_UNIT))

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL_DICT[log_level])

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(_Formatter())
    root.addHandler(stream_handler)

    log_file = log_file or 'mms_app.log'
    file_handler = TimedRotatingFileHandler(log_file, when, interval)
    file_handler.setFormatter(_Formatter(colored=False))
    root.addHandler(file_handler)


class MMS(object):
    """MXNet Model Serving
    """

    # pylint: disable=logging-not-lazy
    def __init__(self, app_name='mms', args=None, models=None):
        """Initialize mxnet model server application.

        Parameters
        ----------
        app_name : str
            App name to initialize mms service.
        args : List of str
            Arguments for starting service. By default it is None
            and commandline arguments will be used. It should follow
            the format recognized by python argparse parse_args method:
            https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args.
            An example for mms arguments:
            ['--models', 'resnet-18=path1', 'inception_v3=path2',
             '--gen-api', 'java', '--port', '8080']
        """
        # Initialize serving frontend and arg parser
        try:
            # Args passed as Namespace object
            self.args = args if args else ArgParser.extract_args()
            self.serving_frontend = ServingFrontend(app_name)
            self.gpu = self.args.gpu
            self.models = models
            # Setup root logger handler and level.
            log_file = self.args.log_file
            log_level = self.args.log_level or "INFO"
            log_rotation_time = self.args.log_rotation_time or "1 H"
            _set_root_logger(log_file, log_level, log_rotation_time)

            logger.info('Initialized model serving.')
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to initialize model serving: ' + str(e))
            exit(1)

    def start_model_serving(self):
        """Start model serving server
        """
        try:
            # Process arguments
            self._arg_process()

            # Start model serving host
            if self.args.gen_api is None:
                logger.info('Service started successfully.')
                logger.info('Service description endpoint: ' + self.host + ':' + str(self.port) + '/api-description')
                logger.info('Service health endpoint: ' + self.host + ':' + str(self.port) + '/ping')

                self.serving_frontend.start_handler(self.host, self.port)

        except SocketServer.socket.error as exc:
            if exc.args[0] != 48:
                raise
            logger.error('Port: ' + str(self.port) + " already in use. exiting...")
            exit(1)

        except Exception as e:  # pylint: disable=broad-except
            logger.error('Failed to start model serving host: ' + str(e))
            exit(1)

    def create_app(self):
        """Create a Flask app object.
        """
        try:
            # Process arguments
            self._arg_process()

            logger.info('Service started successfully.')
            logger.info('Service description endpoint: ' + self.host + ':' + str(self.port) + '/api-description')
            logger.info('Service health endpoint: ' + self.host + ':' + str(self.port) + '/ping')

            # Create app
            return self.serving_frontend.handler.app

        except Exception as e:  # pylint: disable=broad-except
            logger.error('Failed to start model serving host: ' + str(e))
            exit(1)

    def _arg_process(self):
        """Process arguments before starting service or create application.
        """
        try:
            # Port
            self.port = int(self.args.port) if self.args.port else 8080
            self.host = self.args.host or '127.0.0.1'

            # Register user defined model service or default mxnet_vision_service
            manifest = self.models[0][3]
            service_file = os.path.join(self.models[0][2], manifest['Model']['Service'])

            class_defs = self.serving_frontend.register_module(self.args.service or service_file)
            # pylint: disable=deprecated-lambda
            class_defs = list(filter(lambda c: len(c.__subclasses__()) == 0, class_defs))

            if len(class_defs) != 1:
                raise Exception('There should be one user defined service derived from ModelService.')
            model_class_name = class_defs[0].__name__

            # Load models using registered model definitions
            registered_models = self.serving_frontend.get_registered_modelservices()
            ModelClassDef = registered_models[model_class_name]

            self.serving_frontend.load_models(self.models, ModelClassDef, self.gpu)

            # Setup endpoint
            openapi_endpoints = self.serving_frontend.setup_openapi_endpoints(self.host, self.port)

            # Generate client SDK
            if self.args.gen_api is not None:
                ClientSDKGenerator.generate(openapi_endpoints, self.args.gen_api)

            # Generate metrics to target location (log, csv ...), default to log
            MetricsManager.start(self.args.metrics_write_to, self.args.models, Lock())

        except Exception as e:  # pylint: disable=broad-except
            logger.error('Failed to process arguments: ' + str(e))
            exit(1)


def start_serving(app_name='mms', args=None):
    """Start service routing.

    Parameters
    ----------
    app_name : str
        App name to initialize mms service.
    args : List of str
        Arguments for starting service. By default it is None
        and commandline arguments will be used. It should follow
        the format recognized by python argparse parse_args method:
        https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args.
        An example for mms arguments:
        ['--models', 'resnet-18=path1', 'inception_v3=path2',
         '--gen-api', 'java', '--port', '8080']
        """

    # Parse the given arguments
    arguments = ArgParser.extract_args(args)
    # Download and/or Extract the given model files
    models = ModelLoader.load(arguments.models)
    # Instantiate an MMS object and prepare to start serving
    mms = MMS(app_name, args=arguments, models=models)
    mms.start_model_serving()


if __name__ == '__main__':
    start_serving()
