# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import argparse


class StoreDictKeyPair(argparse.Action):
    """This class is a helper class to parse <model-name>=<model-uri> pairs
    """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
          setattr(namespace, 'models', {kv.split('=', 1)[0]: kv.split('=', 1)[1] for kv in values})
        except Exception:
          raise Exception('Failed to parse <model=path>: ' + str(values) +
                          ' Format should be <model-name>=<model-path> (Local file path, URL, S3).')

class ArgParser(object):
    """Argument parser for mxnet-model-server and mxnet-model-export commands
    More detailed example is available at https://github.com/awslabs/mxnet-model-server/blob/master/README.md
    """
    @staticmethod
    def mms_parser():
        """ Argument parser for mxnet-model-server start service
        """
        parser = argparse.ArgumentParser(prog='mxnet-model-server', description='MXNet Model Server')

        parser.add_argument('--models',
                            required=True,
                            action=StoreDictKeyPair,
                            metavar='KEY1=VAL1 KEY2=VAL2...',
                            nargs='+',
                            help='Models to be deployed using name=model_location format. '
                                 'Location can be a URL or a local path to a .model file. '
                                 'Name is arbitrary and used as the API endpoint\'s base name. ')

        parser.add_argument('--service', help='Path to a user defined model service.')

        parser.add_argument('--gen-api', help='Generates API client for the supplied language. '
                                              'Options include Java, C#, JavaScript and Go. '
                                              'For complete list check out '
                                              'https://github.com/swagger-api/swagger-codegen.')

        parser.add_argument('--port', help='Port number. By default it is 8080.')

        parser.add_argument('--host', help='Host. By default it is localhost.')

        parser.add_argument('--gpu', help='ID of GPU device to use for inference. '
                                          'If your machine has N gpus, this number can be 0 to N - 1. '
                                          'If it is not set, cpu will be used.')

        parser.add_argument('--log-file', help='Log file name. By default it is "mms_app.log".')

        parser.add_argument('--log-rotation-time',
                            help='Log rotation time. By default it is "1 H", which means one Hour. '
                                 'Valid format is "interval when", where _when_ can be "S", "M", "H", or "D". '
                                 'For a particular weekday use only "W0" - "W6". '
                                 'For midnight use only "midnight". '
                                 'Check https://docs.python.org/2/library/logging.handlers.html#logging.handlers.TimedRotatingFileHandler '
                                 'for detailed information on values.')

        parser.add_argument('--log-level', help='Log level. By default it is INFO. '
                                                'Possible values are NOTEST, DEBUG, INFO, ERROR AND CRITICAL. '
                                                'Check https://docs.python.org/2/library/logging.html#logging-levels'
                                                'for detailed information on values.')

        parser.add_argument('--metrics-write-to',
                            default='log',
                            choices=['log', 'csv'],
                            help='Target location to write MMS metrics. '
                                 'Log file specified in --log-file or to local CSV files per metric type. ')

        return parser

    @staticmethod
    def export_parser():
        """ Argument parser for mxnet-model-export
        """
        parser_export = argparse.ArgumentParser(prog='mxnet-model-export', description='MXNet Model Export')

        parser_export.add_argument('--model-name',
                                   required=True,
                                   type=str,
                                   help='Exported model name. Exported file will be named as '
                                        'model-name.model and saved in current working directory.')

        parser_export.add_argument('--model-path',
                                   required=True,
                                   type=str,
                                   help='Path to the folder containing model related files. '
                                        'Signature file is required.')

        parser_export.add_argument('--service-file-path',
                                   required=False,
                                   type=str,
                                   default=None,
                                   help='Service file path to handle custom MMS inference logic. '
                                        'if not provided, this tool will package MXNetBaseService if input in signature.json is application/json or '
                                        'MXNetVisionService if input is image/jpeg')
        
        return parser_export
