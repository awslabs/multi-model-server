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
This module parses the arguments given through the mxnet-model-server command-line. This is used by model-server
at runtime.
"""

import argparse


class StoreDictKeyPair(argparse.Action):
    """This class is a helper class to parse <model-name>=<model-uri> pairs
    """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, 'models', {kv.split('=', 1)[0]: kv.split('=', 1)[1] for kv in values})
        except Exception:
            raise Exception('Failed to parse <model=path>: ' + str(values) +
                            'Format should be <model-name>=<model-path> (Local file path, URL, S3).')


class ArgParser(object):
    """Argument parser for mxnet-model-server and mxnet-model-export commands
    More detailed example is available at https://github.com/awslabs/mxnet-model-server/blob/master/README.md
    """
    @staticmethod
    def mms_parser():
        """ Argument parser for mxnet-model-server start service
        """
        parser = argparse.ArgumentParser(prog='mxnet-model-server', description='MXNet Model Server')

        sub_parse = parser.add_mutually_exclusive_group(required=True)
        sub_parse.add_argument('--start', action='store_true', help='Start the model-server')
        sub_parse.add_argument('--stop', action='store_true', help='Stop the model-server')

        parser.add_argument('--mms-config',
                            dest='mms_config',
                            help='Configuration file for model server')

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
                                   dest="service_file_path",
                                   type=str,
                                   default=None,
                                   help='Service file path to handle custom MMS inference logic. '
                                        'If path is not provided and the input defined in signature.json '
                                        'is application/json, this tool will include the MXNetBaseService \
                                        in the archive. '
                                        'Alternatively, if the input defined in signature.json is image/jpeg '
                                        'this tool will include the MXNetVisionService in the archive.')

        return parser_export

    @staticmethod
    def model_service_worker_args():
        """
        Argparser for backend worker. Takes the socket name and socket type.
        :return:
        """
        parser = argparse.ArgumentParser(prog='model-server-worker', description='Model Server Worker')
        parser.add_argument('--sock-type',
                            required=True,
                            dest="sock_type",
                            type=str,
                            choices=["unix", "tcp"],
                            help='Socket type the model service worker would use. The options are\n'
                                 'unix: The model worker expects to unix domain-socket\n'
                                 'tcp: The model worker expects a host-name and port-number')

        parser.add_argument('--sock-name',
                            required=False,
                            dest="sock_name",
                            type=str,
                            help='If \'sock-type\' is \'unix\', sock-name is expected to be a string. '
                                 'Eg: --sock-name \"test_sock\"')

        parser.add_argument('--host',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have a host IP address')

        parser.add_argument('--port',
                            type=str,
                            help='If \'sock-type\' is \'tcp\' this is expected to have the host port to bind on')

        return parser

    @staticmethod
    def extract_args(args=None):
        parser = ArgParser.mms_parser()
        return parser.parse_args(args) if args else parser.parse_args()
