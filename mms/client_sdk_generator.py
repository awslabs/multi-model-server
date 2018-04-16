# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""Client SDK Generator using Swagger codegen tool
"""

import json
import os
import subprocess

from mms.log import get_logger


logger = get_logger()


class ClientSDKGenerator(object):
    """
    Class for client SDK Generator
    """
    @staticmethod
    def generate(openapi_endpoints, sdk_lanugage):
        """Generate client sdk by given OpenAPI specification and target language.

        Parameters
        ----------
        openapi_endpoints : dict
            OpenAPI format api definition
        sdk_lanugage : string
            Target language for client sdk
        """

        # Serialize OpenAPI definition to a file
        try:
            if not os.path.exists('build'):
                os.makedirs('build')
            f = open('build/openapi.json', 'w')
            json.dump(openapi_endpoints, f, indent=4)
            f.flush()

            # Use Swagger codegen tool to generate client sdk in target language
            with open(os.devnull, 'wb') as devnull:
                subprocess.call('java -jar ' + os.path.dirname(os.path.abspath(__file__)) +
                                '/../tools/swagger-codegen-cli-2.2.1.jar generate \
                                -i build/openapi.json \
                                -o build \
                                -l %s' % sdk_lanugage, shell=True, stdout=devnull)

            logger.info("Client SDK for %s is generated", sdk_lanugage)

        except Exception as e:  # pylint: disable=broad-except
            raise Exception('Failed to generate client sdk: ' + str(e))
