import json
import os
import subprocess

from mms.log import get_logger


logger = get_logger(__name__)


class ClientSDKGenerator(object):
    '''Client SDK Generator using Swagger codegen tool
    '''

    @staticmethod
    def generate(openapi_endpoints, sdk_lanugage):
        '''Generate client sdk by given OpenAPI specification and target language.

        Parameters
        ----------
        openapi_endpoints : dict
            OpenAPI format api definition
        sdk_lanugage : string
            Target language for client sdk
        '''

        # Serialize OpenAPI definition to a file
        try:
            
            if not os.path.exists('build'):
                os.makedirs('build')
            f = open('build/openapi.json', 'w')
            json.dump(openapi_endpoints, f, indent=4)
            f.flush()
            
            # Use Swagger codegen tool to generate client sdk in target language
            with open(os.devnull, 'wb') as devnull:
                sdk_proc = subprocess.call('java -jar ' + os.path.dirname(os.path.abspath(__file__)) + '/../tools/swagger-codegen-cli-2.2.1.jar generate \
                                            -i build/openapi.json \
                                            -o build \
                                            -l %s' % sdk_lanugage, shell=True, stdout=devnull)
            
            logger.info('Client SDK for %s is generated.' % sdk_lanugage)
        except Exception as e:
            raise Exception('Failed to generate client sdk: ' + str(e))