"""
File to define the entry point to Model Server
"""

import subprocess
import os
import psutil
from mms.arg_parser import ArgParser

pid_file = '/tmp/.model_server.pid'


def start():
    """
    This is the entry point for model server
    :return:
    """
    args = ArgParser.mms_parser().parse_args()
    if args.start is True:
        os.environ['MODEL_SERVER_HOME'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if args.mms_config is not None:
            os.environ['MMS_CONFIG_FILE'] = args.mms_config
        subprocess.Popen("java -jar %s/frontend/model-server.jar" %
                         os.path.dirname(os.path.abspath(__file__)), shell=True)

    else:  # args.stop is True:
        # TODO: Can we write this in a better way?
        for p in psutil.process_iter():
            if "java" in p.name():
                for pc in p.cmdline():
                    if "model-server.jar" in pc:
                        p.terminate()
