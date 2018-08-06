"""
File to define the entry point to Model Server
"""

import subprocess
import os
import signal
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
        process = subprocess.Popen("java -jar %s/frontend/model-server.jar" %
                                   os.path.dirname(os.path.abspath(__file__)), shell=True)
        pid = process.pid
        with open(pid_file, 'w') as pf:
            pf.write(str(pid))

    else: # args.stop is True:
        if os.path.isfile(pid_file):
            with open(pid_file, 'r') as f:
                try:
                    os.kill(int(f.readline()), signal.SIGKILL)
                except OSError:
                    print("Model server already stopped")
            os.remove(pid_file)
        else:
            print("Model server is not currently running")
