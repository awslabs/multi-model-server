"""
File to define the entry point to Model Server
"""

import os
import subprocess
import tempfile
from builtins import str

import psutil

from mms.arg_parser import ArgParser


def start():
    """
    This is the entry point for model server
    :return:
    """
    args = ArgParser.mms_parser().parse_args()
    pid_file = os.path.join(tempfile.gettempdir(), ".model_server.pid")
    pid = None
    if os.path.isfile(pid_file):
        with open(pid_file, "r") as f:
            pid = int(f.readline())

    if args.stop is True:
        if pid is None:
            print("Model server is not currently running.")
        else:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except OSError:
                print("Model server already stopped.")
            os.remove(pid_file)
    else:
        if pid is not None:
            try:
                psutil.Process(pid)
                print("Model server is already running.")
                exit(1)
            except psutil.Error:
                print("Removing orphan pid file.")
                os.remove(pid_file)

        java_home = os.environ.get("JAVA_HOME")
        java = "java" if java_home is None else "{}/bin/java".format(java_home)

        mms_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = [java, "-Dmodel_server_home={}".format(mms_home)]
        if args.log_config is not None:
            cmd.append("-Dlog4j.configuration={}".format(args.log_config))

        cmd.append("-jar")
        cmd.append("{}/mms/frontend/model-server.jar".format(mms_home))

        if args.mms_config is not None:
            cmd.append("-f")
            cmd.append(args.mms_config)

        if args.models is not None:
            cmd.append("-m")
            cmd.extend(args.models)

        process = subprocess.Popen(cmd)
        pid = process.pid
        with open(pid_file, "w") as pf:
            pf.write(str(pid))


if __name__ == "__main__":
    start()
