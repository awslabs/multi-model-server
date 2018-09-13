"""
File to define the entry point to Model Server
"""

import tempfile
import subprocess
import os
import signal
from mms.arg_parser import ArgParser


def start():
    """
    This is the entry point for model server
    :return:
    """
    args = ArgParser.mms_parser().parse_args()
    pid_file = os.path.join(tempfile.gettempdir(), ".model_server.pid")
    if args.stop is True:
        if os.path.isfile(pid_file):
            with open(pid_file, "r") as f:
                try:
                    os.kill(int(f.readline()), signal.SIGKILL)
                except OSError:
                    print("Model server already stopped")
            os.remove(pid_file)
        else:
            print("Model server is not currently running")

    else:
        mms_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = ["java -jar frontend/model-server.jar"]
        if args.mms_config is not None:
            cmd.append("-f")
            cmd.append(args.mms_config)

        if args.models is not None:
            cmd.append("-m")
            cmd.extend(args.models)

        process = subprocess.Popen(cmd.join(" "), shell=True, cwd=mms_home)
        pid = process.pid
        with open(pid_file, "w") as pf:
            pf.write(str(pid))


if __name__ == "__main__":
    start()
