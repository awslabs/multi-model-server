"""
File to define the entry point to Model Server
"""

import os
import re
import subprocess
import sys
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

    if args.stop:
        if pid is None:
            print("Model server is not currently running.")
        else:
            try:
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                print("Model server stopped.")
            except (OSError, psutil.Error):
                print("Model server already stopped.")
            os.remove(pid_file)
    else:
        if pid is not None:
            try:
                psutil.Process(pid)
                print("Model server is already running, please use mxnet-model-server --stop to stop MMS.")
                exit(1)
            except psutil.Error:
                print("Removing orphan pid file.")
                os.remove(pid_file)

        java_home = os.environ.get("JAVA_HOME")
        java = "java" if not java_home else "{}/bin/java".format(java_home)

        mms_home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        cmd = [java, "-Dmodel_server_home={}".format(mms_home)]
        if args.log_config:
            log_config = os.path.realpath(args.log_config)
            if not os.path.isfile(log_config):
                print("--log-config file not found: {}".format(log_config))
                exit(1)

            cmd.append("-Dlog4j.configuration=file://{}".format(log_config))

        tmp_dir = os.environ.get("TEMP")
        if tmp_dir:
            if not os.path.isdir(tmp_dir):
                print("Invalid temp directory: {}, please check TEMP environment variable.".format(tmp_dir))
                exit(1)

            cmd.append("-Djava.io.tmpdir={}".format(tmp_dir))

        if args.mms_config:
            if not os.path.isfile(args.mms_config):
                print("--mms-config file not found: {}".format(args.mms_config))
                exit(1)

            props = load_properties(args.mms_config)
            vm_args = props.get("vmargs")
            if vm_args:
                cmd.extend(vm_args.split())

        cmd.append("-jar")
        cmd.append("{}/mms/frontend/model-server.jar".format(mms_home))

        # model-server.jar command line parameters
        cmd.append("--python")
        cmd.append(sys.executable)

        if args.mms_config:
            cmd.append("-f")
            cmd.append(args.mms_config)

        if args.model_store:
            if not os.path.isdir(args.model_store):
                print("--model-store directory not found: {}".format(args.model_store))
                exit(1)

            cmd.append("-s")
            cmd.append(args.model_store)

        if args.models:
            cmd.append("-m")
            cmd.extend(args.models)
            if not args.model_store:
                pattern = re.compile(r"(.+=)?http(s)?://.+", re.IGNORECASE)
                for model_url in args.models:
                    if not pattern.match(model_url) and model_url != "ALL":
                        print("--model-store is required to load model locally.")
                        exit(1)

        try:
            process = subprocess.Popen(cmd)
            pid = process.pid
            with open(pid_file, "w") as pf:
                pf.write(str(pid))
        except OSError as e:
            if e.errno == 2:
                print("java not found, please make sure JAVA_HOME is set properly.")
            else:
                print("start java frontend failed:", sys.exc_info())


def load_properties(file_path):
    """
    Read properties file into map.
    """
    props = {}
    with open(file_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                pair = line.split("=", 1)
                if len(pair) > 1:
                    key = pair[0].strip()
                    props[key] = pair[1].strip()

    return props


if __name__ == "__main__":
    start()
