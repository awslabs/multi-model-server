import json
import os
import socket
import struct
import sys
import time
import traceback

from mms.arg_parser import ArgParser
from mms.model_loader import ModelLoader
from mms.service_manager import ServiceManager


class MxNetWorker(object):
    """

    """

    def __init__(self, args=None, models=None):
        """
        Initialize Service Manager.
        """
        self.args = args
        self.models = models
        try:
            self.service_manager = ServiceManager()
            print('Initialized model worker.')
        except Exception as e:
            raise Exception('Failed to initialize model worker: ' + str(e))

    def arg_process(self):
        """Process arguments before starting service or create application.
        """
        try:
            # Register user defined model service or default mxnet_vision_service
            manifest = self.models[0][3]
            service_file = os.path.join(self.models[0][2], manifest['Model']['Service'])

            class_defs = self.register_module(self.args.service or service_file)
            class_defs = list(filter(lambda c: len(c.__subclasses__()) == 0, class_defs))

            if len(class_defs) != 1:
                raise Exception('There should be one user defined service derived from ModelService.')
            model_class_name = class_defs[0].__name__

            registered_models = self.service_manager.get_modelservices_registry()
            model_class_def = registered_models[model_class_name]

            self.load_models(self.models, model_class_def, self.args.gpu)
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to process arguments: ' + str(e))
            exit(1)

    def load_models(self, models, model_class_def, gpu=None):
        """
        Load models by using user passed Model Service Class Definitions.

        Parameters
        ----------
        models : List of model_name, model_path pairs
            List of model_name, model_path pairs that will be initialized.
        model_class_def: python class
            Model Service Class Definition which can initialize a model service.
        gpu : int
            Id of gpu device. If machine has two gpus, this number can be 0 or 1.
            If it is not set, cpu will be used.
        """
        for service_name, model_name, model_path, manifest in models:
            self.service_manager.load_model(service_name, model_name, model_path, manifest, model_class_def, gpu)

    def register_module(self, user_defined_module_file_path):
        """
        Register a python module according to user_defined_module_name
        This module should contain a valid Model Service Class whose
        pre-process and post-process can be derived and customized.

        Parameters
        ----------
        user_defined_module_file_path : Python module file path
            A python module will be loaded according to this file path.


        Returns
        ----------
        List of model service class definitions.
            Those python class can be used to initialize model service.
        """
        model_class_definations = self.service_manager.parse_modelservices_from_module(user_defined_module_file_path)
        assert len(model_class_definations) >= 1, \
            'No valid python class derived from Base Model Service is in module file: %s' % \
            user_defined_module_file_path

        for ModelServiceClassDef in model_class_definations:
            self.service_manager.add_modelservice_to_registry(ModelServiceClassDef.__name__, ModelServiceClassDef)

        return model_class_definations

    def predict_callback(self, model_name, data):
        """
        Callback for predict endpoint

        Parameters
        ----------
        model_name: bytes
            model name.
        data: list
            Input names in request form data.

        Returns
        ----------
        Response
            Http response for predict endpiont.
        """

        model_services = self.service_manager.get_loaded_modelservices()
        modelservice = model_services[str(model_name)]

        return modelservice.inference(data)

    def start(self):
        server_address = "/tmp/.mms.worker.%s" % self.args.port
        try:
            os.unlink(server_address)
        except OSError:
            if os.path.exists(server_address):
                raise

        print("starting up on %s" % server_address)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(server_address)
        sock.listen(1)

        sys.stdout.write("MxNet worker started.\n")
        sys.stdout.flush()

        while True:
            connection, client_address = sock.accept()
            try:
                while True:
                    magic_number = connection.recv(1)
                    if magic_number == b'':
                        print("Connection dropped.")
                        return

                    if magic_number != b'M':
                        print("Invalid magic number: " + magic_number)
                        return

                    model_name = MxNetWorker.read_data(connection)
                    if model_name is None:
                        print('Missing model name: ' + client_address)
                        return

                    request_ids = []
                    request = []

                    length = MxNetWorker.read_length(connection)
                    while length > 0:
                        job_id = connection.recv(length, socket.MSG_WAITALL)
                        if job_id is None:
                            print('Missing payload in message: ' + client_address)
                            return

                        buf = self.read_data(connection)  # request payload
                        if buf is None:
                            print('Missing payload in message: ' + client_address)
                            return

                        request_ids.append(job_id)
                        request.append(buf)
                        length = MxNetWorker.read_length(connection)

                    begin = time.time()

                    try:
                        response = self.predict_callback(model_name, request)
                    except Exception:  # pylint: disable=broad-except
                        print(str(traceback.format_exc()))
                        response = ["Unkown inference error."]

                    print("Predict latency: {}".format((time.time() - begin) * 1000))

                    resp = b'M'
                    resp += struct.pack(">i", len(model_name))
                    resp += model_name
                    for i in range(0, len(response)):
                        resp += struct.pack(">i", len(request_ids[i]))
                        resp += request_ids[i]
                        buf = bytearray(json.dumps(response[i]), 'utf-8')
                        resp += struct.pack(">i", len(buf))
                        resp += buf

                    resp += struct.pack(">i", -1)
                    connection.sendall(resp)
            finally:
                connection.close()

    @staticmethod
    def read_length(connection):
        data = connection.recv(4, socket.MSG_WAITALL)
        if data is None:
            raise Exception('Invalid protocol')

        return struct.unpack(">i", data)[0]

    @staticmethod
    def read_data(connection):
        length = MxNetWorker.read_length(connection)
        if length > 6553500:
            raise Exception('Message size too large: ' + length)

        return connection.recv(length, socket.MSG_WAITALL)


def start_worker(args=None):
    """
    Start worker.

    Parameters
    ----------
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

    models = ModelLoader.load(arguments.models)
    worker = MxNetWorker(args=arguments, models=models)
    worker.arg_process()
    worker.start()


if __name__ == '__main__':
    start_worker()
