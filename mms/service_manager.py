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
This module manages model-services
"""
import inspect

from mms.storage import KVStorage
from mms.model_service.model_service import load_service
from mms.model_service.mxnet_model_service import SingleNodeService


class ServiceManager(object):
    """ServiceManager is responsible for storing information and managing
    model services. ServiceManager calls model services directly.
    In later phase, ServiceManager will also be responsible for model versioning,
    prediction batching and caching.
    """

    def __init__(self):
        """
        Initialize Service Manager.
        """

        # registry for model definition and user defined functions
        self.modelservice_registry = KVStorage('modelservice')
        self.func_registry = KVStorage('func')

        # loaded model services
        self.loaded_modelservices = KVStorage('loaded_modelservices')

    def get_modelservices_registry(self, modelservice_names=None):
        """
        Get all registered Model Service Class Definitions in a dictionary
        from internal registry according to name or list of names.
        If nothing is passed, all registered model services will be returned.

        Parameters
        ----------
        modelservice_names : List, optional
            Names to retrieve registered model services.

        Returns
        ----------
        Dict of name, model service pairs
            Registered model services according to given names.
        """
        if modelservice_names is None:
            return self.modelservice_registry

        return {
            modelservice_name: self.modelservice_registry[modelservice_name]
            for modelservice_name in modelservice_names
        }

    def add_modelservice_to_registry(self, modelservice_name, ModelServiceClassDef):
        """
        Add a model service to internal registry.

        Parameters
        ----------
        modelservice_name : string
            Model service name to be added.
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        """
        self.modelservice_registry[modelservice_name] = ModelServiceClassDef

    def get_loaded_modelservices(self, modelservice_names=None):
        """
        Get all model services which are loaded in the system into a dictionary
        according to name or list of names.
        If nothing is passed, all loaded model services will be returned.

        Parameters
        ----------
        modelservice_names : List, optional
             Model service names to retrieve loaded model services.

        Returns
        ----------
        Dict of name, model service pairs
            Loaded model services according to given names.
        """
        if modelservice_names is None:
            return self.loaded_modelservices

        return {
            modelservice_name: self.loaded_modelservices[modelservice_name]
            for modelservice_name in modelservice_names
        }

    def load_model(self, service_name, model_name, model_path, manifest, ModelServiceClassDef, gpu=None):
        """
        Load a single model into a model service by using
        user passed Model Service Class Definitions.

        Parameters
        ----------
        service_name : string
            Service name
        model_name : string
            Model name
        model_path: stirng
            Model path which can be url or local file path.
        manifest: string
            Model manifest
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        gpu : int
            Id of gpu device. If machine has two gpus, this number can be 0 or 1.
            If it is not set, cpu will be used.
        """
        self.loaded_modelservices[service_name] = ModelServiceClassDef(model_name, model_path, manifest, gpu)

    def unload_models(self, service_name):
        del(self.loaded_modelservices[service_name])
        return self.loaded_modelservices

    def parse_modelservices_from_module(self, service_file):
        """
        Parse user defined module to get all model service classe in it.

        Parameters
        ----------
        service_file : User defined module file path
            A python module which will be parsed by given name.

        Returns
        ----------
        List of model service class definitions.
            Those parsed python class can be used to initialize model service.
        """

        if service_file:
            module = load_service(service_file)
        else:
            raise Exception("Invalid service file given")
        # Parsing the module to get all defined classes
        classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        # Check if class is subclass of base ModelService class
        # pylint: disable=deprecated-lambda
        return list(filter(lambda c: issubclass(c, SingleNodeService), classes))

    def load_models(self, models, ModelServiceClassDef, gpu=None):
        """
        Load models by using user passed Model Service Class Definitions.

        Parameters
        ----------
        models : List of model_name, model_path pairs
            List of model_name, model_path pairs that will be initialized.
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        gpu : int
            Id of gpu device. If machine has two gpus, this number can be 0 or 1.
            If it is not set, cpu will be used.
        """
        for service_name, model_name, model_path, manifest in models:
            self.load_model(service_name, model_name, model_path, manifest, ModelServiceClassDef, gpu)


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
        model_class_definations = self.parse_modelservices_from_module(user_defined_module_file_path)
        assert len(model_class_definations) >= 1, \
            'No valid python class derived from Base Model Service is in module file: %s' % \
            user_defined_module_file_path

        for ModelServiceClassDef in model_class_definations:
            self.add_modelservice_to_registry(ModelServiceClassDef.__name__, ModelServiceClassDef)

        return model_class_definations

    def get_registered_modelservices(self, modelservice_names=None):
        """
        Get all registered Model Service Class Definitions into a dictionary
        according to name or list of names.
        If nothing is passed, all registered model services will be returned.

        Parameters
        ----------
        modelservice_names : str or List, optional
            Names to retrieve registered model services

        Returns
        ----------
        Dict of name, model service pairs
            Registered model services according to given names.
        """
        if not isinstance(modelservice_names, list) and modelservice_names is not None:
            modelservice_names = [modelservice_names]

        return self.get_modelservices_registry(modelservice_names)

    def register_and_load_modules(self, module_file_path, models, gpu):
        """
        Register all the modules and load them. This is a wrapper method around register_module and load_models.
        :param module_file_path:
        :param models:
        :param gpu:
        :return:
        """
        if models is None:
            raise Exception("Models set incorrectly. Models:{}".format(models))

        # Retrieve all the classes defined in the custom service file
        classes = self.register_module(module_file_path)

        # Filter the outer most class defn. Exclude all the superclasses
        classes = list(filter(lambda c: len(c.__subclasses__()) == 0, classes))

        if len(classes) != 1:
            raise Exception("Invalid service file found {}."
                            " Service file should contain only one service class. "
                            "Found {}".format(module_file_path, len(classes)))

        model_class_name = classes[0].__name__

        registered_models = self.get_registered_modelservices()
        model_class_defn = registered_models[model_class_name]

        self.load_models(models, model_class_defn, gpu)