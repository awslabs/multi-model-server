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

import mms.model_service.mxnet_model_service as mxnet_model_service
from mms.model_service.model_service import load_service
from mms.model_service.mxnet_model_service import SingleNodeService
from mms.storage import KVStorage


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
        module = load_service(service_file) if service_file else mxnet_model_service
        # Parsing the module to get all defined classes
        classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        # Check if class is subclass of base ModelService class
        # pylint: disable=deprecated-lambda
        return list(filter(lambda cls: issubclass(cls, SingleNodeService), classes))
