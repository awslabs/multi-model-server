import ast
import inspect
import os
import imp

from mms.storage import KVStorage
from mms.model_service.model_service import ModelService
import mms.model_service.mxnet_vision_service as mxnet_vision_service


class ServiceManager(object):
    '''ServiceManager will be responsible for storing infomation and managing
    model services. ServiceManager will directly talk to model services.
    In later phase, ServiceManager will also be responsible for model versioning,
    prediction batching and caching...
    '''
    def __init__(self):
        '''
        Initialize Service Manager.
        '''

        # registry for model defination and user defined functions
        self.modelservice_registry = KVStorage('modelservice')
        self.func_registry = KVStorage('func')

        # loaded model services
        self.loaded_modelservices = KVStorage('loaded_modelservices')

    def get_modelservices_registry(self, modelservice_names=None):
        '''
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
        '''
        if modelservice_names is None:
            return self.modelservice_registry

        return {
                    modelservice_name: self.modelservice_registry[modelservice_name]
                    for modelservice_name in modelservice_names
                }

    def add_modelservice_to_registry(self, modelservice_name, ModelServiceClassDef):
        '''
        Add a model service to internal registry.

        Parameters
        ----------
        modelservice_name : string
            Model service name to be added.
        ModelServiceClassDef: python class  
            Model Service Class Definition which can initialize a model service.
        '''
        self.modelservice_registry[modelservice_name] = ModelServiceClassDef

    def get_loaded_modelservices(self, modelservice_names=None):
        '''
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
        '''
        if modelservice_names is None:
            return self.loaded_modelservices

        return {
                    modelservice_name: self.loaded_models[modelservice_name] 
                    for modelservice_name in modelservice_names
                }

    def load_model(self, model_name, model_path, ModelServiceClassDef):
        '''
        Load a single model into a model service by using 
        user passed Model Service Class Definitions.

        Parameters
        ----------
        model_name : string
            Model name
        model_path: stirng
            Model path which can be url or local file path.
        ModelServiceClassDef: python class
            Model Service Class Definition which can initialize a model service.
        '''
        self.loaded_modelservices[model_name] = ModelServiceClassDef(model_path)

    def parse_modelservices_from_module(self, user_defined_module_file_path):
        '''
        Parse user defined module to get all model service classe in it.

        Parameters
        ----------
        user_defined_module_file_path : User defined module file path 
            A python module which will be parsed by given name.
            
        Returns
        ----------
        List of model service class definitions.
            Those parsed python class can be used to initialize model service.
        '''
        module =  imp.load_source(
            os.path.splitext(os.path.basename(user_defined_module_file_path))[0],
            user_defined_module_file_path) if user_defined_module_file_path \
            else mxnet_vision_service

        # Parsing the module to get all defined classes
        classes = [cls[1] for cls in inspect.getmembers(module, inspect.isclass)]
        # Check if class is subclass of base ModelService class
        return list(filter(lambda cls: cls is not ModelService and issubclass(cls, ModelService), classes))
