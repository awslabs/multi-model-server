# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Model loader.
"""
import importlib
import inspect
import json
import os
import sys
from abc import ABCMeta, abstractmethod

from mms.mxnet_model_service_error import MMSError
from mms.service import Service
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err


class ModelLoaderFactory(object):
    """
    ModelLoaderFactory
    """

    @staticmethod
    def get_model_loader(service_type):
        if service_type == "mms":
            return MmsModelLoader()
        elif service_type == "legacy_mms":
            return LegacyModelLoader()
        else:
            raise ValueError("Unknown model loader type: {}".format(service_type))


class ModelLoader(object):
    """
    Base Model Loader class.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return: Model
        """
        pass

    @staticmethod
    def list_model_services(module, parent_class=None):
        """
        Parse user defined module to get all model service classes in it.

        :param module:
        :param parent_class:
        :return: List of model service class definitions
        """

        # Parsing the module to get all defined classes
        classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                        member.__module__ == module.__name__)]
        # filter classes that is subclass of parent_class
        if parent_class is not None:
            return [c for c in classes if issubclass(c, parent_class)]

        return classes


class MmsModelLoader(ModelLoader):
    """
    MMS 1.0 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load MMS 1.0 model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return:
        """
        manifest_file = os.path.join(model_dir, "MAR_INF/MANIFEST.json")
        manifest = None
        if os.path.exists(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)

        temp = handler.split(":", 2)
        module_name = temp[0]
        function_name = None if len(temp) == 1 else temp[1]
        module = importlib.import_module(module_name)
        if module is None:
            raise Exception("Unable to load module {}, make sure it is added to python path".format(module_name))
        if function_name is None:
            function_name = "handle"
        if hasattr(module, function_name):
            entry_point = getattr(module, function_name)
            service = Service(model_name, model_dir, manifest, entry_point, gpu_id, batch_size)

            # initialize model at load time
            entry_point(None, service.context)
        else:
            model_class_definitions = ModelLoader.list_model_services(module)
            if len(model_class_definitions) != 1:
                raise MMSError(Err.VALUE_ERROR_WHILE_LOADING,
                               "Expected only one class in custom service code or a function entry point")
            model_class = model_class_definitions[0]
            model_service = model_class()
            handle = getattr(model_service, "handle")
            if handle is None:
                raise MMSError(Err.VALUE_ERROR_WHILE_LOADING,
                               "Expect handle method in class {}".format(str(model_class)),)
            service = Service(model_name, model_dir, manifest, model_service.handle, gpu_id, batch_size)
            initialize = getattr(model_service, "initialize")
            if initialize is not None:
                # noinspection PyBroadException
                try:
                    model_service.initialize(service.context)
                # pylint: disable=broad-except
                except Exception:
                    sys.exc_clear()

        return service


class LegacyModelLoader(ModelLoader):
    """
    MMS 0.4 Model Loader
    """

    def load(self, model_name, model_dir, handler, gpu_id, batch_size):
        """
        Load MMS 0.3 model from file.

        :param model_name:
        :param model_dir:
        :param handler:
        :param gpu_id:
        :param batch_size:
        :return:
        """
        manifest_file = os.path.join(model_dir, "MANIFEST.json")
        with open(manifest_file) as f:
            manifest = json.load(f)
        if not handler.endswith(".py"):
            handler = handler + ".py"

        service_file = os.path.join(model_dir, handler)
        name = os.path.splitext(os.path.basename(service_file))[0]
        if sys.version_info[0] > 2:
            from importlib import util

            spec = util.spec_from_file_location(name, service_file)
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            import imp
            module = imp.load_source(name, service_file)
        if module is None:
            raise MMSError(Err.VALUE_ERROR_WHILE_LOADING, "Unable to load module {}".format(service_file))

        from mms.model_service.mxnet_model_service import SingleNodeService

        model_class_definitions = ModelLoader.list_model_services(module, SingleNodeService)
        module_class = model_class_definitions[0]

        module = module_class(model_name, model_dir, manifest, gpu_id)
        service = Service(model_name, model_dir, manifest, module.handle, gpu_id, batch_size)

        module.initialize(service.context)

        return service
