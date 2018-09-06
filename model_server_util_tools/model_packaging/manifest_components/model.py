# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import json


class Model(object):

    def __init__(self, model_name, description, model_version, extensions, handler):
        self.model_name = model_name
        self.description = description
        self.model_version = model_version
        self.extentions = extensions
        self.handler = handler
        self.model_dict = self.__to_dict__()

    def __to_dict__(self):
        model_dict = dict()
        model_dict['modelName'] = self.model_name
        model_dict['description'] = self.description
        model_dict['modelVersion'] = self.model_version
        model_dict['extensions'] = self.extensions
        model_dict['handler'] = self.handler

        return model_dict

    def __str__(self):
        return json.dumps(self.model_dict)

    def __repr__(self):
        return json.dumps(self.model_dict)