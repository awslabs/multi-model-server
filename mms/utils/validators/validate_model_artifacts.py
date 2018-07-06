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
File to validate all the model artifacts
"""

from mms.mxnet_model_service_error import MMSError
from mms.utils.model_server_error_codes import ModelServerErrorCodes as Err


class MXNetModelArtifactValidator(object):
    """
    Validate the model artifacts for MXNet archive files
    """
    @staticmethod
    def validate_model_metadata(model):
        """
        Validate the metadata retrieved while loading the archive file
        :param model:
        :return:
        """
        if model is None or model[0][0] is None or model[0][1] is None or model[0][2] is None or model[0][3] is None:
            raise MMSError(Err.MISSING_MODEL_ARTIFACTS, "Model artifacts provided in the .model file are invalid.")

    @staticmethod
    def validate_manifest():
        """
        Validate the Manifest file
        :return:
        """
        pass

    @staticmethod
    def validate_signature():
        """
        Validate the signature file, if given.
        :return:
        """
        pass
