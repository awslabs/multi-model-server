# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Image utils
"""
from io import BytesIO
import cv2

import numpy as np
from PIL import Image


def transform_shape(img_arr, dim_order='NHWC'):
    """
    Rearrange image numpy array shape to 'NCHW' or 'NHWC' which
    is valid for TF model input.
    Input image array should have dim_order of 'HWC'.

    :param img_arr: numpy array
        Image in numpy format with shape (height, width, channel)
    :param dim_order: str
        Output image dimension order. Valid values are 'NCHW' and 'NHWC'

    :return: numpy array
        Image in numpy array format with dim_order shape
    """
    assert dim_order in 'NCHW' or dim_order in 'NHWC', "dim_order must be 'NCHW' or 'NHWC'."
    if dim_order == 'NCHW':
        img_arr = np.transpose(img_arr, (2, 0, 1))
    output = np.expand_dims(img_arr, axis=0)
    return output


def read(buf):
    """
    Read and decode an image to a numpy array.
    Input image numpy should have dim_order of 'HWC'.

    :param buf: image bytes
        Binary image data as bytes.
    :return: numpy array
        A numpy array containing the image.
    """
    return np.array(Image.open(BytesIO(buf)))


def resize(src, new_width, new_height, interp=2):
    """
    Resizes image to new_width and new_height.
    Input image numpy array should have dim_order of 'HWC'.

    :param src: numpy array
        Source image in numpy array format
    :param new_width: int
        Width in pixel for resized image
    :param new_height: int
        Height in pixel for resized image
    :param interp: int
        interpolation method for all resizing operations

    :return: numpy array
        An numpy array containing the resized image.
    """
    return cv2.resize(src, dsize=(new_height, new_width), interpolation=interp)