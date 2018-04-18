import sys

import os
import pytest

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../..')

from mms.model_loader import ModelLoader


def test_onnx_fails_fast():
    models = {'onnx': 's3://bucket/prefix/whatever.onnx'}

    with pytest.raises(ValueError) as e:
        ModelLoader.load(models)

    assert 'Convert ONNX model' in str(e.value)

def test_invalid_model_path_input():
    """
    Test to ensure that folder being created is removed,
    when path is invalid
    """
    models = {'squeezenet_v1': 'invalid_model_file_path.model'}
    with pytest.raises(Exception) as e:
        ModelLoader.load(models)
    assert not os.path.exists('invalid_model_file_path')
