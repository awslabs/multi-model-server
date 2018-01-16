import os
import pytest
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_path + '/../..')

from mms.model_loader import ModelLoader

def test_onnx_fails_fast():
    models = { 'onnx': 's3://bucket/prefix/whatever.onnx'}

    with pytest.raises(ValueError) as e:
        ModelLoader.load(models)

    assert 'Convert ONNX model' in str(e.value)
