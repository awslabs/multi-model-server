import pytest
import test_utils as utils


@pytest.fixture(params=utils.mxnet_model_urls.keys())
def model_data(request):
    return request.param


def test_onnx_integ(tmpdir, model_data):
    tmpdir = str(tmpdir)
    urls = utils.filtered_urls([model_data], utils.mxnet_model_urls)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    utils.start_test(
        tmpdir,
        urls,
        port='8080',
        onnx_source_model_zoo=False,
        is_onnx_model=False)
    utils.cleanup(tmpdir)
