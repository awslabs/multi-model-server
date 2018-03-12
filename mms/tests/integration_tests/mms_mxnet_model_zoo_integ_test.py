import pytest
import test_utils as utils


@pytest.fixture(params=utils.mxnet_model_urls.keys())
def model_data(request):
    return request.param


def test_mxnet_integ(tmpdir, model_data):
    tmpdir = str(tmpdir)
    model_urls = utils.filtered_urls([model_data], utils.mxnet_model_urls)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    utils.start_test(
        tmpdir,
        model_urls,
        port='8080',
        model_type='mxnet')
    utils.cleanup(tmpdir)
