import test_utils as utils


# models from onnx-mxnet model zoo

mxnet_models = ['caffenet', 'Inception-BN', 'nin', 'squeezenet_v1.1']


def test_multiple_mxnet_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    model_urls = utils.filtered_urls(mxnet_models, utils.mxnet_model_urls)
    utils.start_test(
        tmpdir,
        model_urls,
        port='8080',
        model_type='mxnet')
    utils.cleanup(tmpdir)
