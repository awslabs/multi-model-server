import test_utils as utils


# models from onnx-mxnet model zoo

mxnet_models = ['caffenet', 'Inception-BN', 'nin', 'squeezenet_v1.1']


def test_onnx_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    urls = utils.filtered_urls(mxnet_models, utils.mxnet_model_urls)
    utils.start_test(
        tmpdir,
        urls,
        port='8080',
        onnx_source_model_zoo=False,
        is_onnx_model=False)
    utils.cleanup(tmpdir)
