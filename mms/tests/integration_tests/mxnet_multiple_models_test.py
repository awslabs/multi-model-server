import test_utils as utils


# models from onnx-mxnet model zoo

mxnet_models = ['caffenet', 'Inception-BN', 'nin', 'squeezenet_v1.1']
mxnet_model_urls = {}


def test_onnx_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    for models in mxnet_models:
        mxnet_model_urls[models] = utils.mxnet_model_urls[models]
    utils.start_test(
        tmpdir,
        mxnet_model_urls,
        None,
        port='8080',
        onnx_source_model_zoo=False,
        is_onnx_model=False,
        test_multiple_models=True)
    utils.cleanup(tmpdir)
