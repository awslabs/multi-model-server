import test_utils as utils


# models from onnx-mxnet model zoo
onnx_models = ['inception_v1', 'inception_v2', 'shufflenet', 'squeezenet']
onnx_model_urls = {}


def test_onnx_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    for models in onnx_model_urls:
        onnx_model_urls[models] = utils.onnx_model_urls[models]
    utils.start_test(
        tmpdir,
        onnx_model_urls,
        None,
        port='8080',
        onnx_source_model_zoo=True,
        is_onnx_model=True,
        test_multiple_models=True)
    utils.cleanup(tmpdir)
