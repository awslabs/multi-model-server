import test_utils as utils


# models from onnx-mxnet model zoo
onnx_models = ['inception_v1', 'inception_v2', 'shufflenet', 'squeezenet']


def test_onnx_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    urls = utils.filtered_urls(onnx_models, utils.onnx_model_urls)
    utils.start_test(
        tmpdir,
        urls,
        port='8080',
        onnx_source_model_zoo=True,
        is_onnx_model=True)
    utils.cleanup(tmpdir)
