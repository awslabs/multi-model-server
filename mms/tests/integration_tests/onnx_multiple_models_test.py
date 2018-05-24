import test_utils as utils


# models from onnx model zoo TODO:temporarily removed shufflenet
onnx_models = ['inception_v1', 'inception_v2'   , 'squeezenet']


def test_onnx_integ(tmpdir):
    tmpdir = str(tmpdir)
    utils._download_file(
        tmpdir,
        "https://s3.amazonaws.com/model-server/inputs/kitten.jpg")
    model_urls = utils.filtered_urls(onnx_models, utils.onnx_model_urls)
    utils.start_test(
        tmpdir,
        model_urls,
        port='8080',
        model_type='onnx')
    utils.cleanup(tmpdir)
