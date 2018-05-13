from mms.model_service.mxnet_vision_service import MXNetVisionService
import wrong_package


class DummyService(MXNetVisionService):

    def __init__(self, model_name, model_dir, manifest, gpu=None):
        pass
    def _preprocess(self,data):
        pass
    def _inference(self,data):
        pass
    def _postprocess(self,data):
        pass
