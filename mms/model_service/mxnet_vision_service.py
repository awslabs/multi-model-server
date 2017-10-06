import mxnet as mx

from mxnet_model_service import MXNetBaseService
from utils.mxnet_utils import Image, NDArray


class MXNetVisionService(MXNetBaseService):
    def _preprocess(self, data):
        img_list = []
        for idx, image in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = Image.read(image)
            img_arr = Image.resize(img_arr, w, h)
            img_arr = Image.transform_shape(img_arr)
            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        return [NDArray.top_probability(d, self.labels, top=5) for d in data]

