# Defining a Custom Service

TODO: introduction

TODO: provide many more examples of pre & post processing.

When you launch the `deep-model-server` CLI using the `service` argument, the path to a service Python file is required. In this file, you specify your own custom service. All customized service classes should be inherited from `MXNetBaseService`:

```python
   class MXNetBaseService(SingleNodeService):
      def __init__(self, path, synset=None, ctx=mx.cpu())

      def _inference(self, data)

      def _preprocess(self, data)

      def _postprocess(self, data, method='predict')
```

Usually you would like to override `_preprocess` and `_postprocess` as features for your application massage the inputs and the outputs. This is going to be bound by the specific domain of your applications. For example, you could add functionality to `_preprocess` for resizing or otherwise modifying images to match your model's input. You could add logic in `_postprocess` for how prediction results are returned to the user. We provide some [utility functions](https://github.com/deep-learning-tools/mxnet-model-server/tree/master/mms/utils) for vision and NLP applications to help user easily build basic preprocess functions.
The following example is for resnet-18 service. In this example, we don't need to change `__init__` or `_inference` methods, which means we just need override `_preprocess` and `_postprocess`. In preprocess, we first convert image to NDArray. Then resize to 224 x 224. In post process, we return top 5 categories:

```python
   import mxnet as mx
   from dms.mxnet_utils import image
   from dms.mxnet_model_service import MXNetBaseService

   class Resnet18Service(MXNetBaseService):
       def _preprocess(self, data):
           img_arr = image.read(data)
           img_arr = image.resize(img_arr, 224, 224)
           return img_arr

       def _postprocess(self, data):
           output = data[0]
           sorted = mx.nd.argsort(output[0], is_ascend=False)
           for i in sorted[0:5]:
               response[output[0][i]] = self.labels[i]
           return response
```
