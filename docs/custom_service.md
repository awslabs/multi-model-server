# Defining a Custom Service

A custom service is a way to customize MMS inference request handling logic. Potential customizations include model initialization, inference request pre-processing, post-processing, and even the inference call itself.

The service code is provided in two possible ways.

1. The `--service` argument, along with the path to the custom service Python file, is used when launching `mxnet-model-server`. If this argument is not used MMS will look in the model archive as described next.
1. A custom service Python file may be included inside the model archive. When the model archive is executed by `mxnet-model-server` it will detect the presence of the custom service and load it.

The `--service` argument overrides any custom service file inside the model archive.

## Vision Service

The simplest custom service example comes from one that is built into MMS, the `mxnet_vision_service`. When you run the resnet-18 or squeezenet example for image inference to get predictions like "what's in this image?", it's actually using the custom vision service. You can check out the [code in its entirety](../mms/model_service/mxnet_vision_service.py), but we'll also cover the highlights here. In the following code snippet, you can see that the vision service is taking in the image and resizing it. The main reasons you want to do this are as follows: you never know what resolution of image someone might submit to the API, models require the input to be the same size/shape that they were trained on, and you don't want to have to deal with that logic in your application. The vision service handles that for you.

```python
input_shape = self.signature['inputs'][idx]['data_shape']
# We are assuming input shape is NCHW
[h, w] = input_shape[2:]
img_arr = image.read(img)
img_arr = image.resize(img_arr, w, h)
img_arr = image.transform_shape(img_arr)
```

### MXNet Image API Wrapper

Take a closer look at the resize code, and you will note that it is pulling the height (h) and the width (w) from the signature data. This signature data describes the model inputs, so the images are being resized to match what the model expects. The resize mechanism comes from `mxnet.img.imresize` via MMS utility wrapper, and it will upsample images smaller than the input size. Note that you can optionally add the `interp` parameter to the `resize` call for different interpolation methods. Details on the options are in the comments for the `resize` function found in [utils/mxnet/image.py](../dms/utils/mxnet/image.py). Images will be stretched by default, so if you need any other image handling like [resizing on the short edge](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py#L229), [center crop](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/image/image.py#L362), etc. you will need to call [MXNet's image API](https://mxnet.incubator.apache.org/api/python/image/image.html) directly.

Now say you want more pre-processing. This is designed to be easy. You have two options: you can extend the vision service, or you can go back to the base service, `MXNetBaseService`, and extend that instead.

## Calling a Custom Service

When you launch the `mxnet-model-server` CLI using the `service` argument, the path to a service Python file is required. For example, if you want to use a local model file and manually call the `mxnet_vision_service` you would use:

```
mxnet-model-server --models squeezenet=squeezenet.model \
                  --service mms/model_service/mxnet_vision_service.py
```

This assumes that you've downloaded the MMS source and you're in the source root directory. In this example you're using the vision service that comes with MMS, but otherwise you don't need the source, and you can specify any Python file that contains your custom service code.

## Designing a Custom Service

All customized service classes should be inherited from `MXNetBaseService`, but you can also extend another custom service class that eventually drills down to `MXNetBaseService` which is outlined here:

```python
class MXNetBaseService(SingleNodeService):
  def __init__(self, path, synset=None, ctx=mx.cpu()):

  def _inference(self, data):

  def _preprocess(self, data):

  def _postprocess(self, data, method='predict'):
```

Usually you would want to override `_preprocess` and `_postprocess` as features for your application, such as massaging the inputs and the outputs. This is going to be bound by the specific domain of your applications. For example, you could add functionality to `_preprocess` for resizing or otherwise modifying images to match your model's input. You could add logic in `_postprocess` for how prediction results are returned to the user. We provide some utility functions in the [utils](../mms/utils/) folder for vision and NLP applications to help you easily build basic pre-process functions.

### An Image Inference Example

The following example is for a resnet-18 service that returns a prediction of what's in the image. In this example, we don't need to change `__init__` or `_inference` methods, which means we just need override `_preprocess` and `_postprocess`. In `_preprocess`, we first read the image data, and then resize to 224 x 224. In `_postprocess`, we return top 5 categories:

```python
   import mxnet as mx
   from mms.utils.mxnet import image
   from mms.model_service.mxnet_model_service import MXNetBaseService

   class Resnet18Service(MXNetBaseService):
       def _preprocess(self, data):
           img_arr = image.read(data[0])
           img_arr = image.resize(img_arr, 224, 224)
           return [img_arr]

       def _postprocess(self, data):
           output = data[0]
           sorted = mx.nd.argsort(output[0], is_ascend=False)
           for i in sorted[0:5]:
               response[output[0][i]] = self.labels[i]
           return response
```

You might be thinking, wait a minute, I thought we used the vision service for the resnet-18 example. Yes, that's right, but what we're showing here is how you might extend the base service and manually define the image size and the number of outputs rather than rely on the vision service.

### An Object Detection Example

Another more complex example using object detection (SSD) is provided in the [examples/ssd](../examples/ssd/README.md) folder. The goal of this example is to use an object detection model to detect several objects in the image, classify them, and also return bounding boxes to show where in the image the objects were detected. As opposed to the resnet-18 example, we take advantage of the vision service and extend it slightly to record a copy of the original image's shape during pre-processing.

```python
def _preprocess(self, data):
    input_image = image.read(data[0])
    self.input_height = input_image.shape[0]
    self.input_width = input_image.shape[1]
    return super(SSDService, self)._preprocess(data)
```

We then use this info in post-processing to provide accurate bounding boxes.

```python
def _postprocess(self, data):
    detections = data[0].asnumpy()
    result = []
    for i in range(detections.shape[0]):
        det = detections[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]
        result.append(res)

    dets = result[0]
    classes = self.labels
    width = self.input_width    # original input image width
    height = self.input_height  # original input image height
    response = []
    for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
            score = dets[i, 1]
            if score > self.threshold:
                xmin = int(dets[i, 2] * width)
                ymin = int(dets[i, 3] * height)
                xmax = int(dets[i, 4] * width)
                ymax = int(dets[i, 5] * height)
                class_name = str(cls_id)
                if classes and len(classes) > cls_id:
                    class_name = classes[cls_id]
                response.append((class_name, xmin, ymin, xmax, ymax))
    return response
```

To get a better picture of how it works, take a look at the [full Python code](../examples/ssd/ssd_service.py) that has more comments, and [try out the example yourself](../examples/ssd/README.md).
