# Loading and serving Gluon models on MXNet Model Server (MMS)

MXNet Model Server (MMS) supports loading and serving MXNet Imperative and Hybrid Gluon models.
This is a short tutorial on how to write a custom Gluon model, and then serve it with MMS.

This tutorial covers the following:
1. [How to serve Gluon models](#how-to-serve-gluon-models)
2. [Load and serve pretrained models](#load-and-serve-a-pretrained-gluon-model)
3. [Load and serve pure Gluon imperative models](#load-and-serve-pure-gluon-imperative-models)
4. [Load and serve a hybridized Gluon model](#load-and-serve-a-hybridized-gluon-model)
5. [How to test your Gluon models](#how-to-test-your-gluon-models)
6. [How to create a model archive](#how-to-create-a-model-archive)

## Prerequisites

* **Basic Gluon knowledge**. If you are using Gluon for the first
time, but are familiar with creating a neural network with MXNet or another framework, you may refer this 10 min Gluon crash-course: [Predict with a pre-trained model](http://gluon-crash-course.mxnet.io/predict.html).
* **Gluon naming**. Fine-tuning pre-trained Gluon models requires some understanding of how the naming conventions work. Take a look at the [Naming of Gluon Parameter and Blocks](https://mxnet.incubator.apache.org/tutorials/gluon/naming.html) tutorial for more information.
* **Basic MMS knowledge**. If you are using MMS for the first time, you should take advantage of the [MMS QuickStart tutorial](https://github.com/awslabs/mxnet-model-server#quick-start).
* **MMS Custom Service knowledge.** Review the [Defining a Custom Service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md) documentation to understand how MMS implements custom pre and post-processing for models.
* **MMS installed**. If you haven't already, [install MMS with pip](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-with-pip) or [install MMS from source](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-from-source-code). Either installation will also install MXNet.


<!----
TODO: move this - seems like TMI at this point?

**NOTE: If you do not use `Symbols` parameter in `MANIFEST.json`, MMS infers this as a imperative Gluon model
(or network defined by run). In other words DO-NOT-USE `Symbols` parameter in `MANIFEST.json` when defining `Gluon`
models.**

---->


## How to access Gluon models for use with MMS

MXNet provides a wide range of pre-trained computer vision models through `mxnet.gluon.model_zoo.vision`. You can access the pre-trained models by [installing MXNet](https://mxnet.incubator.apache.org/install/index.html), however, if you have already installed MMS, you have MXNet as well.

For this tutorial, you will use [AlexNet](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html#mxnet.gluon.model_zoo.vision.alexnet)
for the model. The example code and requisite artifacts needed to create a MMS model archive from a Gluon AlexNet model are in this folder and discussed in more detail in the following sections.

You can access the models with two lines of code, as follows:

```python
from mxnet.gluon.model_zoo import vision
alexnet = vision.alexnet(pretrained=True)
```

Refer to the [MXNet model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) documentation for examples of accessing other models.
## Load and serve a Gluon model

There are three scenarios for serving a Gluon model with MMS:

1. Load and serve a pre-trained pure imperative model.
2. Load and serve a custom imperative model.
3. Load and serve a custom hybrid (imperative/symbol) model.

A pure imperative model is pure as compared to a hybrid model, which has both imperative and symbolic logic.

### Load and serve a pre-trained pure imperative model

Loading and serving a pre-trained pure imperative model is the simplest of the three scenarios. Unlike MXNet's Symbol API, you don't need both `params` and `symbols` to be given as inputs. When acquiring the imperative model from the zoo, only `params` are delivered.

While it is easy to access a model with a couple of lines of code, with MMS you will want to use a [MMS custom service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md) code pattern as follows:

```python
from mxnet.gluon.model_zoo import vision

class AlexnetService():

    def inference(self, data):
        net = vision.alexnet(pretrained=True)
        output = net(data)
        return output.argmax()

```

<!--
TODO: update python code to match MMSGluonImperativeService
TODO: change reference to example to be specific for this section - code shouldn't cover all of the examples in one file.

To see the full custom service for a pre-trained Gluon model in action, refer to the `__init__` section of the `MMSGluonImperativeService` in [gluon_alexnet.py](gluon_alexnet.py).
-->
<!--
Note: commenting out mention of manifest as it is an internal logic notion

Since this is a pretrained model, we wouldn't need to provide `Parameters` and `Symbols` through `MANIFEST.json`.
Remove the `Parameters` and `Symbols` from the `MANIFEST.json` file and create a model-archive.
-->

<!--
Note: There was no info on actually exporting the model so how can you test?

Refer [How to test your models](#how-to-test-your-gluon-models), to test your Gluon model.
-->

## Load and serve pure Gluon imperative models

To load an imperative model for use with MMS, you must activate the network in a MMS custom service. Once activated, MMS can load the pre-trained parameters and start serving the imperative model. You also need to handle pre-processing and post-processing of the image input, but you can take advantage of MMS's default vision service for most of this, and leverage the post-processing code samples for the rest.

<!-- I think that this could be massively simplified. Why not just invoke the model, run once to activate it, then show the inference step where you pass the image to the net?
-->


<!--
TODO: Create a separate section on pre-processing for these vision models, and then incorporate that code widget here. A helper file would work, or adding another model service file that already has it would be even better. Then we can just extend that here.
-->

This custom service template handles pre and post-processing of images.
<!--
TODO: modify this template for use with Gluon directly, so vision and model loading is already there. postprocess and inference need testing
-->

```python
from mms.model_service.mxnet_model_service import MXNetBaseService
from mms.utils.mxnet import image, ndarray
import numpy as np
import mxnet as mx

class MXNetVisionService(MXNetBaseService):
    """MXNetVisionService defines a fundamental service for image classification task.
    In preprocess, input image buffer is read to NDArray and resized respect to input
    shape in signature.
    In post process, top-5 labels are returned.
    """
    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = image.read(img)
            img_arr = mx.image.resize_short(img_arr, 256)
            img_arr, _ = mx.image.center_crop(img_arr, (w, h))
            img_arr = mx.image.color_normalize(img_arr.astype(np.float32)/255,
                                     mean=mx.nd.array([0.485, 0.456, 0.406]),
                                     std=mx.nd.array([0.229, 0.224, 0.225]))
            #img_arr = image.resize(img_arr, w, h)
            img_arr = image.transform_shape(img_arr)

            img_list.append(img_arr)
        return img_list

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability': float(data[0, int(i.asscalar())].asscalar())}
                for i in idx]

```

<!-- Start with a very simple example that extends the vision service that has proper pre and post processing -->

Extend the custom service to include loading a pre-trained model and running inference.

```python
from mxnet.gluon.model_zoo import vision

class ImperativeAlexnetService(MXNetVisionService):

    def inference(self, data):
        net = vision.alexnet(pretrained=True)
        output = net(data)
        return output.argmax()

```

<!--
Commenting out manifest talk.
For this model, we would need `Parameters` file but we would need `Symbols` defined in `MANIFEST.json`.
-->

With this custom service for a pre-trained imperative model, you can create a model archive that loads the Gluon model on the fly. You will still need to include the other standard artifacts: a [synset.txt](), [signature.json]().


<!--
TODO: create alexnet-gluon-imperative.py - which is basically what was just discussed above
TODO: provide the other artifacts in each example subfolder or have the user copy them as they run each example
NOTE: the command below assumes that the required params file is no longer required
TODO: update other docs if model-path is no longer required
such as https://github.com/awslabs/mxnet-model-server/blob/master/docs/export_examples.md#export-example
-->

```bash
cp synset.txt alexnet-gluon-imperative
cp signature.json alexnet-gluon-imperative
cd alexnet-gluon-imperative
mxnet-model-export --model-name alexnet-gluon-imperative  --model-path . --service alexnet-gluon-imperative.py
```

Now you can start the MMS with the newly create model archive.

```bash
mxnet-model-server --models alex-imp=alexnet-gluon-imperative.model
```

Test the model by running requests from another terminal, as follows:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alex-imp/predict -F "data=@kitten.jpg"
```

The output should be close to the following:

```json
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```


## Load and serve a custom imperative model

Now that you have created a custom service for an imperative AlexNet pre-trained model, you can see how to create a custom model starting with AlexNet as your template, then serve it with MMS. In this scenario, you are providing the full network definition, loading the pre-trained parameters into it, then serving the model. This scenario would also allow you to use the standard AlexNet model, but load specific checkpoints from a previous training run, or load a training run that was trained with a different dataset. Of course, you could modify the network itself, such as adding or removing layers, or expanding or contracting connections.

In the example code that follows, you will run a basic AlexNet network without modifications, and load a provided checkpoint of pre-trained parameters.

```python
import mxnet
from mxnet import gluon
from mxnet.gluon import nn

class GluonImperativeAlexnet():
    """
    Fully imperative gluon Alexnet model
    """
    def __init__(self, classes=1000, **kwargs):
        super(GluonImperativeAlexnet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.Sequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(classes)

    def forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

class ImperativeAlexnetService():
    # this should be the same as what was already covered
    def inference(self, data):
        net = GluonImperativeAlexnet()
        net.load_params(self.param_filename, ctx=self.ctx)
        output = net(data)
        return output.argmax()
# full code snippit - no ....
# include code that loads the params

```

Verify you have the required artifacts in the [alexnet-gluon-imperative-custom](alexnet-gluon-imperative-custom) folder.
<!--
TODO: provide a real link to the params (OR)
TODO: create folder and provide artifacts
-->
* alexnet-gluon-imperative-custom.py
* alexnet-gluon-imperative-custom.params
* signature.json (using the same files as the previous example)
* synst.txt (using the same files as the previous example)

Next, you will export the MMS model archive.

```bash
cd alexnet-gluon-imperative-custom
mxnet-model-export --model-name alexnet-gluon-imperative-custom  --model-path . --service alexnet-gluon-imperative-custom.py
```

Now you can start the MMS with the newly create model archive.

```bash
mxnet-model-server --models alex-imp-cust=alexnet-gluon-imperative-custom.model
```

Test the model by running requests from another terminal, as follows:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alex-imp-cust/predict -F "data=@kitten.jpg"
```


## Load and serve a hybridized Gluon model

Similar to above sections, let's consider `gluon_alexnet.py` in `mxnet-model-server/examples/gluon_alexnet` folder.
We first convert the model to a `gluon` hybrid block. For additional background on using `HybridBlocks` and the need to `hybridize` refer to this Gluon [hybridize](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html#) tutorial.


```python
import mxnet
from mxnet import gluon
from mxnet.gluon import nn
class HybridAlexNet(gluon.HybridBlock):
    """
    Hybrid Block gluon model
    """
    def __init__(self, classes=1000, **kwargs):
        super(HybridAlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(64, kernel_size=11, strides=4,
                                            padding=2, activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(192, kernel_size=5, padding=2,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Conv2D(384, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.Conv2D(256, kernel_size=3, padding=1,
                                            activation='relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                self.features.add(nn.Flatten())
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
                self.features.add(nn.Dense(4096, activation='relu'))
                self.features.add(nn.Dropout(0.5))
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

class AlexnetService():
# no dots - use full code
    def inference(self, data):
        net = HybridAlexNet()
        net.hybridize()
        net.load_params(self.param_filename, ctx=self.ctx)
        output = net(data)
        return output.argmax()
# no dots - use full code
```

Similar to imperative models, this model doesn't require `Symbols` as the call to `.hybridize()` compiles the neural net.
This would store the `mxnet.symbols` implicitly.

Create, serve, and test the model archive as you did in the last scenario.
<!--
TODO: provide artifacts for this example in alexnet-gluon-hybrid folder
-->

```bash
cd alexnet-gluon-hybrid
mxnet-model-export --model-name alexnet-gluon-hybrid  --model-path . --service alexnet-gluon-hybrid.py
mxnet-model-server --models alex-hyb=alexnet-gluon-hybrid.model
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alex-hyb/predict -F "data=@kitten.jpg"
```

## Conclusion

In this tutorial you learned how to serve Gluon models in three unique scenarios: a pre-trained imperative model directly from the model zoo, a custom imperative model, and a hybrid model. For further examples of customizing gluon models, try the Gluon tutorial for [Transferring knowledge through fine-tuning](http://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html). For an advanced custom service example, try the MMS [SSD example](https://github.com/awslabs/mxnet-model-server/tree/master/examples/ssd).
