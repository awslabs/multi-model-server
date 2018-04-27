## Loading and serving Gluon models on MXNet Model Server (MMS)

We are glad to announce that MXNet Model Server now supports loading and serving Imperative and Hybrid Gluon models. 
This is a short tutorial on how to write a custom Gluon model and serve it on MMS. If you are using Gluon for the first
time, you could refer this [60 min crash-course](http://gluon-crash-course.mxnet.io/ndarray.html) on gluon to get a 
better understanding.

In this tutorial, we will cover the following:
1. [How to serve Gluon models](#how-to-serve-gluon-models)
2. [Load and serve pretrained models](#load-and-serve-a-pretrained-gluon-model)
3. [Load and serve pure Gluon imperative models](#load-and-serve-pure-gluon-imperative-models)
4. [Load and serve a hybrid'ized gluon model](#load-and-serve-a-hybrid'ized-gluon-model)
5. [How to test your gluon models](#how-to-test-your-gluon-models)
6. [How to create a model archive](#how-to-create-a-model-archive)

**NOTE: If you do not use `Symbols` parameter in `MANIFEST.json`, MMS infers this as a imperative gluon model
(or network defined by run). In other words DO-NOT-USE `Symbols` parameter in `MANIFEST.json` when defining `Gluon` 
models.** 

## How to serve Gluon models
There are three ways of defining our Gluon nerual network/model.
1. Load and serve a pretrained model.
2. Load and serve custom imperative model, which implies that we don't generate/use symbols.
3. Load and serve custom Hybrid models. More on why we need `HybridBlocks` and the need to `hybridize` is explained well
in this [Gluon Document](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html#)

For this tutorial, we use [Alexnet](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) 
to describe the above three scenarios. The example code and supporting files to create a model archive are 
in `mxnet-model-server/examples/gluon_alexnet/*`.  

## Load and serve a pretrained gluon model
Gluon already provides a good range of pretrained models through `mxnet.gluon.model_zoo` and they are rapidly adding 
more. These models are available after simply installing `mxnet`
```bash
# For CPU instances
pip install mxnet-mkl
```

This is the simplest of the three methods to define. Here we don't need `params` or `symbols` to be given as inputs. 
We can access these models from the `service_code` as follows
```text
import mxnet

class AlexnetService():
...
    def inference(self, data):
        net = mxnet.gluon.model_zoo.vision.alexnet(pretrained=True)
        output = net(data)
        return output.argmax()
...        
```

Since this is a pretrained model, we wouldn't need to provide `Parameters` and `Symbols` through `MANIFEST.json`.

To see this in action, refer the `__init__` in the `MMSImperativeService`  in  `gluon_alexnet.py`. 
 
Remove the `Parameters` and `Symbols` from the `MANIFEST.json` file and create a model-archive.
Refer [How to test your models](#how-to-test-your-gluon-models), to test your gluon model.

## Load and serve pure Gluon imperative models
As above, let's consider `gluon_mxnet.py` in this `mxnet-model-server/examples/gluon_alexnet` folder.
To define a imperative neral-net, we would need to define the complete network. We then load the parameters that we 
obtained from training this model and start serving the model.

```text
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

class AlexnetService():
...
    def inference(self, data):
        net = GluonImperativeAlexnet()
        net.load_params(self.param_filename, ctx=self.ctx)
        output = net(data)
        return output.argmax()
...
```

As in the above section, we could test this network by creating a model archive with this service code and running it 
with MMS. For this model, we would nee `Parameters` file but we would need `Symbols` defined in `MANIFEST.json`
Refer [How to test your models](#how-to-test-your-gluon-models), to test your gluon model.

## Load and serve a hybrid'ized gluon model
Similar to above sections, let's consider `gluon_alexnet.py` in `mxnet-model-server/examples/gluon_alexnet` folder.
We first convert the model to a `gluon` hybrid block.
```text
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
...
    def inference(self, data):
        net = HybridAlexNet()
        net.hybridize()
        net.load_params(self.param_filename, ctx=self.ctx)
        output = net(data)
        return output.argmax()
...
```
To understand how `.hybridize()` is useful and its internals, refer to 
[Gluon's tutorial on Hybridize](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html).

Similar to Imperative models, this model doesn't require `Symbols` as the call to `.hybridize()` compiles the nerual-net. 
This would store the `mxnet.symbols` implicitly. 

To test this model, refer to the below section on [how to test your gluon models](#how-to-test-your-gluon-models).
## How to test your gluon models
To test the gluon models, we need to do the following
1. Make sure you have the latest version of `mxnet-model-server`
```bash
pip install mxnet-model-server
``` 
2. Create a model archive file. Refer [How to create a model archive](#how-to-create-a-model-archive) 
for that.
3. Start the model-server with the newly create model archive.
```bash
mxnet-model-server --models gluon=gluon.model
```
4. Test this service by running requests from another terminal, as follows:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/gluon/predict -F "data=@kitten.jpg"
```

The expected output is
```text
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```

## How to create a model archive
1. Go to `mxnet-model-server/examples/gluon_alexnet` folder.
```bash
git clone https://github.com/awslabs/mxnet-model-server.git
```

```bash
# Go to the example
cd mxnet-model-server/examples/gluon_alexnet
```

```bash
# Download the alexnet-params file. This is used when using non-pretrained gluon model.
wget https://s3.amazonaws.com/gluon-mms-model-files/alexnet.params
```

2. Copy relavant files into a model archive
```bash
# Create model archive
zip gluon.model *
```