# Loading and serving Gluon models on MXNet Model Server (MMS)

MXNet Model Server (MMS) supports loading and serving MXNet Imperative and Hybrid Gluon models.
This is a short tutorial on how to write a custom Gluon model, and then serve it with MMS.

This tutorial covers the following:
1. [Prerequisites](#prerequisites)
2. [Serve a Gluon model](#load-and-serve-a-gluon-model)
  * [Load and serve a pre-trained Gluon model](#load-and-serve-a-pre-trained-gluon-model)
  * [Load and serve a custom Gluon model](#load-and-serve-a-custom-gluon-imperative-model)
  * [Load and serve a custom hybrid Gluon model](#load-and-serve-a-hybrid-gluon-model)
3. [Conclusion](#conclusion)

## Prerequisites

* **Basic Gluon knowledge**. If you are using Gluon for the first
time, but are familiar with creating a neural network with MXNet or another framework, you may refer this 10 min Gluon crash-course: [Predict with a pre-trained model](http://gluon-crash-course.mxnet.io/predict.html).
* **Gluon naming**. Fine-tuning pre-trained Gluon models requires some understanding of how the naming conventions work. Take a look at the [Naming of Gluon Parameter and Blocks](https://mxnet.incubator.apache.org/tutorials/gluon/naming.html) tutorial for more information.
* **Basic MMS knowledge**. If you are using MMS for the first time, you should take advantage of the [MMS QuickStart tutorial](https://github.com/awslabs/mxnet-model-server#quick-start).
* **MMS Custom Service knowledge.** Review the [Defining a Custom Service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md) documentation to understand how MMS implements custom pre and post-processing for models.
* **MMS installed**. If you haven't already, [install MMS with pip](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-with-pip) or [install MMS from source](https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md#install-mms-from-source-code). Either installation will also install MXNet.


Refer to the [MXNet model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html) documentation for examples of accessing other models.
## Load and serve a Gluon model

There are three scenarios for serving a Gluon model with MMS:

1. Load and serve a pre-trained Gluon model.
2. Load and serve a custom imperative Gluon model.
3. Load and serve a custom hybrid Gluon model.

To learn more about the differences between gluon and hybrid gluon models refer to [the following document](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html)

### Load and serve a pre-trained Gluon model

Loading and serving a pre-trained Gluon model is the simplest of the three scenarios. These models don't require you to provide `symbols` and `params` files.

While it is easy to access a model with a couple of lines of code, with MMS you will want to use a [MMS custom service](https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md) code pattern as follows:

```python
class MMSPretrainedAlexnet(MXNetVisionService):
    """
    Pretrained alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSPretrainedAlexnet, self).__init__(model_name, model_dir, manifest, gpu)

        self.net = mxnet.gluon.model_zoo.vision.alexnet(pretrained=True)

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = mxnet.img.imdecode(img)
            img_arr = mxnet.image.imresize(img_arr, w, h)
            img_arr = img_arr.astype(np.float32)
            img_arr /= 255
            img_arr = mxnet.image.color_normalize(img_arr,
                                                  mean=mxnet.nd.array([0.485, 0.456, 0.406]),
                                                  std=mxnet.nd.array([0.229, 0.224, 0.225]))
            img_arr = mxnet.nd.transpose(img_arr, (2, 0, 1))
            img_arr = img_arr.expand_dims(axis=0)
            img_list.append(img_arr)
        return img_list

    def _inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data[0])
        return output.softmax()

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability': float(data[0, int(i.asscalar())].asscalar())}
                for i in idx]
```

For an actual code implementation, refer to the custom-service code which uses the [pre-trained Alexnet](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_pretrained_alexnet.py)

### Serve pre-trained model with MMS
To serve pre-trained models with MMS we would need to create an model archive file. Follow the below steps to get the example custom service
loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_pretrained_alexnet.py) 
and other required artifacts to this folder
```bash
cp gluon_pretrained_alexnet.py synset.txt signature.json /tmp/models
```
3. Run the model-export tool on this folder.
```bash
mxnet-model-export --model-name="pretrained-alexnet" --model-path="/tmp/models" --service-file-path="/tmp/models/gluon_pretrained_alexnet.py" --model-type="imperative"
```
This creates a model-archive file `pretrained-alexnet.model`.

4. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --models alexnet=pretrained-alexnet.model
```
5. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

## Load and serve a custom Gluon imperative model

To load an imperative model for use with MMS, you must activate the network in a MMS custom service. Once activated, MMS 
can load the pre-trained parameters and start serving the imperative model. You also need to handle pre-processing and 
post-processing of the image input.

We created a custom imperative model using Gluon. Refer to 
[custom service code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/examples/gluon_alexnet/gluon_alexnet.py)
The network definition, which is defined in the example, is as follows

```python
class ImperativeAlexNet(gluon.Block):
    """
    Fully imperative gluon Alexnet model
    """
    def __init__(self, classes=1000, **kwargs):
        super(ImperativeAlexNet, self).__init__(**kwargs)
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

    def forward(self, x):
	x = self.features(x)
        x = self.output(x)
        return x
```

The pre-process, inference and post-process steps are similar to the service code that we saw in the [above section](#load-and-serve-a-pre-trained-gluon-model).
```python
class MMSImperativeService(MXNetVisionService):
    """
    Gluon alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSImperativeService, self).__init__(model_name, model_dir, manifest, gpu)
        self.net = ImperativeAlexNet()
        if self.param_filename:
            self.net.load_params(os.path.join(model_dir, self.param_filename), ctx=self.ctx)

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = mxnet.img.imdecode(img)
            img_arr = mxnet.image.imresize(img_arr, w, h)
            img_arr = img_arr.astype(np.float32)
            img_arr /= 255
            img_arr = mxnet.image.color_normalize(img_arr,
                                                  mean=mxnet.nd.array([0.485, 0.456, 0.406]),
                                                  std=mxnet.nd.array([0.229, 0.224, 0.225]))
            img_arr = mxnet.nd.transpose(img_arr, (2, 0, 1))
            img_arr = img_arr.expand_dims(axis=0)
            img_list.append(img_arr)
        return img_list

    def _inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data[0])
        return output.softmax()

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability': float(data[0, int(i.asscalar())].asscalar())}
                for i in idx]

``` 
 
### Test your imperative Gluon model service
To serve imperative Gluon models with MMS we would need to create an model archive file. 
Follow the below steps to get the example custom service loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_imperative_alexnet.py) 
and other required artifacts to this folder
```bash
cp gluon_imperative_alexnet.py synset.txt signature.json /tmp/models
```
3. Download/copy the parameters to this `/tmp/models` directory. For this example, we have the parameters file in a S3 bucket.
```bash
wget https://s3.amazonaws.com/gluon-mms-model-files/alexnet.params
mv alexnet.params /tmp/models
```
4. Run the model-export tool on this folder.
```bash
mxnet-model-export --model-name="imperative-alexnet" --model-path="/tmp/models" --service-file-path="/tmp/models/gluon_imperative_alexnet.py" --model-type="imperative"
```
This creates a model-archive file `imperative-alexnet.model`.

5. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --models alexnet=imperative-alexnet.model
```
6. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

The output should be close to the following:

```json
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```

## Load and serve a hybrid Gluon model

To serve hybrid Gluon models with MMS, let's consider `gluon_imperative_alexnet.py` in `mxnet-model-server/examples/gluon_alexnet` folder.
We first convert the model to a `Gluon` hybrid block. 
For additional background on using `HybridBlocks` and the need to `hybridize` refer to this Gluon [hybridize](http://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html#) tutorial.

The above network, after this conversion, would look as follows: 
```python
class GluonHybridAlexNet(HybridBlock):
    """
    Hybrid Block gluon model
    """
    def __init__(self, classes=1000, **kwargs):
        super(GluonHybridAlexNet, self).__init__(**kwargs)
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
```
We could use the same custom service code as in the above section, 

```python
class MMSHybridService(MXNetVisionService):
    """
    Gluon alexnet Service
    """
    def __init__(self, model_name, model_dir, manifest, gpu=None):
        super(MMSHybridService, self).__init__(model_name, model_dir, manifest, gpu)
        self.net = GluonHybridAlexNet()
        if self.param_filename:
            self.net.load_params(os.path.join(model_dir, self.param_filename), ctx=self.ctx)
        self.net.hybridize()
            

    def _preprocess(self, data):
        img_list = []
        for idx, img in enumerate(data):
            input_shape = self.signature['inputs'][idx]['data_shape']
            # We are assuming input shape is NCHW
            [h, w] = input_shape[2:]
            img_arr = mxnet.img.imdecode(img)
            img_arr = mxnet.image.imresize(img_arr, w, h)
            img_arr = img_arr.astype(np.float32)
            img_arr /= 255
            img_arr = mxnet.image.color_normalize(img_arr,
                                                  mean=mxnet.nd.array([0.485, 0.456, 0.406]),
                                                  std=mxnet.nd.array([0.229, 0.224, 0.225]))
            img_arr = mxnet.nd.transpose(img_arr, (2, 0, 1))
            img_arr = img_arr.expand_dims(axis=0)
            img_list.append(img_arr)
        return img_list

    def _inference(self, data):
        # Call forward/hybrid_forward
        output = self.net(data[0])
        return output.softmax()

    def _postprocess(self, data):
        idx = data.topk(k=5)[0]
        return [{'class': (self.labels[int(i.asscalar())]).split()[1], 'probability': float(data[0, int(i.asscalar())].asscalar())}
                for i in idx]
```
Similar to imperative models, this model doesn't require `Symbols` as the call to `.hybridize()` compiles the neural net.
This would store the `symbols` implicitly.

### Test your hybrid Gluon model service
To serve Hybrid Gluon models with MMS we would need to create an model archive file. 
Follow the below steps to get the example custom service loaded and served with MMS.

1. Create a `models` directory
```bash
mkdir /tmp/models
```
2. Copy the [example code](https://github.com/awslabs/mxnet-model-server/examples/gluon_alexnet/gluon_imperative_alexnet.py) 
and other required artifacts to this folder
```bash
cp gluon_hybrid_alexnet.py synset.txt signature.json /tmp/models
```
3. Download/copy the parameters to this `/tmp/models` directory. For this example, we have the parameters file in a S3 bucket.
```bash
wget https://s3.amazonaws.com/gluon-mms-model-files/alexnet.params
mv alexnet.params /tmp/models
```
4. Run the model-export tool on this folder.
```bash
mxnet-model-export --model-name="hybrid-alexnet" --model-path="/tmp/models" --service-file-path="/tmp/models/gluon_hybrid_alexnet.py" --model-type="imperative"
```
This creates a model-archive file `hybrid-alexnet.model`.

5. You could run the server with this model file to serve the pre-trained alexnet.
```bash
mxnet-model-server --models alexnet=hybrid-alexnet.model
```
6. Test your service
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
curl -X POST http://127.0.0.1:8080/alexnet/predict -F "data=@kitten.jpg"
```

The output should be close to the following:

```json
{"prediction":[{"class":"lynx,","probability":0.9411474466323853},{"class":"leopard,","probability":0.016749195754528046},{"class":"tabby,","probability":0.012754007242619991},{"class":"Egyptian","probability":0.011728651821613312},{"class":"tiger","probability":0.008974711410701275}]}
```

## Conclusion

In this tutorial you learned how to serve Gluon models in three unique scenarios: a pre-trained imperative model directly from the model zoo, a custom imperative model, and a hybrid model. For further examples of customizing gluon models, try the Gluon tutorial for [Transferring knowledge through fine-tuning](http://gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html). For an advanced custom service example, try the MMS [SSD example](https://github.com/awslabs/mxnet-model-server/tree/master/examples/ssd).
