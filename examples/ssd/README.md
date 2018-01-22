# Single Shot Multi Object Detection Inference Service

In this example, we show how to use a pre-trained Single Shot Multi Object Detection (SSD) MXNet model for performing real time inference using MMS

The pre-trained model is trained on the [Pascal VOC 2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) The network is a SSD model built on Resnet50 as base network to extract image features. The model is trained to detect the following entities (classes): ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']. For more details about the model, you can refer [here](https://github.com/apache/incubator-mxnet/tree/master/example/ssd).

The inference service would return the response in the format - '[(object_class, xmin, ymin, xmax, ymax)]. Where, xmin, ymin, xmax and ymax are the bounding box coordinates of the detected object.

# Objective

1. Demonstrate how to export and use a pre-trained MXNet model in MMS
2. Demonstrate how to extend MMS with custom pre-processing and post-processing

## Step 1 - Download the pre-trained SSD Model

You will need the model files to use for the export. Check this example's directory in case they're already downloaded. Otherwise, you can `curl` the files or download them via your browser:

```bash
 curl -O https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json
 curl -O https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params
```

Alternatively, use these links to download the Symbol and Params files via your browser:
1. <a href="https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json" download>resnet50_ssd_model-symbol.json</a>
2. <a href="https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params" download>resnet50_ssd_model-0000.params</a>

**Note** params file is around 125 MB.

## Step 2 - Prepare the signature file

Define Input and Output name, type and shape in `signature.json` file. The signature for this example looks like below:

```
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [1, 3, 512, 512]
    }
  ],
  "input_type": "image/jpeg",
  "outputs": [
    {
      "data_name": "detection_output",
      "data_shape": [1, 6132, 6]
    }
  ],
  "output_type": "application/json"
}
```

In the pre-trained model, input name is 'data' and shape is '(1,3,512,512)'. Where, the expected input is a color image (3 channels - RGB) of shape 512*512. We also specify `image/jpeg` as the expected input type since we want the service to support handling of binary JPEG images, to make it easier for clients to use the service. With this setup, MMS will take care of converting binary images to tensor NDArray used by MXNet.

Similarly, in the pre-trained model, output name is 'detection_output' with shape '(1,6132,6)' from the last layer of the network. Here, 6132 is number of detections the network outputs. 6 is the dimension of each detection - (class_id, score, x1, y1, x2, y2).

See [signature.json](signature.json) file for more details.

*Note:* Typically, if you train your own model, you define the Input and Output Layer name and shape when defining the Neural Network. If you are using a pre-trained MXNet model, to get these Input and Output name and dimensions, you can load the Model and extract the Input and Output layer details. Unfortunately, there are no APIs or easy way to extract the Input shape. Example code below:

```python
>>> import mxnet as mx
>>> load_symbol, args, auxs = mx.model.load_checkpoint("resnet50_ssd_model", 000)
>>> mod = mx.mod.Module(load_symbol, label_names=None, context=mx.cpu())
>>> mod.data_names
['data']
>>> mod.bind(data_shapes=[('data', (1, 3, 512, 512))])
>>> mod.set_params(args, auxs)
>>> print(mod.data_names)
>>> print(mod.data_shapes)
>>> print(mod.output_names)
>>> print(mod.output_shapes)
['data']
[DataDesc[data,(1, 3, 512, 512),<class 'numpy.float32'>,NCHW]]
['detection_output']
[('detection_output', (1, 6132, 6))]
```

*Note:* The network generates 6132 detections because we use MXNet's [MultiboxPrior](https://mxnet.incubator.apache.org/api/python/symbol.html#mxnet.contrib.symbol.MultiBoxPrior) to generate the anchor boxes with the following 'Ratios and 'Sizes':
 ```python
    sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
    ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3], \
            [1,2,.5], [1,2,.5]]
 ```
To understand more about the MultiboxPrior, anchor boxes, sizes and ratios, please read [this tutorial](http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)

## Step 3 - Prepare synset.txt with list of class names

[synset.txt](synset.txt) is where we define list of all classes detected by the model. The pre-trained SSD model used in the example is trained to detect 20 classes - person, car, aeroplane, bicycle and more. See synset.txt file for list of all classes.

The list of classes in synset.txt will be loaded by MMS as list of labels in inference logic.

## Step 4 - Extend MMS with custom preprocess and postprocess

MMS allows users to extend the base service functionality and add more custom initialization, pre-processing, inference and post-processing.

In this example, we extend `MXNetVisionService`, provided by MMS for vision inference use-cases, and reuse its input image preprocess functionality to resize and transform the image shape. We only add custom pre-processing and post-processing steps. See [ssd_service.py](ssd_service.py) for more details on how to extend the base service and add custom pre-processing and post-processing.

## Step 5 - Export the model with mxnet-model-export CLI utility

In this step, we package the following:
1. pre-trained MXNet Model we downloaded in Step 1.
2. '[signature.json](signature.json)' file we prepared in step 2.
3. '[synset.txt](synset.txt)' file we prepared in step 3.

as one single 'resnet50_ssd_model.model' file. We use `mxnet-model-export` command line utility (CLI) provided by MMS.

This tool prepares a .model file that will be provided as input to start the inference server.

```bash
mxnet-model-export --model-name resnet50_ssd_model --model-path .
```

## Step 6 - Start the Inference Service

Start the inference service by providing the 'resnet50_ssd_model.model' file we exported in Step 5. We also provide the custom extended service module, 'ssd_service.py'.

MMS then extracts the resources (signature, synset, model symbol and params) we have packaged into .model file and uses the extended custom service, to start the inference server.

By default, the server is started on the localhost at port 8080. You can optionally specify different host and/or port to start the server using the command line options - '--host ', '--port '.

```bash
 mxnet-model-server --models SSD=resnet50_ssd_model.model --service ssd_service.py
```

You will see the output similar to the output listed below.

```
I1025 08:22:13 5986 /usr/local/lib/python2.7/dist-packages/mms/mxnet_model_server.py:__init__:75] Initialized model serving.
I1025 08:22:14 5986 /usr/local/lib/python2.7/dist-packages/mms/serving_frontend.py:add_endpoint:176] Adding endpoint: SSD_predict to Flask
I1025 08:22:14 5986 /usr/local/lib/python2.7/dist-packages/mms/serving_frontend.py:add_endpoint:176] Adding endpoint: ping to Flask
I1025 08:22:14 5986 /usr/local/lib/python2.7/dist-packages/mms/serving_frontend.py:add_endpoint:176] Adding endpoint: apiDescription to Flask
I1025 08:22:14 5986 /usr/local/lib/python2.7/dist-packages/mms/mxnet_model_server.py:start_model_serving:88] Service started at 127.0.0.1:8080
```
Awesome! we have successfully exported a pre-trained MXNet model, extended MMS with custom preprocess/postprocess and started a inference service.

`Note:` In this example, MMS loads the .model file from the local file system. However, you can also store the archive (.model file) over a network-accessible storage such as AWS S3, and use a URL such as http:// or s3:// to indicate the model location. MMS is capable of loading the model archive over such URLs as well.

## Step 7 - Test sample inference

Let us try the inference server we just started. Open another terminal on the same host. Download a sample image, or try any jpeg image that contains the one or more of the object classes mentioned earlier: 'aeroplane', 'bicycle', 'bird', 'boat', etc...

You can also use this image of three dogs on a beach.
![3 dogs on beach](3dogs.jpg)

Use curl to make a prediction call by passing the downloaded image as input to the prediction request.

```bash
curl -X POST http://127.0.0.1:8080/ssd/predict -F "data=@3dogs.jpg"
```

You can expect the response similar to below. The output format is `[(object_class, xmin, ymin, xmax, ymax)]`.
Where, xmin, ymin, xmax and ymax are the bounding box coordinates of the detected object.

```
{
  "prediction": [
    [
      "dog",
      399,
      128,
      570,
      290
    ],
    [
      "dog",
      278,
      196,
      417,
      286
    ],
    [
      "cow",
      205,
      116,
      297,
      272
    ]
  ]
}
```

A consumer application can use this response to identify the objects in the input image and their bounding boxes.

For better visualization on the input and how we can use the inference output, see below:

Input Image

![Street Input Image](dogs-before.jpg)

Output Image

![Street Output Image](dogs-after.jpg)


# References
1. Adapted code and pre-trained model from - https://github.com/apache/incubator-mxnet/tree/master/example/ssd
2. Learn more about SSD in this tutorial - http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html
