# Exporting Models for Use with MMS

## Contents of this Document
* [Overview](#overview)
* [Technical Highlights on Creating a Model Archive](#technical-highlights-on-creating-a-model-archive)
* [MMS Export CLI](#mms-export-command-line-interface)
* [Artifact Details](#artifact-details)
    * [Model Archive Overview](#model-archive-overview)
    * [Model Definition](#model-definition)
    * [Model Parameters and Weights](#model-parameters-and-weights)
    * [Signature](#signature)
    * [Service](#service)
    * [Labels](#labels)

## Other Relevant Documents
* [Export Examples](export_examples.md)
* [Export an ONNX Model](export_from_onnx.md)
* [Exported MMS Model File Tour](export_model_file_tour.md)

## Overview

A key feature of MMS is the ability to export all model artifacts into a single model archive file. It is a separate command line interface (CLI), `mxnet-model-export`, that can take model checkpoints and package them into a `.model` file that can then be redistributed and served by anyone using MMS. It takes in the following model artifacts: a model composed of one or more files, the description of the models' inputs and outputs in the form of a signature file, a service file describing how to handle inputs and outputs, and other optional assets that may be required to serve the model. The CLI creates a `.model` file that MMS's server CLI uses to serve the models.

**Important**: Make sure you try the [Quick Start: Export a Model](../README.md#export-a-model) tutorial for a short example of using `mxnet-model-export`.


## Technical Highlights on Creating a Model Archive

To export a model in MMS, you will need:

1. Model file(s)
    * **MXNet**: a `model-symbol.json` file, which describes the neural network, and much larger `model-0000.params` file containing the parameters and their weights
    * **ONNX**: an `.onnx` file (rename `.pb` or `.pb2` files to `.onnx`)


2. For MMS to understand your model, you must provide a `signature.json` file, which describes the model's inputs and outputs.

3. Most models will require the inputs to go through some pre-processing, and your application will likely benefit from post-processing of the inference results. These functions go into `custom-service.py`. In the cases of image classification, MMS provides one for you.

4. You also have *the option* of providing assets that assist with the inference process. These can be labels for the inference outputs (e.g. synset.txt), key/value vocabulary pairs used in an LSTM model, and so forth.

This gives you the first two assets by providing those files for you to download, or that you've acquired the trained models from a [model zoo](model_zoo.md). In the export examples we provide the latter two files that you would create on your own based on the model you're trying to serve. Don't worry if that sounds ominous; creating those last two files is easy. More details on this can be found in later the **Required Assets** section.

The files in the `model-example.model` file are human-readable in a text editor, with the exception of the `.params` file: this file is binary, and is usually quite large.


## MMS Export Command Line Interface

Now let's cover the details on using the CLI tool: `mxnet-model-export`.

Example usage with the squeezenet_v1.1 model you may have downloaded or exported in the [main README's](../README.md) examples:

```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path models/squeezenet_v1.1
```

### Arguments

```
$ mxnet-model-export -h
usage: mxnet-model-export [-h] --model-name MODEL_NAME --model-path MODEL_PATH
                          [--service-file-path SERVICE_FILE_PATH]

MXNet Model Export

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Exported model name. Exported file will be named as
                        model-name.model and saved in current working
                        directory.
  --model-path MODEL_PATH
                        Path to the folder containing model related files.
                        Signature file is required.
  --service-file-path SERVICE_FILE_PATH
                        Service file path to handle custom MMS inference
                        logic. if not provided, this tool will package
                        MXNetBaseService if input in signature.json is
                        application/json or MXNetVisionService if input is
                        image/jpeg
```

**Required Arguments**

1. model-name: required, prefix of exported model archive file.
2. model-path: required, directory which contains files to be packed into exported archive.


## Artifact Details

### Model Archive Overview

Model archives have the following artifacts:

```
<Model Name>-symbol.json
<Model Name>-<Epoch>.params
signature.json
<Service File>.py
MANIFEST.json
{asset-x.txt, asset-y.txt, ...}
```

### Model Definition
```
<Model Name>-symbol.json
```
This is the model's definition in JSON format. You can name it whatever you want, using a consistent prefix. The pattern expected is `my-awesome-network-symbol.json` or `mnist-symbol.json` so that when you use `mxnet-model-export` you're passing in the prefix and it'll look for prefix-symbol.json. You can generate this file in a variety of ways, but the easiest for MXNet is to use the `.export` feature or the `mms.export_model` method described later.

### Model Parameters and Weights
```
<Model Name>-<Epoch>.params
```
This is the model's hyper-parameters and weights. It will be created when you use MXNet's `.export` feature or the `mms.export_model` method described later.

### Signature
```
signature.json
```

1. **input**: Contains MXNet model input names and input shapes. It is a list contains { data_name : name, data_shape : shape } maps. Client side inputs should have the same order with the input order defined here. **Note**: the default `data_name` for MXNet is `data`, and for ONNX it is `input_0`.
1. **input_type**: Defines the MIME content type for client side inputs. Currently all inputs must have the same content type. Only two MIME types are currently supported: "image/jpeg" and "application/json".
1. **output**: Similar to input, it contains MXNet model output names and output shapes.
1. **output_type**: Similar to input_type. Currently all outputs must have the same content type. Only two MIME types are currently supported: "image/jpeg" and "application/json".

Using the squeezenet_v1.1 example, you can view the [signature.json](https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/signature.json) file in the folder that was extracted once you dowloaded and served the model for the first time. The input is an image with 3 color channels and size 224 by 224. The output is named 'softmax' with length 1000 (one for every class that the model can recognize).

```json
{
 "inputs": [
   {
     "data_name": "data",
     "data_shape": [0, 3, 224, 224]
   }
 ],
 "input_type": "image/jpeg",
 "outputs": [
   {
     "data_name": "softmax",
     "data_shape": [0, 1000]
   }
 ],
 "output_type": "application/json"
}
```

The `data_shape` is a list of integers. It should contain batch size as the first dimension as in NCHW. Also, 0 is a placeholder for MXNet shape and means any value is valid. Batch size should be set as 0.

Note that the signature output isn't necessarily the final output that is returned to the user. This is directed by your model service class, which defaults to the [MXNet Model Service](../mms/model_service/mxnet_model_service.py). In this signature.json example your output_type is json and a shape of 1000 results, but API's response is actually limited to the top 5 results via the vision service. In the [object detection example](../examples/ssd/README.md), it is using a [signature.json](../examples/ssd/signature.json) that has `"data_shape": [1, 6132, 6]` and has a [custom service](../examples/ssd/ssd_service.py) to modify the output to the API in such a way as to identify the objects AND their locations, e.g. `[(person, 555, 175, 581, 242), (dog, 306, 446, 468, 530)]`.

### Service
```
<Service File>.py
```

This is a stubbed out version of the class extension you would use to override the `SingleNodeService` as seen in [mxnet_model_service.py](https://github.com/awslabs/mxnet-model-server/blob/manifest_docs/mms/model_service/mxnet_model_service.py). You may instead want to override the `MXNetBaseService` as seen in [mxnet_vision_service.py](https://github.com/awslabs/mxnet-model-server/blob/manifest_docs/mms/model_service/mxnet_vision_service.py)

```python
class MXNetBaseService(SingleNodeService):
  def __init__(self, path, synset=None, ctx=mx.cpu()):

  def _inference(self, data):

  def _preprocess(self, data):

  def _postprocess(self, data, method='predict'):
```

Further details and specifications are found on the [custom service](custom_service.md) page.

### Labels
```
synset.txt
```
This optional text file is for classification labels. Simply put, if it were for MNIST, it would be 0 through 9 where each number is on its own line. For a more complex example take a look at the [synset for Imagenet-11k](https://github.com/tornadomeet/ResNet/blob/master/predict/synset.txt).


If `synset.txt` is included in exported archive file and each line represents a category, `MXNetBaseModel` will load this file and create `labels` attribute automatically. If this file is named differently or has a different format, you need to override `__init__` method and manually load it.

### Dependent/Nested Models

In some cases, there is a need to nest multiple models in the same Model Archive. To package multiple models in the same archive, MMS export tool requires the following structure:

```bash
/Model-folder/
  main-model-symbol-file
  main-model-parameter-file
  main-model-signature-file
  dependency-model-1-sub-folder/
    dependency-model-1-symbol-file
    dependency-model-1-parameter-file
  dependency-model-2-sub-folder/
      dependency-model-2-symbol-file
      dependency-model-2-parameter-file
```

The dependency models can also be part of a single sub folder

```bash
/Model-folder/
  main-model-symbol-file
  main-model-parameter-file
  main-model-signature-file
  dependency-model-sub-folder/
    dependency-model-1-symbol-file
    dependency-model-1-parameter-file
    dependency-model-2-symbol-file
    dependency-model-2-parameter-file
```

It is recommended that all the custom code for the model is added into the custom service file [custom service](custom_service.md). 
