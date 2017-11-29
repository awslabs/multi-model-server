# Exporting Models for Use with MMS

A key feature of MMS is the ability to export all model artifacts into a single model archive file. It is a separate CLI that takes in model artifacts: network definitions in the form of a JSON file, the trained network weight in the form of a parameters file, the description of the models' inputs and outputs in the form of a signature JSON file, a service file describing how to handle inputs and outputs, an overall manifest file for metadata, and other optional assets that may be required to serve the model. The CLI creates a `.model` file that MMS's server CLI uses to serve the models.

## Technical Details of a Model Archive

To export a model in MMS, you will need:

1. A `model-symbol.json` file, which describes the neural network,

2. A much larger `model-0000.params` file containing the parameters and their weights

    * For the purpose of a quick example, we'll pretend that you've already saved a checkpoint which is numbered by those digits in the middle of `model-0000.params` filename.


3. For MMS to understand your model, you must provide a `signature.json` file, which describes the model's inputs and outputs.

4. Most models will require the inputs to go through some pre-processing, and your application will likely benefit from post-processing of the inference results. These functions go into `custom-service.py`.

5. The glue that holds these files together is the `MANIFEST.json` file. This describes each of the artifacts, the deep learning engine to use, and metadata about versions, the model archive author, and more.

6. You also have *the option* of providing assets in the `/assets` folder that assist with the inference process. These can be labels for the inference outputs, key/value vocabulary pairs used in an LSTM model, and so forth.

This gives you the first two assets by providing those files for you to download, or that you've acquired the trained models from a [model zoo](model_zoo.md). We'll also provide the latter two files that you would create on your own based on the model you're trying to serve. Don't worry if that sounds ominous; creating those last two files is easy. More details on this can be found in later the **Required Assets** section.

The files in the `model-example.model` file are human-readable in a text editor, with the exception of the `.params` file: this file is binary, and is usually quite large.

## Example Model File Exploration

In the quick start export example on the main [README](../README.md), we provide a zip file of all of the artifacts you need to run your first export. However, when you start using MMS and downloading and sharing models you will be using `.model` files, so here we'll start with one of those.

**Note**: A `.model` file is a zip file under the hood, so if you have trouble extracting it, change the extension to `.zip` first and then extract it. It might be worth assigning your favorite unzip program to the `.model` filetype.

**Download and extract a model file:**

* [SqueezeNet v1.1](https://s3.amazonaws.com/model-server/models/squeezenet_v1.1/squeezenet_v1.1.model) - 5 MB
* Or choose a model file from the [model zoo](model_zoo.md)

**Once the model archive has been extracted you can review the following files:**

* **Model Structure** (json file) - contains the description of the layers and overall structure of the neural network
  * Example: [squeezenet_v1.1-symbol.json](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-symbol.json) - the name, or prefix, here is "squeezenet_v1.1"
* **Model Params and Weights** (binary params file) - contains the parameters and the weights
  * Example: [squeezenet_v1.1-0000.params](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-0000.params) - again, the prefix is "squeezenet_v1.1"
* **Model Signature** (json file) - defines the inputs and outputs that MMS is expecting to hand-off to the API
  * Example: [signature.json](https://s3.amazonaws.com/model-server/models/model-example/signature.json) - in this case for squeezenet_v1, it expects images of 224x224 pixels and will output a tensor of 1,000 probabilities.
* **Custom Service** (py file) - customizes the inference request handling for both pre-processing and post-processing
  * Example: [custom-service.py](#) - in this case, it is a copy of the [MXNet vision service](https://github.com/awslabs/mxnet-model-server/blob/master/mms/model_service/mxnet_vision_service.py) which does standard image pre-processing to match the input required and limits the output results to the top 5 instead of the full 1,000.
* **MANIFEST** (json file) - contains metadata about the files in the model archive. Inspired by the [JAR](https://en.wikipedia.org/wiki/JAR_(file_format)) manifest.
  * Example: [MANIFEST.json](#)
* **assets** (folder) - folder containing auxiliary files that support model inference such as vocabularies, labels, etc. Will vary depending on the model.
  * Example:  [synset.txt](https://s3.amazonaws.com/model-server/models/model-example/synset.txt) - an *optional* list of labels (one per line) specific to a image recognition model, in this case based on the ImageNet dataset.
  * Example:  [vocab_dict.txt](https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt) - an *optional* list of word/index pairs specific to an LSTM model, in this case based on the PenTreeBank dataset.

## Export Example

Given the files downloaded in the exploration section above, you can use the `mxnet-model-export` CLI to generate a `.model` file that can be used with MMS.

To try this out, open your terminal and go to the folder you just extracted.

Using the zip file and its directory structure can help you keep things organized. In this next example we'll go into the `model-example` folder and run `mxnet-model-export`. We're going to tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets, which are in the current working directory, so we'll use `.` for the path.

```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path .
```

This will output `squeezenet_v1.1.model` in the current working directory. Try serving it with:

```bash
mxnet-model-server \
  --models squeezenet=squeezenet_v1.1.model
```

## Export Example with Customizations

To give you an idea of how you might download another's model, modify it, then serve it, let's try out a simple use case. The example we have been using will serve the SqueezeNet model, and upon inference requests it will return the top 5 results. Let's change the **custom service** so that it returns 10 results instead.

Open the `custom-service.py` file in your text editor.

Find the function for `_postprocess` and the line that says the following:

```python
return [ndarray.top_probability(d, self.labels, top=5) for d in data]
```

Change the `top=5` to `top=10`, then save the file. Run the export process again, and then serve the created model file. Then in a different terminal window, upload an image file the API, and see your results. Instead of the top 5 results, you will now get the top 10!

This is just one example of customization. There are many variations, but here are a couple of ideas to get your creative juices flowing:

* You might decide that you want to take a model, grab the params as a checkpoint and retrain it using additional training images. This is often called fine tuning a model. A fine tuning tutorial using MXNet with Gluon can be found in [The Straight Dope's computer vision section](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter08_computer-vision/fine-tuning.ipynb).

* You also might decide that you like two models and you want to combine the model archive to contain both and setup the configuration so that the API that is served now has the features of both models. More info on serving multiple models is in the [Running the Model Server](server.md) page.

## Artifact Details

### Model Archive Overview

Model archives have the following artifacts:

```
<Model Name>-symbol.json
<Model Name>-<Epoch>.params
signature.json
<Service File>.py
MANIFEST.json
manifest-schema.json
assets/
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

1. **input**: Contains MXNet model input names and input shapes. It is a list contains { data_name : name, data_shape : shape } maps. Client side inputs should have the same order with the input order defined here.
1. **input_type**: Defines the MIME content type for client side inputs. Currently all inputs must have the same content type. Only two MIME types are currently supported: "image/jpeg" and "application/json".
1. **output**: Similar to input, it contains MXNet model output names and output shapes.
1. **output_type**: Similar to input_type. Currently all outputs must have the same content type. Only two MIME types are currently supported: "image/jpeg" and "application/json".

Using the squeezenet_v1.1 example, you can view the `signature.json` file in the folder that was extracted once you dowloaded and served the model for the first time. The input is an image with 3 color channels and size 224 by 224. The output is named 'softmax' with length 1000 (one for every class that the model can recognize).

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

```python
class MXNetBaseService(SingleNodeService):
  def __init__(self, path, synset=None, ctx=mx.cpu()):

  def _inference(self, data):

  def _preprocess(self, data):

  def _postprocess(self, data, method='predict'):
```

Further details and specifications are found on the [custom service](custom_service.md) page.

### Manifest
```
MANIFEST.json
```

```json
{
    "Model-Archive-Version": 0.1,
    "Model-Archive-Description": "Resnet-18",
    "License": "Apache 2.0",
    "Created-By": {
        "Model-Server": 0.2,
        "Author": "mxnet-sdk",
        "Author-Email": "mxnet-sdk-dev@amazon.com"
    },
    "Model": {
        "Parameters": "resnet-18-0000.params",
        "Symbol": "resnet-18-symbol.json",
        "Signature": "signature.json",
        "Description": "Resnet-18",
        "Model-Format": "MXNet-Symbolic",
        "Service": "mxnet_vision_service.py"
    },
    "Engine": {
        "MXNet": "0.12.1"
    }
}
```

### Manifest Schema
```
manifest-schema.json
```

```json
{
    "$schema": "http://json-schema.org/schema#",
    "type": "object",
    "definitions": {
        "Created-By": {
            "type": "object",
            "properties": {
                "Model-Server": {"type": "number"},
                "Author": {"type": "string"},
                "Author-Email": {"type": "string"}
            },
            "required": ["Model-Server"]
        },
        "Model": {
            "type": "object",
            "properties": {
                "Parameters": {"type": "string"},
                "Symbol": {"type": "string"},
                "Signature": {"type": "string"},
                "Description": {"type": "string"},
                "Model-Format": {"type": "string"},
                "Model-Name": {"type": "string"}
            },
            "required": ["Parameters", "Symbol", "Signature", "Model-Format", "Model-Name"]
        },
        "Service-Files": {
            "type": "object",
            "properties": {
                "File-Name": {"type": "string"},
                "Description": {"type": "string"}
            },
            "required": ["File-Name"]
        },
        "Engine": {
            "type": "object"
        },
        "Assets": {
            "type": "array"
        }
    },
    "properties": {
        "Model-Archive-Version": {"type": "number"},
        "Model-Archive-Description": {"type": "string"},
        "License": {"type": "string"},
        "Created-By": { "$ref": "#/definitions/Created-By" },
        "Model": { "$ref": "#/definitions/Model" },
        "Engine": { "$ref": "#/definitions/Engine" }
    },
    "required": ["Model-Archive-Version", "License", "Created-By", "Model", "Engine"]
}
```

### Labels (synset.txt)
```
assets/synset.txt
```
This optional text file is for classification labels. Simply put, if it were for MNIST, it would be 0 through 9 where each number is on its own line. For a more complex example take a look at the [synset for Imagenet-11k](https://github.com/tornadomeet/ResNet/blob/master/predict/synset.txt).


If `synset.txt` is included in exported archive file and each line represents a category, `MXNetBaseModel` will load this file and create `labels` attribute automatically. If this file is named differently or has a different format, you need to override `__init__` method and manually load it.


## Using Your Own Trained Models and Checkpoints

While all of these features are super exciting you've probably been asking yourself, so how do I create these fabulous MMS model files for my own trained models? We'll provide some MXNet code examples for just this task.

There are two main routes for this: 1) export a checkpoint or use the new `.export` function, or 2) using a MMS Python class to export your model directly.

The Python method to export model is to use `export_serving` function while completing training:

```python
   import mxnet as mx
   from mms.export_model import export_serving

   mod = mx.mod.Module(...)
   # Training process
   ...

   # Export model
   signature = { "input_type": "image/jpeg", "output_type": "application/json" }
   export_serving(mod, 'resnet-18', signature, aux_files=['synset.txt'])
```

In MXNet with version higher than 0.12.0, you can export a Gluon model directly, as long as your model is Hybrid:

```python
from mxnet import gluon
net = gluon.nn.HybridSequential() # this mode will allow you to export the model

with net.name_scope():
    net.add(gluon.nn.Dense(128, activation="relu")) # an example first layer
    # Add the rest of your network architecture here

net.hybridize() # hybridize your network so that it can be exported

# Then train your network before moving on to exporting

signature = {
                "input_type": "application/json",
                "inputs" : [
                    {
                        "data_name": "data",
                        "data_shape": [1, 100]
                    }
                ],
                "outputs" : [
                    {
                        "data_name": "softmax",
                        "data_shape": [1, 128]
                    }
                ],
                "output_type": "application/json"
            }

export_serving(net, 'gluon_model', signature, aux_files=['synset.txt'])
```

**Note**: be careful with versions. If you export a v0.12 model and try to run it with MMS running v0.11 of MXNet, the server will probably throw errors and you won't be able to use the model.


## MMS Export Command Line Interface

Now let's cover the details on using `mxnet-model-export`. This CLI can take model checkpoints and package them into a `.model` file that can then be redistributed and served by anyone using MMS.

Example usage with the squeezenet_v1.1 model you may have downloaded or exported in the [main README's](../README.md) examples:

```bash
mxnet-model-export --model-name squeezenet_v1.1 --model-path models/squeezenet_v1.1
```

### Arguments

```bash
mxnet-model-export -h
usage: mxnet-model-export [-h] --model-name MODEL_NAME --model-path MODEL_PATH

MXNet Model Export

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Exported model name. Exported file will be named as
                        model-name.model and saved in current working
                        directory.
  --model-path MODEL_PATH
                        Path to the folder containing model related files.
                        Signature file is required
```

1. model-name: required, prefix of exported model archive file.
2. model-path: required, directory which contains files to be packed into exported archive.
