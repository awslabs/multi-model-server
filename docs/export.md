# Exporting Models for Use with MMS

A key feature of MMS is the ability to export all model artifacts into a single model archive file. It is a separate CLI that takes in model artifacts: network definitions in the form of a JSON file, the trained network weight in the form of a parameters file, the description of the models' inputs and outputs in the form of a signature JSON file, and other optional files that may be required to serve the model. The CLI creates a `.model` file that MMS's server CLI uses to serve the models.

## Technical Details

When you export a model in MXNet, you will have a `model-symbol.json` file (1), which describes the neural network, and a larger `model-0000.params` file containing the parameters and their weights (2). In addition to these two files, for MMS to work with your model, you must provide a `signature.json` file (3), which describes your inputs and your outputs. You also have *the option* of providing labels for the outputs in a `synset.txt` file (4), as well as other files such as a custom service code. For the purpose of a quick example, we'll pretend that you've already saved a checkpoint which gives you the first two assets by providing those files for you to download, or that you've acquired the trained models from a [model zoo](model_zoo.md). We'll also provide the latter two files that you would create on your own based on the model you're trying to serve. Don't worry if that sounds ominous; creating those last two files is easy. More details on this can be found in later the **Required Assets** section.

The files in the `model-example.model` file are human-readable in a text editor, with the exception of the `.params` file: this file is binary, and is usually quite large. Download and extract the provided model file. It is a zip file under the hood, so if you have trouble extracting it, change the extension to .zip first and then extract it.

* [model-example.model](https://s3.amazonaws.com/model-server/models/model-example/model-example.model) - contains the following four files
* [squeezenet_v1.1-symbol.json](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-symbol.json) - contains the layers and overall structure of the neural network; the name, or prefix, here is "squeezenet_v1.1"
* [squeezenet_v1.1-0000.params](https://s3.amazonaws.com/model-server/models/model-example/squeezenet_v1.1-0000.params) - contains the parameters and the weights; again, the prefix is "squeezenet_v1.1"
* [signature.json](https://s3.amazonaws.com/model-server/models/model-example/signature.json) - defines the inputs and outputs that MMS is expecting to hand-off to the API
* [synset.txt](https://s3.amazonaws.com/model-server/models/model-example/synset.txt) - an *optional* list of labels (one per line)

Given these files you can use the `mxnet-model-export` CLI to generate a `.model` file that can be used with MMS. This file is essentially a zip archive, so changing the extension from `.model` to `.zip` will let you manually extract the files from any MMS model file.

To try this out, open your terminal and go to the folder you just extracted. Using the zip file and its directory structure can help you keep things organized. In this next example we'll go into the `model-example` folder and run `mxnet-model-export`. We're going to tell it our model's prefix is `squeezenet_v1.1` with the `model-name` argument. Then we're giving it the `model-path` to the model's assets, which are in the current working directory, so we'll use `.` for the path.

```bash
cd model-example
mxnet-model-export --model-name squeezenet_v1.1 --model-path .
```

This will output `squeezenet_v1.1.model` in the current working directory.

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

### Required Assets

#### Assets Overview
In order for the model file to be created, you need to provide these three or four assets:

1. signature.json - required; the inputs and outputs of the model and the service inputs and outputs
1. name-symbol.json - required; the model's definition (layers, etc.); name is any prefix you give it
1. name-0000.params - required; the model's hyper-parameters and weights; name must match the name from the previous JSON file
1. synset.txt - optional; a list of prediction classes IDs and names

**signature.json**

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

**name-symbol.json**

  This is the model's definition in JSON format. You can name it whatever you want, using a consistent prefix. The pattern expected is `my-awesome-network-symbol.json` or `mnist-symbol.json` so that when you use `mxnet-model-export` you're passing in the prefix and it'll look for prefix-symbol.json. You can generate this file in a variety of ways, but the easiest for MXNet is to use the `.export` feature or the `mms.export_model` method described later.

**name-0000.params**

  This is the model's hyper-parameters and weights. It will be created when you use MXNet's `.export` feature or the `mms.export_model` method described later.

**synset.txt**

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

Note: be careful with versions. If you export a v0.12 model and try to run it with MMS running v0.11 of MXNet, the server will probably throw errors and you won't be able to use the model.
