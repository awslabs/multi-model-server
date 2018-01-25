# Export an ONNX Model
[![onnx](images/onnx_logo_50.png)](http://onnx.ai)
ONNX model serving with MMS is very simple. You can download a model from the [ONNX Model Zoo](https://github.com/onnx/models) then use `mxnet-model-export` to covert it to a `.model` file.

**Note**: Some ONNX model authors upload their models to the zoo in the `.pb` or `.pb2` format. Just change the extension to `.onnx` before attempting an export.

Let's use the SqueezeNet ONNX model as an example. To export a model for MMS, recall that you also need to download or create a signature file, and optionally provide a labels file (synset.txt). So here you will utilize the following three files:
* [SqueezeNet ONNX model](https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.onnx): a `.onnx` model file from the [ONNX Model Zoo](https://github.com/onnx/models)
* [signature file](https://s3.amazonaws.com/model-server/models/onnx-squeezenet/signature.json): defines the input layer and the outputs
* [label file](https://s3.amazonaws.com/model-server/models/onnx-squeezenet/synset.txt): has the labels for 1,000 ImageNet classes

Create a new directory and download the files:

```bash
mkdir onnx-squeezenet && cd onnx-squeezenet
curl -O https://s3.amazonaws.com/model-server/models/onnx-squeezenet/squeezenet.onnx
curl -O https://s3.amazonaws.com/model-server/models/onnx-squeezenet/signature.json
curl -O https://s3.amazonaws.com/model-server/models/onnx-squeezenet/synset.txt
```

Since the model has the `.onnx` extension, it will be detected and the managed accordingly using the [onnx-mxnet converter](https://github.com/onnx/onnx-mxnet). The converter comes preinstalled with MMS v0.2 or later.

Now you can use the typical export command to output `onnx-squeezenet.model` in the current working directory.

```bash
mxnet-model-export --model-name onnx-squeezenet --model-path .
```

Now start the server:

```bash
mxnet-model-server --models squeezenet=onnx-squeezenet.model
```

After your server starts, you can use the following command to see the prediction results.

```bash
curl -X POST http://127.0.0.1:8080/squeezenet/predict -F "input_0=@kitten.jpg"
```

**Note**: The data name for ONNX models is `input_0` instead of `data` as with other examples. Make sure you update this in your `curl` call. The reason is that ONNX models' input layer defaults to `input_0`, and you can't override this in your `signature.json` file when you define `data_name`. You can compare this by looking at the signature file you downloaded for this ONNX example, versus the signature you downloaded for an MXNet model example.
