# Tensor Flow Saved Model Inference Service

In this example, we show how to use a pre-trained Tensorflow MobileNet V2 model in the saved model format for performing real time inference using MMS

# Objective

1. Demonstrate how to package a a pre-trained TensorFlow saved model in MMS
2. Demonstrate how to create custom service with pre-processing and post-processing

# Pre-requisite
Install tensorflow

```
pip install tensorflow==1.15
```

## Step 1 - Download the pre-trained MobileNet V2 Model

You will need the model files to use for the export. Check this example's directory in case they're already downloaded. Otherwise, you can `curl` the files or download them via your browser:

```bash
cd multi-model-server/examples/tf_vision

curl -O http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
tar -xvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
cp ssd_mobilenet_v1_coco_2017_11_17/saved_model/saved_model.pb .
```


## Step 2 - Prepare the signature file

Define model input name and shape in `signature.json` file. The signature for this example looks like below:

```json
{
  "inputs": [
    {
      "data_precision": "UINT8",
      "data_name": "inputs",
      "data_shape": [
        1,
        224,
        224,
        3
      ]
    }
  ]
}
```

## Step 3 - Create custom service class

We provid a custom service class template code in this folder:
1. [model_handler.py](./model_handler.py) - A generic based service class.
2. [tensorflow_saved_model_service.py](./tensorflow_saved_model_service.py) - A Tensorflow saved model base service class.
3. [tensorflow_vision_service.py](./tensorflow_vision_service.py) - A Tensorflow Vision service class.
4. [image.py](./image.py) - Utils for reshaping

In this example, you can simple use the provided tensorflow_vision_service.py as user model archive entry point.

## Step 4 - Package the model with `model-archiver` CLI utility

In this step, we package the following:
1. pre-trained TensorFlow Saved Model we downloaded in Step 1.
2. signature.json file we prepared in step 2.
3. custom model service files we mentioned in step 3.

We use `model-archiver` command line utility (CLI) provided by MMS.
Install `model-archiver` in case you have not:

```bash
pip install model-archiver
```

This tool create a .mar file that will be provided to MMS for serving inference requests. In following command line, we specify 'tensorflow_vision_service:handle' as model archive entry point.

```bash
cd multi-model-server/examples
model-archiver --model-name mobilenetv2 --model-path tf_vision --handler tensorflow_vision_service:handle
```

## Step 5 - Start the Inference Service

Start the inference service by providing the 'mobilenetv2.mar' file we created in Step 4.

MMS then extracts the resources (signature, saved model) we have packaged into .mar file and uses the extended custom service, to start the inference server.

By default, the server is started on the localhost at port 8080.

```bash
cd multi-model-server
multi-model-server --start --model-store examples --models ssd=mobilenetv2.mar
```

Awesome! we have successfully exported a pre-trained TF saved model model, extended MMS with custom preprocess/postprocess and started a inference service.

**Note**: In this example, MMS loads the .mar file from the local file system. However, you can also store the archive (.mar file) over a network-accessible storage such as AWS S3, and use a URL such as http:// or https:// to indicate the model archive location. MMS is capable of loading the model archive over such URLs as well.

## Step 6 - Test sample inference

Let us try the inference server we just started. Open another terminal on the same host. Download a sample image, or try any jpeg.

You can also use this image of three dogs on a beach.
![3 dogs on beach](../../docs/images/3dogs.jpg)

Use curl to make a prediction call by passing the downloaded image as input to the prediction request.

```bash
cd multi-model-server
curl -X POST http://127.0.0.1:8080/predictions/ssd -T docs/images/3dogs.jpg
```
